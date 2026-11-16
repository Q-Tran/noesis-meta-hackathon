"""
LOCOMO Benchmark Runner (Noesis)
================================

Adapted from mem0ai/memory-benchmarks (benchmarks/locomo/run.py). Runs the
same LOCOMO-10 benchmark (Snap Research, ACL 2024) and J-score judge
methodology, but against Noesis's own graph-native evidence pipeline
(`benchmarks.common.noesis_client.NoesisClient`) instead of Mem0.

Flow (per conversation):
    1. Parse into chronological sessions/turns.
    2. Ingest each turn as a `Claim` via FeedbackRepository, scoped by user_id.
    3. For each question (categories 1-4; 5=adversarial is excluded):
        - hybrid_search Noesis -> retrieved claims
        - Generate answer (Noesis's configured LLM_PROVIDER)
        - Judge answer vs ground truth (same LLM)
    4. Compute metrics (by category, by cutoff) and write a unified JSON.

Usage:
    python -m benchmarks.locomo.run --project-name smoke --conversations 0 --max-questions 5
    python -m benchmarks.locomo.run --project-name full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.common.metrics import compute_locomo_metrics, display_results
from benchmarks.common.noesis_client import NoesisClient, format_search_results
from benchmarks.common.utils import Checkpoint, cutoff_label, parse_cutoffs, save_result_json

from .prompts import (
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    JudgeResult,
    get_answer_generation_prompt,
    get_judge_prompt,
    preprocess_answer,
)

DEFAULT_DATASET_PATH = "benchmarks_data/locomo10.json"


# ===============================================================================
# DATASET / CONVERSATION PARSING (unchanged from mem0ai/memory-benchmarks)
# ===============================================================================


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_locomo_date(date_str: str) -> datetime | None:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def locomo_date_to_epoch(date_str: str) -> int | None:
    parsed = parse_locomo_date(date_str)
    if parsed:
        return int(parsed.replace(tzinfo=timezone.utc).timestamp())
    return None


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []
    for key in session_keys:
        date_key = f"{key}_date_time"
        date_str = conversation.get(date_key, "")
        turns = conversation[key]
        paired.append((key, date_str, turns))

    def sort_key(item: tuple) -> tuple:
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def session_to_messages(turns: list[dict], speaker_a: str, speaker_b: str) -> list[dict]:
    messages = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"
        else:
            photo_tag = ""
        if photo_tag:
            text = f"{text} {photo_tag}" if text else photo_tag
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"{speaker}: {text}"})
    return messages


# ===============================================================================
# INGESTION
# ===============================================================================


async def ingest_conversation(
    conv_idx: int,
    entry: dict,
    client: NoesisClient,
    run_id: str,
) -> tuple[str, int]:
    """Ingest every turn of a LOCOMO conversation into Noesis, one Claim per turn.

    Returns (user_id, total_turns_ingested).
    """
    conversation = entry["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    user_id = f"locomo_{conv_idx}_{run_id}"

    sorted_sessions = get_sorted_sessions(conversation)
    total = 0
    pbar = tqdm(sorted_sessions, desc=f"Ingest conv {conv_idx}", leave=True)
    for session_key, date_str, turns in pbar:
        messages = session_to_messages(turns, speaker_a, speaker_b)
        if not messages:
            continue
        session_epoch = locomo_date_to_epoch(date_str)
        # One add() call per turn keeps each Claim's timestamp accurate
        # (a whole session sharing one timestamp would blur temporal-reasoning
        # answers), mirroring the CHUNK_SIZE=1 default upstream uses.
        for message in messages:
            resp = await client.add([message], user_id, timestamp=session_epoch)
            if resp is not None:
                total += 1
    return user_id, total


# ===============================================================================
# SEARCH + ANSWER + JUDGE
# ===============================================================================


async def process_question(
    qa: dict,
    qa_idx: int,
    conv_idx: int,
    user_id: str,
    client: NoesisClient,
    llm: Any,
    cutoffs: list[int],
    top_k: int,
    reference_date_human: str | None,
    predict_only: bool,
) -> dict[str, Any]:
    question_id = f"conv{conv_idx}_q{qa_idx}"
    question = qa["question"]
    category = qa["category"]
    answer = str(qa["answer"])

    start = time.monotonic()
    search_results = await client.search(question, user_id, top_k=top_k)
    search_latency = (time.monotonic() - start) * 1000

    formatted = format_search_results(search_results)

    result: dict[str, Any] = {
        "question_id": question_id,
        "conversation_idx": conv_idx,
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "unknown"),
        "question": question,
        "ground_truth_answer": answer,
        "evidence": qa.get("evidence", []),
        "user_id": user_id,
        "reference_date": reference_date_human,
        "retrieval": {
            "search_query": question,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1),
            "total_results": len(formatted),
        },
    }

    if predict_only:
        return result

    cutoff_results: dict[str, dict] = {}
    processed_answer = preprocess_answer(category, answer)

    for c in cutoffs:
        sliced = formatted[:c]
        label = cutoff_label(c)

        gen_prompt = get_answer_generation_prompt(question, sliced, reference_date=reference_date_human)
        generated_answer = await asyncio.to_thread(llm.complete, gen_prompt)
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        judge_prompt = get_judge_prompt(question, processed_answer, generated_answer)
        raw = await asyncio.to_thread(llm.extract, judge_prompt, JudgeResult, JUDGE_SYSTEM_PROMPT)

        correct = raw.label.upper() == "CORRECT"
        score = 1.0 if correct else 0.0
        judgment = "CORRECT" if correct else "WRONG"

        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": generated_answer,
            "memories_evaluated": len(sliced),
            "reason": raw.reasoning,
        }

    result["cutoff_results"] = cutoff_results
    return result


# ===============================================================================
# CLI
# ===============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOCOMO-10 against Noesis's evidence pipeline")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-k-cutoffs", default="5,10,20,50")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="benchmarks/results/locomo")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--categories", default="1,2,3,4")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions per conversation")
    parser.add_argument("--cleanup", action="store_true", help="Delete ingested Claims for this run when done")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    cutoffs = parse_cutoffs(args.top_k_cutoffs)
    categories = [int(c) for c in args.categories.split(",")]
    conv_indices = [int(c) for c in args.conversations.split(",")]

    run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"LOCOMO (Noesis) | project={args.project_name} run_id={run_id}")
    print(f"  Conversations: {args.conversations}")
    print(f"  Cutoffs: {args.top_k_cutoffs}")

    dataset = load_dataset(args.dataset_path)

    from llm import get_llm

    llm = get_llm()
    checkpoint = Checkpoint(output_dir)

    client = NoesisClient(max_workers=args.max_workers)
    all_evaluations: list[dict] = []
    ingested_user_ids: list[str] = []

    try:
        for conv_idx in conv_indices:
            if conv_idx >= len(dataset):
                print(f"  Conversation {conv_idx} out of range, skipping")
                continue
            entry = dataset[conv_idx]
            conversation = entry["conversation"]

            user_id, n_turns = await ingest_conversation(conv_idx, entry, client, run_id)
            ingested_user_ids.append(user_id)
            print(f"  Conversation {conv_idx}: ingested {n_turns} turns (user_id={user_id})")

            sorted_sessions = get_sorted_sessions(conversation)
            ref_date_human = sorted_sessions[-1][1] if sorted_sessions else None

            questions = entry.get("qa", entry.get("qa_pairs", []))
            conv_questions = [
                (qi, qa) for qi, qa in enumerate(questions) if qa.get("category") in categories
            ]
            if args.max_questions is not None:
                conv_questions = conv_questions[: args.max_questions]

            sem = asyncio.Semaphore(args.max_workers)
            pbar = tqdm(total=len(conv_questions), desc=f"Questions conv {conv_idx}")

            async def run_one(qi: int, qa: dict) -> dict:
                qid = f"conv{conv_idx}_q{qi}"
                cached = checkpoint.load(qid)
                if cached is not None:
                    pbar.update(1)
                    return cached
                async with sem:
                    result = await process_question(
                        qa=qa,
                        qa_idx=qi,
                        conv_idx=conv_idx,
                        user_id=user_id,
                        client=client,
                        llm=llm,
                        cutoffs=cutoffs,
                        top_k=args.top_k,
                        reference_date_human=ref_date_human,
                        predict_only=args.predict_only,
                    )
                checkpoint.save(qid, result)
                pbar.update(1)
                return result

            conv_results = await asyncio.gather(*[run_one(qi, qa) for qi, qa in conv_questions])
            pbar.close()
            all_evaluations.extend(conv_results)

        if not args.predict_only and all_evaluations:
            has_cutoffs = any("cutoff_results" in e for e in all_evaluations)
            if has_cutoffs:
                metrics = compute_locomo_metrics(all_evaluations, cutoffs)
                display_results(metrics, cutoffs)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unified_path = os.path.join(args.output_dir, f"locomo_results_{timestamp}.json")
                save_result_json(unified_path, {
                    "metadata": {
                        "benchmark": "locomo",
                        "system": "noesis",
                        "project_name": args.project_name,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "top_k": args.top_k,
                        "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                        "total_questions": len(all_evaluations),
                        "categories": categories,
                        "conversations": conv_indices,
                    },
                    "metrics_by_cutoff": metrics,
                    "evaluations": all_evaluations,
                })
                print(f"\nResults saved to: {unified_path}")

        print(f"\nTotal questions processed: {len(all_evaluations)}")
    finally:
        if args.cleanup:
            for uid in ingested_user_ids:
                await client.delete_user(uid)
            print(f"  Cleaned up {len(ingested_user_ids)} conversation(s)")
        await client.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
