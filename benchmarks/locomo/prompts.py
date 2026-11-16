"""LOCOMO benchmark prompts.

Vendored (with light trims) from mem0ai/memory-benchmarks
(benchmarks/locomo/prompts.py), which uses the industry-standard J-score
methodology: a binary LLM judge (CORRECT/WRONG) on categories 1-4.

Category mapping (unchanged from the LOCOMO-10 dataset, Snap Research,
ACL 2024):
    1 = multi-hop
    2 = temporal reasoning
    3 = open-domain
    4 = single-hop
    5 = adversarial (excluded from scoring)
"""

from __future__ import annotations

from datetime import datetime as _datetime

from pydantic import BaseModel

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]


class JudgeResult(BaseModel):
    label: str = "WRONG"
    reasoning: str = ""


ANSWER_GENERATION_PROMPT = """You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity.

## Step 3: COMBINE AND CROSS-REFERENCE
Combine facts from multiple memories about the same topic. For listing/counting questions, extract EVERY distinct item from ALL memories.

## Step 4: SELECT THE BEST ANSWER
Do NOT assume the highest-ranked memory is correct. ALWAYS choose the MOST SPECIFIC detail available.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024. Calculate time relative to this date, NOT today.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found relevant items during reasoning that you're tempted to exclude, include them unless you have STRONG evidence they are wrong.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified" or "not mentioned" if ANY memory contains relevant information.

{memories}

Question: {question}

Work through Steps 1-7, then give your final answer after "ANSWER:".
"""

ANSWERER_MEMORY_LIMIT = 200


def _to_human_date(iso_str: str) -> str:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _datetime.strptime(iso_str[:26].rstrip("Z"), fmt.replace("%z", "")).strftime(
                "%A, %B %d, %Y"
            )
        except ValueError:
            continue
    return iso_str[:10]


def get_answer_generation_prompt(
    question: str,
    search_results: list,
    reference_date: str | None = None,
) -> str:
    if reference_date is None:
        reference_date = "2023"

    if not search_results:
        memories_text = "(No relevant memories found)"
    else:
        top_results = search_results[:ANSWERER_MEMORY_LIMIT]
        sorted_results = sorted(top_results, key=lambda x: x.get("created_at", "") or "")
        lines = ["The following memories are presented in chronological order (oldest to newest).", ""]
        for result in sorted_results:
            memory = result.get("memory", "")
            created_at = result.get("created_at", "")
            if created_at:
                date_str = _to_human_date(created_at)
                lines.append(f"({date_str}) {memory}")
            else:
                lines.append(f"(unknown date) {memory}")
        memories_text = "\n".join(lines)

    return ANSWER_GENERATION_PROMPT.format(
        memories=memories_text,
        question=question,
        reference_date=reference_date,
    )


JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

_JUDGE_TEMPLATE = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT.
2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT.
3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT.
4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT.
5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer.
6. **SAME REFERENT**: If the generated answer mentions the same named entity/concept as the gold answer, mark CORRECT.
7. **FOCUS ON KNOWLEDGE, NOT WORDING**: Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG)."""


def get_judge_prompt(question: str, answer: str, response: str) -> str:
    return _JUDGE_TEMPLATE.format(question=question, answer=answer, response=response)


def preprocess_answer(category: int, answer: str) -> str:
    """Category 3 (open-domain): use only the first part before semicolon."""
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return answer
