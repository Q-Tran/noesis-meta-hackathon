"""Pluggable LLM interface: null | openai | ollama.

`complete()` returns free text; `extract()` returns a validated Pydantic model
via JSON. The `null` provider keeps everything runnable fully offline (returns
empty text / empty structures so callers fall back to heuristics).
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import httpx
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from config import settings

T = TypeVar("T", bound=BaseModel)


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    # strip ```json fences if present
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        # try to locate the first {...} block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
        return {}


class BaseLLM:
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        raise NotImplementedError

    def rerank(self, query: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
        if not settings.enable_reranking or not candidates:
            return candidates
        try:
            from sentence_transformers import CrossEncoder

            if not hasattr(self, "_reranker"):
                model_path = Path(settings.reranker_model_path).expanduser()
                if not model_path.is_absolute():
                    # Model weights live in the repo root; resolve against this
                    # file's directory so engines run from any CWD (starting
                    # inside demo/ is exactly how the weights go "missing").
                    model_path = Path(__file__).resolve().parent / model_path
                if not model_path.resolve().is_dir():
                    raise RuntimeError(
                        f"Local reranker model is missing at {model_path.resolve()}. "
                        "Run scripts/download_reranker_model.py or set RERANKER_MODEL_PATH."
                    )
                self._reranker = CrossEncoder(str(model_path.resolve()), local_files_only=True)

            pairs = [(query, str(candidate.get(text_key) or "")) for candidate in candidates]
            scores = self._reranker.predict(pairs)
            scored = []
            for candidate, score in zip(candidates, scores):
                updated = dict(candidate)
                updated["rerank_score"] = float(score)
                scored.append(updated)
            scored.sort(key=lambda item: item["rerank_score"], reverse=True)
            return scored
        except Exception as e:
            print(f"DEBUG: Reranking failed: {e}")
            return candidates

    def rerank_candidate_count(self, top_k: int, *, dedupe: bool = False) -> int:
        if dedupe:
            return max(top_k, settings.dedupe_rerank_candidates)
        candidate_k = max(top_k * settings.rerank_candidate_multiplier, settings.rerank_min_candidates)
        return min(candidate_k, settings.rerank_max_candidates)

    def embed(self, text: str) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, "_encoder"):
                model_path = Path(settings.embedding_model_path).expanduser()
                if not model_path.is_absolute():
                    # Weights live in the repo root; resolve against this
                    # file's directory so engines run from any CWD.
                    model_path = Path(__file__).resolve().parent / model_path
                if not model_path.resolve().is_dir():
                    raise RuntimeError(
                        f"Local embedding model is missing at {model_path.resolve()}. "
                        "Run scripts/download_embedding_model.py or set EMBEDDING_MODEL_PATH."
                    )
                self._encoder = SentenceTransformer(
                    str(model_path.resolve()), local_files_only=True
                )
            
            embedding = self._encoder.encode(text)
            return embedding.tolist()
        except Exception as e:
            # Fallback to a deterministic pseudo-embedding using a simple hash trick
            # This allows the graph to still cluster similar strings while offline
            import hashlib
            dim = settings.embedding_dimension
            vec = [0.0] * dim
            
            # Simple feature hashing (hashing trick) on words
            words = text.lower().split()
            if not words:
                return vec
                
            for word in words:
                # Hash the word to a bucket
                h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
                bucket = h % dim
                sign = 1 if (h // dim) % 2 == 0 else -1
                vec[bucket] += sign
            
            # L2 normalize
            magnitude = sum(x*x for x in vec) ** 0.5
            if magnitude > 0:
                vec = [x / magnitude for x in vec]
                
            return vec

    def extract(self, prompt: str, model: Type[T], system: Optional[str] = None) -> T:
        schema = json.dumps(model.model_json_schema())
        sys = (system or "You extract structured data.") + (
            f"\nReturn ONLY valid JSON matching this JSON schema:\n{schema}"
        )
        raw = self.complete(prompt, system=sys)
        data = _parse_json(raw)
        try:
            return model.model_validate(data)
        except Exception as e:
            print(f"DEBUG: Validation failed: {e}")
            return model()  # empty/default instance keeps callers safe


class NullLLM(BaseLLM):
    """Offline no-op provider."""

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        return ""


class OpenAILLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        
        # A local server (llama.cpp, Strata, Ollama) can be busy with agent traffic and
        # take minutes for one request. The client default is read=600s, which kills the
        # request while the server is still generating. LLM_TIMEOUT raises it; 0 means
        # no client-side timeout at all, so the server decides when a request ends.
        timeout = settings.llm_timeout if settings.llm_timeout > 0 else httpx.Timeout(None)
        if settings.openai_base_url:
            self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                                 timeout=timeout)
        else:
            self.client = OpenAI(api_key=settings.openai_api_key, timeout=timeout)

        self.model = settings.openai_model

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""


class OllamaLLM(BaseLLM):
    def __init__(self):
        import ollama

        self.client = ollama.Client(host=settings.ollama_host)
        self.model = settings.ollama_model

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat(model=self.model, messages=messages)
        return resp["message"]["content"]


class LiteLLMProvider(BaseLLM):
    def __init__(self):
        import litellm
        self.litellm = litellm
        self.model = settings.litellm_model

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.litellm.completion(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""  # type: ignore


# The gateway fails transiently often enough that one attempt is not enough,
# and rarely enough that three is plenty.
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.0


def get_llm() -> BaseLLM:
    provider = (settings.llm_provider or "null").lower()
    if provider == "openai":
        return OpenAILLM()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "litellm":
        return LiteLLMProvider()
    # The "att" provider from the author's deployment is excluded from this
    # public snapshot: it fetched OAuth tokens with real credentials.
    return NullLLM()
