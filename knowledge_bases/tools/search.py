"""Search the graph-registered `Tool` catalog by capability/need.

Every Noesis tool is already fully discoverable through the MCP protocol's
own `list_tools()` (see `knowledge_bases.mcp.load_tool_definitions`) -- but that
returns the *entire* catalog with no way to ask "what can help me do X?".
This is that question, answered in three stages:

1. **Keyword** overlap of the need with each skill's `name`/`key`/
   `description` (deterministic, always available).
2. **Semantic** similarity against the `noesis_skills` vector index
   (`knowledge_bases.tools.catalog_index`), so paraphrases that share no
   token with a description still surface. Skipped silently without
   Milvus/an embedding model.
3. **Rerank** of the fused candidate pool with the same cross-encoder
   `hybrid_search` uses, when available.

Every Noesis skill is graph metadata pointing at code already deployed
everywhere a harness's `noesis-orchestrator` process runs (see
`knowledge_bases/tools/resolver.py`) -- so once a matching skill is found,
there is nothing further to "install": it's already callable through
`call_noesis_tool`/the tool's own MCP entry with no client-side setup step.
"""
from __future__ import annotations

import re
from typing import Any

from graph import Graph
from knowledge_bases.tools.repository import ToolRepository

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "by", "can", "do", "for", "from", "how", "i",
    "in", "is", "it", "me", "my", "need", "of", "on", "or", "should", "that",
    "the", "this", "to", "use", "want", "what", "with",
}
RRF_K = 60


def search_tokens(text: str) -> set[str]:
    return {
        token for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _short_name(skill_key: str, knowledge_base: str) -> str:
    prefix = f"{knowledge_base}."
    return skill_key[len(prefix):] if skill_key.startswith(prefix) else skill_key


def _score_skill(need_tokens: set[str], skill: Any) -> tuple[float, list[str]]:
    name_tokens = search_tokens(skill.name)
    description_tokens = search_tokens(skill.description)
    key_tokens = search_tokens(skill.key.replace(".", " ").replace("_", " "))

    name_matches = sorted(need_tokens & name_tokens)
    description_matches = sorted(need_tokens & description_tokens)
    key_matches = sorted(need_tokens & key_tokens)

    score = len(name_matches) * 3 + len(key_matches) * 2 + len(description_matches)
    reasons = []
    if name_matches:
        reasons.append("name:" + ",".join(name_matches))
    if key_matches:
        reasons.append("key:" + ",".join(key_matches))
    if description_matches:
        reasons.append("description:" + ",".join(description_matches))
    return float(score), reasons


def _skill_allowed(skill_key: str, patterns: list[str] | None) -> bool:
    if patterns is None:
        return True
    return any(
        pattern == skill_key
        or (pattern.endswith(".*") and skill_key.startswith(pattern[:-1]))
        for pattern in patterns
    )


def rank_skills(
    need: str,
    skills: list[Any],
    semantic: dict[str, float] | None = None,
    llm: Any | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fuse keyword and semantic rankings (reciprocal rank fusion), then
    rerank with the cross-encoder when one is available. Pure function of
    its inputs so it is unit-testable without Milvus or a model."""
    need_tokens = search_tokens(need)
    keyword: dict[str, tuple[float, list[str]]] = {}
    for skill in skills:
        score, reasons = _score_skill(need_tokens, skill)
        if score > 0:
            keyword[skill.key] = (score, reasons)
    semantic = {key: sim for key, sim in (semantic or {}).items() if sim > 0}

    fused: dict[str, float] = {}
    for rank, key in enumerate(sorted(keyword, key=lambda k: -keyword[k][0]), start=1):
        fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
    for rank, key in enumerate(sorted(semantic, key=lambda k: -semantic[k]), start=1):
        fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)

    by_key = {skill.key: skill for skill in skills}
    candidates = []
    for key in sorted(fused, key=lambda k: -fused[k]):
        skill = by_key.get(key)
        if skill is None:
            continue
        kw_score, reasons = keyword.get(key, (0.0, []))
        if key in semantic:
            reasons = reasons + [f"semantic:{semantic[key]:.2f}"]
        candidates.append({
            "tool_name": _short_name(skill.key, skill.knowledge_base),
            "tool_key": skill.key,
            # The name this field had when a tool was called a skill. Kept so a
            # caller that already reads `skill_key` keeps working; new callers
            # should read `tool_key`.
            "skill_key": skill.key,
            "knowledge_base": skill.knowledge_base,
            "name": skill.name,
            "description": skill.description,
            "input_schema": skill.input_schema,
            "score": round(fused[key], 6),
            "keyword_score": kw_score,
            "similarity": semantic.get(key),
            "reasons": reasons,
            "text": f"{skill.name}. {skill.description}",
        })
    limit = max(1, min(limit, 50))
    pool = candidates[: max(limit * 3, 10)]
    if llm is not None and len(pool) > 1:
        try:
            pool = llm.rerank(need, pool, text_key="text")
        except Exception:
            pass
    for entry in pool:
        entry.pop("text", None)
    return pool[:limit]


def _default_index() -> Any:
    from knowledge_bases.tools.catalog_index import CatalogIndex

    return CatalogIndex()


def _default_llm() -> Any | None:
    try:
        from llm import get_llm

        return get_llm()
    except Exception:
        return None


def search_noesis_tools(
    need: str, limit: int = 10, graph: Graph | None = None,
    knowledge_base: str | None = None,
    expert_key: str | None = None,
    index: Any | None = None,
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """Rank every active `Tool` against a plain-language description of
    what the caller is trying to do (see module docstring for the three
    stages). `knowledge_base` restricts to one knowledge base; `expert_key`
    restricts to the skills that graph-defined expert is allowed to call.
    Every returned entry already includes the tool's real MCP name and
    knowledge base, so a caller with a match in hand can call it
    immediately via `call_noesis_tool`.
    """
    graph = graph or Graph()
    skills = ToolRepository(graph).load_tools(knowledge_base)
    if expert_key:
        from knowledge_bases.orchestration.repository import RoutingRepository

        expert = RoutingRepository(graph).get_expert(expert_key)
        if expert is not None:
            skills = [s for s in skills if _skill_allowed(s.key, expert.allowed_tools)]
    index = index or _default_index()
    semantic = index.search_skills(need, top_k=max(20, limit * 3))
    llm = llm if llm is not None else _default_llm()
    return rank_skills(need, skills, semantic=semantic, llm=llm, limit=limit)
