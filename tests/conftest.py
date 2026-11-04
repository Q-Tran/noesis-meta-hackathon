"""Keep unit tests off the optional retrieval infrastructure.

The semantic catalog index (Milvus) and the cross-encoder/embedding models
are accelerators with keyword-only fallbacks; unit tests exercise the
fallback deterministically. Tests that want the real thing (e.g. live graph
round-trips) construct it explicitly.
"""
import pytest


class _NoIndex:
    def search_skills(self, query, top_k=20):
        return {}

    def search_experts(self, query, top_k=30):
        return {}

    def index_skills(self, skills):
        return 0

    def index_experts(self, experts):
        return 0


@pytest.fixture(autouse=True)
def _offline_catalog_index(monkeypatch):
    import knowledge_bases.tools.search as search

    monkeypatch.setattr(search, "_default_index", lambda: _NoIndex())
    monkeypatch.setattr(search, "_default_llm", lambda: None)
    yield
