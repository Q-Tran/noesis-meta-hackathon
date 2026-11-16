"""Noesis memory-recall benchmarks.

Adapted from mem0ai/memory-benchmarks (https://github.com/mem0ai/memory-benchmarks)
so Noesis's graph-native evidence pipeline can be scored on the same
datasets/methodology used to report Mem0's LOCOMO numbers.

Unlike Mem0 (a dedicated conversational-memory product with an `/add` /
`/search` API purpose-built for chat turns), Noesis is a decision-intelligence
knowledge graph. There is no built-in "memory" primitive; this package maps
the benchmark's ingest/search steps onto Noesis's closest existing
primitives instead of introducing a new memory subsystem:

  ingest -> knowledge_bases.epistemology.feedback.FeedbackRepository.record()
            (each conversation turn becomes a `Claim` node, provenance-linked
            to a per-conversation `SourceNode`, tagged with a `user_id`
            property for isolation)
  search -> knowledge_bases.orchestration.hybrid_search.hybrid_search()
            (the same fulltext+vector+cross-encoder-rerank retrieval every
            other knowledge base in Noesis uses), scoped with
            filters={"user_id": ...}

This intentionally does NOT vendor Mem0's LLM-based fact-extraction/dedup
step -- Noesis's evidence pipeline stores raw turns as `Claim`s rather than
distilling them into atomic facts first. See benchmarks/README.md for
important caveats about comparing these numbers to Mem0's.
"""
