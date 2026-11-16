"""Create graph indexes, apply migrations, and build vector collections.

An ENGINE script: it must work on any product, including one that installs no
versioning ontology. It used to import `methodology.search` at module scope, so it
crashed on import for such a product -- and the graph indexes it creates first have
nothing to do with versioning.

Reindexing is therefore resolved as a capability. Absent means there are no versioned
entities to reindex, which is a correct outcome rather than a degraded one, so this
reports and continues instead of failing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import Graph
from knowledge_bases.registry import capability
from migrations import apply_migrations
from vector_store import VectorStore

with Graph() as g:
    applied = apply_migrations(g)
    g.ensure_indexes()
    print(f"Migrations applied: {len(applied)}. Graph indexes created.")

vs = VectorStore()
vs.ensure_collections()
print("Vector collections created.")

reindex = capability("versioning.reindex")
if reindex is None:
    print(
        "No versioning ontology installed, so there are no managed entities to "
        "reindex. This is expected for a product without `methodology`."
    )
else:
    with Graph() as g:
        print(f"Managed entities indexed: {reindex(g, vs)}")
