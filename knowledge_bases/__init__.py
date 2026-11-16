"""Bounded knowledge bases hosted by Noesis.

Re-exports are lazy (PEP 562): importing this package -- which happens when
any `knowledge_bases.<kb>` submodule or any `package.py` declaration is
loaded -- must not eagerly compose the registry, or `registry` could never
import those declarations itself.

This package is EXTENSIBLE. `noesis-core` ships the composition machinery and the
control plane; a separately distributed ontology contributes `knowledge_bases.<key>`
from its own installation directory. `extend_path` is what makes that one import path
span several distributions, so the 200-odd `from knowledge_bases.methodology import ...`
statements inside the ontology packages keep working unchanged after the split.

A plain namespace package (dropping this file) would do the same, but it would also drop
the lazy re-exports above and force every consumer to know which distribution a
knowledge base came from. The import path is part of the contract; where the code is
installed is not.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

__all__ = ["KNOWLEDGE_BASES", "KnowledgeBase"]


def __getattr__(name: str):
    if name in __all__:
        from knowledge_bases import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
