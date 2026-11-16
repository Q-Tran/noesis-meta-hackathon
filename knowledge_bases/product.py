"""The product this repository ships.

A product is a declaration, not a fork: `noesis-core` is ontology-free, each
ontology is an installable package, and a `ProductProfile`
(`knowledge_bases/base.py`) binds a set of them together with its own routing
posture and workflow profile. Everything derived from the package set --
the graph ontology, the search-type registry, the seeded tool and expert
catalogs, the versionable kinds, the migration streams -- follows from the
`packages` tuple below, so adding or removing a product capability is one edit
here rather than a change spread across those surfaces.

This file is the seam Phase 2/3 splits on: when the ontology packages become
separately installed distributions, `PROFILE` is what a product distribution
replaces, and the engine keeps reading it through `knowledge_bases/registry.py`
exactly as it does now.

The profile is code on purpose. Choosing which ontologies are installed decides
the label and relationship vocabulary of the whole graph, so it is an
architecture decision requiring a reviewed commit, not runtime-mutable data
(`package_composition` in `knowledge_bases/boundary.py`).
"""
from __future__ import annotations

from knowledge_bases.base import ProductProfile

PROFILE = ProductProfile(
    key="noesis-enterprise",
    name="Noesis",
    # Registration order; defines `KNOWLEDGE_BASES` iteration order. Seed
    # catalogs keep their own deliberately stable historical order (see
    # `tools/catalog.py` and `orchestration/experts_catalog.py`).
    packages=(
        "semantic_layer",
        "business_dictionary",
        "epistemology",
        "strategy",
        "architecture",
        "project_management",
        "tools",
        "orchestration",
        "provenance",
        "methodology",
        "development",
    ),
    # `None` keeps the routable set exactly as the installed packages declare
    # it, so a newly installed package's expert is routable without editing a
    # second list here.
    routable_experts=None,
    workflow_profile="knowledge_bases.methodology.phases:WORKFLOW_PHASES",
    # Self-improvement-authored handlers land in this product's own `tools`
    # package, which ships in this source tree. Never an installed ontology
    # distribution: that source lives in `site-packages`, so an appended
    # handler would be unwritable at worst and silently lost on upgrade at
    # best (see `knowledge_bases/tools/source_editor.py`).
    generated_tool_target="tools",
    # What this product registers as with Prime, and the prefix of its exported
    # Agent Skill directories (`.prime/agent/skills/noesis-<kb>/`).
    mcp_server_name="noesis",
)
