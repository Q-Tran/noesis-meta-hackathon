# Ontology audit (prerequisite for session memory)

Run before tightening `upsert_rel`. Two *different* causes of off-ontology data,
which need opposite responses.

## 1. The registry is incomplete (legitimate structure, not drift)

`knowledge_bases/methodology/models.py` declares 30 `RelationshipType`s and 20
`EntityType`s. At audit time `registry.py`'s `bmad` entry declared only 12
relationships and 7 labels. Every "off-ontology" methodology edge in the graph
is written by *committed repository code* and validated by
`methodology/validation.py`:

    HAS_KNOWLEDGE 43   Project -> BmadEntity      methodology/repository.py:393
    HAS_RUN       10   Project -> WorkflowRun     methodology/repository.py:82
    MUST_SATISFY   7   Requirement -> AcceptanceCriterion
    MEASURES       5   SuccessMeasure -> Goal
    ADDRESSES      5   Requirement -> Problem
    SATISFIES      5   Capability -> Requirement
    HAS_BASELINE   2   Project -> Baseline        methodology/repository.py:306
    ADVANCES       1   ProductPlan -> Goal
    RECORDED       1   WorkflowRun -> Claim       methodology/repository.py:227

These are real ontology that was never mirrored into the registry. Response:
*declare them*. Deleting them would break methodology validation.

## 2. Extraction can genuinely mint ontology (the actual hazard)

    AFFECTS      223   Insight -> BusinessTerm, Insight -> Project, ...

`AFFECTS` is a legitimate methodology relationship (Decision -> Architecture), but
`extract.py` also names it -- along with `WORKS_FOR`, which exists nowhere -- in
the free-text `relationship_type` hint it gives the LLM:

    relationship_type: str = Field(..., description="UPPER_SNAKE_CASE e.g.
        ASSERTS, ..., WORKS_FOR, AFFECTS, ...")

and `graph.upsert_rel()` interpolates whatever comes back straight into Cypher:

    MERGE (a)-[r:`{rel_type}`]->(b)

with no validation. So an LLM can add a relationship type to the graph at
runtime. `boundary.py` declares graph ontology an IMMUTABLE surface -- a human
writing code and a reviewed commit -- so this is a hole in that guarantee, and
it predates the session-memory work.

Response: validate against the registry at the write boundary, and record
rejected triplets as reviewable evidence rather than dropping them silently.

## Labels

At audit time the 20 `EntityType`s were undeclared (`Requirement`,
`Capability`, `Problem`, ...). They are always co-labelled `:BmadEntity`,
which *is* declared, so `mcp.py`'s label checks still work -- but the
registry should say so. Since the audit, ownership was split out of the
monolith: every `EntityType` label is now declared by exactly one *domain*
knowledge base (`strategy`, `architecture`, `project_management`) --
enforced by `tests/test_knowledge_bases.py` -- while `methodology` declares
only the versioning machinery (`BmadEntity`, `Revision`, `WorkflowRun`,
`ChangeSet`, `Baseline`, `BaselineRelationship`).
`SchemaMigration` is internal bookkeeping and deliberately stays undeclared.

## Derived labels (deliberately undeclared)

`<Kind>Revision` (`ReviewFindingRevision`, `RequirementRevision`, ...) is a
*derived* label: every such node is also `:Revision`, which is declared. The
label is generated per entity kind rather than enumerated, so declaring each one
would duplicate `EntityType` without adding a guarantee. The allowlist therefore
accepts a label that is declared OR whose node also carries a declared label.
`SchemaMigration` is internal migration bookkeeping and stays undeclared.

## Shared kinds (now fully split)

At audit time `Goal`, `Decision`, `Epic`, `UserStory`, `Task` were shared:
declared by `strategy`/`execution` and reused by BMAD as
`(:Goal:BmadEntity)`, deliberately not re-declared under `bmad`. That
sharing has since become the ownership model itself: every managed entity
carries its domain label plus `:BmadEntity`, and each domain label has
exactly one owning knowledge base, so "who owns this entity" is always
answerable.

## Result

After declaring the methodology's real ontology, off-ontology relationships
in the live graph: **0** (was 302 across 10 types).
