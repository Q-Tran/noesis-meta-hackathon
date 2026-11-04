# Extraction efficiency audit -- ontology-driven triplet extraction

Repo: `/home/k1ng/Projects/noesis`. All code run with the repo's own interpreter
`.venv/bin/python`. Suite state at audit time: **916 passed, 14 skipped** (`.venv/bin/python -m
pytest tests -q`). **No source file was changed.** Scratch scripts: `/tmp/noesis_research/m1.py`
.. `m8.py`.

Every number below is marked **[measured]** (produced by running repo code) or **[estimate]**
(arithmetic on top of measured numbers plus a stated assumption). Token counts are estimates
throughout, because they are `chars / 4` -- the brief's convention -- not a tokenizer result.

The renderer under audit is `knowledge_bases/triplets.py:71` `render_ontology_prompt(labels,
relationships, include_internal)`.

---

## 0. Correction to the briefed figures

Two briefed numbers do not match what the code produces today.

| Quantity | Briefed | Measured | Source |
|---|---|---|---|
| full prompt chars | ~56,000 | **38,257** | `m1.py` [measured] |
| `include_internal=True` chars | -- | **59,177** | `m1.py` [measured] |
| labels total / extractable | 61 / 33 | **61 / 33** (confirmed) | `m1.py` [measured] |
| relationships total / extractable | 85 / 61 | **85 / 61** (confirmed) | `m1.py` [measured] |
| `not_to_be_confused_with` entries | 459 | **459 total** (181 entity + 278 rel); **311** of those are on extractable types (102 entity + 209 rel) | `m1.py` [measured] |

The "~56k" figure matches `include_internal=True` (59,177) far better than the default prompt.
The docstring at `knowledge_bases/epistemology/mcp_tools.py:88-90` states "the full ontology is 56k
characters while one knowledge base is around 7k". **The 56k half of that is now wrong: the
default (extractable-only) prompt is 38,257 chars.** The 7k half is right for
`semantic_layer` (7,360). This is a documentation drift, not a behaviour bug -- most likely the
comment predates the `extractable` filter at `knowledge_bases/triplets.py:88-90`.

---

## 1. Prompt size: full, internal, and per knowledge base

All [measured] (`m1.py`, `m4.py`). Tokens = chars/4 [estimate].

| Scope | Chars | ~Tokens | % of full |
|---|---|---|---|
| `include_internal=True` (61 labels, 85 rels) | 59,177 | 14,794 | 155% |
| **full default** (33 extractable labels, 61 extractable rels) | **38,257** | **9,564** | 100% |

### Per knowledge base, ranked by cost

Rendered as `render_ontology_prompt(labels=kb.labels, relationships=kb.relationships)`, i.e.
what `describe_ontology_tool` (`knowledge_bases/epistemology/mcp_tools.py:82`) returns for a single
key. Labels/rels columns are *declared*; the renderer then drops non-extractable ones.

| Rank | Knowledge base | Chars | ~Tokens | % of full | labels declared / extractable | rels declared / extractable |
|---|---|---|---|---|---|---|
| 1 | `project_management` | 14,216 | 3,554 | 37.2% | 14 / 14 | 20 / 20 |
| 2 | `semantic_layer` | 7,360 | 1,840 | 19.2% | 4 / 4 | 14 / 14 |
| 3 | `strategy` | 6,965 | 1,741 | 18.2% | 6 / 6 | 11 / 11 |
| 4 | `claims` | 5,339 | 1,335 | 14.0% | 4 / 3 | 7 / 7 |
| 5 | `development` | 3,696 | 924 | 9.7% | 8 / 0 | 10 / 9 |
| 6 | `architecture` | 3,506 | 876 | 9.2% | 3 / 3 | 5 / 5 |
| 7 | `business_dictionary` | 1,925 | 481 | 5.0% | 2 / 2 | 2 / 2 |
| 8 | `provenance` | 1,424 | 356 | 3.7% | 1 / 1 | 1 / 1 |
| 9 | `methodology` | 929 | 232 | 2.4% | 6 / 0 | 16 / 2 |
| 10 | `orchestration` | 592 | 148 | 1.5% | 10 / 0 | 10 / 1 |
| 11 | `tools` | 268 | 67 | 0.7% | 3 / 0 | 3 / 0 |

Notes on this table, all [measured]:

- **`project_management` alone is 37% of the whole prompt.** It is the single most expensive
  scope and the one most likely to be routed for a meeting note.
- The four cheapest KBs (`methodology`, `orchestration`, `tools`, plus most of `development`)
  are cheap *because* the `extractable` filter empties them. `tools` renders 268 chars: pure
  scaffolding, zero types. `orchestration` contributes 1 extractable relationship out of 10
  declared. Routing a document to these KBs buys nothing.
- Sum of per-KB renders = 46,220 chars vs 38,257 for the union [measured]. The 7,963-char
  excess is scaffolding (264 chars x 11) plus labels/rels declared by more than one KB.

### Multi-KB scopes (what routing actually produces)

Mean over all combinations of the 7 KBs with extractable content [measured, `m4.py`]:

| Scope size | Mean chars | Min | Max | Mean % of full |
|---|---|---|---|---|
| 1 domain KB | 6,144 | 1,925 | 14,216 | 16.1% |
| 2 domain KBs | 11,790 | 5,163 | 20,821 | 30.8% |
| 3 domain KBs | 17,219 | 8,591 | 26,866 | 45.0% |

Scoping stays a win up to 3 KBs; at 4+ the fixed cost of a second LLM pass starts to dominate
(see section 3).

---

## 2. Where the bytes go

Method: re-implemented the exact line-construction of
`render_ontology_prompt` (`knowledge_bases/triplets.py:99-141`) and attributed every character
to a category (`m2.py`). Attribution total is 38,808 vs actual 38,257 -- **+1.4% over-count**,
because a few fixed line prefixes (`- `, `: `, `    NOT: `) are counted in two categories.
Percentages are of the attributed total.

### Default prompt (extractable only) -- [measured]

| Category | Chars | % | What it is |
|---|---|---|---|
| **relationship descriptions** | **16,141** | **41.6%** | `RelationshipType.description` prose |
| entity descriptions | 9,877 | 25.5% | `EntityType.description` prose |
| relationship endpoint lists | 2,803 | 7.2% | `(Subject, ...) -> (Object, ...)` label names |
| `NOT:` contrasts (relationships) | 2,686 | 6.9% | `not_to_be_confused_with` on rels |
| bare property-name lists | 1,943 | 5.0% | `properties: a, b, c` |
| relationship name/arrow syntax | 1,527 | 3.9% | `- () -[:NAME]-> (): ` boilerplate |
| `NOT:` contrasts (entities) | 1,386 | 3.6% | `not_to_be_confused_with` on labels |
| described property lines | 1,291 | 3.3% | `    - name: description` |
| entity label names | 433 | 1.1% | |
| section headers + closing instruction | 264 | 0.7% | |
| newlines | 231 | 0.6% | |
| property enums | 226 | 0.6% | `One of: a, b, c.` |

### `include_internal=True` -- [measured]

| Category | Chars | % |
|---|---|---|
| relationship descriptions | 21,358 | 35.6% |
| entity descriptions | 17,892 | 29.8% |
| bare property-name lists | 4,542 | 7.6% |
| `NOT:` contrasts (relationships) | 3,610 | 6.0% |
| relationship endpoint lists | 3,559 | 5.9% |
| `NOT:` contrasts (entities) | 2,510 | 4.2% |
| described property lines | 2,428 | 4.0% |
| relationship name/arrow syntax | 2,145 | 3.6% |
| entity label names | 863 | 1.4% |
| property enums | 422 | 0.7% |
| newlines + scaffolding | 633 | 1.0% |

### The single biggest contributor

**Relationship descriptions: 16,141 chars, 41.6% of the prompt** [measured]. Descriptions as a
whole (relationship + entity prose) are **26,018 chars, 67.1%**. Mean lengths [measured]: 265
chars per extractable relationship description, 299 chars per extractable entity description.
The longest single descriptions are `FOUND_IN` (437), `Claim` (433), `SourceNode` (420),
`CONFLICTS_WITH` (390), `DEPENDS_ON` (386).

**`not_to_be_confused_with` is not the problem.** All 311 contrasts on extractable types render
as 4,072 chars combined = **10.5%** of the prompt [measured]. It is the fourth and seventh line
items, not the first. Any optimization aimed at the `NOT:` lines is attacking a tenth of the
cost while removing the disambiguation the module's own docstring
(`knowledge_bases/triplets.py:17-22`) says is load-bearing. **Descriptions are where the bytes
are.**

---

## 3. What a realistic document needs, and the two-pass arithmetic

### Waste for a 5-label document

Two plausible meeting-note label sets [measured, `m3.py`]. "Endpoint-filtered rels" keeps only
relationships where at least one legal subject *and* one legal object label is in the set.

| Document scenario | Labels | Rels kept (of 61) | Scoped chars | % of full | **Wasted if full prompt used** |
|---|---|---|---|---|---|
| PM note (`Person`, `Task`, `Decision`, `Insight`, `Project`) | 5 | 12 | 8,533 | 22.3% | **77.7%** |
| Metrics note (`Person`, `Metric`, `BusinessTerm`, `Decision`, `Insight`) | 5 | 12 | 8,021 | 21.0% | **79.0%** |
| PM note, labels scoped but **all** 61 rels offered | 5 | 61 | 26,073 | 68.2% | 31.8% |

Two things follow, both [measured]:

1. **If extraction truly needs only 5 labels, 77-79% of the full prompt is waste** -- about
   29,700 chars / ~7,400 tokens per call.
2. **Scoping labels without scoping relationships recovers only a third of that** (68.2% of the
   prompt remains). Relationship descriptions are 41.6% of the bytes, so *the relationship set
   is what must be narrowed*. This is the most actionable finding in the report.

### Is two-pass cheaper than one-pass?

Stated assumptions, all explicit:

- 4 chars = 1 token [assumption, brief's convention].
- Document body = 4,000 chars (a meeting note) [assumption].
- Extraction completion = 2,000 chars [assumption].
- Classification completion = 50 chars (a short key list) [assumption].
- Input and output tokens priced the same [assumption -- **this is conservative against
  two-pass**, since real APIs bill output 2-5x, and two-pass adds almost no output].
- Classification prompt = KB key + `KnowledgeBase.description` for all 11 KBs = **3,049 chars**
  [measured, `m3.py`]. A key+name-only variant is 348 chars [measured], but is too thin to
  route on.
- No prompt caching at the provider [assumption]. Provider-side caching of a stable prefix
  would shift this heavily toward one-pass; see the caching win in section 4.

One-pass baseline: `(38,257 + 4,000 + 2,000) / 4` = **11,064 tokens** [estimate].

Two-pass: `pass1 = (3,049 + 4,000 + 50)/4 = 1,775` tokens, then `pass2 = (scoped + 4,000 +
2,000)/4`.

| Routed scope | Scoped chars | Pass 1 | Pass 2 | Two-pass total | vs one-pass |
|---|---|---|---|---|---|
| 1 KB (`strategy`) | 6,965 | 1,775 | 3,241 | **5,016** | **-54.7%** |
| ideal 5-label + endpoint-filtered rels | 8,533 | 1,775 | 3,633 | **5,408** | **-51.1%** |
| 2 KBs (`project_management`+`strategy`) | 20,261 | 1,775 | 6,565 | **8,340** | **-24.6%** |
| 3 KBs (`pm`+`strategy`+`claims`) | 24,195 | 1,775 | 7,549 | **9,324** | **-15.7%** |

**Conclusion: two-pass wins at every realistic fan-out, by 16-55%** [estimate, from measured
prompt sizes]. The break-even scope is where `scoped_chars > 38,257 - 3,049 - 50 = 35,158`
chars -- i.e. two-pass only loses if routing selects ~92% of the whole ontology, which no single
routing decision does (3 KBs max measured = 26,866 chars). The fixed cost of pass 1 is 1,775
tokens, and even the 3-KB case saves 1,740 tokens against it.

Caveat, honest: this arithmetic counts tokens only. It does not count the second round-trip's
latency, and it does not measure whether pass 1 routes correctly -- which is the real risk
(section 5). Extraction *quality* under scoping was not measured; the repo has no extraction
quality benchmark I could run.

### Batching multiple documents per call

Amortizes the prompt over N documents [estimate, same assumptions]:

| Prompt | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| full (38,257) | 11,064 tok/doc | 6,282 (-43%) | 3,891 (-65%) | 2,696 (-76%) |
| 2-KB scoped (20,261) | 6,565 tok/doc | 4,033 (-39%) | 2,766 (-58%) | 2,133 (-68%) |

Batching and scoping compose: batching 8 documents against a 2-KB prompt is 2,133 tok/doc, vs
11,064 for one-pass-full-per-document -- an **81% reduction** [estimate]. Batching is the single
largest lever on paper, and the one whose correctness risk is highest (cross-document entity
bleed; see section 5).

---

## 4. Concrete wins, ranked by (measured saving / risk)

### Win 1 -- Scope relationships by satisfiable endpoints, not just by knowledge base

- **Where**: `knowledge_bases/triplets.py:71` `render_ontology_prompt`; caller
  `knowledge_bases/epistemology/mcp_tools.py:100` in `describe_ontology_tool`.
- **Mechanism**: today the caller passes `relationships=tuple(kb.relationships)`
  (`claims/mcp_tools.py:97`). Additionally drop any relationship whose legal `subject` has no
  member in the offered label set, or whose legal `object` has none. Such a relationship cannot
  produce a valid triplet -- `validate_triplet`
  (`knowledge_bases/triplets.py:180-192`) would reject every attempt.
- **Measured saving**: for a 5-label scope, 26,073 -> 8,533 chars, **-67.3%** of the scoped
  prompt and **-45.8% of the full prompt**. At KB granularity [measured, `m7.py`]: `strategy`
  offers 11 relationships of which **6 cannot be satisfied within its own labels**
  (`MEASURED_BY`, `MADE_BY`, `HELD_BY`, `SUPPORTS`, `CHALLENGES`, `AFFECTS`);
  `claims` 2 of 7; `methodology` 2 of 2; `architecture` 2 of 5; `project_management` 2 of 20.
- **Risk**: **low-medium.** 19 of 61 extractable relationships have endpoints in more than one
  knowledge base [measured]. Dropping an unsatisfiable relationship removes an edge the model
  *could* have used had the other endpoint's label been offered. So this must be applied to the
  final offered label set, not per KB in isolation -- otherwise scoping to `strategy` alone
  silently removes `MADE_BY`, and "Alice decided X" loses its actor. The mitigation is to expand
  the label set to the union of endpoints first, then filter relationships; that costs bytes but
  keeps the graph connected.

### Win 2 -- Two-pass: classify knowledge bases, then extract scoped

- **Where**: new caller logic; renderer needs no change. Uses the existing scoping arguments at
  `knowledge_bases/triplets.py:71-75` and the existing `knowledge_bases` argument of
  `describe_ontology_tool` (`knowledge_bases/epistemology/mcp_tools.py:91-99`). A routing pass already
  exists in the product (`route_noesis_question`), so the machinery is not new.
- **Mechanism**: pass 1 classifies the document against the 11 `KnowledgeBase.description`
  strings (3,049 chars [measured]); pass 2 renders only the selected KBs.
- **Estimated saving**: **-54.7% at 1 KB, -24.6% at 2 KBs, -15.7% at 3 KBs** (table above).
- **Risk**: **medium.** Misrouting silently loses facts, and the loss is invisible because a
  type absent from the prompt produces no rejection -- there is nothing to reject. Detection is
  section 5. Mitigation: always include `provenance` (1,424 chars, carries `SourceNode` and
  `FOUND_IN`) and bias pass 1 toward over-selection; at 3 KBs it still saves 16%.

### Win 3 -- Batch N documents per extraction call

- **Where**: new caller; `ingest_information_tool`
  (`knowledge_bases/epistemology/mcp_tools.py:33`) already accepts a list of triplets, so the write
  side needs no change. The legacy per-document driver is `worker.py:47` calling
  `extract_claims`.
- **Mechanism**: one rendered prompt, N document bodies, one response with a document id per
  triplet.
- **Estimated saving**: **-43% at N=2, -65% at N=4, -76% at N=8** on the full prompt (table
  above).
- **Risk**: **high.** Entity identity bleeds across documents -- append-only ingestion converges
  on identity (`EntityType.identity`, `knowledge_bases/base.py:78-95`), so two documents that
  both say "the API project" merge, which is sometimes right and sometimes wrong. Provenance is
  also at stake: `ingest_triplets` takes a single `source_id`
  (`knowledge_bases/ingestion_service.py:72`), so a batch needs per-triplet source attribution
  or every fact gets the wrong `FOUND_IN`. **Do not batch across unrelated sources.** Batching
  chunks of one document is the safe subset.

### Win 4 -- Shorten the rendered description form

- **Where**: `knowledge_bases/triplets.py:103` (entity description line) and
  `knowledge_bases/triplets.py:128` (relationship line).
- **Mechanism**: render only the first sentence of each `description`, keeping the full text
  available through `include_internal`-style diagnostic paths.
- **Measured saving**: entity descriptions 9,877 -> 4,935 (-50.0%), relationship descriptions
  16,141 -> 7,162 (-55.6%); combined **-13,921 chars = -36.4% of the full prompt** [measured,
  `m3.py`].
- **Risk**: **medium-high, and unmeasured.** The module docstring
  (`knowledge_bases/triplets.py:17-22`) and `EntityType`'s docstring
  (`knowledge_bases/base.py:66-75`) both argue that the *boundary against near neighbours* is
  what makes a label choosable -- and that boundary is usually stated in the second or third
  sentence, exactly what first-sentence truncation deletes. I could not measure the quality
  cost: there is no extraction-accuracy benchmark in the repo to run. Treat the 36.4% as an
  upper bound on saving and a real risk to precision between near neighbours
  (`Insight`/`Claim`, `Capability`/`Requirement`).

### Win 5 -- Prune `not_to_be_confused_with` names that are out of scope

- **Where**: `knowledge_bases/triplets.py:118-121` (entities) and
  `knowledge_bases/triplets.py:130-131` (relationships).
- **Mechanism**: filter each `NOT:` list to names actually present in the rendered vocabulary.
  Contrasting against a type the model was never offered is dead weight.
- **Measured saving**: on the full prompt only **350 chars (-0.9%)**; on a 5-label scope **458
  and 470 chars (-5.4% / -5.9%)** [measured, `m3.py`]. Also [measured]: **8 of 102** entity
  contrast names and **24 of 209** relationship contrast names on extractable types point at
  types the default prompt withholds, so they are unconditionally dead today.
- **Risk**: **low**, with one caveat -- the contrast may be *why* the model avoids a wrong type
  it knows from pre-training (e.g. `NOT: Revision` warns against a type the prompt never
  offers). Keeping contrasts that name withheld types is arguably correct. Pruning the
  in-vocabulary-only subset is safe; the saving is small either way. **This is the briefed
  hypothesis that the measurement does not support as a major win.**

### Win 6 -- Cache the rendered prompt

- **Answer to the brief's CHECK: yes, it is recomputed on every call, and no, it does not
  matter.**
- `render_ontology_prompt` (`knowledge_bases/triplets.py:71`) carries no decorator; it rebuilds
  the whole `lines` list every call. `hasattr(render_ontology_prompt, 'cache_info')` is `False`
  [measured]. Its inputs *are* cached: `entity_types` (`knowledge_bases/registry.py:172-173`)
  and `relationship_types` (`knowledge_bases/registry.py:193-194`) are both
  `@lru_cache(maxsize=None)`, as is `_load_declaration` (`registry.py:72`).
- **Measured cost**: **0.073 ms per call**, mean of 200 calls after warm-up [measured, `m3.py`].
- **Measured saving from caching: ~0.073 ms of CPU.** Zero tokens. An LLM call costs hundreds of
  milliseconds to seconds, so this is 4-5 orders of magnitude below the noise. **Do not add an
  `lru_cache` here for performance.** The real caching win is *provider-side prompt caching*:
  put the rendered ontology in a stable prefix so the provider can cache it across calls. That
  is a caller-side decision, not a renderer change, and it argues for keeping the prompt
  *stable* -- which is in direct tension with per-document scoping (wins 1-2). Worth noting:
  if provider prefix caching is available and cheap, the full stable prompt may beat scoping.
  I could not measure that -- it depends on the provider's cache pricing.

### Win 7 (already done) -- Withholding non-extractable types: verified and quantified

- **Where**: `knowledge_bases/triplets.py:88-90`, the `if not include_internal:` filter.
- **Verified**: `include_internal=True` renders 59,177 chars; the default renders 38,257
  [measured]. **Saving: 20,920 chars = 35.4% of the unfiltered prompt.** In tokens, ~5,230 per
  call [estimate]. It removes 28 labels and 24 relationships from the offered vocabulary
  [measured].
- Enforced by tests: `tests/test_ontology_descriptions.py:120-133`
  (`test_internal_types_are_withheld_from_the_extraction_prompt`) asserts `SUPERSEDES`,
  `CURRENT`, `INCLUDES_RELATIONSHIP`, `- Span:`, `- RoutingTrace:` are absent;
  `tests/test_ontology_descriptions.py:135-144` asserts they remain reachable with
  `include_internal=True`. Defence in depth exists at write time too:
  `validate_triplet` re-checks `extractable` (`knowledge_bases/triplets.py:167-179`), so a model
  naming a withheld type from memory is still refused.
- **Risk: none remaining** -- this is the highest-value change already shipped, and it is the
  reason the prompt is 38k rather than 59k.

### Ranking summary

| # | Win | Measured/estimated saving | Risk | Saving/risk |
|---|---|---|---|---|
| 7 | withhold non-extractable types | -35.4% [measured] | none (shipped, tested) | done |
| 1 | endpoint-satisfiable relationship scoping | -45.8% of full at 5 labels [measured] | low-medium | **best remaining** |
| 2 | two-pass classify-then-scope | -16% to -55% [estimate] | medium | high |
| 5 | prune out-of-scope `NOT:` names | -0.9% full / -5.9% scoped [measured] | low | low value, low risk |
| 4 | first-sentence descriptions | -36.4% [measured] | medium-high, unmeasured quality cost | uncertain |
| 3 | batch N docs | -43% to -76% [estimate] | high (identity + provenance) | conditional |
| 6 | cache rendered prompt | 0.073 ms, 0 tokens [measured] | none | **not worth doing** |

Two more small measured items, not worth their own win:

- **Endpoint-list deduplication**: 61 extractable relationships use 54 distinct endpoint
  signatures; hoisting the 7 repeats saves at most **223 chars (0.6%)** [measured]. Not worth
  it.
- **Legacy comparison**: `extract.py:211-222` sends 737 chars of hand-written instruction plus
  a 6,512-char `EnhancedExtraction` JSON schema (`extract.py:105`) = **7,249 chars fixed
  overhead** [measured, `m5.py`]. That is 19% of the ontology prompt's size -- the legacy path
  is genuinely cheaper *per call*. What it buys with the saving is coverage: its 10 hardcoded
  classes (`extract.py:17-74`) cover **10 of 33 extractable labels = 30%**; the 23 it cannot
  express include `Requirement`, `Capability`, `Goal`, `Metric`, `Claim`, `Architecture`,
  `Component`, `UserStory`, `TestCase` [measured, `m8.py`]. The ontology prompt costs ~5.3x more
  per call for 3.3x the vocabulary, plus enum-checked properties and endpoint enforcement.
  **A cheaper prompt that cannot name the type is not an efficiency win.**

---

## 5. Correctness risks, and how to detect each

The governing asymmetry: **an invalid triplet is loud, a missing one is silent.**
`partition` (`knowledge_bases/triplets.py:242`) returns rejections with reasons, and
`ingest_triplets` surfaces them as `"rejected"`
(`knowledge_bases/ingestion_service.py:128-132`). Nothing anywhere reports a fact that was
never extracted because its type was not offered. Every scoping optimization trades loud
failure for silent failure.

| Optimization | Failure mode | Detection |
|---|---|---|
| Win 1 (endpoint scoping) | A relationship dropped as unsatisfiable was the only edge for a real fact; the entity lands isolated. | **Already instrumented**: `unmet_kind_requirements` (`knowledge_bases/triplets.py:280`) returns exactly this -- a `Goal` with no inbound `MEASURES`, a `Decision` with no outgoing `AFFECTS`. It is surfaced as `"incomplete"` at `knowledge_bases/ingestion_service.py:142`. **A rise in `incomplete` after enabling scoping is the signal.** Also check for orphan nodes: entities written with zero edges in the batch. |
| Win 1 / Win 2 (label absent) | The label was never offered, so the model cannot even name it. No rejection, no `incomplete` -- the sentence is just gone. | **Needs new instrumentation. This is the one real gap.** Two detectors: (a) **shadow audit** -- re-extract a sampled 1-5% of documents with the full prompt and diff triplet sets; any triplet the scoped run missed names the label that scoping should have included. (b) **routing recall harness** -- label a fixed corpus with its true KB set, then measure pass-1 recall directly. Neither exists today. |
| Win 2 (misrouting) | Pass 1 picks the wrong KBs. | Same shadow audit. Cheap proxy: if pass 1 returns 0-1 KBs for a document whose text is long or topically mixed, escalate to the full prompt. Bias pass 1 to over-select -- 3 KBs still saves 16%, and the asymmetry in cost is far smaller than the asymmetry in harm. |
| Win 3 (batching) | Cross-document entity merge, and wrong provenance -- `ingest_triplets` takes one `source_id` (`knowledge_bases/ingestion_service.py:72`). | Require a document id per triplet and assert every triplet maps to exactly one source before writing. Detect merges by watching for entities whose `FOUND_IN` count jumps within a single batch. Refuse to batch documents from different `SourceNode`s. |
| Win 4 (short descriptions) | Near-neighbour confusion: `Insight` written where `Claim` was meant. Both are valid types, so `validate_triplet` passes and nothing complains. | **`validate_triplet` cannot catch this at all** -- it checks existence and endpoint legality (`knowledge_bases/triplets.py:147-192`), never semantic fit. Needs a labelled gold set the repo does not have. **Do not ship win 4 before that gold set exists.** The `not_to_be_confused_with` pairs (459 of them) are a ready-made source of confusion pairs for such a set. |
| Win 5 (`NOT:` pruning) | The model falls back on a pre-training type not in the vocabulary. | Already loud: `validate_triplet` raises `"is not a described entity type"` (`knowledge_bases/triplets.py:161-164`) and it lands in `rejected`. Watch the rejection reason distribution. |
| Win 6 (caching) | An `lru_cache` on the renderer would freeze the prompt against a hot-reloaded package. | Not applicable -- do not add it. Note `knowledge_bases/tools/resolver.py:22-30` documents this exact hazard as the reason it avoids `lru_cache`. |

**Standing recommendation**: whatever scoping ships, log the offered label/relationship set
alongside each extraction. Without it, a silently dropped fact is undiagnosable after the fact --
you cannot tell whether the model failed or was never given the chance.

---

## What I could not measure

Stated plainly, not guessed:

- **Extraction quality under any optimization.** There is no extraction-accuracy benchmark or
  gold triplet set in the repo. Every quality claim above is a risk argument, not a measurement.
- **Real token counts.** All token figures are `chars/4`. No tokenizer was run.
- **Real money.** No provider pricing is applied; input and output are counted at par, which
  understates two-pass's advantage and understates batching's advantage.
- **Provider-side prompt caching economics**, which could reverse the ranking of wins 1-2
  against win 6's stable-prefix variant.
- **Pass-1 routing accuracy.** The 16-55% two-pass saving assumes routing is correct. Its error
  rate is the single most important unmeasured number in this report.
