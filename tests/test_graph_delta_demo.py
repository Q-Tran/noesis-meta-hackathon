"""The recorded demo is data, so its invariants can be checked without a browser.

Guards what would silently ruin the demo: events out of order, an event pointing
at a node that is not in the payload, an edge coloured by the wrong ontology
(`FOUND_IN` is the point of the piece), and a stale `index.html` that no longer
matches `events.json`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo" / "graph_delta"
DATA = json.loads((DEMO / "events.json").read_text())
NODES = {n["ref"]: n for n in DATA["nodes"]}
EDGES = {e["key"]: e for e in DATA["edges"]}
KINDS = {e["kind"] for e in DATA["events"]}
ONTOLOGIES = {o["key"] for o in DATA["ontologies"]}

# The ground of the graph: evidence and telemetry are what justifies everything
# else, and claims/arguments *are* the justification. Everything else is a
# knowledge entity and must be concluded by an argument.
GROUND = {"SourceNode", "Chunk", "RoutingTrace", "Session", "Span", "Tool",
          "TraceOutcome"}
JUSTIFICATION = {"Claim", "Argument"}


def _knowledge_entities():
    return [n for n in DATA["nodes"]
            if n["label"] not in GROUND | JUSTIFICATION and not n.get("pre")]


def _edges(kind):
    return [e for e in DATA["edges"] if e["type"] == kind]


NODE_EVENTS = {e["ref"]: e for e in DATA["events"] if e["kind"] == "node.create"}
EDGE_EVENTS = {e["ref"]: e for e in DATA["events"] if e["kind"] == "edge.create"}
NODE_BY_REF = {n["ref"]: n for n in DATA["nodes"]}


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def test_the_recording_is_ordered_and_complete():
    events = DATA["events"]
    assert len(events) > 30, "the demo should exercise a session, not a single write"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    times = [e["t_ms"] for e in events]
    assert times == sorted(times), "events must replay in the order they happened"
    assert len(set(times)) > 1, "a single timestamp is not a delta"
    assert {"node.create", "edge.create", "node.update", "edge.update"} <= KINDS


def test_every_event_resolves_to_a_recorded_entity():
    for event in DATA["events"]:
        if event["kind"].startswith("edge."):
            assert event["ref"] in EDGES, f"event {event['seq']} references {event['ref']}"
        else:
            assert event["ref"] in NODES, f"event {event['seq']} references {event['ref']}"


def test_ontology_colours_are_declared_and_the_spine_belongs_to_provenance():
    for node in DATA["nodes"]:
        assert node["ontology"] in ONTOLOGIES, node["ref"]
    for edge in DATA["edges"]:
        assert edge["ontology"] in ONTOLOGIES, edge["key"]
    spine = [e for e in DATA["edges"] if e["type"] == "FOUND_IN"]
    assert spine, "the demo must show the provenance spine"
    assert {e["ontology"] for e in spine} == {"provenance"}


def test_creations_carry_the_graph_timestamp_and_mutations_say_observed():
    for event in DATA["events"]:
        if event["kind"].endswith(".create"):
            assert event["source"] == "graph"
            assert abs(_epoch(event["t"]) * 1000 - event["t_ms"]) < 1
            stamped = event["props"].get("created_at")
            assert stamped, f"creation {event['seq']} has no graph timestamp"
            assert abs(_epoch(stamped) * 1000 - event["t_ms"]) < 1
        elif event["kind"].endswith((".update", ".delete")):
            assert event["source"] == "observed"


def test_property_changes_name_the_property_and_both_values():
    updates = [e for e in DATA["events"] if e["kind"] == "node.update"]
    assert updates, "the demo must show a property change"
    assert any("confidence" in e["changed"] for e in updates)
    assert any("reliability" in e["changed"] for e in updates)
    reweight = [e for e in DATA["events"]
                if e["kind"] == "edge.update"
                and e["changed"].get("weight") == [1, 0.35]]
    assert reweight, "the demo must show an edge weight change to 0.35"


def test_index_html_is_the_current_recording():
    template = (DEMO / "template.html").read_text()
    assert "__DATA__" in template, "the template lost its injection point"
    html = (DEMO / "index.html").read_text()
    assert "__DATA__" not in html, "index.html was copied from the template unrendered"
    assert "const DATA = " in html, "index.html has no embedded recording"
    embedded = html.split("const DATA = ", 1)[1].split(";</script>", 1)[0]
    assert json.loads(embedded)["events"] == DATA["events"], (
        "index.html is stale: run python3 demo/graph_delta/render.py"
    )
    assert "const CORPUS = " in html, "index.html has no embedded corpus"
    corpus = json.loads(html.split("const CORPUS = ", 1)[1].split(";</script>", 1)[0])
    assert corpus["meta"]["pages"] == 204
    assert corpus["meta"]["chunks"] == 3194
    if corpus["meta"].get("claims"):
        assert corpus["meta"]["claims"] > 5000
        assert corpus["meta"]["entities"] > 1000


def test_every_knowledge_entity_is_concluded_by_an_argument():
    entities = _knowledge_entities()
    assert len(entities) >= 8, "the demo should show the ontology's breadth"
    concluded = {e["to"] for e in _edges("CONCLUDES")}
    missing = [n["ref"] for n in entities if n["ref"] not in concluded]
    assert not missing, f"entities with no argument for their existence: {missing}"


def test_every_argument_has_premises_and_exactly_one_conclusion():
    arguments = [n for n in DATA["nodes"] if n["label"] == "Argument"]
    assert arguments, "the demo must show arguments, not just claims"
    for argument in arguments:
        premises = [e for e in _edges("USES_PREMISE") if e["from"] == argument["ref"]]
        conclusions = [e for e in _edges("CONCLUDES") if e["from"] == argument["ref"]]
        resolution = (argument["props"].get("candidate_id")
                      or argument["props"].get("chunk_id"))
        if not resolution:
            assert premises, f"{argument['ref']} argues from nothing"
            assert len(conclusions) == 1, (
                f"{argument['ref']} concludes {len(conclusions)} things"
            )
        else:
            # Resolution and feedback arguments may attack without concluding.
            assert len(conclusions) <= 1
        assert all(n["label"] == "Claim" for n in DATA["nodes"]
                   for e in premises if n["ref"] == e["to"]), (
            "an argument's premises must be claims"
        )


def test_every_claim_is_grounded_once_in_a_chunk_or_a_conversation():
    claims = [n for n in DATA["nodes"] if n["label"] == "Claim"]
    assert len(claims) >= 8
    node_label = {n["ref"]: n["label"] for n in DATA["nodes"]}
    grounded_in_chunk = grounded_in_trace = 0
    for claim in claims:
        grounds = [e for e in _edges("FOUND_IN") if e["from"] == claim["ref"]]
        assert len(grounds) == 1, (
            f"{claim['ref']} has {len(grounds)} grounds; it must have exactly one"
        )
        target = node_label[grounds[0]["to"]]
        assert target in {"Chunk", "RoutingTrace"}, (
            f"{claim['ref']} is grounded in a {target}, not a chunk or a trace"
        )
        grounded_in_chunk += target == "Chunk"
        grounded_in_trace += target == "RoutingTrace"
    assert grounded_in_chunk and grounded_in_trace, (
        "the demo must show both evidence paths: document and conversation"
    )


def test_every_business_claim_reaches_an_entity_through_an_argument():
    """No orphan business claims; resolution claims are conclusions instead."""
    used = {e["to"] for e in _edges("USES_PREMISE")}
    business = {n["ref"] for n in DATA["nodes"]
                if n["label"] == "Claim"
                and n["props"].get("claim_type") != "derivation"}
    assert business <= used, f"claims that justify nothing: {sorted(business - used)}"
    derivation = {n["ref"] for n in DATA["nodes"]
                  if n["label"] == "Claim"
                  and n["props"].get("claim_type") == "derivation"}
    concluded = {e["to"] for e in _edges("CONCLUDES")}
    assert derivation <= concluded, (
        f"resolution claims with no argument: {sorted(derivation - concluded)}"
    )


def test_the_recording_is_a_sequence_of_atomic_bundles():
    """One write of a justified entity is one beat, not a chain of micro-steps."""
    batches = DATA["batches"]
    assert DATA["meta"]["bundles"] == len(batches)
    assert len(batches) < len(DATA["events"]) , "bundles must group events"
    for info in batches:
        group = [e for e in DATA["events"] if e["batch"] == info["id"]]
        assert group, f"bundle {info['id']} is empty"
        assert info["first_seq"] == group[0]["seq"]
        assert info["last_seq"] == group[-1]["seq"]
        assert info["count"] == len(group)
        assert [e["batch"] for e in group] == [info["id"]] * len(group)
        # bundles are contiguous in replay order
    ordered = [e["seq"] for e in DATA["events"]]
    assert ordered == list(range(1, len(ordered) + 1))
    for before, after in zip(batches, batches[1:]):
        assert after["first_seq"] == before["last_seq"] + 1


def test_an_entity_is_created_in_the_same_bundle_as_its_argument():
    """The epistemology layer is born with the entity, not later."""
    for entity in _knowledge_entities():
        conclusions = [e for e in _edges("CONCLUDES") if e["to"] == entity["ref"]]
        assert len(conclusions) == 1, entity["ref"]
        argument = NODE_BY_REF[conclusions[0]["from"]]
        entity_batch = NODE_EVENTS[entity["ref"]]["batch"]
        assert NODE_EVENTS[argument["ref"]]["batch"] == entity_batch, (
            f"{argument['ref']} arrives in a different bundle than {entity['ref']}"
        )
        assert EDGE_EVENTS[conclusions[0]["key"]]["batch"] == entity_batch, (
            f"the conclusion of {entity['ref']} lands in a different bundle"
        )
        premises = [e for e in _edges("USES_PREMISE")
                    if e["from"] == argument["ref"]]
        assert premises, "the argument on the entity's bundle has no premises"
        assert all(EDGE_EVENTS[p["key"]]["batch"] == entity_batch for p in premises)


def test_a_claim_is_created_in_the_same_bundle_as_its_ground():
    """A claim never exists for a moment without the evidence that backs it."""
    for claim in [n for n in DATA["nodes"] if n["label"] == "Claim"]:
        grounds = [e for e in _edges("FOUND_IN") if e["from"] == claim["ref"]]
        assert len(grounds) == 1
        assert EDGE_EVENTS[grounds[0]["key"]]["batch"] == NODE_EVENTS[claim["ref"]]["batch"]


def test_a_correction_repoints_one_reference_and_keeps_the_old_one_as_history():
    """The canonical edge moves; the truth layer keeps the rejected resolution."""
    superseded = [e for e in DATA["edges"]
                  if (e["props"] or {}).get("status") == "superseded"]
    assert superseded, "the correction did not supersede an edge"
    old = next(e for e in superseded if e["type"] == "EXPOSES")
    new = next(e for e in DATA["edges"]
               if (e["props"] or {}).get("supersedes") == old["props"].get("id"))
    assert new["type"] == old["type"] == "EXPOSES"
    assert new["to"] == old["to"] and new["from"] != old["from"]
    assert (new["props"] or {}).get("status") == "active"

    # The rejected metric no longer carries the property or the mention.
    old_node = NODE_BY_REF[old["from"]]
    new_node = NODE_BY_REF[new["from"]]
    assert "availability" not in old_node["props"]
    assert new_node["props"].get("availability") == "Model API"

    chunk_id = old["props"].get("chunk_id")
    chunk_ref = next(n["ref"] for n in DATA["nodes"]
                     if n["props"].get("id") == chunk_id)
    mentions = [e for e in _edges("MENTIONS")
                if e["from"] == chunk_ref and not e.get("deleted")]
    assert not any(e["to"] == old_node["ref"] for e in mentions), (
        "the chunk still points at the rejected canonical entity"
    )
    assert any(e["to"] == new_node["ref"] for e in mentions), (
        "the corrected mention was not rewired"
    )


def test_the_two_layers_are_tagged_and_canonical_entities_are_all_argued():
    layers = {n["layer"] for n in DATA["nodes"]}
    assert {"canonical", "truth", "telemetry"} <= layers
    canonical = [n for n in DATA["nodes"] if n["layer"] == "canonical"]
    assert len(canonical) >= 10
    concluded = {e["to"] for e in _edges("CONCLUDES")}
    assert all(n["ref"] in concluded for n in canonical)
    assert any(n["layer"] == "truth" and n["label"] in {"Argument", "Claim"}
               for n in DATA["nodes"])


def test_the_full_extraction_is_reported_when_present():
    """The extraction is a run artifact; validate its committed summary."""
    summary_path = DEMO / "sources" / "extraction-summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    if summary.get("chunks_total", 0) < 3000:
        return  # the committed summary is still the small validation run
    assert summary["model"] == "muse-spark-1.3"
    assert summary.get("ok", summary.get("chunks_ok", 0)) >= 3000
    assert summary["claims"] > 5000
    assert summary["failed"] == 0
    report_path = DEMO / "sources" / "extraction-report.json"
    report = json.loads(report_path.read_text())
    assert report["claims_ungrounded"] == 0
    assert report["arguments_without_premises"] == 0
    assert report["arguments_without_conclusion"] == 0
    assert report["entities_without_argument"] == 0
