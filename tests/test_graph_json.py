import json

from neo4j.spatial import CartesianPoint
from neo4j.time import DateTime

from graph import Graph, json_safe


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def run(self, query, **params):
        return [FakeRecord({
            "nested": ({"created_at": DateTime(2026, 8, 5, 12, 30)},),
            "location": CartesianPoint((1, 2)),
        })]


class FakeDriver:
    def session(self):
        return FakeSession()


def test_json_safe_normalizes_nested_neo4j_temporal_and_spatial_values():
    value = json_safe({
        "values": [{"at": DateTime(2026, 8, 5, 12, 30)}],
        "points": {CartesianPoint((1, 2))},
    })

    assert value["values"][0]["at"] == "2026-08-05T12:30:00.000000000"
    assert value["points"] == [{
        "type": "CartesianPoint",
        "srid": 7203,
        "coordinates": [1.0, 2.0],
    }]
    json.dumps(value)


def test_json_safe_encodes_binary_values_without_data_loss():
    value = json_safe({
        "bytes": b"\x00\xff",
        "bytearray": bytearray(b"noesis"),
        "memoryview": memoryview(b"graph"),
    })

    assert value == {
        "bytes": {"type": "bytes", "base64": "AP8="},
        "bytearray": {"type": "bytearray", "base64": "bm9lc2lz"},
        "memoryview": {"type": "memoryview", "base64": "Z3JhcGg="},
    }
    json.dumps(value, allow_nan=False)


def test_json_safe_normalizes_non_finite_floats():
    value = json_safe([float("nan"), float("inf"), float("-inf"), 1.5])

    assert value == ["NaN", "Infinity", "-Infinity", 1.5]
    json.dumps(value, allow_nan=False)


def test_graph_run_normalizes_record_data():
    graph = Graph.__new__(Graph)
    graph._driver = FakeDriver()

    rows = graph.run("RETURN value")

    assert rows[0]["nested"][0]["created_at"].startswith("2026-08-05T12:30:00")
    assert rows[0]["location"]["coordinates"] == [1.0, 2.0]
    json.dumps(rows)
