"""build_filter tool loop, driven by a scripted fake client (no API calls)."""

import json
from types import SimpleNamespace

import duckdb
import pytest

import build_filter as bf
from filter_spec import FIELDS

B737 = {"where": {"aircraft_family": ["B737"]}}


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(filters, id="t1"):
    return SimpleNamespace(type="tool_use", id=id, name="build_filter", input={"filters": filters})


def response(*content, stop_reason=None):
    has_tool = any(b.type == "tool_use" for b in content)
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), *content],
        stop_reason=stop_reason or ("tool_use" if has_tool else "end_turn"),
        model=bf.MODEL,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
                              cache_read_input_tokens=100),
    )


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE reports (acn BIGINT, year BIGINT, aircraft_family VARCHAR, human_factors VARCHAR[])")
    con.execute("CREATE TABLE narratives (acn BIGINT, part BIGINT, text VARCHAR)")
    con.executemany("INSERT INTO reports VALUES (?, ?, ?, ?)", [
        (1, 2019, "B737", ["Fatigue"]), (2, 2023, "B737", []), (3, 2023, "A320 family", ["Fatigue"]),
    ])
    con.executemany("INSERT INTO narratives VALUES (?, ?, ?)", [
        (1, 1, "Item deferred per MEL."), (2, 1, "Shift turnover."), (3, 1, "Deferred again."),
    ])
    return con


def run_tool(con, *responses, log_path=None):
    client = FakeClient(*responses)
    return bf.build_filter("question", client=client, con=con, log_path=log_path), client


def test_schema_offers_every_filter_field():
    spec = bf.TOOL["input_schema"]["properties"]["filters"]["items"]
    assert set(spec["properties"]["where"]["properties"]) == set(FIELDS)
    assert set(spec["properties"]["group_by"]["enum"]) == set(FIELDS)


def test_system_prompt_lists_values():
    prompt = bf.system_prompt()
    assert "<<FIELD_VALUES>>" not in prompt
    for field in FIELDS:
        assert f"- `{field}`:" in prompt
    assert '"Electrical Wiring & Connectors"' in prompt


def test_accepted_call(con):
    record, client = run_tool(con, response(tool_use([B737])), response(text("737 family filter.")))
    assert record["status"] == "ok"
    assert record["filters"] == [B737]
    assert record["results"][0]["acns"] == [1, 2]
    assert record["explanation"] == "737 family filter."
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is False
    assert json.loads(tool_result["content"]) == [{"count": 2}]
    assert record["usage"]["cache_read_input_tokens"] == 200


def test_request_shape(con):
    _, client = run_tool(con, response(text("no call")))
    req = client.requests[0]
    assert req["model"] == bf.MODEL
    assert req["tools"] == [bf.TOOL]
    assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}  # forced choice is a 400 on Opus 5.5
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["messages"] == [{"role": "user", "content": "question"}]


def test_rejected_call_is_explained_and_corrected(con):
    bad = {"where": {"make_model": ["Spacecraft"]}}
    record, client = run_tool(con,
                              response(tool_use([bad], id="t1")),
                              response(tool_use([{"text_any": ["spacecraft"]}], id="t2")),
                              response(text("No coded value; text search instead.")))
    first_result = client.requests[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] is True
    assert "filters[0]: 'Spacecraft' is not a value of 'make_model'" in first_result["content"]
    assert record["status"] == "ok"
    assert record["filters"] == [{"text_any": ["spacecraft"]}]
    assert record["results"][0]["count"] == 0
    assert [bool(c["errors"]) for c in record["calls"]] == [True, False]


def test_comparison_returns_one_result_per_filter(con):
    filters = [{"year_min": 2019, "year_max": 2019, "text_any": ["defer"]},
               {"year_min": 2023, "year_max": 2023, "text_any": ["defer"]}]
    record, _ = run_tool(con, response(tool_use(filters)), response(text("2019 vs 2023.")))
    assert [r["acns"] for r in record["results"]] == [[1], [3]]


def test_group_by_shows_top_groups(con):
    spec = {"group_by": "aircraft_family"}
    _, client = run_tool(con, response(tool_use([spec])), response(text("Grouped.")))
    output = json.loads(client.requests[1]["messages"][-1]["content"][0]["content"])
    assert output == [{"count": 3, "top_groups": {"B737": 2, "A320 family": 1}}]


def test_gives_up_after_max_calls(con):
    bad = response(tool_use([{"where": {"aircraft_family": ["Boeing 737"]}}]))
    record, client = run_tool(con, *[bad] * bf.MAX_CALLS)
    assert len(client.requests) == bf.MAX_CALLS
    assert record["status"] == "invalid"
    assert record["filters"] is None


def test_later_rejected_call_keeps_earlier_accepted_one(con):
    record, _ = run_tool(con,
                         response(tool_use([B737])),
                         response(tool_use([{"where": {"aircraft_family": ["Boeing 737"]}}])),
                         response(text("Done.")))
    assert record["status"] == "ok"
    assert record["filters"] == [B737]


def test_no_call(con):
    record, _ = run_tool(con, response(text("I can't help with that.")))
    assert record["status"] == "no_call"
    assert record["filters"] is None


def test_refusal(con):
    record, _ = run_tool(con, response(stop_reason="refusal"))
    assert record["status"] == "refusal"


@pytest.mark.parametrize("tool_input, message", [
    ({}, "'filters' must be a non-empty list"),
    ({"filters": []}, "'filters' must be a non-empty list"),
    ({"filters": B737}, "'filters' must be a non-empty list"),
    ({"filters": [B737, {"where": ["B737"]}]}, "filters[1]: 'where' must be an object"),
])
def test_execute_rejects_malformed_input(con, tool_input, message):
    call = bf.execute(tool_input, con)
    assert any(message in e for e in call["errors"])
    assert "results" not in call


def test_log_records_calls_without_acn_lists(con, tmp_path):
    log = tmp_path / "log.jsonl"
    run_tool(con, response(tool_use([B737])), response(text("Done.")), log_path=log)
    run_tool(con, response(text("no call")), log_path=log)
    first, second = [json.loads(line) for line in log.read_text().splitlines()]
    assert first["status"] == "ok" and second["status"] == "no_call"
    assert first["calls"] == [{"input": {"filters": [B737]}, "errors": [], "counts": [2]}]
    assert first["results"] is None
