"""Provider layer: Claude and Ollama behind one chat, driven by scripted responses (no API or server)."""

import io
import json
import urllib.error

import duckdb
import pytest

import answer as ans
import build_filter as bf
import llm
from fakes import FakeClient, response, text, tool_use

B737 = {"where": {"aircraft_family": ["B737"]}}


def ollama_reply(content="", tool_calls=None, **extra):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": "gemma4:12b", "message": message, "prompt_eval_count": 100, "prompt_eval_duration": 2e9,
            "eval_count": 20, "eval_duration": 4e9, **extra}


def filter_call(arguments):
    return {"function": {"name": "build_filter", "arguments": arguments}}


@pytest.fixture
def server(monkeypatch):
    """Stands in for Ollama: `server.replies` are returned in order, `server.bodies` records each request."""
    class Server:
        replies, bodies = [], []
    Server.replies, Server.bodies = [], []

    def post(body):
        Server.bodies.append({**body, "messages": list(body["messages"])})
        reply = Server.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(llm, "post", post)
    return Server


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE reports (acn BIGINT, year BIGINT, aircraft_family VARCHAR[], make_model VARCHAR, synopsis VARCHAR)")
    con.execute("CREATE TABLE narratives (acn BIGINT, part BIGINT, text VARCHAR)")
    con.execute("INSERT INTO reports VALUES (1, 2023, ['B737'], 'B737-800', 'Synopsis.')")
    con.execute("INSERT INTO narratives VALUES (1, 1, 'Text.')")
    return con


def test_model_name_chooses_the_provider():
    assert isinstance(llm.open_chat("claude-haiku-5-5", "s", "u", client=FakeClient()), llm.ClaudeChat)
    assert isinstance(llm.open_chat("gemma4:12b", "s", "u"), llm.OllamaChat)


def test_claude_chat_normalizes_the_response():
    client = FakeClient(response(text("Calling."), tool_use([B737])), response(text(" Done. ")))
    chat = llm.open_chat("claude-haiku-5-5", "sys", "question", tools=[bf.TOOL], client=client)
    reply = chat.complete()
    assert (reply.text, reply.stop_reason, reply.calls[0].input) == ("Calling.", "tool_use", {"filters": [B737]})
    assert reply.usage == {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 100}
    chat.add_tool_results([(reply.calls[0].id, "output", True)])
    assert chat.complete().text == "Done."
    last = client.requests[1]["messages"][-1]["content"][0]
    assert (last["type"], last["tool_use_id"], last["content"], last["is_error"]) == ("tool_result", "t1", "output", True)
    assert client.requests[0]["system"] == "sys"


def test_claude_chat_without_tools_sends_no_tool_fields():
    client = FakeClient(response(text("ok")))
    llm.open_chat("claude-haiku-5-5", "sys", "question", client=client).complete()
    assert "tools" not in client.requests[0] and "tool_choice" not in client.requests[0]


def test_ollama_request_matches_the_screening_setup(server):
    server.replies = [ollama_reply("ok")]
    llm.open_chat("gemma4:12b", "sys", "question", tools=[bf.TOOL]).complete()
    body = server.bodies[0]
    assert body["options"] == {"num_ctx": 32768, "temperature": 0}
    assert body["stream"] is False and body["think"] is False
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}]
    assert body["tools"][0]["function"]["parameters"] == bf.TOOL["input_schema"]


@pytest.mark.parametrize("model, think, expected", [
    ("gemma4:12b", True, True), ("qwen3:30b-a3b", False, False), ("gpt-oss:20b", False, "low"), ("llama3.1:8b", False, None),
])
def test_ollama_thinking_setting(server, model, think, expected):
    server.replies = [ollama_reply("ok")]
    llm.open_chat(model, "sys", "question", think=think).complete()
    assert server.bodies[0].get("think") == expected


def test_ollama_reply_is_normalized(server):
    server.replies = [ollama_reply("Answer.", done_reason="stop")]
    reply = llm.open_chat("gemma4:12b", "sys", "question").complete()
    assert (reply.text, reply.model, reply.stop_reason, reply.calls) == ("Answer.", "gemma4:12b", "end_turn", [])
    assert reply.usage == {"input_tokens": 100, "output_tokens": 20, "input_seconds": 2, "output_seconds": 4}


def test_ollama_tool_arguments_sent_as_text_are_decoded(server):
    server.replies = [ollama_reply(tool_calls=[filter_call(json.dumps({"filters": [B737]}))]),
                      ollama_reply(tool_calls=[filter_call("not json")])]
    chat = llm.open_chat("gemma4:12b", "sys", "question", tools=[bf.TOOL])
    assert chat.complete().calls[0].input == {"filters": [B737]}
    assert chat.complete().calls[0].input == {"_unparseable": "not json"}


def test_ollama_parse_failure_with_tools_is_an_error_reply(server):
    server.replies = [urllib.error.HTTPError("url", 500, "error", {}, io.BytesIO(b"error parsing tool call"))]
    reply = llm.open_chat("gpt-oss:20b", "sys", "question", tools=[bf.TOOL]).complete()
    assert reply.error == "server error 500: error parsing tool call"


def test_ollama_server_error_without_tools_is_raised(server):
    server.replies = [urllib.error.HTTPError("url", 500, "error", {}, io.BytesIO(b"out of memory"))]
    with pytest.raises(urllib.error.HTTPError):
        llm.open_chat("gemma4:12b", "sys", "question").complete()


@pytest.mark.parametrize("reply_text, args", [
    ('{"name": "build_filter", "arguments": {"where": {"aircraft_family": ["B737"]}}}', ("build_filter", B737)),
    ('Reasoning first.\n<tool_call>\n{"name": "build_filter", "arguments": {"filters": [{}]}}\n</tool_call>',
     ("build_filter", {"filters": [{}]})),
    ('{"name": "build_filter", "arguments": "{\\"filters\\": [{}]}"}', ("build_filter", {"filters": [{}]})),  # JSON text
    ('{}\nNo filter fits this question.', None),  # bare JSON is not a call
    ('{"name": "other_tool", "arguments": {}}', None),
    ("No JSON here.", None),
])
def test_parse_text_call(reply_text, args):
    assert llm.parse_text_call(reply_text, ["build_filter"]) == args


def test_build_filter_runs_on_ollama(con, server):
    server.replies = [ollama_reply(tool_calls=[filter_call({"filters": [{"year_min": 1}]})]),
                      ollama_reply(tool_calls=[filter_call({"filters": [B737]})]),
                      ollama_reply("737 reports.")]
    record = bf.build_filter("question", con=con, model="gemma4:12b", log_path=None)
    assert record["status"] == "ok" and record["filters"] == [B737] and record["results"][0]["acns"] == [1]
    assert record["model"] == "gemma4:12b" and record["explanation"] == "737 reports."
    assert record["usage"]["input_tokens"] == 300 and record["usage"]["output_seconds"] == 12
    rejected = server.bodies[1]["messages"][-1]
    assert (rejected["role"], rejected["tool_name"]) == ("tool", "build_filter")
    assert rejected["content"].startswith("Rejected, nothing was run")


def test_build_filter_reads_a_call_written_as_text(con, server):
    written = json.dumps({"name": "build_filter", "arguments": B737})
    server.replies = [ollama_reply(written), ollama_reply("Done.")]
    record = bf.build_filter("question", con=con, model="qwen3:30b-a3b", log_path=None)
    assert record["status"] == "ok" and record["calls"][0]["from_text"] is True


def test_build_filter_records_a_tool_call_the_server_cannot_parse(con, server):
    server.replies = [urllib.error.HTTPError("url", 500, "error", {}, io.BytesIO(b"bad call"))]
    record = bf.build_filter("question", con=con, model="gpt-oss:20b", log_path=None)
    assert record["status"] == "invalid" and record["calls"] == [{"input": None, "errors": ["server error 500: bad call"]}]


class FakeRetriever:
    def search(self, question, acns, k, mode):
        return [{"acn": 1, "part": 1, "text": "Text.", "score": 1.0}]


def test_each_pipeline_step_can_use_a_different_model(con, server):
    client = FakeClient(response(tool_use([B737])), response(text("Done.")))
    server.replies = [ollama_reply("Reports describe this [ACN 1].")]
    record = ans.answer("question", client=client, retriever=FakeRetriever(), con=con, log_path=None,
                        filter_log_path=None, model="gemma4:12b", filter_model="claude-haiku-5-5")
    assert record["status"] == "ok" and record["citations"]["verified"] == [1]
    assert record["model"] == "gemma4:12b"
    assert client.requests[0]["model"] == "claude-haiku-5-5"
    assert record["usage"]["answer"]["output_seconds"] == 4
    assert [m["role"] for m in server.bodies[0]["messages"]] == ["system", "user"]
    assert "tools" not in server.bodies[0]
