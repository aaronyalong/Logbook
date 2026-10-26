"""Streamlit app: citation formatting, plus a smoke test with a fake client and a tiny database."""

import json
from pathlib import Path

import anthropic
import duckdb
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import answer as ans
import app
import build_filter as bf
import preprocess
import retrieve
from fakes import FakeClient, response, text, tool_use

APP = Path(app.__file__)
CITATIONS = {"cited": [1, 2, 9], "verified": [1, 2], "not_shown": [], "not_in_corpus": [9]}


def test_format_answer_marks_each_citation():
    out = app.format_answer("Fatigue [ACN 2, ACN 1] and [ACN 9].", CITATIONS)
    assert out == "Fatigue :blue-background[ACN 1] :blue-background[ACN 2] and :red-background[ACN 9 ⚠]."


def test_format_answer_escapes_dollar_signs():
    assert app.format_answer("Cost $5 to $10 [ACN 1]", CITATIONS) == r"Cost \$5 to \$10 :blue-background[ACN 1]"


def test_examples_come_from_the_query_set():
    assert len(app.examples()) == len(app.EXAMPLE_IDS)


class FakeRetriever:
    def search(self, question, acns=None, k=10, mode="hybrid"):
        return [{"acn": 1, "part": 1, "text": "Missed the MLG inspection after a double shift.", "score": 0.03}]


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    db = tmp_path / "asrs.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE reports (acn BIGINT, year BIGINT, make_model VARCHAR, synopsis VARCHAR, "
                    "human_factors VARCHAR[])")
        con.execute("CREATE TABLE narratives (acn BIGINT, part BIGINT, text VARCHAR)")
        con.execute("INSERT INTO reports VALUES (1, 2019, 'B737-800', 'Technician missed an inspection item.', "
                    "['Fatigue'])")
        con.execute("INSERT INTO narratives VALUES (1, 1, 'Missed the MLG inspection after a double shift.')")

    client = FakeClient(response(tool_use([{"where": {"human_factors": ["Fatigue"]}}])), response(text("Chosen.")),
                        response(text("Fatigue led to a missed item [ACN 1].")))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(preprocess, "DB_PATH", db)
    monkeypatch.setattr(retrieve, "load", lambda con: FakeRetriever())
    monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
    monkeypatch.setattr(ans, "LOG_PATH", tmp_path / "answer.jsonl")
    monkeypatch.setattr(bf, "LOG_PATH", tmp_path / "filter.jsonl")
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def test_question_produces_cited_answer_and_feedback(fake_app):
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("What do reporters say about fatigue?").run()
    assert not at.exception

    markdown = " ".join(m.value for m in at.markdown)
    assert "Fatigue led to a missed item :blue-background[ACN 1]." in markdown
    assert "**Sources** (1 reports)" in markdown
    assert [e.label for e in at.expander] == ["ACN 1 · 2019 · B737-800", "How this answer was built"]
    assert json.loads((fake_app / "answer.jsonl").read_text())["citations"]["verified"] == [1]

    assert at.button(key="send-0").disabled  # nothing to send yet
    at.text_input(key="comment-0").input("Clear and well cited.").run()
    at.button(key="send-0").click().run()
    entry = json.loads((fake_app / "feedback.jsonl").read_text())
    assert (entry["comment"], entry["mode"]) == ("Clear and well cited.", "hybrid")
    assert any("Feedback sent" in c.value for c in at.caption)


def test_missing_api_key_stops_with_instructions(fake_app, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert "ANTHROPIC_API_KEY is not set" in at.error[0].value
