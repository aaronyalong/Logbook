"""Consistency checks for the evaluation query set."""

import json
from collections import Counter
from pathlib import Path

from filter_spec import validate

QUERIES = json.loads((Path(__file__).resolve().parent.parent / "evaluation/queries.json").read_text())["queries"]


def test_ids_are_1_to_25():
    assert [q["id"] for q in QUERIES] == list(range(1, 26))


def test_category_sizes():
    assert Counter(q["category"] for q in QUERIES) == {
        "common": 5, "tool_use": 6, "ambiguous": 5, "edge_case": 5, "should_not_answer": 4,
    }


def test_gold_filters_are_valid():
    for q in QUERIES:
        gold = q["gold_filter"]
        for spec in gold if isinstance(gold, list) else [gold] if gold else []:
            assert validate(spec) == [], f"query {q['id']}: {validate(spec)}"


def test_only_query_19_is_scored_without_a_filter():
    assert [q["id"] for q in QUERIES if q["filter_scored"] and q["gold_filter"] is None] == [19]
