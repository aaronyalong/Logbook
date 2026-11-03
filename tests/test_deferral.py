"""Deferral classifier: dataset, split, keyword rule, and metrics (tiny in-memory data, no training)."""

import json

import duckdb
import numpy as np
import pytest

import deferral as dfr


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE reports (acn BIGINT, year BIGINT, maintenance_deferred VARCHAR)")
    con.execute("CREATE TABLE narratives (acn BIGINT, part BIGINT, text VARCHAR)")
    con.executemany("INSERT INTO reports VALUES (?, ?, ?)", [(1, 2019, "Y"), (2, 2020, "N"), (3, 2021, None)])
    con.executemany("INSERT INTO narratives VALUES (?, ?, ?)", [
        (1, 2, "Second reporter."), (1, 1, "Item deferred per MEL."), (2, 1, "Routine check."), (3, 1, "Uncoded."),
    ])
    return con


def test_dataset_joins_parts_in_order_and_skips_uncoded(con):
    assert dfr.load_dataset(con) == [
        {"acn": 1, "year": 2019, "text": "Item deferred per MEL.\n\nSecond reporter.", "label": 1},
        {"acn": 2, "year": 2020, "text": "Routine check.", "label": 0},
    ]


def test_split_is_stratified_disjoint_and_repeatable():
    data = [{"acn": i, "label": int(i % 4 == 0)} for i in range(100)]
    split = dfr.make_split(data)
    assert not set(split["train"]) & set(split["test"])
    assert len(split["test"]) == 20
    assert sum(a % 4 == 0 for a in split["test"]) == 5  # same 25% positive rate as the whole set
    assert dfr.make_split(data) == split


def test_saved_split_must_match_the_data(tmp_path):
    data = [{"acn": i, "label": i % 2} for i in range(10)]
    path = tmp_path / "split.json"
    assert dfr.load_split(data, path) == json.loads(path.read_text())
    with pytest.raises(SystemExit, match="does not match"):
        dfr.load_split(data[:-1], path)


def test_keyword_rule_matches_whole_words_and_prefixes():
    texts = ["Item deferred per MEL.", "Deferral form signed.", "MELs reviewed.", "Extremely tight access.", "None."]
    assert dfr.keyword_predict(texts).tolist() == [1, 1, 1, 0, 0]


def test_scores_treat_y_as_the_positive_class():
    s = dfr.scores([1, 1, 0, 0], [1, 0, 1, 0])
    assert (s["precision"], s["recall"], s["f1"], s["accuracy"]) == (0.5, 0.5, 0.5, 0.5)
    assert dfr.scores([1, 0], [0, 0])["f1"] == 0.0


def test_f1_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    pred = np.where(rng.random(200) < 0.8, y, 1 - y)
    lo, hi = dfr.f1_interval(y, pred, n=200)
    assert 0 <= lo <= dfr.scores(y, pred)["f1"] <= hi <= 1
