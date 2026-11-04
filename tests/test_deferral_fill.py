"""Filled-in deferral flags for query 10: candidate sets and their breakdown (tiny in-memory data)."""

import duckdb
import pytest

import deferral_fill as fill


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("CREATE TABLE reports (acn BIGINT, year BIGINT, maintenance_deferred VARCHAR)")
    con.execute("CREATE TABLE narratives (acn BIGINT, part BIGINT, text VARCHAR)")
    con.executemany("INSERT INTO reports VALUES (?, ?, ?)", [
        (1, 2019, "Y"), (2, 2019, "N"), (3, 2019, None), (4, 2019, None), (5, 2019, None), (6, 2020, None),
    ])
    con.executemany("INSERT INTO narratives VALUES (?, ?, ?)", [
        (1, 1, "Placed the pack on MEL."), (2, 1, "Repair was deferred, then done."), (3, 1, "Deferred per MEL."),
        (4, 1, "Routine check."), (5, 1, "Left the fix for the next night."), (6, 1, "Deferred."),
    ])
    return con


def test_candidate_sets_fill_only_the_blanks(con):
    spec = {"year_min": 2019, "year_max": 2019, "text_any": ["defer*"]}
    labels = {a: {"label": int(a == 5)} for a in (3, 4, 5)}
    sets, flags = fill.candidate_sets(con, 2019, spec, labels)
    assert sets == {"keyword": {2, 3}, "flag": {1}, "flag + rule": {1, 3}, "flag + llm": {1, 5}}
    assert fill.breakdown(sets["flag + rule"], flags, sets["keyword"]) == \
        {"reports": 2, "coded_y": 1, "coded_n": 0, "blank": 1, "in_keyword": 1}


def test_candidate_sets_need_every_blank_labeled(con):
    with pytest.raises(SystemExit, match="1 blank 2019 reports have no LLM label"):
        fill.candidate_sets(con, 2019, {"year_min": 2019, "year_max": 2019}, {3: {"label": 0}, 4: {"label": 0}})
