import duckdb
import pytest

from filter_spec import run, validate

REPORTS = [
    # acn, year, family, human_factors, anomaly, time_of_day, component
    (1, 2022, "B737", ["Fatigue"], ["Aircraft Equipment Problem Critical"], "1801-2400", "Main Gear"),
    (2, 2023, "B737", ["Time Pressure"],
     ["Aircraft Equipment Problem Critical", "Deviation / Discrepancy - Procedural Maintenance"], "0601-1200", "Main Gear"),
    (3, 2019, "A320 family", ["Time Pressure", "Fatigue"],
     ["Deviation / Discrepancy - Procedural Maintenance"], None, "Turbine Engine"),
    (4, 2024, None, [], ["Aircraft Equipment Problem Less Severe"], "0001-0600", None),
]
NARRATIVES = [
    (1, 1, "Shift turnover was rushed and the MLG inspection was missed."),
    (2, 1, "Item deferred per MEL."),
    (3, 1, "Borescope inspection of the engine."),
    (4, 1, "Routine check."),
    (4, 2, "Second reporter: the HANDOFF to the night crew was incomplete."),
]


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("""CREATE TABLE reports (acn BIGINT, year BIGINT, aircraft_family VARCHAR, human_factors VARCHAR[],
                   anomaly VARCHAR[], time_of_day VARCHAR, component VARCHAR)""")
    con.execute("CREATE TABLE narratives (acn BIGINT, part BIGINT, text VARCHAR)")
    con.executemany("INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?)", REPORTS)
    con.executemany("INSERT INTO narratives VALUES (?, ?, ?)", NARRATIVES)
    return con


@pytest.mark.parametrize("spec, acns", [
    ({}, [1, 2, 3, 4]),
    ({"year_min": 2022, "year_max": 2023}, [1, 2]),                                # bounds are inclusive
    ({"where": {"aircraft_family": ["B737", "A320 family"]}}, [1, 2, 3]),            # OR within a field
    ({"where": {"human_factors": ["Fatigue"]}}, [1, 3]),                             # list column: any overlap
    ({"where": {"aircraft_family": ["B737"], "human_factors": ["Fatigue"]}}, [1]),  # AND across fields
    ({"where": {"time_of_day": ["1801-2400", "0001-0600"]}}, [1, 4]),
    ({"text_any": ["turnover"]}, [1]),
    ({"text_any": ["handoff"]}, [4]),                                                # case-insensitive, any narrative
    ({"text_any": ["turnover", "handoff"]}, [1, 4]),
    ({"acn": [3]}, [3]),
    ({"acn": [999]}, []),
])
def test_run_matches(con, spec, acns):
    result = run(spec, con)
    assert result["acns"] == acns
    assert result["count"] == len(acns)


def test_group_by_list_column(con):
    result = run({"where": {"human_factors": ["Time Pressure"]}, "group_by": "anomaly"}, con)
    assert result["count"] == 2
    assert result["groups"] == {
        "Deviation / Discrepancy - Procedural Maintenance": 2,
        "Aircraft Equipment Problem Critical": 1,
    }


def test_group_by_scalar_column(con):
    assert run({"group_by": "component"}, con)["groups"] == {"Main Gear": 2, "Turbine Engine": 1}


def test_text_is_parameterized(con):
    assert run({"text_any": ["'; DROP TABLE reports; --"]}, con)["count"] == 0
    assert con.execute("SELECT count(*) FROM reports").fetchone()[0] == len(REPORTS)


def test_run_rejects_invalid_spec(con):
    with pytest.raises(ValueError, match="not a value of 'make_model'"):
        run({"where": {"make_model": ["Spacecraft"]}}, con)


@pytest.mark.parametrize("spec, message", [
    ("B737", "must be a JSON object"),
    ({"aircraft": "B737"}, "unknown key 'aircraft'"),
    ({"where": {"aircraft_type": ["B737"]}}, "unknown field 'aircraft_type'"),
    ({"where": {"aircraft_family": ["Boeing 737"]}}, "'Boeing 737' is not a value of 'aircraft_family'"),
    ({"where": {"aircraft_family": []}}, "must be a non-empty list"),
    ({"where": {"aircraft_family": "B737"}}, "must be a non-empty list"),
    ({"year_min": 1995}, "'year_min' must be an integer"),
    ({"year_max": "2024"}, "'year_max' must be an integer"),
    ({"year_min": 2024, "year_max": 2020}, "'year_min' is after 'year_max'"),
    ({"acn": ["1884087"]}, "'acn' must be a list of integers"),
    ({"text_any": [""]}, "'text_any' must be a list of non-empty strings"),
    ({"group_by": "airline"}, "cannot group by unknown field 'airline'"),
    # malformed shapes an LLM might produce: reported, not raised
    ({"where": ["B737"]}, "'where' must be an object"),
    ({"where": {"aircraft_family": [["B737"]]}}, "is not a value of 'aircraft_family'"),
    ({"group_by": ["anomaly"]}, "cannot group by unknown field"),
    ({"year_min": "2022", "year_max": 2024}, "'year_min' must be an integer"),
])
def test_validate_rejects(spec, message):
    assert any(message in e for e in validate(spec))


def test_validate_accepts_full_spec():
    spec = {"acn": [1], "year_min": 2019, "year_max": 2024, "where": {"human_factors": ["Fatigue"]},
            "text_any": ["inspection"], "group_by": "anomaly"}
    assert validate(spec) == []
