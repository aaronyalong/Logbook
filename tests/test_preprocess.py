import pandas as pd
import pytest

from preprocess import COLUMNS, aircraft_family, build_tables, load_abbreviations, make_expander, split_values


@pytest.mark.parametrize("cell, expected", [
    ("Taxi", ["Taxi"]),
    ("Parked; Taxi", ["Parked", "Taxi"]),
    ("Taxi; Other Tow/Tug to Hangar", ["Taxi", "Other"]),
    ("Other / Unknown", ["Other / Unknown"]),
    ("Other Tow; Other Towing", ["Other"]),
    ("Conflict Ground Conflict; Critical; Ground Incursion Runway",
     ["Conflict Ground Conflict Critical", "Ground Incursion Runway"]),
    (float("nan"), []),
])
def test_split_values(cell, expected):
    assert split_values(cell) == expected


@pytest.mark.parametrize("make_model, family", [
    ("B737-800", "B737"),
    ("B737 MAX 9", "B737"),
    ("B737 Next Generation Undifferentiated", "B737"),
    ("A321", "A320 family"),
    ("Airbus 318/319/320/321 Undifferentiated", "A320 family"),
    ("Regional Jet 900 (CRJ900)", "CRJ"),
    ("EMB ERJ 170/175 ER/LR", "E-Jet"),
    ("EMB ERJ 145 ER/LR", "ERJ 135/145"),
    ("MD-80 Series (DC-9-80) Undifferentiated or Other Model", "MD-80"),
    ("MD-88", "MD-80"),
    ("Q400", "Dash 8"),
    ("MD Helicopter 500/C/D/E/L", None),
    ("Commercial Fixed Wing", None),
    (None, None),
])
def test_aircraft_family(make_model, family):
    assert aircraft_family(make_model) == family


@pytest.fixture(scope="module")
def expand():
    return make_expander(load_abbreviations())


@pytest.mark.parametrize("text, expected", [
    ("Deferred per MEL.", "Deferred per MEL (Minimum Equipment List)."),
    ("Checked the AMM and IPC", "Checked the AMM (Aircraft Maintenance Manual) and IPC (Illustrated Parts Catalog)"),
    ("Wrong P/N installed", "Wrong P/N (Part Number) installed"),
    ("Per MM section", "Per MM (Maintenance Manual) section"),  # supplement overrides ASRS "Middle Marker"
])
def test_expander_expands(expand, text, expected):
    assert expand(text) == expected


@pytest.mark.parametrize("text", [
    "The AC was inoperative",        # ambiguous: do_not_expand
    "Departed ZZZ1 for ZZZ",         # de-identification placeholders
    "Temperature 20 C at Z time",    # single letters are never expanded
    "the MELT point",                # abbreviation inside a longer token
    "per the mel",                   # match is case-sensitive
])
def test_expander_leaves_alone(expand, text):
    assert expand(text) == text


def test_abbreviation_table():
    table = load_abbreviations()
    assert table["ATA"].startswith("ATA chapter")
    assert "AC" not in table and "TR" not in table
    assert all(len(abbr) > 1 for abbr in table)


def test_build_tables():
    raw = pd.DataFrame([{col: None for col in COLUMNS.values()} for _ in range(2)])
    raw["acn"] = ["1000001", "1000002"]
    raw["date"] = ["202203", "201911"]
    raw["make_model"] = ["B737-800", "Commercial Fixed Wing"]
    raw["anomaly"] = ["Aircraft Equipment Problem Critical", None]
    raw["narrative_1"] = ["First narrative per MEL.", "Only narrative."]
    raw["narrative_2"] = ["Second reporter.", None]

    reports, narratives = build_tables(raw)

    assert reports["acn"].tolist() == [1000001, 1000002]
    assert reports[["year", "month"]].values.tolist() == [[2022, 3], [2019, 11]]
    assert reports["aircraft_family"].iloc[0] == "B737"
    assert pd.isna(reports["aircraft_family"].iloc[1])
    assert reports["anomaly"].tolist() == [["Aircraft Equipment Problem Critical"], []]
    assert narratives[["acn", "part"]].values.tolist() == [[1000001, 1], [1000001, 2], [1000002, 1]]
    assert narratives["text"].iloc[0] == "First narrative per MEL."
    assert "MEL (Minimum Equipment List)" in narratives["text_expanded"].iloc[0]
