"""Load the raw ASRS export into DuckDB.

Builds two tables in data/asrs.duckdb:
  reports     one row per ACN: coded metadata fields (multi-valued fields as lists),
              synopsis, and the original narratives
  narratives  one row per reporter narrative (retrieval unit): original text for
              display/citation, and abbreviation-expanded text for indexing

Usage: python code/preprocess.py
"""

import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data/raw/asrs_maint_2010_2026.csv"
ASRS_ABBR = ROOT / "data/reference/asrs_abbreviations.json"
MAINT_ABBR = ROOT / "data/reference/maintenance_abbreviations.json"
DB_PATH = ROOT / "data/asrs.duckdb"

# (header row 1, header row 2) -> output column. The export has a two-row header;
# columns not listed here are dropped (mostly UAS, ATC, and Aircraft 2 fields, all near-empty).
COLUMNS = {
    ("", "ACN"): "acn",
    ("Time", "Date"): "date",
    ("Time", "Local Time Of Day"): "time_of_day",
    ("Place", "State Reference"): "state",
    ("Environment", "Work Environment Factor"): "work_environment",
    ("Environment", "Light"): "light",
    ("Aircraft 1", "Aircraft Operator"): "aircraft_operator",
    ("Aircraft 1", "Make Model Name"): "make_model",
    ("Aircraft 1", "Operating Under FAR Part"): "far_part",
    ("Aircraft 1", "Mission"): "mission",
    ("Aircraft 1", "Flight Phase"): "flight_phase",
    ("Aircraft 1", "Maintenance Status.Maintenance Deferred"): "maintenance_deferred",
    ("Aircraft 1", "Maintenance Status.Records Complete"): "records_complete",
    ("Aircraft 1", "Maintenance Status.Released For Service"): "released_for_service",
    ("Aircraft 1", "Maintenance Status.Required / Correct Doc On Board"): "correct_doc_on_board",
    ("Aircraft 1", "Maintenance Status.Maintenance Type"): "maintenance_type",
    ("Aircraft 1", "Maintenance Status.Maintenance Items Involved"): "maintenance_items",
    ("Component", "Aircraft Component"): "component",
    ("Component", "Manufacturer"): "component_manufacturer",
    ("Component", "Problem"): "component_problem",
    ("Person 1", "Location Of Person"): "reporter_location",
    ("Person 1", "Reporter Organization"): "reporter_organization",
    ("Person 1", "Function"): "reporter_function",
    ("Person 1", "Qualification"): "reporter_qualification",
    ("Person 1", "Human Factors"): "human_factors",
    ("Person 1", "Communication Breakdown"): "communication_breakdown",
    ("Events", "Anomaly"): "anomaly",
    ("Events", "Detector"): "detector",
    ("Events", "When Detected"): "when_detected",
    ("Events", "Result"): "result",
    ("Assessments", "Contributing Factors / Situations"): "contributing_factors",
    ("Assessments", "Primary Problem"): "primary_problem",
    ("Report 1", "Narrative"): "narrative_1",
    ("Report 1", "Callback"): "callback_1",
    ("Report 2", "Narrative"): "narrative_2",
    ("Report 2", "Callback"): "callback_2",
    ("Report 1", "Synopsis"): "synopsis",
}

# Coded fields that hold several values separated by "; ".
MULTI_VALUED = [
    "work_environment", "mission", "flight_phase", "maintenance_items", "component_problem",
    "reporter_function", "reporter_qualification", "human_factors",
    "communication_breakdown", "anomaly", "detector", "when_detected", "result",
    "contributing_factors",
]

# make_model has ~225 distinct values; map the common transport families so a
# question about "737s" matches every variant. First match wins.
AIRCRAFT_FAMILIES = [
    ("B717", r"^B717"), ("B727", r"^B727"), ("B737", r"^B737"), ("B747", r"^B747"),
    ("B757", r"^B757"), ("B767", r"^B767"), ("B777", r"^B777"), ("B787", r"^B787"),
    ("A220", r"^A220"), ("A300", r"^A300"), ("A310", r"^A310"),
    ("A320 family", r"^(A318|A319|A320|A321|Airbus 318/319/320/321)"),
    ("A330", r"^A330"), ("A340", r"^A340"), ("A380", r"^A380"),
    ("CRJ", r"^Regional Jet"), ("E-Jet", r"^EMB ERJ 1[79]0"), ("ERJ 135/145", r"^EMB ERJ 1[34][05]"),
    ("MD-80", r"^MD-8"), ("MD-90", r"^MD-90"), ("MD-11", r"^MD-11"),
    ("DC-9", r"^DC-9"), ("DC-10", r"^DC-10"),
    ("Dash 8", r"^(Dash 8|Q400)"), ("ATR", r"^ATR"),
]

# The taxonomy's "Other" option carries analyst free text ("Other During Maintenance",
# "Other Tow", ...), which would otherwise turn into hundreds of one-off filter values.
OTHER_FREE_TEXT = re.compile(r"^Other\b(?! / Unknown).*")

# Severity qualifiers exported as their own list item ("Conflict Ground Conflict; Critical").
SEVERITY_QUALIFIERS = {"Critical", "Less Severe"}


def load_raw():
    df = pd.read_csv(RAW_CSV, header=[0, 1], dtype=str)
    df.columns = [("" if a.startswith("Unnamed") else a.strip(), b) for a, b in df.columns]
    df = df[list(COLUMNS)]
    df.columns = list(COLUMNS.values())
    return df


def aircraft_family(make_model):
    if isinstance(make_model, str):
        for family, pattern in AIRCRAFT_FAMILIES:
            if re.search(pattern, make_model):
                return family
    return None


def split_values(cell):
    """'Taxi; Other Tow' -> ['Taxi', 'Other'] (deduplicated, order kept).

    A standalone severity qualifier is reattached to the value before it:
    'Conflict Ground Conflict; Critical' -> ['Conflict Ground Conflict Critical'].
    """
    if not isinstance(cell, str):
        return []
    values = []
    for v in (OTHER_FREE_TEXT.sub("Other", v.strip()) for v in cell.split(";")):
        if v in SEVERITY_QUALIFIERS and values:
            values[-1] = f"{values[-1]} {v}"
        elif v:
            values.append(v)
    return list(dict.fromkeys(values))


def load_abbreviations():
    """Merge the ASRS list with the maintenance supplement into {ABBR: expansion}.

    ASRS entries with several senses are skipped unless the supplement resolves them;
    single letters (C, E, Z, ...) are skipped because they collide with ordinary tokens.
    """
    asrs = json.loads(ASRS_ABBR.read_text())["abbreviations"]
    maint = json.loads(MAINT_ABBR.read_text())

    table = {abbr: senses[0] for abbr, senses in asrs.items() if len(senses) == 1}
    table.update(maint["overrides"])
    table.update(maint["abbreviations"])
    for abbr in maint["do_not_expand"]:
        table.pop(abbr, None)
    return {abbr: exp for abbr, exp in table.items() if len(abbr) > 1}


def make_expander(table):
    # Case-sensitive whole-token match; "/" counts as part of a token so "P/N" and "L/H" match whole.
    alternation = "|".join(re.escape(a) for a in sorted(table, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w/])({alternation})(?![\w/])")

    def expand(text):
        return pattern.sub(lambda m: f"{m.group(1)} ({table[m.group(1)]})", text)

    return expand


def build_tables(df):
    reports = df.copy()
    reports["acn"] = reports["acn"].astype(int)
    reports["year"] = reports["date"].str[:4].astype(int)
    reports["month"] = reports["date"].str[4:6].astype(int)
    reports = reports.drop(columns="date")
    reports["aircraft_family"] = reports["make_model"].map(aircraft_family)
    for col in MULTI_VALUED:
        reports[col] = reports[col].map(split_values)

    expand = make_expander(load_abbreviations())
    rows = []
    for r in reports.itertuples():
        for part in (1, 2):
            text = getattr(r, f"narrative_{part}")
            if isinstance(text, str) and text.strip():
                rows.append({"acn": r.acn, "part": part, "text": text, "text_expanded": expand(text)})
    narratives = pd.DataFrame(rows)

    leading = ["acn", "year", "month", "aircraft_family"]
    reports = reports[leading + [c for c in reports.columns if c not in leading]]
    return reports, narratives


def main():
    reports, narratives = build_tables(load_raw())

    DB_PATH.unlink(missing_ok=True)
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("CREATE TABLE reports AS SELECT * FROM reports")
        con.execute("CREATE TABLE narratives AS SELECT * FROM narratives")
        n_reports = con.execute("SELECT count(*) FROM reports").fetchone()[0]
        n_narr = con.execute("SELECT count(*) FROM narratives").fetchone()[0]
    print(f"Wrote {n_reports} reports and {n_narr} narratives to {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
