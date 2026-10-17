import json

import pytest

from build_abbreviations import OUT_PATH, is_incomplete, normalize_abbr, parse_column


def word(text, x, y):
    return {"text": text, "x0": x, "top": y}


def test_parse_column_rebuilds_wrapped_rows():
    words = [
        # Single-line row.
        word("AC", 50, 100), word("Air", 130, 100), word("Conditioning", 150, 100),
        # Expansion wraps onto two lines; abbreviation is centred between them.
        word("ACARS", 50, 125),
        word("Automatic", 130, 120), word("Communications", 180, 120),
        word("Addressing", 130, 130), word("&", 185, 130), word("Reporting", 195, 130), word("System", 245, 130),
        # Abbreviation wraps onto two lines; expansion is centred between them.
        word("BAL", 50, 150), word("(S,ED,", 70, 150), word("ING)", 50, 160),
        word("Balance", 130, 155),
    ]
    assert parse_column(words) == [
        ("AC", "Air Conditioning"),
        ("ACARS", "Automatic Communications Addressing & Reporting System"),
        ("BAL (S,ED, ING)", "Balance"),
    ]


@pytest.mark.parametrize("abbr, expected", [
    ("APU", "APU"), ("ALT(S)", "ALT"), ("BAL (S,ED, ING)", "BAL"), ("APCH CTL", "APCH CTL"),
])
def test_normalize_abbr(abbr, expected):
    assert normalize_abbr(abbr) == expected


@pytest.mark.parametrize("abbr, incomplete", [("BAL (S,ED,", True), ("CHK (S", True), ("ALT(S)", False), ("APU", False)])
def test_is_incomplete(abbr, incomplete):
    assert is_incomplete(abbr) == incomplete


def test_generated_json_spot_checks():
    table = json.loads(OUT_PATH.read_text())["abbreviations"]
    assert table["ACARS"] == ["Automatic Communications Addressing & Reporting System"]
    assert table["TRACON"] == ["Terminal Radar Approach Control Facility"]
    assert table["ECAM"] == ["Electronic Centralized Aircraft Monitoring"]
    assert table["MIKE"] == table["MIC"] == ["Microphone"]
    assert table["AC"] == ["Air Conditioning", "Alternating Current"]
