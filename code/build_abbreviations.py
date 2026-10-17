"""Parse the ASRS Abbreviations PDF (Decode section) into JSON.

The Decode section (pages 8-13) is a two-column table of ABBREVIATION -> Expansion.
Long expansions wrap onto two lines with the abbreviation vertically centred between
them, and some abbreviations wrap too (e.g. "BAL (S,ED," / "ING)"), so rows are
rebuilt from word coordinates rather than from extracted text lines.

Usage: python code/build_abbreviations.py
Output: data/reference/asrs_abbreviations.json
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data/reference/ASRS_Abbreviations.pdf"
OUT_PATH = ROOT / "data/reference/asrs_abbreviations.json"

DECODE_PAGES = range(7, 13)  # 0-indexed: PDF pages 8-13
LINE_TOLERANCE = 3  # points; words within this vertical distance share a line

# The PDF letter-spaces this entry, which extraction splits into fragments.
EXTRACTION_FIXES = {"ECAM": ["Electronic Centralized Aircraft Monitoring"]}


def group_lines(words):
    """Group words into lines by vertical position. Returns [(y, text)] sorted by y."""
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1]["y"] - w["top"]) <= LINE_TOLERANCE:
            lines[-1]["words"].append(w)
        else:
            lines.append({"y": w["top"], "words": [w]})
    return [(ln["y"], " ".join(w["text"] for w in sorted(ln["words"], key=lambda w: w["x0"])))
            for ln in lines]


def is_incomplete(abbr):
    """An abbreviation fragment that continues on the next line, e.g. 'BAL (S,ED,'."""
    return abbr.endswith(",") or abbr.count("(") > abbr.count(")")


def parse_column(words):
    """Rebuild (abbreviation, expansion) rows for one column of the Decode table."""
    abbr_x = min(w["x0"] for w in words)
    # Expansion column starts at the most common x0 well to the right of the abbreviations.
    term_x = Counter(round(w["x0"]) for w in words if w["x0"] > abbr_x + 30).most_common(1)[0][0]

    abbr_lines = group_lines([w for w in words if w["x0"] < term_x - 5])
    term_lines = group_lines([w for w in words if w["x0"] >= term_x - 5])

    # Join wrapped abbreviation fragments ("BAL (S,ED," + "ING)").
    merged = []
    for y, text in abbr_lines:
        if merged and is_incomplete(merged[-1][1]):
            prev_y, prev_text = merged[-1]
            merged[-1] = ((prev_y + y) / 2, f"{prev_text} {text}")
        else:
            merged.append((y, text))

    # Each expansion line belongs to the vertically nearest abbreviation.
    parts = defaultdict(list)
    for y, text in term_lines:
        nearest = min(range(len(merged)), key=lambda i: abs(merged[i][0] - y))
        parts[nearest].append((y, text))

    rows = []
    for i, (_, abbr) in enumerate(merged):
        if i in parts:
            expansion = " ".join(t for _, t in sorted(parts[i]))
            rows.append((abbr, expansion))
    return rows


def normalize_abbr(abbr):
    """'ALT(S)' -> 'ALT', 'BAL (S,ED, ING)' -> 'BAL'. Suffix notation lists inflections."""
    return re.sub(r"\s*\(.*$", "", abbr).strip()


def main():
    rows = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page_num in DECODE_PAGES:
            page = pdf.pages[page_num]
            words = [w for w in page.extract_words()
                     if w["text"] not in ("Decode",) and "Encode/Decode" not in w["text"]]
            # Drop the page footer ("ASRS Abbreviations - Encode/Decode • May 2024  Page N of 14").
            footer_y = max(w["top"] for w in words if w["text"] == "Page")
            words = [w for w in words if w["top"] < footer_y - LINE_TOLERANCE]
            mid = page.width / 2
            for column in ([w for w in words if w["x0"] < mid], [w for w in words if w["x0"] >= mid]):
                if column:
                    rows.extend(parse_column(column))

    decode = defaultdict(list)
    for abbr, expansion in rows:
        for key in normalize_abbr(abbr).split(", "):  # "MIKE, MIC" lists two abbreviations
            if expansion not in decode[key]:
                decode[key].append(expansion)
    decode.update(EXTRACTION_FIXES)

    out = {
        "source": "ASRS Abbreviations - Encode/Decode, May 2024 (Decode section)",
        "abbreviations": dict(sorted(decode.items())),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    ambiguous = {k: v for k, v in decode.items() if len(v) > 1}
    print(f"Wrote {len(decode)} abbreviations ({len(ambiguous)} ambiguous) to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
