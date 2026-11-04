"""Fill in the blank deferral flags and compare the report sets evaluation query 10 would search.

Query 10 compares what reporters said about deferred maintenance in 2019 and 2023. Its gold
filter is a keyword search (text_any "defer*") because the maintenance_deferred flag is blank for
62% of reports. For each year this compares four candidate sets:

  keyword            the gold filter: a narrative contains "defer*"
  flag               reports the analysts coded Y (blanks are left out)
  flag + rule        coded Y, plus blank reports the keyword rule ("defer*", "MEL") labels Y
  flag + llm         coded Y, plus blank reports Claude Haiku labels Y (nlp/deferral_llm.py --split blank)

Each set is broken down by the analysts' flag (Y, N, blank), so it shows how much of each set the
analysts confirm and how many coded-Y reports the keyword search misses. Then the pipeline's hybrid
retrieval runs the query over each set, to show how much the top TOP_K narratives (what the answer
model reads) actually change. Nothing in the pipeline changes; this measures whether a filled-in
flag would be worth wiring in.

Usage: python nlp/deferral_fill.py   (needs the Haiku labels for the two years' blank reports)
Output: nlp/results/deferral_fill.json, and a table on stdout.
"""

import json

import duckdb

import deferral as dfr
import deferral_llm as llm
import filter_spec as fs
import retrieve
from preprocess import DB_PATH

QUERIES_PATH = dfr.ROOT / "evaluation/queries.json"
FILL_PATH = dfr.ROOT / "nlp/results/deferral_fill.json"
QUERY_ID = 10
TOP_K = 10  # narratives the pipeline retrieves per filter (code/answer.py)


def query(path=QUERIES_PATH, query_id=QUERY_ID):
    return next(q for q in json.loads(path.read_text())["queries"] if q["id"] == query_id)


def candidate_sets(con, year, keyword_spec, llm_labels):
    """{name: set of ACNs} for one year, plus each report's analyst flag ("Y", "N", or None)."""
    flags = dict(con.execute("SELECT acn, maintenance_deferred FROM reports WHERE year = ?", [year]).fetchall())
    blank = [d for d in dfr.load_dataset(con, blank=True) if d["year"] == year]
    missing = [d["acn"] for d in blank if d["acn"] not in llm_labels]
    if missing:
        raise SystemExit(f"{len(missing)} blank {year} reports have no LLM label; run: "
                         f"python nlp/deferral_llm.py --split blank --years {year}")
    coded_y = {a for a, f in flags.items() if f == "Y"}
    rule = dfr.keyword_predict([d["text"] for d in blank])
    sets = {
        "keyword": set(fs.run(keyword_spec, con)["acns"]),
        "flag": set(fs.run({"year_min": year, "year_max": year, "where": {"maintenance_deferred": ["Y"]}}, con)["acns"]),
        "flag + rule": coded_y | {d["acn"] for d, r in zip(blank, rule) if r},
        "flag + llm": coded_y | {d["acn"] for d in blank if llm_labels[d["acn"]]["label"] == 1},
    }
    return sets, flags


def breakdown(acns, flags, keyword):
    return {"reports": len(acns), "coded_y": sum(flags[a] == "Y" for a in acns),
            "coded_n": sum(flags[a] == "N" for a in acns), "blank": sum(flags[a] is None for a in acns),
            "in_keyword": len(acns & keyword)}


def evaluate(con):
    labels = llm.load_labels()
    q = query()
    retriever = retrieve.load(con)
    results = {}
    for spec in q["gold_filter"]:
        year = spec["year_min"]
        sets, flags = candidate_sets(con, year, spec, labels)
        # the reports behind the top TOP_K narratives; a report with two narratives can appear twice
        top = {name: set(h["acn"] for h in retriever.search(q["query"], sorted(acns), TOP_K, "hybrid"))
               for name, acns in sets.items()}
        results[year] = {
            "reports_in_year": len(flags),
            "coded_y_in_year": sum(f == "Y" for f in flags.values()),
            "blank_in_year": sum(f is None for f in flags.values()),
            "sets": {name: breakdown(acns, flags, sets["keyword"]) for name, acns in sets.items()},
            "retrieved": {name: breakdown(acns, flags, top["keyword"]) for name, acns in top.items()},
        }
    return results


def table(results):
    lines = []
    for year, r in results.items():
        lines += [f"{year}: {r['reports_in_year']} reports, {r['coded_y_in_year']} coded Y, {r['blank_in_year']} blank",
                  "| Set | Reports | Coded Y | Coded N | Blank | Also in keyword set | Retrieved reports "
                  "(coded Y / blank) | Also retrieved from keyword set |", "|---|---|---|---|---|---|---|---|"]
        for name, s in r["sets"].items():
            t = r["retrieved"][name]
            lines.append(f"| {name} | {s['reports']} | {s['coded_y']} | {s['coded_n']} | {s['blank']} | {s['in_keyword']} "
                         f"| {t['reports']} ({t['coded_y']} / {t['blank']}) | {t['in_keyword']} |")
        lines.append("")
    return "\n".join(lines)


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    results = evaluate(con)
    FILL_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(table(results))


if __name__ == "__main__":
    main()
