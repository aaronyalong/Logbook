"""Answer a question from the corpus: filter -> retrieve -> Claude writes a cited answer.

1. build_filter (code/build_filter.py) chooses the reports the question is about.
2. Hybrid retrieval (code/retrieve.py) ranks their narratives, TOP_K per filter.
3. Claude answers from those reports only (prompts/answer_system.md), citing [ACN n].
4. Every cited ACN is checked after generation: shown to the model (verified), in the
   corpus but not shown (unsupported), or not in the corpus at all (fabricated).

If no valid filter could be built, the whole corpus is searched and the model is told so.
Each answer is logged to logs/answer.jsonl; the filter step logs its calls separately.

Usage: python code/answer.py "What do reporters say about how fatigue affected their work?"
Requires ANTHROPIC_API_KEY.
"""

import argparse
import json
import re
import time
from html import escape

import anthropic
import duckdb

import build_filter as bf
import retrieve
from preprocess import DB_PATH, ROOT

MODEL = "claude-opus-5-5"
EFFORT = "medium"
TOP_K = 10  # narratives retrieved per filter
TOP_GROUPS = 10
PROMPT_PATH = ROOT / "prompts/answer_system.md"
LOG_PATH = ROOT / "logs/answer.jsonl"

# "[ACN 1234567]", "[ACN 1234567, ACN 2345678]", and the loose "[ACN 1234567, 2345678]" or "ACN #1234567".
CITE_GROUP = re.compile(r"\[\s*ACN[^\]]*\]")
CITE_LOOSE = re.compile(r"ACN\s*#?\s*(\d+)")


def cited_acns(text):
    found = {int(n) for group in CITE_GROUP.findall(text) for n in re.findall(r"\d+", group)}
    return sorted(found | {int(n) for n in CITE_LOOSE.findall(text)})


def check_citations(text, shown, con):
    """Sort every cited ACN by whether the model was shown it and whether it exists at all."""
    cited = cited_acns(text)
    exists = {a for (a,) in con.execute("SELECT acn FROM reports WHERE acn IN (SELECT unnest(?))", [cited]).fetchall()}
    return {
        "cited": cited,
        "verified": [a for a in cited if a in shown],
        "not_shown": [a for a in cited if a not in shown and a in exists],
        "not_in_corpus": [a for a in cited if a not in exists],
    }


def report_info(acns, con):
    rows = con.execute(
        "SELECT acn, year, make_model, synopsis FROM reports WHERE acn IN (SELECT unnest(?))", [list(acns)]
    ).fetchall()
    return {acn: {"acn": acn, "year": year, "aircraft": model or "unknown", "synopsis": syn or ""}
            for acn, year, model, syn in rows}


def describe_filter(i, spec, result):
    constraint = json.dumps(spec) if spec else "{} (no constraints; the whole corpus)"
    lines = [f"Filter {i}: {constraint} matched {result['count']} reports in this corpus."]
    if "groups" in result:
        top = "; ".join(f"{v} ({n})" for v, n in list(result["groups"].items())[:TOP_GROUPS])
        lines.append(f"Most common {spec['group_by']} values among them: {top}")
    return "\n".join(lines)


def render_context(filter_record, sides, info):
    if filter_record["status"] == "ok":
        filters = "\n".join(describe_filter(i, s["spec"], s["result"]) for i, s in enumerate(sides, 1))
    else:
        filters = "No valid filter could be built for this question, so the whole corpus was searched."

    blocks = []
    for i, side in enumerate(sides, 1):
        by_report = {}  # rank order of each report's best narrative
        for hit in side["hits"]:
            by_report.setdefault(hit["acn"], []).append(hit)
        reports = []
        for acn, hits in by_report.items():
            r = info[acn]
            narratives = "\n".join(f'<narrative part="{h["part"]}">\n{escape(h["text"], quote=False)}\n</narrative>'
                                   for h in sorted(hits, key=lambda h: h["part"]))
            reports.append(f'<report acn="{acn}" year="{r["year"]}" aircraft="{escape(r["aircraft"])}">\n'
                           f'<synopsis>{escape(r["synopsis"], quote=False)}</synopsis>\n{narratives}\n</report>')
        label = f' filter="{i}"' if len(sides) > 1 else ""
        blocks.append(f"<reports{label}>\n" + "\n".join(reports) + "\n</reports>")
    return f"<filter>\n{filters}\n</filter>\n" + "\n".join(blocks)


def answer(question, client=None, retriever=None, con=None, mode="hybrid",
           log_path=LOG_PATH, filter_log_path=bf.LOG_PATH, on_step=None):
    """Returns a record with status ("ok" or "refusal"), answer text, citation check, sources
    (the verified reports), the filter used, and what was retrieved.

    on_step, if given, is called with "filter", "retrieve", and "answer" as each stage starts.
    """
    client = client or anthropic.Anthropic()
    con = con or duckdb.connect(str(DB_PATH), read_only=True)
    retriever = retriever or retrieve.load(con)
    step = on_step or (lambda stage: None)

    step("filter")
    filt = bf.build_filter(question, client=client, con=con, log_path=filter_log_path)
    record = {"question": question, "status": "refusal", "answer": "", "citations": None, "sources": [],
              "filter": {"status": filt["status"], "filters": filt["filters"],
                         "counts": [r["count"] for r in filt["results"] or []]},
              "retrieved": [], "model": None, "filter_model": filt["model"], "usage": {"filter": filt["usage"]}}
    if filt["status"] == "refusal":
        return _finish(record, log_path)

    if filt["status"] == "ok":
        sides = [{"spec": s, "result": r} for s, r in zip(filt["filters"], filt["results"])]
    else:
        sides = [{"spec": None, "result": None}]
    step("retrieve")
    for side in sides:
        acns = side["result"]["acns"] if side["result"] else None
        side["hits"] = retriever.search(question, acns, TOP_K, mode)
    shown = {h["acn"] for side in sides for h in side["hits"]}
    info = report_info(shown, con)

    step("answer")
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=PROMPT_PATH.read_text(),
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"<context>\n{render_context(filt, sides, info)}\n</context>\n\n"
                                              f"<question>\n{question}\n</question>"}],
    )
    record["model"] = response.model
    record["usage"]["answer"] = {k: getattr(response.usage, k, None) or 0 for k in ("input_tokens", "output_tokens")}
    record["retrieved"] = [[{k: h[k] for k in ("acn", "part", "score")} for h in s["hits"]] for s in sides]
    if response.stop_reason == "refusal":
        return _finish(record, log_path)

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    citations = check_citations(text, shown, con)
    record |= {"status": "ok", "answer": text, "citations": citations,
               "sources": [info[a] for a in citations["verified"]]}
    return _finish(record, log_path)


def _finish(record, log_path):
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--mode", choices=retrieve.MODES, default="hybrid")
    args = parser.parse_args()
    record = answer(args.question, mode=args.mode)

    f = record["filter"]
    print(f"filter: {f['status']} {json.dumps(f['filters'])} -> {f['counts']} reports\n")
    if record["status"] == "refusal":
        print("The model declined to answer this request.")
        return
    print(record["answer"])
    print("\nSources:")
    for s in record["sources"]:
        print(f"  ACN {s['acn']} ({s['year']}, {s['aircraft']}): {s['synopsis'][:120]}")
    c = record["citations"]
    if c["not_shown"]:
        print(f"\nWARNING: cited reports the model was not shown: {c['not_shown']}")
    if c["not_in_corpus"]:
        print(f"WARNING: cited ACNs that do not exist in the corpus: {c['not_in_corpus']}")


if __name__ == "__main__":
    main()
