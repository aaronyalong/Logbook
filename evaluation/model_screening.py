"""Model screening: can a model drive the build_filter tool?

Runs only the filter step, through build_filter() itself, so the system prompt, tool schema,
validation, and execution are the pipeline's; only the model changes (code/llm.py). Thinking stays
off for local models that allow it unless --think is given. Results are appended to
evaluation/results/model_screening.jsonl; see evaluation/model_screening.md.

Usage: python evaluation/model_screening.py MODEL [QUERY_ID ...] [--runs N] [--think]
       (default queries: 6 10 19)
Claude models require ANTHROPIC_API_KEY; others require a running Ollama server with MODEL pulled.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import duckdb  # noqa: E402

import build_filter as bf  # noqa: E402
from filter_spec import run  # noqa: E402

OUT = ROOT / "evaluation/results/model_screening.jsonl"
ROUND = 3  # 1: before the backlog section 8 fixes; 2: after them; 3: after reading tool calls written as text (8.6)


def check(model, query, con, think=False):
    start = time.time()
    record = bf.build_filter(query["query"], con=con, model=model, log_path=None, think=think)
    return {"model": model, "id": query["id"], "seconds": round(time.time() - start, 1), "status": record["status"],
            "filters": record["filters"], "counts": [r["count"] for r in record["results"] or []] or None,
            "calls": [{"input": c["input"], "errors": c["errors"], "counts": [r["count"] for r in c.get("results", [])],
                       "from_text": c.get("from_text", False)} for c in record["calls"]],
            "final_text": record["explanation"][:400], "usage": record["usage"], "served_by": record["model"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("ids", nargs="*", type=int, default=[6, 10, 19])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--think", action="store_true", help="local models: turn thinking on")
    args = parser.parse_args()
    queries = {q["id"]: q for q in json.loads((ROOT / "evaluation/queries.json").read_text())["queries"]}
    con = duckdb.connect(str(ROOT / "data/asrs.duckdb"), read_only=True)
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    for run_no in range(1, args.runs + 1):
        for qid in args.ids:
            query = queries[qid]
            r = check(args.model, query, con, args.think)
            gold = query["gold_filter"]
            r["gold_counts"] = [run(s, con)["count"] for s in (gold if isinstance(gold, list) else [gold])] if gold else None
            r.update(think=args.think or None, round=ROUND, run=run_no, commit=commit, source="screening run")
            with OUT.open("a") as f:
                f.write(json.dumps(r) + "\n")
            usage = r["usage"]
            speed = f"out {usage.get('output_tokens', 0)} tok"
            if usage.get("output_seconds"):
                speed += f" @ {usage['output_tokens'] / usage['output_seconds']:.1f}/s"
            print(f"{args.model:16} run {run_no} Q{qid:<2} {r['status']:8} {r['seconds']:5}s  calls={len(r['calls'])} "
                  f"rejected={sum(bool(c['errors']) for c in r['calls'])}  counts={r['counts']} "
                  f"gold={r['gold_counts']}  {speed}", flush=True)
            print(f"   filters: {json.dumps(r['filters'])}", flush=True)
            for c in r["calls"]:
                if c["errors"]:
                    print(f"   rejected: {c['errors'][:3]}", flush=True)
            if r["status"] != "ok":
                print(f"   text: {r['final_text']!r}", flush=True)


if __name__ == "__main__":
    main()
