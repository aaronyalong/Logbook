"""Model screening: can a model drive the build_filter tool?

Runs only the filter step, with the same system prompt, tool schema, validation, and execution
as code/build_filter.py; only the model changes. Claude models go through build_filter() itself;
any other model goes to a local Ollama server with 32K context, temperature 0, and thinking off
where the model allows it (gpt-oss only goes down to "low"), unless --think is given. Results are
appended to evaluation/results/model_screening.jsonl; see evaluation/model_screening.md.

Usage: python evaluation/model_screening.py MODEL [QUERY_ID ...] [--runs N] [--think]
       (default queries: 6 10 19)
Claude models require ANTHROPIC_API_KEY; others require a running Ollama server with MODEL pulled.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import duckdb  # noqa: E402

import build_filter as bf  # noqa: E402
from filter_spec import run  # noqa: E402

THINKING = {"qwen3", "qwen3.5", "gemma4"}  # think can be switched off; gpt-oss only goes down to "low"
OUT = ROOT / "evaluation/results/model_screening.jsonl"
ROUND = 2  # round 1: 2026-10-30, before the fixes in backlog section 8


def chat(model, messages, think=False):
    body = {"model": model, "messages": messages, "stream": False,
            "tools": [{"type": "function", "function": {"name": bf.TOOL["name"], "description": bf.TOOL["description"],
                                                        "parameters": bf.TOOL["input_schema"]}}],
            "options": {"num_ctx": 32768, "temperature": 0}}
    family = model.split(":")[0]
    if think:
        body["think"] = True
    elif family in THINKING:
        body["think"] = False
    elif family == "gpt-oss":
        body["think"] = "low"
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode())
    return json.load(urllib.request.urlopen(req, timeout=3600))


def check_claude(model, query, con):
    start = time.time()
    record = bf.build_filter(query["query"], con=con, model=model, log_path=None)
    return {"model": model, "id": query["id"], "seconds": round(time.time() - start, 1), "status": record["status"],
            "filters": record["filters"], "counts": [r["count"] for r in record["results"] or []] or None,
            "calls": [{"input": c["input"], "errors": c["errors"], "counts": [r["count"] for r in c.get("results", [])]}
                      for c in record["calls"]],
            "final_text": record["explanation"][:400], "usage": record["usage"], "served_by": record["model"]}


def check(model, query, con, think=False):
    messages = [{"role": "system", "content": bf.system_prompt()}, {"role": "user", "content": query["query"]}]
    calls, accepted, timing, text = [], None, {"prompt": 0, "prompt_s": 0.0, "out": 0, "out_s": 0.0}, ""
    start = time.time()
    for _ in range(bf.MAX_CALLS):
        try:
            r = chat(model, messages, think)
        except urllib.error.HTTPError as e:
            # Ollama answers 500 when it cannot parse the model's output as a tool call.
            calls.append({"name": None, "input": None, "errors": [f"server error {e.code}: {e.read().decode()[:300]}"],
                          "counts": []})
            break
        timing["prompt"] += r.get("prompt_eval_count", 0)
        timing["prompt_s"] += r.get("prompt_eval_duration", 0) / 1e9
        timing["out"] += r.get("eval_count", 0)
        timing["out_s"] += r.get("eval_duration", 0) / 1e9
        msg = r["message"]
        text = msg.get("content", "")
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        messages.append(msg)
        for tc in tool_calls:
            args = tc["function"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparseable": args}
            call = bf.execute(args, con)
            calls.append({"name": tc["function"].get("name"), "input": args, "errors": call["errors"],
                          "counts": [x["count"] for x in call.get("results", [])]})
            if not call["errors"]:
                accepted = call
            messages.append({"role": "tool", "tool_name": "build_filter", "content": bf.tool_output(call)})

    t = timing
    return {"model": model, "id": query["id"], "seconds": round(time.time() - start),
            "status": "ok" if accepted else ("invalid" if calls else "no_call"),
            "filters": accepted["filters"] if accepted else None,
            "counts": [x["count"] for x in accepted["results"]] if accepted else None,
            "calls": calls, "final_text": text[:400],
            "tokens": {"prompt": t["prompt"], "output": t["out"], "output_per_s": round(t["out"] / max(t["out_s"], 1e-9), 1)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("ids", nargs="*", type=int, default=[6, 10, 19])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--think", action="store_true", help="local models: turn thinking on")
    args = parser.parse_args()
    queries = {q["id"]: q for q in json.loads((ROOT / "evaluation/queries.json").read_text())["queries"]}
    con = duckdb.connect(str(ROOT / "data/asrs.duckdb"), read_only=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    claude = args.model.startswith("claude")
    for run_no in range(1, args.runs + 1):
        for qid in args.ids:
            query = queries[qid]
            r = check_claude(args.model, query, con) if claude else check(args.model, query, con, args.think)
            gold = query["gold_filter"]
            r["gold_counts"] = [run(s, con)["count"] for s in (gold if isinstance(gold, list) else [gold])] if gold else None
            r.update(think=args.think or None, round=ROUND, run=run_no, commit=commit, source="screening run")
            with OUT.open("a") as f:
                f.write(json.dumps(r) + "\n")
            speed = (f"out {r['usage']['output_tokens']} tok" if claude
                     else f"out {r['tokens']['output']} tok @ {r['tokens']['output_per_s']}/s")
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
