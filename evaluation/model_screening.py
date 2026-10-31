"""Model screening: can a local Ollama model drive the build_filter tool?

Runs only the filter step, with the same system prompt, tool schema, validation, and execution
as code/build_filter.py; only the model changes. Settings: 32K context, temperature 0, thinking
off where the model allows it (gpt-oss only goes down to "low"). Results are appended to
evaluation/results/model_screening.jsonl; see evaluation/model_screening.md.

Claude rows in the results file come from build_filter.build_filter(question, model=...).

Usage: python evaluation/model_screening.py MODEL [QUERY_ID ...]    (default queries: 6 10 19)
Requires a running Ollama server with MODEL pulled.
"""

import json
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


def chat(model, messages):
    body = {"model": model, "messages": messages, "stream": False,
            "tools": [{"type": "function", "function": {"name": bf.TOOL["name"], "description": bf.TOOL["description"],
                                                        "parameters": bf.TOOL["input_schema"]}}],
            "options": {"num_ctx": 32768, "temperature": 0}}
    family = model.split(":")[0]
    if family in THINKING:
        body["think"] = False
    elif family == "gpt-oss":
        body["think"] = "low"
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode())
    return json.load(urllib.request.urlopen(req, timeout=3600))


def check(model, query, con):
    messages = [{"role": "system", "content": bf.system_prompt()}, {"role": "user", "content": query["query"]}]
    calls, accepted, timing, text = [], None, {"prompt": 0, "prompt_s": 0.0, "out": 0, "out_s": 0.0}, ""
    start = time.time()
    for _ in range(bf.MAX_CALLS):
        try:
            r = chat(model, messages)
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

    gold = query["gold_filter"]
    gold_counts = [run(s, con)["count"] for s in (gold if isinstance(gold, list) else [gold])] if gold else None
    return {"model": model, "id": query["id"], "seconds": round(time.time() - start),
            "status": "ok" if accepted else ("invalid" if calls else "no_call"),
            "filters": accepted["input"].get("filters") if accepted else None,
            "counts": [x["count"] for x in accepted["results"]] if accepted else None,
            "gold_counts": gold_counts, "calls": calls, "final_text": text[:400], "timing": timing}


def main():
    model, ids = sys.argv[1], [int(i) for i in sys.argv[2:]] or [6, 10, 19]
    queries = {q["id"]: q for q in json.loads((ROOT / "evaluation/queries.json").read_text())["queries"]}
    con = duckdb.connect(str(ROOT / "data/asrs.duckdb"), read_only=True)
    for qid in ids:
        r = check(model, queries[qid], con)
        t = r.pop("timing")
        r["tokens"] = {"prompt": t["prompt"], "output": t["out"], "output_per_s": round(t["out"] / max(t["out_s"], 1e-9), 1)}
        r["source"] = "screening run"
        with OUT.open("a") as f:
            f.write(json.dumps(r) + "\n")
        t = r["tokens"]
        print(f"{model:16} Q{qid:<2} {r['status']:8} {r['seconds']:4}s  calls={len(r['calls'])} "
              f"rejected={sum(bool(c['errors']) for c in r['calls'])}  counts={r['counts']} gold={r['gold_counts']}  "
              f"out {t['output']} tok @ {t['output_per_s']}/s", flush=True)
        print(f"   filters: {json.dumps(r['filters'])}", flush=True)
        for c in r["calls"]:
            if c["errors"]:
                print(f"   rejected: {c['errors'][:3]}", flush=True)
        if r["status"] != "ok":
            print(f"   text: {r['final_text']!r}", flush=True)


if __name__ == "__main__":
    main()
