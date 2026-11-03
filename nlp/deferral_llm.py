"""Zero-shot deferral labels from Claude: the flag's definition and the narratives, no training examples.

Labels the reports in one split (the test split by default) and appends one record per report to
nlp/results/deferral_llm.jsonl. Reports already labeled with the same model and prompt are
skipped, so an interrupted run resumes where it stopped. nlp/deferral.py scores the labels
once every test report has one.

The prompt (nlp/prompts/deferral_label_system.md) was written once from the field's definition
and not tuned on the test set; changing it changes prompt_sha, which starts a fresh set of labels.

Usage: python nlp/deferral_llm.py [--model claude-haiku-5-5] [--split test] [--limit N]
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import duckdb

import deferral as dfr
from preprocess import DB_PATH

PROMPT_PATH = dfr.ROOT / "nlp/prompts/deferral_label_system.md"
LABELS_PATH = dfr.ROOT / "nlp/results/deferral_llm.jsonl"
MODEL = "claude-haiku-5-5"
EFFORT = "medium"
WORKERS = 8
TOOL = {
    "name": "record_label",
    "description": "Record the Maintenance Deferred label for this report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "evidence": {"type": "string", "description": "The words from the narratives that decide the label, "
                                                          "quoted exactly; empty if nothing in them bears on it."},
            "deferred": {"type": "string", "enum": ["Y", "N"]},
        },
        "required": ["evidence", "deferred"],
    },
}


def prompt_sha():
    """Identifies the prompt and tool, so labels from an older prompt are never mixed in."""
    text = PROMPT_PATH.read_text() + json.dumps(TOOL, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def label_report(report, client, model=MODEL):
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=PROMPT_PATH.read_text(),
        tools=[TOOL],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"<narratives>\n{report['text']}\n</narratives>"}],
    )
    call = next((b.input for b in response.content if b.type == "tool_use" and b.name == TOOL["name"]), None)
    label = {"Y": 1, "N": 0}.get(call.get("deferred")) if call else None
    return {
        "acn": report["acn"], "model": model, "response_model": response.model, "prompt_sha": prompt_sha(),
        "label": label, "evidence": call.get("evidence", "") if call else None, "stop_reason": response.stop_reason,
        "usage": {k: getattr(response.usage, k, None) or 0 for k in ("input_tokens", "output_tokens")},
    }


def load_labels(path=LABELS_PATH, model=MODEL):
    """{acn: record} for the current prompt and `model`; the latest record wins."""
    if not path.exists():
        return {}
    sha = prompt_sha()
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["acn"]: r for r in records if r["model"] == model and r["prompt_sha"] == sha}


def label_split(reports, client, model=MODEL, path=LABELS_PATH, workers=WORKERS):
    """Labels every report not yet labeled, appending each record as it arrives. Returns the new records."""
    done = load_labels(path, model)
    todo = [r for r in reports if r["acn"] not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = []
    with ThreadPoolExecutor(workers) as pool, path.open("a") as f:
        for record in pool.map(lambda r: label_report(r, client, model), todo):
            f.write(json.dumps(record) + "\n")
            f.flush()
            new.append(record)
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, help="label only the first N reports of the split")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    data = dfr.load_dataset(con)
    by_acn = {d["acn"]: d for d in data}
    reports = [by_acn[a] for a in dfr.load_split(data)[args.split]][:args.limit]
    new = label_split(reports, anthropic.Anthropic(), args.model)

    unlabeled = sum(r["label"] is None for r in new)
    tokens = {k: sum(r["usage"][k] for r in new) for k in ("input_tokens", "output_tokens")}
    print(f"Labeled {len(new)} reports with {args.model} ({unlabeled} without a label); "
          f"{tokens['input_tokens']} input and {tokens['output_tokens']} output tokens")
    agree = [r for r in new if r["label"] is not None]
    if agree:
        correct = sum(r["label"] == by_acn[r["acn"]]["label"] for r in agree)
        print(f"Matches the analysts' flag on {correct} of {len(agree)}")


if __name__ == "__main__":
    main()
