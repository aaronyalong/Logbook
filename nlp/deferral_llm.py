"""Zero-shot deferral labels from Claude: the flag's definition and the narratives, no training examples.

Labels the reports in one split (the test split by default) and appends one record per report to
nlp/results/deferral_llm.jsonl. Reports already labeled with the same model and prompt are
skipped, so an interrupted run resumes where it stopped. nlp/deferral.py scores the labels
once every test report has one.

Prompts live in nlp/prompts/deferral_label_<version>.md. v1 was written once from the field's
definition. Later versions are developed on the dev split (a fixed sample of training reports)
and scored on the test split only once finished. Editing a prompt changes its prompt_sha, which
starts a fresh set of labels.

`--split blank` labels uncoded reports instead (optionally only `--years`), for filling in the flag
(nlp/deferral_fill.py); those have no analyst label to score against.

Usage: python nlp/deferral_llm.py [--model claude-haiku-5-5] [--prompt v1] [--split test|dev|train|blank]
                                  [--years 2019 2023] [--limit N]
"""

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import duckdb

import deferral as dfr
from preprocess import DB_PATH

PROMPT_DIR = dfr.ROOT / "nlp/prompts"
PROMPT = "v1"
DEV_SIZE = 200
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


def prompt_text(prompt=PROMPT):
    return (PROMPT_DIR / f"deferral_label_{prompt}.md").read_text()


def prompt_sha(prompt=PROMPT):
    """Identifies the prompt's text and the tool, so labels from an edited prompt are never mixed in."""
    text = prompt_text(prompt) + json.dumps(TOOL, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def dev_split(split):
    """A fixed random sample of the training reports, for developing prompts without touching the test set."""
    return sorted(random.Random(dfr.SEED).sample(split["train"], DEV_SIZE))


def label_report(report, client, model=MODEL, prompt=PROMPT):
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=prompt_text(prompt),
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
        "acn": report["acn"], "model": model, "response_model": response.model,
        "prompt": prompt, "prompt_sha": prompt_sha(prompt),
        "label": label, "evidence": call.get("evidence", "") if call else None, "stop_reason": response.stop_reason,
        "usage": {k: getattr(response.usage, k, None) or 0 for k in ("input_tokens", "output_tokens")},
    }


def load_labels(path=LABELS_PATH, model=MODEL, prompt=PROMPT):
    """{acn: record} for the current text of `prompt` and `model`; the latest record wins."""
    if not path.exists():
        return {}
    sha = prompt_sha(prompt)
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["acn"]: r for r in records if r["model"] == model and r["prompt_sha"] == sha}


def label_split(reports, client, model=MODEL, prompt=PROMPT, path=LABELS_PATH, workers=WORKERS):
    """Labels every report not yet labeled, appending each record as it arrives. Returns the new records."""
    done = load_labels(path, model, prompt)
    todo = [r for r in reports if r["acn"] not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = []
    with ThreadPoolExecutor(workers) as pool, path.open("a") as f:
        for record in pool.map(lambda r: label_report(r, client, model, prompt), todo):
            f.write(json.dumps(record) + "\n")
            f.flush()
            new.append(record)
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", default=PROMPT, help="version: nlp/prompts/deferral_label_<version>.md")
    parser.add_argument("--split", choices=["train", "dev", "test", "blank"], default="test")
    parser.add_argument("--years", type=int, nargs="+", help="with --split blank: only reports from these years")
    parser.add_argument("--limit", type=int, help="label only the first N reports of the split")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    if args.split == "blank":
        data = [d for d in dfr.load_dataset(con, blank=True) if not args.years or d["year"] in args.years]
        acns = [d["acn"] for d in data]
    else:
        data = dfr.load_dataset(con)
        split = dfr.load_split(data)
        acns = dev_split(split) if args.split == "dev" else split[args.split]
    by_acn = {d["acn"]: d for d in data}
    reports = [by_acn[a] for a in acns][:args.limit]
    new = label_split(reports, anthropic.Anthropic(), args.model, args.prompt)

    unlabeled = sum(r["label"] is None for r in new)
    tokens = {k: sum(r["usage"][k] for r in new) for k in ("input_tokens", "output_tokens")}
    print(f"Labeled {len(new)} reports with {args.model}, prompt {args.prompt} ({unlabeled} without a label); "
          f"{tokens['input_tokens']} input and {tokens['output_tokens']} output tokens")
    if args.split == "blank":
        return
    labels = load_labels(model=args.model, prompt=args.prompt)
    done = [a for a in acns[:args.limit] if a in labels]
    y = [by_acn[a]["label"] for a in done]
    pred = [labels[a]["label"] or 0 for a in done]
    s = dfr.scores(y, pred)
    print(f"{args.split}: {len(done)} reports, F1 {s['f1']:.2f}, precision {s['precision']:.2f}, "
          f"recall {s['recall']:.2f}, accuracy {s['accuracy']:.2f}")


if __name__ == "__main__":
    main()
