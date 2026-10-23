"""build_filter as a Claude tool call: question -> validated filter specs -> matching reports.

Claude sees the field reference and every observed value (prompts/build_filter_system.md)
and calls the build_filter tool with a list of filter specs (code/filter_spec.py): one for
most questions, one per side for a comparison. Each call is validated before it runs; a
rejected call goes back to Claude with the problems listed so it can correct itself, up to
MAX_CALLS calls. The last accepted call is the result.

Every question is logged to logs/build_filter.jsonl with each call, its validation errors,
and its match counts, so a bad filter can be told apart from bad retrieval later.

Usage: python code/build_filter.py "Show me 737 reports from 2022 to 2024"
Requires ANTHROPIC_API_KEY.
"""

import argparse
import json
import time

import anthropic
import duckdb

from filter_spec import FIELDS, FILTER_VALUES, run, validate
from preprocess import DB_PATH, ROOT

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_CALLS = 3
TOP_GROUPS = 10  # group_by values shown to the model per filter
PROMPT_PATH = ROOT / "prompts/build_filter_system.md"
LOG_PATH = ROOT / "logs/build_filter.jsonl"

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "acn": {"type": "array", "items": {"type": "integer"}, "description": "Report numbers to look up."},
        "year_min": {"type": "integer", "description": "Earliest incident year, inclusive."},
        "year_max": {"type": "integer", "description": "Latest incident year, inclusive."},
        "where": {
            "type": "object",
            "properties": {f: {"type": "array", "items": {"type": "string"}} for f in FIELDS},
            "additionalProperties": False,
            "description": "Coded field -> listed values. OR within a field, AND across fields.",
        },
        "text_any": {
            "type": "array", "items": {"type": "string"},
            "description": "Match reports where any narrative contains one of these terms (case-insensitive).",
        },
        "group_by": {"type": "string", "enum": list(FIELDS), "description": "Return value counts for this field."},
    },
    "additionalProperties": False,
}

TOOL = {
    "name": "build_filter",
    "description": (
        "Select the ASRS reports that the answer step will search. Takes one filter spec, or one per side "
        "of a comparison, and returns the number of reports each matched (value counts when group_by is "
        "set). Invalid fields or values are rejected with an explanation. A later call replaces an earlier one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filters": {
                "type": "array", "items": SPEC_SCHEMA, "minItems": 1,
                "description": "One filter, or one per side of a comparison in the order the question names them.",
            },
        },
        "required": ["filters"],
        "additionalProperties": False,
    },
}


def system_prompt():
    values = "\n".join(f"- `{f}`: {json.dumps(v)}" for f, v in FILTER_VALUES.items() if f != "year")
    return PROMPT_PATH.read_text().replace("<<FIELD_VALUES>>", values)


def execute(tool_input, con):
    """Validate and run one build_filter call. Returns {"input", "errors"} plus "results" if accepted."""
    filters = tool_input.get("filters") if isinstance(tool_input, dict) else None
    if not (isinstance(filters, list) and filters):
        errors = ["'filters' must be a non-empty list of filter specs"]
    else:
        errors = [f"filters[{i}]: {e}" for i, spec in enumerate(filters) for e in validate(spec)]
    call = {"input": tool_input, "errors": errors}
    if not errors:
        call["results"] = [run(spec, con) for spec in filters]
    return call


def tool_output(call):
    """What the model sees: the problems, or the match count (and top groups) per filter."""
    if call["errors"]:
        return "Rejected, nothing was run:\n" + "\n".join(f"- {e}" for e in call["errors"])
    summary = []
    for r in call["results"]:
        s = {"count": r["count"]}
        if "groups" in r:
            s["top_groups"] = dict(list(r["groups"].items())[:TOP_GROUPS])
        summary.append(s)
    return json.dumps(summary)


def build_filter(question, client=None, con=None, model=MODEL, log_path=LOG_PATH):
    """Ask Claude for the filter for `question`.

    Returns a record with:
      status    "ok" (a call was accepted), "invalid" (every call rejected),
                "no_call" (the model never called the tool), or "refusal"
      filters   the accepted specs (None unless status is "ok")
      results   filter_spec.run() output for each accepted spec
      calls     every call with its validation errors and match counts
    """
    client = client or anthropic.Anthropic()
    con = con or duckdb.connect(str(DB_PATH), read_only=True)
    messages = [{"role": "user", "content": question}]
    calls, accepted, usage = [], None, {}

    for _ in range(MAX_CALLS):
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
            tools=[TOOL],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            output_config={"effort": EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        )
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            usage[key] = usage.get(key, 0) + (getattr(response.usage, key, None) or 0)
        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            call = execute(block.input, con)
            calls.append(call)
            if not call["errors"]:
                accepted = call
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": tool_output(call), "is_error": bool(call["errors"])})
        messages.append({"role": "user", "content": results})

    if response.stop_reason == "refusal":
        status = "refusal"
    elif accepted:
        status = "ok"
    else:
        status = "invalid" if calls else "no_call"
    record = {
        "question": question,
        "model": response.model,
        "status": status,
        "filters": accepted["input"]["filters"] if accepted else None,
        "results": accepted["results"] if accepted else None,
        "calls": calls,
        "explanation": "".join(b.text for b in response.content if b.type == "text").strip(),
        "stop_reason": response.stop_reason,
        "usage": usage,
    }
    if log_path:
        _log(record, log_path)
    return record


def _log(record, log_path):
    # Counts only: ACN lists are reproducible by re-running the logged spec.
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record, "results": None}
    entry["calls"] = [
        {"input": c["input"], "errors": c["errors"], "counts": [r["count"] for r in c.get("results", [])]}
        for c in record["calls"]
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    record = build_filter(parser.parse_args().question)
    print(f"status: {record['status']}  ({len(record['calls'])} call(s), model {record['model']})")
    for i, call in enumerate(record["calls"], 1):
        print(f"call {i}: {json.dumps(call['input'])}")
        for e in call["errors"]:
            print(f"  rejected: {e}")
    for spec, result in zip(record["filters"] or [], record["results"] or []):
        print(f"filter: {json.dumps(spec)} -> {result['count']} reports")
    if record["explanation"]:
        print(record["explanation"])


if __name__ == "__main__":
    main()
