"""Zero-shot deferral labeler, driven by a scripted fake client (no API calls)."""

import json
from types import SimpleNamespace

import deferral_llm as llm
from fakes import FakeClient, response, text

REPORT = {"acn": 7, "text": "Placed the pack on MEL."}


def label_call(deferred, evidence="Placed the pack on MEL."):
    return SimpleNamespace(type="tool_use", id="t1", name="record_label",
                           input={"evidence": evidence, "deferred": deferred})


def test_label_report_reads_the_tool_call():
    client = FakeClient(response(label_call("Y")))
    record = llm.label_report(REPORT, client)
    assert (record["acn"], record["label"], record["evidence"]) == (7, 1, "Placed the pack on MEL.")
    assert record["prompt_sha"] == llm.prompt_sha()
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 5}
    req = client.requests[0]
    assert req["model"] == llm.MODEL and req["tools"] == [llm.TOOL]
    assert REPORT["text"] in req["messages"][0]["content"]


def test_label_report_without_a_call_has_no_label():
    record = llm.label_report(REPORT, FakeClient(response(text("Y"))))
    assert record["label"] is None and record["evidence"] is None


def test_load_labels_keeps_current_prompt_and_model_and_latest_record(tmp_path):
    path = tmp_path / "labels.jsonl"
    sha = llm.prompt_sha()
    rows = [
        {"acn": 1, "model": llm.MODEL, "prompt_sha": sha, "label": 0},
        {"acn": 1, "model": llm.MODEL, "prompt_sha": sha, "label": 1},
        {"acn": 2, "model": llm.MODEL, "prompt_sha": "old", "label": 1},
        {"acn": 3, "model": "other", "prompt_sha": sha, "label": 1},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert {a: r["label"] for a, r in llm.load_labels(path).items()} == {1: 1}
    assert llm.load_labels(tmp_path / "missing.jsonl") == {}


def test_label_split_skips_reports_already_labeled(tmp_path):
    path = tmp_path / "labels.jsonl"
    client = FakeClient(response(label_call("Y")), response(label_call("N")))
    llm.label_split([REPORT], client, path=path, workers=1)
    new = llm.label_split([REPORT, {"acn": 8, "text": "Replaced the pack."}], client, path=path, workers=1)
    assert [r["acn"] for r in new] == [8]
    assert len(client.requests) == 2
    assert {a: r["label"] for a, r in llm.load_labels(path).items()} == {7: 1, 8: 0}


def test_dev_split_is_a_fixed_sample_of_train():
    split = {"train": list(range(500)), "test": list(range(500, 600))}
    dev = llm.dev_split(split)
    assert len(dev) == llm.DEV_SIZE and set(dev) <= set(split["train"])
    assert llm.dev_split(split) == dev


def test_each_prompt_version_has_its_own_labels(tmp_path):
    path = tmp_path / "labels.jsonl"
    llm.label_split([REPORT], FakeClient(response(label_call("Y"))), prompt="v1", path=path, workers=1)
    llm.label_split([REPORT], FakeClient(response(label_call("N"))), prompt="v2", path=path, workers=1)
    assert llm.load_labels(path, prompt="v1")[7]["label"] == 1
    assert llm.load_labels(path, prompt="v2")[7]["label"] == 0
