"""Hybrid retrieval, with a toy embedder so no model download is needed."""

import numpy as np
import pandas as pd
import pytest

import retrieve
from retrieve import RRF_K, Retriever, check_index, corpus_hash, ranks, tokenize

# Toy embedding space: one axis per concept, so synonyms share a direction (tired ~ fatigue).
CONCEPTS = [("gear", "landing"), ("pump", "fuel"), ("tired", "fatigue", "sleep"), ("door", "slide")]


def embed(text):
    words = set(tokenize(text))
    v = np.array([float(any(w in words for w in c)) for c in CONCEPTS]) + 1e-3
    return v / np.linalg.norm(v)


NARRATIVES = pd.DataFrame([
    (1, 1, "Main gear inspection missed.", "Main gear inspection missed."),
    (1, 2, "Second reporter on the gear.", "Second reporter on the gear."),
    (2, 1, "Fuel pump replaced per AMM.", "Fuel pump replaced per AMM (Aircraft Maintenance Manual)."),
    (3, 1, "Tired after a double shift.", "Tired after a double shift."),
    (4, 1, "MLG strut serviced.", "MLG (Main Landing Gear) strut serviced."),
    (5, 1, "Slide door armed.", "Slide door armed."),
], columns=["acn", "part", "text", "text_expanded"])


@pytest.fixture(scope="module")
def r():
    return Retriever(NARRATIVES, [embed(t) for t in NARRATIVES["text_expanded"]], embed)


def test_tokenize():
    assert tokenize("The P/N was in the hand-off log; IT was MISSED.") == ["p/n", "hand-off", "log", "missed"]


def test_ranks_break_ties_by_corpus_order():
    assert ranks(np.array([0.5, 0.9, 0.5, 0.1])).tolist() == [2, 1, 3, 4]


def test_bm25_needs_shared_terms(r):
    assert [h["acn"] for h in r.search("fuel pump", mode="bm25", k=1)] == [2]
    assert r.search("fatigue", mode="bm25", k=1)[0]["score"] == 0


def test_dense_matches_synonyms(r):
    assert r.search("fatigue", mode="dense", k=1)[0]["acn"] == 3


def test_hybrid_gives_no_bm25_credit_without_shared_terms(r):
    hit = r.search("fatigue", k=1)[0]
    assert hit["acn"] == 3
    assert hit["score"] == pytest.approx(1 / (RRF_K + hit["dense_rank"]))


def test_hybrid_fuses_both_rankers(r):
    hit = r.search("fuel pump", k=1)[0]
    assert (hit["acn"], hit["bm25_rank"], hit["dense_rank"]) == (2, 1, 1)
    assert hit["score"] == pytest.approx(2 / (RRF_K + 1))


def test_query_is_abbreviation_expanded(r):
    # "landing" only appears in the expanded text of ACN 4, so it can only match via MLG -> Main Landing Gear.
    hits = r.search("landing", mode="bm25", k=1)
    assert hits[0]["acn"] == 4
    assert r.search("MLG", mode="bm25", k=1)[0]["acn"] == 4


def test_restricted_to_acns(r):
    hits = r.search("gear", acns=[1, 5])
    assert {h["acn"] for h in hits} == {1, 5}
    assert {(h["acn"], h["part"]) for h in hits} >= {(1, 1), (1, 2)}  # both narratives of a report


def test_empty_acns_returns_nothing(r):
    assert r.search("gear", acns=[]) == []


def test_hits_carry_original_text_and_respect_k(r):
    hits = r.search("gear", k=2)
    assert len(hits) == 2
    assert all(h["text"] in set(NARRATIVES["text"]) for h in hits)
    assert set(hits[0]) == {"acn", "part", "text", "score", "bm25_rank", "dense_rank"}
    assert set(r.search("gear", k=1, mode="dense")[0]) == {"acn", "part", "text", "score", "dense_rank"}


def test_rejects_unknown_mode(r):
    with pytest.raises(ValueError, match="mode must be one of"):
        r.search("gear", mode="sparse")


def index_for(narratives, model=retrieve.EMBED_MODEL):
    return {"acn": narratives["acn"].to_numpy(), "part": narratives["part"].to_numpy(), "model": np.array(model),
            "text_sha256": np.array(corpus_hash(narratives["text_expanded"]))}


def test_check_index():
    assert check_index(index_for(NARRATIVES), NARRATIVES) == []
    assert "expected BAAI" in check_index(index_for(NARRATIVES, model="other-model"), NARRATIVES)[0]
    edited = NARRATIVES.assign(text_expanded=NARRATIVES["text_expanded"].str.upper())
    assert check_index(index_for(NARRATIVES), edited) == ["narrative text changed since the index was built"]
    assert check_index(index_for(NARRATIVES), NARRATIVES.iloc[1:]) == [
        "index narratives do not match the narratives table"]


def test_committed_index_is_well_formed():
    index = np.load(retrieve.INDEX_PATH)
    assert str(index["model"]) == retrieve.EMBED_MODEL
    assert index["embeddings"].shape == (len(index["acn"]), 384)
    norms = np.linalg.norm(index["embeddings"].astype(np.float32), axis=1)
    assert np.allclose(norms, 1, atol=1e-2)
