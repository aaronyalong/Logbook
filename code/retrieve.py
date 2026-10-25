"""Hybrid retrieval over narratives: BM25 + dense embeddings, fused by reciprocal rank.

The retrieval unit is one reporter narrative (the narratives table). Both rankers score the
abbreviation-expanded text, and the question is expanded the same way. A search can be
restricted to the ACNs a filter selected; with no restriction it covers the whole corpus.

Modes: "hybrid" (default), "bm25", or "dense". Dense over the whole corpus is the naive-RAG
baseline for the retrieval comparison.

Dense vectors for every narrative are precomputed by code/build_index.py into
data/index/narrative_embeddings.npz, which is committed so the assistant runs without
rebuilding it.

Usage: python code/retrieve.py "question" [--mode hybrid|bm25|dense] [-k 10] [--acn 1884087 ...]
"""

import argparse
import hashlib
import re

import duckdb
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from preprocess import DB_PATH, ROOT, load_abbreviations, make_expander

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = ROOT / ".cache/fastembed"  # fastembed defaults to the system temp dir, which is cleared on reboot
INDEX_PATH = ROOT / "data/index/narrative_embeddings.npz"
MODES = ("hybrid", "bm25", "dense")
RRF_K = 60  # standard reciprocal-rank-fusion constant; damps the weight of the very top ranks

TOKEN = re.compile(r"[a-z0-9]+(?:[/'-][a-z0-9]+)*")
STOPWORDS = set("""
a about after all also an and any are as at be been before but by can could did do does during for from
had has have he her his how i if in into is it its me my no not of on or our out over she so than that
the their them then there these they this those to up was we were what when where which while who why
will with would you your
""".split())


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def corpus_hash(texts):
    return hashlib.sha256("\0".join(texts).encode()).hexdigest()


def ranks(scores):
    """1-based rank of each score, highest first; ties keep corpus order."""
    r = np.empty(len(scores), dtype=int)
    r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return r


class Retriever:
    def __init__(self, narratives, embeddings, embed_query, expand=None):
        """narratives: DataFrame with acn, part, text, text_expanded (row i <-> embeddings[i]).
        embed_query: str -> unit vector in the same space as the embeddings."""
        self.narratives = narratives.reset_index(drop=True)
        self.acn = self.narratives["acn"].to_numpy()
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embed_query = embed_query
        self.expand = expand or make_expander(load_abbreviations())
        self.bm25 = BM25Okapi([tokenize(t) for t in self.narratives["text_expanded"]])

    def search(self, question, acns=None, k=10, mode="hybrid"):
        """Top-k narratives for the question, restricted to `acns` unless it is None.

        An empty `acns` (a filter that matched nothing) returns nothing. Each hit has
        acn, part, text, score, and the rank from each ranker that ran.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        idx = np.arange(len(self.acn)) if acns is None else np.flatnonzero(np.isin(self.acn, list(acns)))
        if len(idx) == 0:
            return []

        query = self.expand(question)
        found = {}
        if mode in ("hybrid", "bm25"):
            found["bm25"] = self.bm25.get_scores(tokenize(query))[idx]
        if mode in ("hybrid", "dense"):
            found["dense"] = self.embeddings[idx] @ np.asarray(self.embed_query(query), dtype=np.float32)
        rank = {name: ranks(s) for name, s in found.items()}

        if mode == "hybrid":
            # A narrative sharing no terms with the query gets no BM25 credit, only its dense rank.
            score = 1 / (RRF_K + rank["dense"]) + np.where(found["bm25"] > 0, 1 / (RRF_K + rank["bm25"]), 0)
        else:
            score = found[mode]

        hits = []
        for i in np.lexsort((np.arange(len(idx)), -score))[:k]:
            row = self.narratives.iloc[idx[i]]
            hit = {"acn": int(row.acn), "part": int(row.part), "text": row.text, "score": float(score[i])}
            hit |= {f"{name}_rank": int(r[i]) for name, r in rank.items()}
            hits.append(hit)
        return hits


def read_narratives(con):
    return con.execute("SELECT acn, part, text, text_expanded FROM narratives ORDER BY acn, part").df()


def check_index(index, narratives):
    """Problems that make the saved index unusable with these narratives; empty if it matches."""
    problems = []
    if str(index["model"]) != EMBED_MODEL:
        problems.append(f"index built with {index['model']}, expected {EMBED_MODEL}")
    if not (np.array_equal(index["acn"], narratives["acn"]) and np.array_equal(index["part"], narratives["part"])):
        problems.append("index narratives do not match the narratives table")
    elif str(index["text_sha256"]) != corpus_hash(narratives["text_expanded"]):
        problems.append("narrative text changed since the index was built")
    return problems


def load(con=None, index_path=INDEX_PATH):
    con = con or duckdb.connect(str(DB_PATH), read_only=True)
    narratives = read_narratives(con)
    index = np.load(index_path)
    problems = check_index(index, narratives)
    if problems:
        raise RuntimeError(f"{'; '.join(problems)}. Rebuild with: python code/build_index.py")
    model = TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_CACHE))
    return Retriever(narratives, index["embeddings"], lambda q: next(iter(model.query_embed([q]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--acn", type=int, nargs="+", help="restrict to these reports (e.g. a filter's result)")
    args = parser.parse_args()
    for n, hit in enumerate(load().search(args.question, args.acn, args.k, args.mode), 1):
        ranks_ = ", ".join(f"{k.removesuffix('_rank')} #{v}" for k, v in hit.items() if k.endswith("_rank"))
        snippet = " ".join(hit["text"].split())[:200]
        print(f"{n:2}. ACN {hit['acn']} (part {hit['part']})  score {hit['score']:.4f}  [{ranks_}]\n    {snippet}...")


if __name__ == "__main__":
    main()
