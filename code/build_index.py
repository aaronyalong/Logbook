"""Embed every narrative for dense retrieval.

Writes data/index/narrative_embeddings.npz: one float16 vector per narrative in narratives
table order, plus the model name and a hash of the embedded text, so code/retrieve.py can
refuse an index that no longer matches the database.

Usage: python code/build_index.py           # embeds the abbreviation-expanded narratives on CPU
       python code/build_index.py --check   # verify the committed index matches data/asrs.duckdb
"""

import argparse
import sys
import time

import duckdb
import numpy as np
from fastembed import TextEmbedding

from preprocess import DB_PATH, ROOT
from retrieve import EMBED_MODEL, INDEX_PATH, MODEL_CACHE, check_index, corpus_hash, read_narratives


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only verify the existing index")
    args = parser.parse_args()
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        narratives = read_narratives(con)

    if args.check:
        problems = check_index(np.load(INDEX_PATH), narratives)
        print("; ".join(problems) or f"{INDEX_PATH.relative_to(ROOT)} matches the narratives table")
        sys.exit(1 if problems else 0)

    start = time.time()
    texts = narratives["text_expanded"].tolist()
    embeddings = np.stack(list(TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_CACHE)).passage_embed(texts, batch_size=32)))
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        acn=narratives["acn"].to_numpy(), part=narratives["part"].to_numpy(),
        embeddings=embeddings.astype(np.float16),
        model=np.array(EMBED_MODEL), text_sha256=np.array(corpus_hash(texts)),
    )
    print(f"Embedded {len(texts)} narratives with {EMBED_MODEL} in {time.time() - start:.0f}s "
          f"-> {INDEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
