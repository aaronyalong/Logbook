"""Predict the ASRS `maintenance_deferred` flag (Y/N) from report narratives.

ASRS analysts coded the flag for only 38% of reports; the rest are blank, which limits
filters on deferred maintenance (evaluation query 10). This trains text classifiers on the
coded reports and compares them on one fixed, stratified test split:

  majority   always predicts the most common label (N)
  keyword    Y if a narrative contains a deferral word ("defer*", "MEL"), the rule a keyword filter uses
  tfidf      TF-IDF words + logistic regression; n-gram range and C chosen by 5-fold cross-validation on train
  embedding  bge-small sentence embeddings (the retrieval index, averaged per report) + logistic regression
  embedding+keyword  the same, with the keyword rule's output as one more feature

The input is the reporters' narratives only. The synopsis is left out because ASRS analysts
write it at coding time, so it can restate the label.

Usage: python nlp/deferral.py
Output: nlp/results/deferral_metrics.json, and a results table on stdout.
"""

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import retrieve  # noqa: E402
from filter_spec import term_pattern  # noqa: E402
from preprocess import DB_PATH  # noqa: E402

SPLIT_PATH = ROOT / "nlp/data/deferral_split.json"
METRICS_PATH = ROOT / "nlp/results/deferral_metrics.json"
SEED = 259
TEST_SIZE = 0.2
KEYWORDS = ["defer*", "MEL"]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
NGRAMS = [(1, 1), (1, 2)]
BOOTSTRAP = 1000


def load_dataset(con):
    """Coded reports with their narratives joined in part order: acn, year, text, label (1 = Y)."""
    rows = con.execute("""
        SELECT r.acn, r.year, string_agg(n.text, '\n\n' ORDER BY n.part) AS text, r.maintenance_deferred = 'Y' AS y
        FROM reports r JOIN narratives n ON n.acn = r.acn
        WHERE r.maintenance_deferred IS NOT NULL
        GROUP BY r.acn, r.year, r.maintenance_deferred
        ORDER BY r.acn
    """).fetchall()
    return [{"acn": acn, "year": year, "text": text, "label": int(y)} for acn, year, text, y in rows]


def make_split(data, test_size=TEST_SIZE, seed=SEED):
    """Stratified train/test split by ACN."""
    acns = [d["acn"] for d in data]
    train, test = train_test_split(acns, test_size=test_size, stratify=[d["label"] for d in data], random_state=seed)
    return {"seed": seed, "train": sorted(train), "test": sorted(test)}


def load_split(data, path=SPLIT_PATH):
    """The committed split, so every model and every later step uses the same test set."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_split(data)) + "\n")
    split = json.loads(path.read_text())
    if set(split["train"]) | set(split["test"]) != {d["acn"] for d in data}:
        raise SystemExit(f"{path} does not match the coded reports; delete it to make a new split")
    return split


def keyword_predict(texts, keywords=KEYWORDS):
    patterns = [re.compile(term_pattern(k)) for k in keywords]
    return np.array([int(any(p.search(t) for p in patterns)) for t in texts])


def report_vectors(index, acns):
    """One unit vector per report: the mean of its narratives' embeddings from the retrieval index."""
    embeddings = index["embeddings"].astype(np.float32)
    vectors = []
    for acn in acns:
        v = embeddings[index["acn"] == acn].mean(axis=0)
        vectors.append(v / np.linalg.norm(v))
    return np.stack(vectors)


def embedding_model():
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(class_weight="balanced", max_iter=5000)
    return GridSearchCV(lr, {"C": C_GRID}, scoring="f1", cv=folds, n_jobs=-1)


def tfidf_model():
    pipe = make_pipeline(
        TfidfVectorizer(min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = {"tfidfvectorizer__ngram_range": NGRAMS, "logisticregression__C": C_GRID}
    return GridSearchCV(pipe, grid, scoring="f1", cv=folds, n_jobs=-1)


def scores(y, pred):
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred)}


def f1_interval(y, pred, n=BOOTSTRAP, seed=SEED):
    """95% bootstrap interval for F1 over the test set; the test set is small, so F1 is noisy."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    samples = [f1_score(y[i], pred[i], zero_division=0) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def evaluate(con):
    data = load_dataset(con)
    split = load_split(data)
    by_acn = {d["acn"]: d for d in data}
    train = [by_acn[a] for a in split["train"]]
    test = [by_acn[a] for a in split["test"]]
    x_train, y_train = [d["text"] for d in train], [d["label"] for d in train]
    x_test, y_test = [d["text"] for d in test], [d["label"] for d in test]

    index = np.load(retrieve.INDEX_PATH)
    problems = retrieve.check_index(index, retrieve.read_narratives(con))
    if problems:
        raise SystemExit(f"{'; '.join(problems)}. Rebuild with: python code/build_index.py")
    e_train, e_test = report_vectors(index, split["train"]), report_vectors(index, split["test"])
    k_train, k_test = keyword_predict(x_train)[:, None], keyword_predict(x_test)[:, None]

    fitted = {
        "tfidf": tfidf_model().fit(x_train, y_train),
        "embedding": embedding_model().fit(e_train, y_train),
        "embedding+keyword": embedding_model().fit(np.hstack([e_train, k_train]), y_train),
    }
    preds = {
        "majority": np.full(len(y_test), int(np.mean(y_train) > 0.5)),
        "keyword": k_test[:, 0],
        "tfidf": fitted["tfidf"].predict(x_test),
        "embedding": fitted["embedding"].predict(e_test),
        "embedding+keyword": fitted["embedding+keyword"].predict(np.hstack([e_test, k_test])),
    }
    results = {name: {**scores(y_test, p), "f1_95ci": f1_interval(y_test, p)} for name, p in preds.items()}
    for name, model in fitted.items():
        results[name]["params"] = {k.split("__")[-1]: v for k, v in model.best_params_.items()}
        results[name]["cv_f1"] = float(model.best_score_)
    return {
        "data": {"train": len(train), "test": len(test), "train_y": int(sum(y_train)), "test_y": int(sum(y_test))},
        "models": results,
    }


def table(metrics):
    lines = ["| Model | Precision | Recall | F1 (95% CI) | Accuracy |", "|---|---|---|---|---|"]
    for name, m in metrics["models"].items():
        lo, hi = m["f1_95ci"]
        lines.append(f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} ({lo:.2f}–{hi:.2f}) "
                     f"| {m['accuracy']:.2f} |")
    return "\n".join(lines)


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    metrics = evaluate(con)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    d = metrics["data"]
    print(f"train {d['train']} reports ({d['train_y']} Y), test {d['test']} reports ({d['test_y']} Y)\n")
    print(table(metrics))
    print()
    for name, m in metrics["models"].items():
        if "params" in m:
            print(f"{name}: {m['params']}, 5-fold CV F1 on train = {m['cv_f1']:.2f}")


if __name__ == "__main__":
    main()
