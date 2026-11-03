# NLP Experiments

Experiments on the ASRS narratives that sit beside the question-answering pipeline. They use the same database (`data/asrs.duckdb`) but do not change the pipeline unless a result is wired in on purpose.

Install with `pip install -r nlp/requirements.txt`, which adds scikit-learn to the pipeline's requirements.

## Predicting the deferral flag

**Task.** Predict the ASRS `maintenance_deferred` flag (Y/N) from the reporters' narratives. Analysts coded the flag for only 1,183 of 3,080 reports (38%). The blanks make deferred-maintenance questions hard to filter: evaluation query 10 has to fall back on keyword matching. A classifier that fills them in would give the filter a field it can rely on.

**Data.** The 1,183 coded reports: 837 N and 346 Y (29% Y). Each report's input is its narratives joined in order. The analyst-written synopsis is left out, because it is written at coding time and can restate the label. One stratified 80/20 split by ACN (seed 259) is committed in [data/deferral_split.json](data/deferral_split.json), so every model is scored on the same 237 test reports (69 Y).

Coding coverage varies a lot by year: over 40% of reports from 2010–2012 and 2019–2022, under 20% from 2016–2018 and 2025–2026. The blanks may therefore differ from the coded reports, and scores on coded reports may not carry over fully to them.

**Run:** `python nlp/deferral.py` (about 10 seconds). Writes [results/deferral_metrics.json](results/deferral_metrics.json).

### Models

| Model | What it does |
|---|---|
| majority | Always predicts N, the most common label. The floor any model must beat. |
| keyword | Y if a narrative contains "defer*" (deferred, deferral) or "MEL" as a whole word: the rule a keyword filter applies, with the same matching as the pipeline. Nothing is learned. |
| tfidf | TF-IDF word features + logistic regression with balanced class weights. The n-gram range (unigrams, or unigrams and bigrams) and regularization strength C are chosen by 5-fold cross-validation on the training set only. |

### Results (test set, Y as the positive class)

| Model | Precision | Recall | F1 (95% CI) | Accuracy |
|---|---|---|---|---|
| majority | 0.00 | 0.00 | 0.00 | 0.71 |
| keyword | 0.79 | 0.80 | **0.79** (0.71–0.86) | 0.88 |
| tfidf (unigrams, C = 1) | 0.78 | 0.75 | 0.76 (0.68–0.84) | 0.86 |

The confidence intervals come from 1,000 bootstrap resamples of the test set. With 69 positive examples, F1 differences under about 0.1 are within noise.

### Findings

1. **Accuracy is misleading here.** Always answering N scores 71% accuracy and finds no deferrals. F1 on the Y class is the measure that matters.
2. **A one-line domain rule matches bag-of-words machine learning.** The keyword rule scored 0.79 F1, and TF-IDF + logistic regression scored 0.76. Their intervals overlap, so neither is clearly better. The same held in cross-validation on the training set: no TF-IDF setup tried (unigrams or bigrams, word or character n-grams, C from 0.01 to 30) got past about 0.75, against 0.80 for the rule.
3. **Why:** the label depends on a few specific words, and the rule folds the word forms together ("defer*" covers deferred, deferral, and deferring) where TF-IDF learns each form separately from about 950 examples. Any learned model has to beat this rule to be worth using.

### Next

- Sentence embeddings (bge-small, the retrieval model) + logistic regression: does a pretrained transformer representation beat the rule?
- Error analysis: the words the TF-IDF model weights most, a confusion matrix, and the reports both approaches get wrong.
- Fill in the blank flags and measure the effect on query 10.
