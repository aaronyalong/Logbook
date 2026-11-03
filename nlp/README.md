# NLP Experiments

Experiments on the ASRS narratives that sit beside the question-answering pipeline. They use the same database (`data/asrs.duckdb`) but do not change the pipeline unless a result is wired in on purpose.

Install with `pip install -r nlp/requirements.txt`, which adds scikit-learn to the pipeline's requirements.

## Predicting the deferral flag

**Task.** Predict the ASRS `maintenance_deferred` flag (Y/N) from the reporters' narratives. Analysts coded the flag for only 1,183 of 3,080 reports (38%). The blanks make deferred-maintenance questions hard to filter: evaluation query 10 has to fall back on keyword matching. A classifier that fills them in would give the filter a field it can rely on.

**Data.** The 1,183 coded reports: 837 N and 346 Y (29% Y). Each report's input is its narratives joined in order. The analyst-written synopsis is left out, because it is written at coding time and can restate the label. One stratified 80/20 split by ACN (seed 259) is committed in [data/deferral_split.json](data/deferral_split.json), so every model is scored on the same 237 test reports (69 Y).

Coding coverage varies a lot by year: over 40% of reports from 2010–2012 and 2019–2022, under 20% from 2016–2018 and 2025–2026. The blanks may therefore differ from the coded reports, and scores on coded reports may not carry over fully to them.

**Run:** `python nlp/deferral.py` (about 10 seconds; it needs the embedding index from `code/build_index.py`). Writes [results/deferral_metrics.json](results/deferral_metrics.json).

### Models

| Model | What it does |
|---|---|
| majority | Always predicts N, the most common label. The floor any model must beat. |
| keyword | Y if a narrative contains "defer*" (deferred, deferral) or "MEL" as a whole word: the rule a keyword filter applies, with the same matching as the pipeline. Nothing is learned. |
| tfidf | TF-IDF word features + logistic regression with balanced class weights. The n-gram range (unigrams, or unigrams and bigrams) and regularization strength C are chosen by 5-fold cross-validation on the training set only. |
| embedding | Sentence embeddings from `BAAI/bge-small-en-v1.5`, the pretrained transformer the retrieval step uses, + logistic regression. Each report's vector is the mean of its narratives' vectors in the retrieval index, so the classifier sees exactly what retrieval sees. C is chosen the same way. |
| embedding+keyword | The embedding model with the keyword rule's output added as one more feature. Tests whether the embeddings add anything to the rule. |

### Results (test set, Y as the positive class)

| Model | Precision | Recall | F1 (95% CI) | Accuracy |
|---|---|---|---|---|
| majority | 0.00 | 0.00 | 0.00 | 0.71 |
| keyword | 0.79 | 0.80 | **0.79** (0.71–0.86) | 0.88 |
| tfidf (unigrams, C = 1) | 0.78 | 0.75 | 0.76 (0.68–0.84) | 0.86 |
| embedding (C = 10) | 0.67 | 0.80 | 0.73 (0.65–0.81) | 0.83 |
| embedding+keyword (C = 3) | 0.79 | 0.81 | **0.80** (0.72–0.87) | 0.88 |

The confidence intervals come from 1,000 bootstrap resamples of the test set. With 69 positive examples, F1 differences under about 0.1 are within noise.

### Findings

1. **Accuracy is misleading here.** Always answering N scores 71% accuracy and finds no deferrals. F1 on the Y class is the measure that matters.
2. **A one-line domain rule matches bag-of-words machine learning.** The keyword rule scored 0.79 F1, and TF-IDF + logistic regression scored 0.76. Their intervals overlap, so neither is clearly better. The same held in cross-validation on the training set: no TF-IDF setup tried (unigrams or bigrams, word or character n-grams, C from 0.01 to 30) got past about 0.75, against 0.80 for the rule.
3. **Why:** the label depends on a few specific words, and the rule folds the word forms together ("defer*" covers deferred, deferral, and deferring) where TF-IDF learns each form separately from about 950 examples. Any learned model has to beat this rule to be worth using.
4. **Pretrained sentence embeddings did worse than TF-IDF** (0.73 F1, with the lowest precision of the learned models). bge-small was trained to place texts about the same topic close together. Whether an item was deferred is an administrative detail inside the story, not its topic: a deferred hydraulic leak and a repaired one read alike. The embeddings find reports *about* deferrals, but also many that only resemble them.
5. **Adding the embeddings to the rule changed almost nothing** (0.80 against 0.79, within noise; cross-validation gave 0.80 for both). The rule already carries the signal the embeddings have.
6. **Lesson:** the best representation depends on what the label depends on. For a label decided by a few words, a rule or sparse word features match or beat a general-purpose transformer. A pretrained model needs fine-tuning on this task, or a model that reads the text the way an analyst does, to have a chance of doing better.

### Next

- An LLM as a zero-shot classifier: give Claude Haiku the narrative and the ASRS definition of the flag, with no training examples.
- Error analysis: the words the TF-IDF model weights most, a confusion matrix, and the reports both approaches get wrong.
- Fill in the blank flags and measure the effect on query 10.
