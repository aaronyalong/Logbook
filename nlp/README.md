# NLP Experiments

Experiments on the ASRS narratives that sit beside the question-answering pipeline. They use the same database (`data/asrs.duckdb`) but do not change the pipeline unless a result is wired in on purpose.

Install with `pip install -r nlp/requirements.txt`, which adds scikit-learn to the pipeline's requirements.

## Predicting the deferral flag

**Task.** Predict the ASRS `maintenance_deferred` flag (Y/N) from the reporters' narratives. Analysts coded the flag for only 1,183 of 3,080 reports (38%). The blanks make deferred-maintenance questions hard to filter: evaluation query 10 has to fall back on keyword matching. A classifier that fills them in would give the filter a field it can rely on.

**Data.** The 1,183 coded reports: 837 N and 346 Y (29% Y). Each report's input is its narratives joined in order. The analyst-written synopsis is left out, because it is written at coding time and can restate the label. One stratified 80/20 split by ACN (seed 259) is committed in [data/deferral_split.json](data/deferral_split.json), so every model is scored on the same 237 test reports (69 Y).

Coding coverage varies a lot by year: over 40% of reports from 2010–2012 and 2019–2022, under 20% from 2016–2018 and 2025–2026. The blanks may therefore differ from the coded reports, and scores on coded reports may not carry over fully to them.

**Run:** `python nlp/deferral.py` (about 10 seconds; it needs the embedding index from `code/build_index.py`). Writes [results/deferral_metrics.json](results/deferral_metrics.json).

The LLM labels come from a separate script that calls the Claude API: `python nlp/deferral_llm.py` (about a minute for the 237 test reports; needs `ANTHROPIC_API_KEY`). It appends to [results/deferral_llm.jsonl](results/deferral_llm.jsonl) and skips reports already labeled with the same model and prompt, so `deferral.py` rescoring is free. Prompt versions are in [prompts/](prompts). New versions are developed with `--split dev`, a fixed random sample of 200 training reports, and only a version that beats v1 there is run on the test set.

### Models

| Model | What it does |
|---|---|
| majority | Always predicts N, the most common label. The floor any model must beat. |
| keyword | Y if a narrative contains "defer*" (deferred, deferral) or "MEL" as a whole word: the rule a keyword filter applies, with the same matching as the pipeline. Nothing is learned. |
| tfidf | TF-IDF word features + logistic regression with balanced class weights. The n-gram range (unigrams, or unigrams and bigrams) and regularization strength C are chosen by 5-fold cross-validation on the training set only. |
| embedding | Sentence embeddings from `BAAI/bge-small-en-v1.5`, the pretrained transformer the retrieval step uses, + logistic regression. Each report's vector is the mean of its narratives' vectors in the retrieval index, so the classifier sees exactly what retrieval sees. C is chosen the same way. |
| embedding+keyword | The embedding model with the keyword rule's output added as one more feature. Tests whether the embeddings add anything to the rule. |
| llm zero-shot (haiku) | Claude Haiku 5.5 reads each report's narratives and a definition of the flag ([prompts/deferral_label_v1.md](prompts/deferral_label_v1.md)), then calls a tool with a Y/N label and the quoted words that decide it. No training examples. The prompt was written once from the field's definition and not revised after seeing test results. |

### Results (test set, Y as the positive class)

| Model | Precision | Recall | F1 (95% CI) | Accuracy |
|---|---|---|---|---|
| majority | 0.00 | 0.00 | 0.00 | 0.71 |
| keyword | 0.79 | 0.80 | **0.79** (0.71–0.86) | 0.88 |
| tfidf (unigrams, C = 1) | 0.78 | 0.75 | 0.76 (0.68–0.84) | 0.86 |
| embedding (C = 10) | 0.67 | 0.80 | 0.73 (0.65–0.81) | 0.83 |
| embedding+keyword (C = 3) | 0.79 | 0.81 | **0.80** (0.72–0.87) | 0.88 |
| llm zero-shot (haiku) | 0.79 | 0.86 | **0.82** (0.74–0.89) | 0.89 |

The confidence intervals come from 1,000 bootstrap resamples of the test set. With 69 positive examples, F1 differences under about 0.1 are within noise.

### Findings

1. **Accuracy is misleading here.** Always answering N scores 71% accuracy and finds no deferrals. F1 on the Y class is the measure that matters.
2. **A one-line domain rule matches bag-of-words machine learning.** The keyword rule scored 0.79 F1, and TF-IDF + logistic regression scored 0.76. Their intervals overlap, so neither is clearly better. The same held in cross-validation on the training set: no TF-IDF setup tried (unigrams or bigrams, word or character n-grams, C from 0.01 to 30) got past about 0.75, against 0.80 for the rule.
3. **Why:** the label depends on a few specific words, and the rule folds the word forms together ("defer*" covers deferred, deferral, and deferring) where TF-IDF learns each form separately from about 950 examples. Any learned model has to beat this rule to be worth using.
4. **Pretrained sentence embeddings did worse than TF-IDF** (0.73 F1, with the lowest precision of the learned models). bge-small was trained to place texts about the same topic close together. Whether an item was deferred is an administrative detail inside the story, not its topic: a deferred hydraulic leak and a repaired one read alike. The embeddings find reports *about* deferrals, but also many that only resemble them.
5. **Adding the embeddings to the rule changed almost nothing** (0.80 against 0.79, within noise; cross-validation gave 0.80 for both). The rule already carries the signal the embeddings have.
6. **Lesson:** the best representation depends on what the label depends on. For a label decided by a few words, a rule or sparse word features match or beat a general-purpose transformer. A pretrained model needs fine-tuning on this task, or a model that reads the text the way an analyst does, to have a chance of doing better.
7. **An LLM with only a definition scored highest, but not by a significant margin.** Claude Haiku, given no examples, reached 0.82 F1 against 0.79 for the rule. On the same 237 reports the two disagree on 13: Haiku alone is right on 8 and the rule alone on 5 (McNemar exact test, p = 0.58). It labeled all 237 reports in about a minute, using about 316K input and 77K output tokens.
8. **Haiku wins where reading matters.** It correctly says N for "It could have been deferred per MEL but the outbound weight ... precluded this option" and "non-deferrable unserviceable FCDC", where the rule fires on the word. It correctly says Y for deferrals worded without the keywords: a CDL item, "Item not to be addressed at this time per customer request", and a test postponed to the next day. Its 5 extra errors are N reports it read as deferrals, such as parts ordered "to change them at a later date".
9. **Most remaining errors are about what the analysts meant by the flag.** Both methods get the same 21 reports wrong. In 11 of them the analysts coded N although a deferral is plainly mentioned, usually because the reporter was *working on* an item deferred earlier ("Working a deferred item for the C-1 hydraulic pump", "replace the Call Horn, which was on MEL"). The analysts appear to code Y only when this event deferred something. In the other 10 the analysts coded Y but the narratives describe no deferral in any words, so the label probably rests on information outside the narratives. Neither group can be fixed by a better classifier of these words.
10. **A sharper definition did not help, because the analysts are not consistent either.** Prompt v2 ([prompts/deferral_label_v2.md](prompts/deferral_label_v2.md)) said the flag means a deferral *in this event*, and that work on an earlier deferral is N. On the 200 dev reports it scored 0.71 F1 against 0.73 for v1 (precision 0.69 → 0.72, recall 0.79 → 0.70). Of the 15 dev reports v2 moved from Y to N, mostly earlier-deferral cases, the analysts had coded 6 Y and 9 N. They do not follow one rule for this case, so no single definition can match them. v2 was rejected on the dev set and never run on the test set. The flag's labels, not the classifiers, set the ceiling: the keyword rule (0.74 on dev) and both prompts all land within a few points of each other.

### Filling the blanks for query 10

Query 10 asks: "Compare what reporters said about deferred maintenance in 2019 versus 2023." Its gold filter searches narratives for `defer*`, because the flag is blank for most reports (176 of 273 in 2019, 81 of 120 in 2023). [deferral_fill.py](deferral_fill.py) compares four candidate sets per year: that keyword filter; the analysts' Y flags alone; the flags with the blanks filled by the keyword rule; and the flags with the blanks filled by Claude Haiku (prompt v1, run on the 257 blank reports of the two years with `deferral_llm.py --split blank --years 2019 2023`). It then runs the pipeline's hybrid retrieval with the query over each set and reports the top 10 narratives, which are what the answer model reads. Nothing in the pipeline changes. Results are in [results/deferral_fill.json](results/deferral_fill.json).

| 2019 set | Reports | Coded Y | Coded N | Blank | Also in keyword set | Retrieved reports (coded Y / blank) | Also retrieved from keyword set |
|---|---|---|---|---|---|---|---|
| keyword (`defer*`) | 30 | 14 | 1 | 15 | 30 | 10 (7 / 3) | 10 |
| flag | 36 | 36 | 0 | 0 | 14 | 10 (10 / 0) | 7 |
| flag + rule | 61 | 36 | 0 | 25 | 29 | 10 (7 / 3) | 8 |
| flag + llm | 58 | 36 | 0 | 22 | 25 | 10 (7 / 3) | 7 |

| 2023 set | Reports | Coded Y | Coded N | Blank | Also in keyword set | Retrieved reports (coded Y / blank) | Also retrieved from keyword set |
|---|---|---|---|---|---|---|---|
| keyword (`defer*`) | 12 | 7 | 1 | 4 | 12 | 9 (6 / 3) | 9 |
| flag | 13 | 13 | 0 | 0 | 7 | 9 (9 / 0) | 6 |
| flag + rule | 19 | 13 | 0 | 6 | 11 | 9 (6 / 3) | 8 |
| flag + llm | 19 | 13 | 0 | 6 | 11 | 9 (4 / 5) | 6 |

(Two reports in 2023 have two narratives each in the top 10, so 9 reports.)

11. **The keyword filter misses most coded deferrals.** `defer*` finds 14 of the 36 reports the analysts coded Y in 2019 and 7 of 13 in 2023. The rest say "MEL" or describe the deferral in other words. The flag filter, which query 10's notes also accept, finds a largely different set: the two share 14 of their 30 and 36 reports.
12. **Filling the blanks roughly doubles the 2019 candidate set** (30 → 58 or 61) and adds about half to 2023 (12 → 19). The rule and Haiku fill in mostly the same blank reports; they differ on 13 in 2019 and 4 in 2023. With the classifiers' test precision of about 0.79, roughly one in five filled-in reports would not be a deferral.
13. **Retrieval absorbs most of the difference.** The pipeline answers from the top 10 narratives, and those change much less than the candidate sets: 6 to 8 of the 10 reports are the same as with the keyword filter, because the query's own words already rank deferral reports first. So filling the flag would change query 10's answer only a little. It would matter more for counting questions ("how many deferrals in 2019?"), where the keyword filter's count, 30, undercounts the coded deferrals alone, 36.
14. **Not wired in.** A filled-in field would have to be marked as predicted wherever it is shown, since about a fifth of the filled values are wrong, and the gain for query 10 is small. The finding for the evaluation is the gap in finding 11 (backlog 7.2).
