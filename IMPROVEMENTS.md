# Improvements Backlog

Improvements identified while building and smoke-testing the assistant, in priority order. Each item records what was observed, the decision to make, where the change goes, and how to show that it worked.

Items tagged with a grading area strengthen one of the course's Advanced areas. Do section 1 before the full evaluation run, so the scored run reflects the fixed system. Record each fix as "found in testing, then fixed", with before and after numbers; that is evidence for the evaluation and reflection sections of the report.

## 1. Fix before the full evaluation run

### 1.1 Keyword matching matches inside other words
*Grading area: tool/data integration*

- **Observed:** `text_any` matches substrings (`ILIKE '%term%'`). "MEL" also matches *extremely*, *timely*, *smell*, and *melted*: 503 reports instead of the 400 that contain MEL as a word. This inflated query 10's counts.
- **Constraint:** some terms need prefix matching. "defer" must match *deferred* and *deferral*: 330 reports as a substring, but only 86 as a whole word.
- **Decision:** match at the start of a word (`\bterm`), match whole words with an optional plural, or let the model mark each term as exact or prefix.
- **Where:** `to_sql()` in [code/filter_spec.py](code/filter_spec.py), the `text_any` guidance in [prompts/build_filter_system.md](prompts/build_filter_system.md), and tests in [tests/test_filter_spec.py](tests/test_filter_spec.py). The gold filters for queries 7, 10, and 18 use `text_any`, so rerun `evaluation/check_gold_filters.py` and commit the changed counts.
- **Measure:** gold-filter counts before and after, plus a spot check of the reports each term matches.

### 1.2 The filter searches for words that are not topics
*Grading area: robustness*

- **Observed:** for query 19 (spacecraft), the filter searched for "NASA" and matched 8 reports in which mechanics mention filing a *NASA ASRS* form. The answer step caught it and correctly said that no reports involve spacecraft.
- **Decision:** add a general note to the filter prompt: reporters often mention NASA, ASRS, or "this report" when referring to the reporting program, so those words do not indicate a topic. This failure was found on an evaluation query, so record the fix as such; it slightly contaminates that query.
- **Where:** [prompts/build_filter_system.md](prompts/build_filter_system.md).

### 1.3 Failed citations are flagged but not enforced
*Grading area: grounding*

- **Observed:** all 60 citations across 6 smoke-test queries were verified. The local model is expected to do worse.
- **Decision:** what happens when a citation fails the check: regenerate once with the problem stated, remove the unsupported sentence, or keep only the warning.
- **Where:** `answer()` in [code/answer.py](code/answer.py).
- **Measure:** how often citations fail for each model, and whether regenerating fixes them.

## 2. Evaluation rigor
*Grading areas: evaluation, grounding*

### 2.1 Score filter correctness by overlap, not exact match
Query 10's generated filter differs from gold but is defensible: it matched 61 and 25 reports, against gold's 30 and 12. Score each generated filter by how much its matched reports overlap the gold filter's (Jaccard similarity, and recall of the gold reports). Record acceptable alternatives in [evaluation/queries.json](evaluation/queries.json); query 10's notes already name one.

### 2.2 Judge groundedness claim by claim
Have a separate LLM judge check each cited sentence against the narrative it cites: does that report actually say this? Hand-label about 20 judgments and report how often the judge agrees with you. This follows the citation-precision method of Gao et al. (2023), *Enabling Large Language Models to Generate Text with Citations* (ALCE).

### 2.3 Label retrieval relevance
For about 10 queries, mark which of the top 10 results are relevant under hybrid, BM25, and dense retrieval, and report precision@10. This turns the Axis B comparison into numbers.

### 2.4 Repeat runs
Run each configuration 2–3 times before claiming that one is better, so a difference can be told apart from run-to-run variation.

### 2.5 Report cost and latency per configuration
Token usage is already logged in `logs/answer.jsonl`. The smoke test on Claude cost about $0.08–0.12 and took 12–31 seconds per question. Spec 4 asks for tradeoffs, and a statement such as "free but N× slower and X% worse at citing" needs these numbers.

## 3. Retrieval quality
*Grading area: tool/data integration*

### 3.1 Rerank the top candidates
Rescore the top 30 hybrid results with a local cross-encoder reranker (`fastembed` supports them). Measure the change with the labels from 2.3.

### 3.2 Split long narratives for the dense ranker
499 of 3,632 narratives (14%) exceed the embedding model's 512-token input, so their vectors cover only the opening. Embed overlapping chunks and score each narrative by its best chunk. This requires rebuilding the index.

### 3.3 Write a search query separate from the question
Query 24 ("is the aircraft safe to return to service?") makes a poor search query, and retrieval returned loosely related reports. The filter step could also return a search query written for the narratives. This changes the tool schema and its tests.

### 3.4 Keep the results diverse
Two narratives of the same report can take two of the 10 slots. Cap results per report, or use maximal marginal relevance.

## 4. Usability
*Grading area: usability*

### 4.1 Follow-up questions
Each question is answered on its own, so "tell me more about the second one" does not work. Adding conversation memory improves usability, but makes the evaluation multi-turn and harder to reproduce. If you add it, evaluate multi-turn behavior separately.

### 4.2 Stream the answer
Answers take 15–30 seconds. Streaming the text into the page as it is written would shorten the wait the user perceives. Citation highlighting would then apply once the answer is complete.

### 4.3 Link citations to source cards
Clicking a cited ACN in the answer could jump to or open its source card.

## 5. Hosting

The app runs locally. A hosted copy would make peer testing easier, but hosting is not a course requirement.

| Option | Cost | Notes |
|---|---|---|
| Tunnel from the laptop (`cloudflared`, `ngrok`) | Free | A temporary public link that works only while the app runs. Best for a test session or the demo. |
| Streamlit Community Cloud | Free | Deploys from GitHub. The repo belongs to the class organization, so a personal copy may be needed. Sleeps when idle. |
| Hugging Face Spaces | Free tier | More memory. Uses a Docker template. Sleeps when idle. |
| Container hosting (Render, Railway, Fly.io, Cloud Run) | A few dollars a month | Always on and fully controlled. Needs a Dockerfile. |

Changes needed for any hosted option:
- Store the API key as a secret on the host, never in the repo.
- Build `data/asrs.duckdb` on first start if it is missing.
- Add a shared password and a cap of about 15 questions per session.
- Use a separate API key with a spending limit set in the Anthropic Console, and delete it afterward.

## 6. Comparison scope

### 6.1 Choose the local model for Axis A deliberately
Choose a 7–8B model that supports tool calling in Ollama. On a laptop without a GPU it will be slow. If its tool calls are unreliable, use the fallback from proposal Section 8: constrained decoding or a simplified filter schema. An unreliable local model is itself a reportable finding.

### 6.2 Axis C: strict or permissive citation prompting
This is now only a prompt variant in [prompts/](prompts). Do it only after the items above.

## 7. Data quality

### 7.1 Review the abbreviation expansion by hand
This was in progress at check-in 1. Read expansions in a sample of narratives, and fix mis-expansions with `overrides` or `do_not_expand` in [maintenance_abbreviations.json](data/reference/maintenance_abbreviations.json). Then rebuild the database and the embedding index. Until the index is rebuilt, retrieval refuses to load it, which is intended.
