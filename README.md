# Grounded Question Answering over Aviation Maintenance Safety Narratives

An LLM assistant for exploring the NASA Aviation Safety Reporting System (ASRS) maintenance corpus. Analysts ask questions in natural language. The assistant translates them into structured filters over ASRS coded fields, runs hybrid retrieval over the narrative text, and synthesizes an answer in which every claim cites a report by ACN.

[Project proposal](proposal/CMPE259-Project-Proposal.pdf)

## How It Works

```
User question
  → LLM tool call: build_filter(question) → JSON filter spec      code/build_filter.py
  → Spec validated against observed field values, run in DuckDB  code/filter_spec.py
  → Hybrid retrieval (BM25 + dense) over the filtered narratives  code/retrieve.py
  → LLM synthesis with mandatory ACN citation per claim           code/answer.py
  → Every cited ACN checked against the retrieved set             code/answer.py
  → Answer + cited source list
```

Both LLM steps run on Claude through the Anthropic API or on a local model through Ollama ([code/llm.py](code/llm.py)), and each step can use a different model. The reasoning behind each design choice is in [docs/design.md](docs/design.md).

## Data

The corpus is a CSV export from [ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html). The data is public and de-identified by NASA.

**Search criteria** ([asrs_maint_2010_2026_criteria.txt](data/raw/asrs_maint_2010_2026_criteria.txt)): incident date January 2010 to December 2026; Reporter Function is any maintenance role (Technician, Lead Technician, Inspector, Instructor, Trainee, Quality Assurance / Audit, Parts / Stores Personnel, Other / Unknown).

| | |
|---|---|
| Reports | 3,080 (incident dates 2010-01 to 2026-05; 2026 is partial) |
| Narratives | 3,632 (18% of reports include a second reporter's account) |
| Median narrative length | 181 words |

Full statistics are in [data/profile/corpus_profile.md](data/profile/corpus_profile.md).

The search interface caps exports at 5,000 records per download (the ASRS documentation states 10,000), so larger date windows must be split across several exports.

## Setup

Requires Python 3.11 or later.

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt        # add -dev for the test suite
export ANTHROPIC_API_KEY=...                 # for Claude models only; tests do not need it
```

Build the database and generated reference files from the raw export. `data/asrs.duckdb` is not committed; it rebuilds in a few seconds. The embedding index is committed, so it does not need building.

```bash
python code/build_abbreviations.py    # ASRS Abbreviations PDF → data/reference/asrs_abbreviations.json
python code/preprocess.py             # raw CSV → data/asrs.duckdb (reports, narratives tables)
python code/profile_data.py           # → data/profile/corpus_profile.md, filter_values.json
python evaluation/check_gold_filters.py   # validate and run every gold filter
```

## Running the Assistant

```bash
streamlit run code/app.py      # opens http://localhost:8501
```

Type a question or pick an example from the sidebar. Each answer shows:
- cited reports highlighted (blue if verified, red with ⚠ if a citation failed the check)
- source cards with the ASRS synopsis and full narratives
- a "How this answer was built" panel with the filter, match counts, and retrieved narratives

The sidebar switches retrieval between hybrid and the dense-only baseline. Thumbs up/down and comments on each answer are saved to `logs/feedback.jsonl`. A question takes about 15–30 seconds and costs roughly $0.10 in API usage.

To ask from the command line, choose models, or run a single step, see [docs/usage.md](docs/usage.md).

## Testing

```bash
pip install -r code/requirements-dev.txt
pytest
```

No API key, model server, or model download is needed. GitHub Actions runs the tests on every push that touches `project/`; see [docs/usage.md](docs/usage.md#tests) for what they cover.

## Repository Layout

| Path | Contents |
|------|----------|
| [proposal/](proposal) | Project proposal |
| [checkins/](checkins) | Progress check-in submissions |
| [code/](code) | Pipeline source: abbreviation parsing, preprocessing, profiling, filter spec, filter tool call, model providers, retrieval, answer step, Streamlit interface |
| [data/raw/](data/raw) | ASRS export (CSV) and the search criteria used to produce it |
| [data/reference/](data/reference) | ASRS Coding Taxonomy and Abbreviations (PDF), parsed abbreviation tables (JSON) |
| [data/profile/](data/profile) | Generated corpus statistics and the observed value inventory for each filterable field |
| [data/index/](data/index) | Prebuilt narrative embeddings for dense retrieval |
| [prompts/](prompts) | System prompts and prompt variants |
| [evaluation/](evaluation) | Query set, gold filters, evaluation scripts, and model screening results |
| [nlp/](nlp) | NLP experiments beside the pipeline: predicting a coded field from the narratives |
| [docs/](docs) | Design notes, command-line usage, and project progress |
| [tests/](tests) | Unit tests |
| [demo/](demo) | Demo recordings and screenshots |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Known issues and future work, in priority order |

## Known Limitations

The [improvements backlog](IMPROVEMENTS.md) records the planned fix for each.

- **Valid filters can still be wrong.** Validation rejects invented fields and values, but not a filter that drops a constraint or splits one filter into several. The [model screening](evaluation/model_screening.md) saw both from local models.
- **Failed citations are flagged, not corrected.**
- **Long narratives are partly embedded.** 14% exceed the embedding model's input, so dense retrieval sees only their opening.
- **No follow-up questions.** Each question is answered independently.
- **Local only.** The app runs on the machine it is started on.
- **The data cannot support rates.** ASRS reports are voluntary and unverified, so the assistant reports corpus counts only, never real-world frequencies. This limitation is designed in, not a defect.

## Stack

Python · DuckDB (metadata filter) · `rank_bm25` and `BAAI/bge-small-en-v1.5` via `fastembed`, with NumPy similarity search (hybrid retrieval) · Anthropic API and local models through Ollama (filter tool call and answer, [code/llm.py](code/llm.py)) · Streamlit (interface)
