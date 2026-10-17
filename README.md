# Grounded Question Answering over Aviation Maintenance Safety Narratives

An LLM assistant for exploring the NASA Aviation Safety Reporting System (ASRS) maintenance corpus. Analysts ask questions in natural language. The assistant translates them into structured filters over ASRS coded fields, runs hybrid retrieval over the narrative text, and synthesizes an answer in which every claim cites a report by ACN.

See the full [project proposal](proposal/CMPE259-Project-Proposal.pdf).

## Status

| Phase | State |
|-------|-------|
| Data assembly: export, reference parsing, preprocessing, profiling | Complete |
| Filter tool: spec, validation, execution against DuckDB | Complete (LLM tool call not yet connected) |
| Evaluation query set with gold filters | Complete |
| Hybrid retrieval, citation-enforced synthesis, Streamlit interface | Planned |
| Model comparison, safety testing, peer testing | Planned |

## Pipeline

```
User question
  → LLM tool call: build_filter(question) → JSON filter spec      code/filter_spec.py
  → Spec validated against observed field values, run in DuckDB  code/filter_spec.py
  → Hybrid retrieval (BM25 + dense) over the filtered narratives  planned
  → LLM synthesis with mandatory ACN citation per claim           planned
  → Answer + cited source list
```

## Structure

| Path | Contents |
|------|----------|
| [proposal/](proposal) | Project proposal |
| [checkins/](checkins) | Progress check-in submissions |
| [code/](code) | Pipeline source: abbreviation parsing, preprocessing, profiling, filter spec |
| [data/raw/](data/raw) | ASRS export (CSV) and the search criteria used to produce it |
| [data/reference/](data/reference) | ASRS Coding Taxonomy and Abbreviations (PDF), parsed abbreviation tables (JSON) |
| [data/profile/](data/profile) | Generated corpus statistics and the observed value inventory for each filterable field |
| [prompts/](prompts) | System prompts and prompt variants |
| [evaluation/](evaluation) | Query set, gold filters, and evaluation scripts |
| [tests/](tests) | Unit tests |
| [demo/](demo) | Demo recordings and screenshots |

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
```

## Usage

Build the database and generated reference files from the raw export:

```bash
python code/build_abbreviations.py    # ASRS Abbreviations PDF → data/reference/asrs_abbreviations.json
python code/preprocess.py             # raw CSV → data/asrs.duckdb (reports, narratives tables)
python code/profile_data.py           # → data/profile/corpus_profile.md, filter_values.json
python evaluation/check_gold_filters.py   # validate and run every gold filter
```

`data/asrs.duckdb` is not committed; it rebuilds in a few seconds.

Running a filter spec directly:

```python
from filter_spec import run   # with code/ on the path
run({"year_min": 2022, "year_max": 2024, "where": {"aircraft_family": ["B737"]}})
# {'count': 109, 'acns': [...]}
```

The spec format is documented in [code/filter_spec.py](code/filter_spec.py).

## Testing

```bash
pip install -r code/requirements-dev.txt
pytest
```

The unit tests cover abbreviation parsing and expansion, value normalization, aircraft family mapping, filter validation and execution, and consistency of the evaluation query set. A GitHub Actions workflow ([project-tests.yml](../.github/workflows/project-tests.yml)) runs them on every push that touches `project/`. It also rebuilds the pipeline from the raw export and fails if any committed generated file is out of date.

## Design Notes

Decisions made after examining the exported data, relative to the proposal:

- **Abbreviation normalization.** Narratives from 2010 onward are mixed-case prose; abbreviations make up about 2% of words. The ASRS list is oriented to flight operations and covers 24% of abbreviation tokens in this corpus. A hand-curated maintenance supplement ([maintenance_abbreviations.json](data/reference/maintenance_abbreviations.json)) raises coverage to 60%, and overrides ASRS senses that are wrong in a maintenance context (e.g. MM is Maintenance Manual, not Middle Marker). Expansion is applied only to the text used for indexing; original narratives are kept for display and citation.
- **Chunking.** Narratives are short enough that each reporter's account is a single retrieval unit.
- **Filter spec.** Beyond metadata constraints, the spec supports keyword matching (`text_any`) and aggregation (`group_by`), which several evaluation queries require. Values are validated against those observed in the corpus, so an invented category is rejected rather than silently returning no results.
- **Aircraft family.** `make_model` has 225 distinct values; a derived `aircraft_family` column groups variants (e.g. B737-800, B737 MAX 9) so a question about "737s" matches all of them.

## Planned Stack

Python · DuckDB (metadata filter) · `rank_bm25` and a local sentence-embedding model with NumPy similarity search (hybrid retrieval) · Streamlit · Anthropic API and Ollama (model comparison)
