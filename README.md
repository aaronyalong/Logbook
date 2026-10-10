# Grounded Question Answering over Aviation Maintenance Safety Narratives

An LLM assistant for exploring the NASA Aviation Safety Reporting System (ASRS) maintenance corpus. Analysts ask questions in natural language. The assistant translates them into structured filters over ASRS coded fields, runs hybrid retrieval over the narrative text, and synthesizes an answer in which every claim cites a report by ACN.

See the full [project proposal](proposal/CMPE259-Project-Proposal.pdf).

## Structure

| Folder | Contents |
|--------|----------|
| [proposal/](proposal) | Project proposal |
| [checkins/](checkins) | Progress check-in submissions |
| [code/](code) | Assistant source code (filter tool, retrieval, Streamlit app) |
| [data/](data) | ASRS export, taxonomy/abbreviation references, prebuilt index |
| [prompts/](prompts) | System prompts and prompt variants |
| [evaluation/](evaluation) | Query set, evaluation scripts, and results |
| [demo/](demo) | Demo recordings and screenshots |

## Planned Stack

Python · Streamlit · DuckDB (metadata filter) · Chroma or FAISS + `rank_bm25` (hybrid retrieval) · Anthropic API and Ollama (model comparison)

## Setup

_TODO: fill in once the code is in place._

1. Install dependencies: `pip install -r code/requirements.txt`
2. Set the API key for the hosted model: `export ANTHROPIC_API_KEY=...`
3. For the local model path, install [Ollama](https://ollama.com) and pull the chosen model.

## Usage

_TODO: how to launch the Streamlit app, rebuild the index, and run the evaluation._

## Data

The corpus is a manual CSV export of maintenance-category reports from [ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html). The data is public and de-identified by NASA.
