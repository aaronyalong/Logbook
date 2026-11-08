# Command-Line Usage

Each pipeline step can be run on its own. Run these from `project/` with the virtual environment active, after the build steps in the [README's Setup section](../README.md#setup).

## Choosing models

The filter and answer steps run on any model in [code/llm.py](../code/llm.py). A name starting with `claude` uses the Anthropic API and needs `ANTHROPIC_API_KEY`. Any other name is sent to a local [Ollama](https://ollama.com) server, which must be running with that model pulled. Local models run with 32K context, temperature 0, and thinking off where the model allows it. The two steps can use different models.

## Embedding index

The embedding index ([data/index/narrative_embeddings.npz](../data/index)) is committed, because building it takes about 7 minutes on a laptop CPU. Rebuild it only when the narratives change; retrieval refuses to load an index that no longer matches the database.

```bash
python code/build_index.py            # → data/index/narrative_embeddings.npz
python code/build_index.py --check    # does the committed index match data/asrs.duckdb?
```

## Running a filter spec

```python
from filter_spec import run   # with code/ on the path
run({"year_min": 2022, "year_max": 2024, "where": {"aircraft_family": ["B737"]}})
# {'count': 109, 'acns': [...]}
```

The spec format is documented in [code/filter_spec.py](../code/filter_spec.py).

## Building a filter from a question

```bash
python code/build_filter.py "Show me maintenance reports involving Boeing 737 aircraft between 2022 and 2024."
python code/build_filter.py "..." --model gemma4:12b     # a local model
```

It prints each tool call the model made, any validation errors, and the match count for each accepted filter. Every run appends a record to `logs/build_filter.jsonl` (not committed): each tool call, its validation errors, and its match counts.

## Searching the narratives

No API key is needed. The first search downloads the embedding model (65 MB) into `.cache/`, which is not committed; after that, retrieval runs offline.

```bash
python code/retrieve.py "errors during landing gear inspections"           # hybrid, whole corpus
python code/retrieve.py "how fatigue affected their work" --acn 1161791 1072746   # restricted to given reports
python code/retrieve.py "shift turnover" --mode dense -k 5                 # dense only (naive-RAG baseline)
```

## Asking a question (the full pipeline)

```bash
python code/answer.py "What do reporters say about how fatigue affected their work?"
python code/answer.py "..." --filter-model claude-haiku-5-5                  # a cheaper model for the filter step
python code/answer.py "..." --model gemma4:12b --filter-model gemma4:12b     # fully local
python code/answer.py "..." --mode bm25                                      # keyword-only retrieval
```

It prints the filter used, the answer, and a source list of the cited reports. It also warns about any cited ACN the model was not shown, or one that does not exist in the corpus. Each answer is logged to `logs/answer.jsonl`.

## Tests

The unit tests cover:
- abbreviation parsing and expansion, value normalization, and aircraft family mapping
- filter validation and execution
- the build_filter tool loop and the answer step's context building and citation check, against a scripted fake client, so no API key is needed
- the provider layer, against scripted Claude and Ollama responses, so no server is needed
- retrieval ranking and fusion, with a toy embedder, so no model download is needed
- the Streamlit app, as a smoke test through Streamlit's test harness with a fake client and a tiny database
- the NLP experiments' data handling and scoring
- consistency of the evaluation query set

A GitHub Actions workflow ([project-tests.yml](../../.github/workflows/project-tests.yml)) runs them on every push that touches `project/`. It also rebuilds the pipeline from the raw export and fails if any committed generated file, including the embedding index, is out of date.
