# Progress

Where each part of the project stands. Planned work and known issues are in the [improvements backlog](../IMPROVEMENTS.md).

| Phase | State |
|-------|-------|
| Data assembly: export, reference parsing, preprocessing, profiling | Complete |
| Filter tool: spec, validation, execution against DuckDB, model tool call | Complete (generated filters not yet scored against gold) |
| Evaluation query set with gold filters | Complete |
| Hybrid retrieval: BM25 + local embeddings, filtered, rank fusion | Complete |
| Answer step: cited synthesis, post-generation citation check | Complete (smoke-tested on 6 queries) |
| Streamlit interface | Complete |
| Model comparison | In progress: filter-step [screening](../evaluation/model_screening.md) of 2 Claude and 5 local models done; the whole pipeline now runs on Claude or local models, with a separate model per step; full runs planned |
| NLP experiments: predicting the blank deferral flag from narratives | In progress: [baselines, embeddings, an LLM zero-shot classifier, and filling the blanks for query 10](../nlp/README.md) done (keyword rule 0.79 F1, TF-IDF 0.76, bge-small embeddings 0.73, Claude Haiku zero-shot 0.82; filling the blanks doubles query 10's candidate set but changes few retrieved reports). Planned: error analysis (top TF-IDF words, a confusion matrix) |
| Safety testing, peer testing | Planned |
