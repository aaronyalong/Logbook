# Project Check-in 1 — Individual Report

**Project:** Grounded Question Answering over Aviation Maintenance Safety Narratives
**Author:** Aaron Yalong · CMPE 259, Fall 2026 · October 17, 2026

## Project Activities Completed

**Proposal feedback.** The proposal was approved without requested changes, so work proceeded on the proposed design. The refinements below came from examining the actual data.

**Data acquisition.** Exported the corpus from ASRS Database Online: all reports whose reporter function is a maintenance role, with incident dates from 2010 to 2026. The result is 3,080 reports and 3,632 narratives, since 18% of reports include a second reporter's account. The export file and the exact search criteria are committed for reproducibility.
- The planned 2019–2024 window returned only 1,066 reports, too few for the year-over-year queries, so the window was widened.
- The export limit turned out to be 5,000 records per download, not the 10,000 stated in the ASRS documentation.

**Corpus analysis and resulting design changes.**
- *Abbreviations:* Narratives from 2010 onward are mixed-case prose, not dense shorthand; abbreviations are 2.3% of words. I parsed the ASRS Abbreviations PDF into 501 entries. Its table layout has wrapped rows, so the parser rebuilds rows from word coordinates. That list is oriented to flight operations and covers only 24% of abbreviation tokens in this corpus. I added a hand-curated maintenance supplement of 117 terms (AMM, IPC, RII, MLG, ...) with 6 overrides where the ASRS meaning is wrong for maintenance (e.g. MM is Maintenance Manual, not Middle Marker). Coverage is now 60%. Expansion is applied only to the text used for search; original narratives are kept for citation.
- *Chunking:* The median narrative is 181 words, so each reporter's account is one retrieval unit, and the complex chunking strategy anticipated in the proposal is unnecessary.
- *Metadata:* Added a derived aircraft-family field that maps 225 make/model variants to families, so "737" matches the -800, -900, and MAX variants. Normalized analyst free-text "Other" entries, which reduced the "when detected" field from 187 distinct values to 7. Documented coverage gaps; for example, aircraft family is known for only 67% of reports, which is a concrete recall risk for filters on aircraft type.

**Preprocessing pipeline.** The raw CSV is loaded into a DuckDB database with a reports table (coded fields, with multi-value fields stored as lists) and a narratives table (original and expanded text). A profiling script produces corpus statistics and an inventory of observed values for each of 23 filterable fields.

**Filter tool (execution layer).** Defined the JSON filter specification that the LLM's `build_filter` tool call will produce. A validator rejects unknown fields and invented values, for example "Spacecraft" or "Boeing 737" instead of "B737"; this is the robustness measure from proposal Section 4.2. Valid filters are translated to parameterized SQL. Writing the gold filters showed that metadata constraints alone cannot answer several evaluation queries, so the spec also supports keyword matching (shift handoffs have no coded field) and aggregation (queries asking which categories are most common).

**Evaluation query set.** Refined all 25 queries; the proposal's text said 24 but listed 25. Placeholder ACNs were replaced with real reports, and gold filters were written and run against the corpus for 11 queries, giving expected result counts. The corpus contradicts the false premise in query 21: night-time incidents are 45% of reports with a recorded time.

**Testing and documentation.** Wrote 74 unit tests covering abbreviation parsing and expansion, value normalization, filter validation and execution (including an SQL-injection case), and consistency of the query set. A GitHub Actions workflow runs the tests on every push, rebuilds the pipeline from the raw export, and fails if any committed generated file is out of date. Updated the project README with status, usage, and design notes.

## Project Activities in Progress

- Connecting `build_filter` as an LLM tool call (Anthropic API). The tool schema's allowed values come from the corpus value inventory, and failed validations are logged separately from retrieval results so filter errors can be told apart from retrieval errors (proposal Section 8).
- Manual review of abbreviation expansion on sampled narratives, to catch context-dependent mis-expansions.

## Project Activities Planned

- **Hybrid retrieval:** BM25 plus dense embeddings over the filtered narratives. Embeddings will use a local sentence-transformer model so the offline model path in the model comparison (Axis A) stays genuinely offline. With about 3,600 narratives, plain NumPy similarity search replaces the Chroma/FAISS vector store.
- **Citation-enforced synthesis:** every cited ACN is checked after generation against the retrieved set.
- **Streamlit interface.**
- **Baseline and comparison:** a naive dense-RAG baseline (Axis B) and the local Ollama model path (Axis A).
- **Evaluation run:** all 25 queries on both models, scored on citation validity, filter correctness, refusal behavior, and groundedness.
- **Safety testing:** the six tests from proposal Section 7, including the poisoned-record prompt-injection test.
- **Peer testing:** sessions with two classmates. *The peer section is left blank for this check-in because testers have not yet been arranged.*
- **Literature review:** Lewis et al. (2020) on retrieval-augmented generation; Gao et al. (2023), *Enabling Large Language Models to Generate Text with Citations* (ALCE), for citation evaluation; Greshake et al. (2023) on indirect prompt injection.

## GitHub/Google Drive links

- Project repository: https://github.com/fa26-sjsu-cmpe259/cmpe259-014480362/tree/main/project
- Corpus profile: https://github.com/fa26-sjsu-cmpe259/cmpe259-014480362/blob/main/project/data/profile/corpus_profile.md
- Evaluation query set: https://github.com/fa26-sjsu-cmpe259/cmpe259-014480362/blob/main/project/evaluation/queries.json
- Gold filter results: https://github.com/fa26-sjsu-cmpe259/cmpe259-014480362/blob/main/project/evaluation/gold_filter_results.md
