# Model Screening: Filter Step

A screening run, done before choosing which models to include in the full Axis A comparison (hosted Claude against local open-weight models). It covers only the **filter step** (the `build_filter` tool call), on **3 queries**, mostly with **one run per model**. It shows which models can drive the tool at all and how they fail. It is not the scored evaluation.

Run on 2026-10-30. Raw records: [results/model_screening.jsonl](results/model_screening.jsonl). Script for the local models: [model_screening.py](model_screening.py).

## Setup

Every model got the same system prompt ([build_filter_system.md](../prompts/build_filter_system.md)), the same tool schema, and the same validation and execution as [code/build_filter.py](../code/build_filter.py), with up to 3 tool calls per question.

| | Claude models | Local models |
|---|---|---|
| Served by | Anthropic API | Ollama 0.40.2 on a laptop CPU (Ryzen 7 PRO 8840U, 29 GB RAM, no GPU) |
| Settings | Effort `medium`, adaptive thinking (the model's default) | 32K context, temperature 0, thinking off where the model allows it (`gpt-oss` only goes down to `low`) |

The queries were chosen to exercise different parts of the filter spec:

| Query | Question | Correct filter |
|---|---|---|
| 6 | Boeing 737 reports, 2022–2024 | `aircraft_family = B737`, years 2022–2024: **109 reports** |
| 10 | Deferred maintenance, 2019 vs 2023 | Two filters, one per year. Gold uses `text_any: ["defer"]` (**30 / 12**); `maintenance_deferred = Y` per year (**36 / 13**) is the listed acceptable alternative |
| 19 | Maintenance reports involving spacecraft | No coded value exists; an invented value must be rejected. The ideal result is **0 reports** |

## Results

✅ correct · 🟡 valid but imprecise · ❌ wrong or no valid filter

| Model | Q6 | Q10 | Q19 | Accepted calls / total | Seconds per question* |
|---|---|---|---|---|---|
| **Claude Opus 5.5** (hosted) | ✅ 109 | 🟡 61 / 25: broader keywords (added "MEL", "placard") | ❌ 8: searched "NASA" | 3 / 3 | not measured separately† |
| **Claude Haiku 5.5** (hosted) | ✅ 109 | ✅ 30 / 12, exact gold | 🟡 4 (run 2) / ❌ 10, searched "NASA" (run 1) | 3 / 3 | 2–4 |
| **gemma4:12b** | ✅ 109, one call | ❌ pasted the value-count table as a value | ❌ 1,114: the term "ISS" matched inside *missing* and *issue* | 2 / 3 | 16–565 (median 28) |
| **gpt-oss:20b** | ❌ split into 3 separate filters (506 / 652 / 2,868) | ✅ 36 / 13, alternative, after 1 rejection | 🟡 3 (spacecraft, satellite, UAV, UAS) | 3 / 5, plus 1 server error | 22–299 (median 217) |
| **qwen3:30b-a3b** | ❌ no tool call: wrote its reasoning as text | ✅ 36 / 13, alternative | ❌ no tool call: wrote its reasoning as text | 1 / 1 | 213–595 (median 467) |
| **qwen3.5:9b** | ❌ 506: after rejection, dropped the years | ❌ 15: one filter, years dropped | ❌ invented "Spacecraft" rejected, then stopped | 2 / 6 | 39–320 (median 91) |
| **llama3.1:8b** | ❌ correct filter, sent as a text string, ×3 | ❌ correct alternative, sent as a text string, ×3 | ❌ invented "Spacecraft", sent as a text string, ×3 | 0 / 9 | 96–816 (median 160) |
| qwen2.5:7b (Q6 only) | ❌ years inside `where`, rejected twice | – | – | 0 / 2 | 308 |

\* Filter step only. The first question for each local model includes loading the model and reading the full prompt; later questions reuse the cached prompt and run faster. `llama3.1:8b` generated at 2.4 tokens/s, against 6.6–9.7 for other models of similar size, so its timings are probably affected by other load on the machine. Local output speed: 3.2–4.1 tokens/s for `qwen3:30b-a3b`, 5.0 for `qwen2.5:7b`, and 6.6–9.7 for the others.

† The Opus rows come from the 2026-10-25 smoke test, which timed the whole question (12–31 s, filter step and answer together).

## Findings

1. **The hosted models produced a valid call every time; no local model was reliable.** Claude Haiku matched gold most closely. The small hosted model beat Opus on query 10, because Opus added keywords beyond the gold filter, so a larger model did not mean a better filter.
2. **Every local model failed in its own way:**
   - `llama3.1` sent the filter list as a text string.
   - `qwen3.5` deleted constraints to get past validation.
   - `gpt-oss` split one filter into several.
   - `gemma4` copied the prompt's value-count table.
   - `qwen3:30b-a3b` wrote its reasoning instead of calling the tool.
3. **The content was often right when the format was wrong.** `llama3.1` chose the exact gold filter for query 6 and the accepted alternative for query 10, but sent both as text, and the rejection message did not tell it what was wrong.
4. **Validation stopped every invented value.** "Spacecraft" never reached the database. But validation cannot catch filters that are valid and wrong:
   - dropped constraints (`qwen3.5`)
   - one filter split into several (`gpt-oss`)
   - substring over-matching (`gemma4`, 1,114 reports)
5. **The same model gave different filters on different runs.** Claude Haiku searched for "NASA" on query 19 in one run but not in the next. Single runs are not enough to rank models.
6. **"NASA" is a prompt gap, not one model's quirk.** Opus and one of the two Haiku runs both searched for it; reporters use the word when they mention filing the ASRS form.

## Fixes on the pipeline side

These came out of the screening and are recorded in the [improvements backlog](../IMPROVEMENTS.md) (section 8). Apply the same fix to every model, so the comparison stays fair.

- **Accept a `filters` list sent as a JSON string,** or reject it with a message that names the problem. This would likely have made `llama3.1` pass queries 6 and 10.
- **Make rejection messages say where a field belongs.** For example: "`year_min` goes at the top level of the filter, not inside `where`". This addresses the `qwen2.5` and `qwen3.5` failures.
- **List field values as plain lists in the prompt, with counts shown separately.** `gemma4` copied the `{value: count}` table.
- **Match keywords on word boundaries** (backlog 1.1). This addresses "ISS" and "MEL".
- **Note in the prompt that NASA and ASRS are not topics** (backlog 1.2).
- **Test `qwen3:30b-a3b` with thinking on,** as a separate configuration. With thinking off, it wrote its reasoning as text instead of calling the tool.

## Next

Apply the fixes above, then rerun this screening, with 2–3 runs per model. Choose the local models for the full 25-query run from those results. Based on this run, `gemma4:12b` and `gpt-oss:20b` are the strongest candidates. `llama3.1:8b` deserves another chance once string-encoded lists are handled.
