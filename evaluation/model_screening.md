# Model Screening: Filter Step

A screening, done before choosing which models to include in the full Axis A comparison (hosted Claude against local open-weight models). It covers only the **filter step** (the `build_filter` tool call), on **3 queries**. It shows which models can drive the tool at all and how they fail. It is not the scored evaluation.

There were two rounds:
- **Round 1** (2026-10-30): one run per model, mostly. It found failures on the pipeline side, which were then fixed.
- **Round 2** (2026-11-01): the same queries after the fixes, with 3 runs per Claude model and 2 per local model.

Raw records: [results/model_screening.jsonl](results/model_screening.jsonl). Round 2 records carry `round`, `run`, and the `commit` they ran on. Script: [model_screening.py](model_screening.py).

## Setup

Every model got the same system prompt ([build_filter_system.md](../prompts/build_filter_system.md)), the same tool schema, and the same validation and execution as [code/build_filter.py](../code/build_filter.py), with up to 3 tool calls per question.

| | Claude models | Local models |
|---|---|---|
| Served by | Anthropic API | Ollama 0.40.2 on a laptop CPU (Ryzen 7 PRO 8840U, 29 GB RAM, no GPU) |
| Settings | Effort `medium`, adaptive thinking (the model's default) | 32K context, temperature 0, thinking off where the model allows it (`gpt-oss` only goes down to `low`); `qwen3:30b-a3b` was also run with thinking on |
| Runs in round 2 | 3 | 2. Temperature 0 makes output nearly deterministic, so the second run mainly checks that. |

Load one local model at a time. In round 2, `qwen3:30b-a3b` (18 GB) was killed for lack of memory while `gpt-oss:20b` (13 GB) was still loaded. Those runs were redone after unloading it.

The queries were chosen to exercise different parts of the filter spec:

| Query | Question | Correct filter |
|---|---|---|
| 6 | Boeing 737 reports, 2022–2024 | `aircraft_family = B737`, years 2022–2024: **109 reports** |
| 10 | Deferred maintenance, 2019 vs 2023 | Two filters, one per year. Gold uses `text_any: ["defer*"]` (**30 / 12**); `maintenance_deferred = Y` per year (**36 / 13**) is the listed acceptable alternative |
| 19 | Maintenance reports involving spacecraft | No coded value exists; an invented value must be rejected. The ideal result is **0 reports** |

✅ correct · 🟡 valid but imprecise · ❌ wrong or no valid filter

## Round 2: after the fixes

| Model | Q6 | Q10 | Q19 | Accepted calls / total | Seconds per question* |
|---|---|---|---|---|---|
| **Claude Haiku 5.5** ×3 | ✅ 109 ×3 | ✅ 30 / 12, exact gold ×3 | 🟡 2, 🟡 2, ❌ 43: searched "space" and "NASA" | 10 / 10 | 2.0–5.1 (median 2.9) |
| **Claude Opus 5.5** ×3 | ✅ 109 ×3 | 🟡 50 / 17, 50 / 17, 53 / 17: added MEL, CDL, and DMI or placard | ✅ 0, 🟡 10: "booster", ✅ 0 | 12 / 12 | 4.1–9.8 (median 6.1) |
| **gemma4:12b** ×2 | ❌ 4 ×2: also required the narrative to contain "B737*" | 🟡 14 / 7 ×2: required both `defer*` and `maintenance_deferred = Y` | 🟡 2 ×2 | 6 / 6 | 17–345 (median 29) |
| **qwen3.5:9b** ×2 | ✅ 109 ×2, after moving the years out of `where` | ❌ ×2: moved keys back and forth for 3 calls | ✅ 0 ×2, after one empty call | 4 / 14 | 69–269 (median 76) |
| **llama3.1:8b** ×2 | ✅ 109 ×2: list sent as text, now decoded | ❌ ×2: four filters, with the years separated from the deferral (346 / 273 / 346 / 120) | ❌ ×2: invented "Spacecraft" and "Space Shuttle" for 3 calls | 4 / 10 | 45–259 (median 64) |
| **gpt-oss:20b** ×2 | ✅ 109 after 2 rejections (run 1); ❌ 3 rejections (run 2) | ❌ ×2: Ollama could not parse its tool call (a closing brace missing); the content was the accepted alternative | 🟡 2 ×2 | 3 / 10, including 2 server errors | 17–194 (median 23) |
| **qwen3:30b-a3b**, thinking off ×2 | ❌ no tool call ×2 | ✅ 36 / 13 ×2, alternative, after 2 rejections | ❌ no tool call ×2 | 2 / 6 | 139–593 (median 260) |
| **qwen3:30b-a3b**, thinking on ×2 | ❌ ×2: wrote the exact gold filter as plain JSON instead of a tool call | ✅ 36 / 13 ×2, alternative, after 2 rejections | ❌ ×2: wrote `{}` as plain text, with a correct explanation | 2 / 6 | 111–707 (median 316) |

\* Filter step only. The first question for each local model includes loading the model. Local output speed: 3.5–3.8 tokens/s for `llama3.1:8b`, 4.0–6.5 for `qwen3:30b-a3b`, 5.1–5.9 for `gemma4:12b`, 5.8–6.1 for `qwen3.5:9b`, and 8.7–8.8 for `gpt-oss:20b`.

What the 🟡 counts on query 19 matched: "satellite" matches 2 reports (a SatCom system and a "satellite area" of a hangar), "booster" matches 8 (oxygen booster carts), "space" matches 35 (hangar space, confined spaces), and "NASA" matches 8 (filing the ASRS form). The answer step is expected to find no spacecraft in any of them.

### Effect of each fix

| Fix | Before (round 1) | After (round 2) |
|---|---|---|
| Decode a `filters` list sent as text (backlog 8.1) | `llama3.1`: 0 of 9 calls accepted | `llama3.1` Q6 exact on both runs; its other failures are now about content |
| Rejection messages say where a key belongs (8.2) | `qwen3.5` deleted the years to pass validation (506 reports) | `qwen3.5` moved the years and matched 109; `gpt-oss` corrected Q6 in run 1. On Q10, `qwen3.5` still moved keys back and forth. |
| Plain value lists in the prompt (8.3) | `gemma4` pasted the `{value: count}` table into Q10 | No model copied the value lists. But `gemma4` Q6 went from ✅ to ❌ (added "B737*" as a keyword). |
| Whole-word keyword matching (1.1) | "ISS" matched 1,114 reports for `gemma4` | Over-matching inside words is gone; Opus's broad Q10 filter fell from 61 / 25 to 50 / 17 |
| NASA is not a topic (1.2) | Searched by Opus (1 of 1 runs) and Haiku (1 of 2) | Opus 0 of 3, Haiku 1 of 3. Reduced, not eliminated. |
| `qwen3:30b-a3b` with thinking on (8.4) | No tool call on Q6 and Q19 | Same filters as with thinking off. Thinking only moved the reasoning out of the reply, which showed the real failure: the tool call written as plain JSON. |

### Findings

1. **The Claude models produced a valid call on every question (22 of 22 calls accepted).** Claude Haiku matched gold exactly on Q6 and Q10 in all 3 runs. Opus was consistent on Q6 but kept adding keywords on Q10; Haiku varied only on Q19.
2. **No local model produced a valid filter for all three questions.** Each one had at least one question that failed on every run.
3. **Most local failures are in the format, not the content.** `qwen3:30b-a3b` wrote the exact gold filter for Q6 as plain JSON. `gpt-oss` chose the accepted Q10 alternative but sent JSON that Ollama could not parse. `llama3.1` sent its list as text, which is now handled.
4. **Valid filters can still be wrong, and validation cannot catch them.** `gemma4` added the aircraft type as a narrative keyword (4 reports instead of 109), and `llama3.1` split one filter per year into four.
5. **Clear rejection messages work for models that read them.** `qwen3.5` and `gpt-oss` fixed misplaced keys after one message. `llama3.1` repeated an invented value three times.
6. **Temperature 0 is not fully deterministic.** `gpt-oss` gave different second and third calls on Q6 in its two runs. Every other local model repeated itself exactly.
7. **Words unrelated to the topic still leak into keyword filters.** "NASA" became rarer, but "space", "booster", and "satellite" all match reports about other things. Whole-word matching cannot fix this, because the words are used correctly, just in a different sense. The answer step has to reject them.

### Choosing local models for the full run

Not decided yet. One more fix on the pipeline side would change the picture: reading a tool call that a model writes as plain JSON in its reply (backlog 8.6). With it, `qwen3:30b-a3b` would likely be the strongest local model on content, but it is also the slowest (about 4–5 minutes per filter). Without it, `gemma4:12b` is the most usable: every call was accepted, its mistakes were imprecise rather than invalid, and it took about 30 seconds per filter.

## Round 1: before the fixes

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

Round 1 used substring keyword matching, and its Q10 gold was `text_any: ["defer"]` (30 / 12 under substring matching).

\* Filter step only. The first question for each local model includes loading the model and reading the full prompt. `llama3.1:8b` generated at 2.4 tokens/s, so its timings are probably affected by other load on the machine.

† The Opus rows come from the 2026-10-25 smoke test, which timed the whole question (12–31 s, filter step and answer together).

Round 1 findings, in short:
- The hosted models produced a valid call every time, and no local model was reliable.
- Each local model failed in its own way: `llama3.1` sent the list as a string, `qwen3.5` deleted constraints to pass validation, `gpt-oss` split one filter into several, `gemma4` copied the value-count table, and `qwen3:30b-a3b` wrote its reasoning instead of calling the tool.
- Validation stopped every invented value, but not valid-and-wrong filters.
- "NASA" was a gap in the prompt, not one model's quirk.

These led to backlog items 1.1, 1.2, and 8.1–8.5, which round 2 measured.
