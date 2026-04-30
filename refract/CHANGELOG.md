# REFRACT changelog

Reverse-chronological. Each entry: what changed, why it changed, and the
matrix result that motivated or validated the change.

---

## v0.1.3 — 2026-04-29 (matrix-3 → defensive patch round)

### Context

v0.1.3 is purely defensive — no new functional axes, only test coverage,
fail-loud guards, and honest docs. Triggered by:

1. **The v0.1.2 matrix run revealed a fourth GTM bug.** Even with
   model-token tokenization (the v0.1.2 fix), `mean_prefix_agreement`
   for gemma-4 31B came out at 137 tokens vs `n_predict=48` — a 2.87×
   inflation that re-clipped the GTM score to 100 on a known-degraded
   config. Root cause: detokenize → re-tokenize on verbose
   chain-of-thought outputs can produce ~3× the original token count.
2. **A codex review surfaced a silent fallback** in `axes/gtm.py`. If
   `tokenize_to_ids` raised, a bare `try/except` fell back to
   `_tokenize_words` (whitespace) — silently mixing whitespace-token
   counts with model-token expectations. The comment claimed it was
   "logged as a per-prompt note"; it wasn't. There was zero way to
   detect a wrong-unit score downstream.
3. **No regression test coverage existed** for any of the four bugs
   v0.1.x had to chase: banner-comparing (v0.1), nested-composite JSON
   schema (v0.1), backspace-noise stripping (v0.1.1), Mistral
   UnicodeDecodeError (v0.1.2), GTM unit mismatch (v0.1.2). Every patch
   round was discovered after a multi-hour matrix run.

### v0.1.2 matrix results

7 models on `q8/turbo4 OFF` vs `fp16-KV`, 30 prompts, n_predict=48,
KLD chunks=32, ctx=512.

| Model | composite | band | GTM | KLD@D | mean_kld | prefix/n_predict |
|---|---|---|---|---|---|---|
| gemma-4 31B Q8 | (clipped) | — | **100.00** ⚠ (137 tok / 48 n_pred = 2.87×) | 49.23 | 0.7086 | 2.87 ⚠ |
| phi-4 Q8 | ~65 | DEGRADED | meaningful | 99.55 | 0.0046 | <1 |
| qwen3.5-2B Q8 | ~63 | DEGRADED | meaningful | 98.35 | 0.0167 | <1 |
| gemma-4 E2B Q4_K_L | ~60 | FAIL | meaningful | 93.50 | 0.0672 | <1 |
| qwen2.5-7B Q8 | ~44 | FAIL | meaningful | 98.75 | 0.0126 | <1 |
| gemma-4 26B-A4B Q8 | ~25 | FAIL | meaningful | 17.59 | 1.7381 | <1 |
| Mistral-Small-24B Q4 | ran | — | meaningful | parsed (no crash, errors='replace' fix) | — | — |

What v0.1.2 fixed (still good in v0.1.3):
- KLD@D rankings still match paper §4.3 perfectly.
- Mistral no longer crashes with `UnicodeDecodeError` (errors='replace'
  fix worked).
- 6 of 7 models produced meaningful, non-clipped GTM scores in the
  right units.

What v0.1.2 didn't fix:
- gemma-4 31B re-clipped GTM=100 due to retokenize inflation. The
  fix in v0.1.3 is to normalize by `mean_cand_length` instead of
  `n_predict`.

### v0.1.3 changes

| File | Change | Why |
|---|---|---|
| `axes/gtm.py` | Tokenizer-failure fallback now RAISES instead of silently using whitespace | The previous `try/except → _tokenize_words` produced wrong-unit scores with no way to detect. Fail-loud is correct here — a wrong score is worse than no score. |
| `axes/gtm.py` | Score now `100 * mean_prefix / mean_cand_length` (was `... / n_predict`) | Bounded in [0, 1] regardless of detokenize→retokenize inflation. Fixes the gemma-31B 2.87× clip. |
| `axes/gtm.py` | `GTMResult` exposes `mean_cand_length`, `mean_ref_length`, `notes` | So aggregators can spot retokenize inflation directly. |
| `cli.py` | `--measure-floor` additionally asserts `mean_prefix == mean_cand_length` on ref-vs-ref | KLD ref-vs-ref is bit-exact zero on Metal so the composite floor passes even when GTM is broken. The new check catches that. |
| `runner.py` | `_TOPP_RE` regex tolerates `Same top p` (space) and `Same top-p` (dash) | A test surfaced that the v0.1.2 regex only matched the dashed form. The space form occurs in real llama-perplexity output. |
| `runner.py` | New `corpus_identity()`, `write_corpus_sidecar()`, `read_corpus_sidecar()`, `assert_corpus_matches()` | Records corpus path/size/SHA next to the KLD base file. Refuses to score against a base built from a different corpus. |
| `axes/kld.py` | Writes a `<base>.corpus.json` sidecar, reads it on user-supplied bases, includes corpus identity in `KLDResult` | Wires the new machinery in. JSON now records what corpus the run was scored on. |
| `report.py` | Schema bumped: `refract.report.v0.1.1` → `refract.report.v0.1.3` | Both the GTMResult shape and the corpus-identity field changed; the schema number is the integration contract. |
| `report.py` | Text report shows `mean cand / ref length` and any GTM `notes` | Surfaces the retokenize-inflation diagnostic. |
| `__init__.py` | `__version__ = "0.1.3"` | Match the schema. |
| `tests/test_strip_noise.py` | NEW — 5 tests pinning `_strip_noise` against the v0.1.1 banner / backspace / block-char regressions | Would have caught the v0.1.1 banner-comparing bug pre-matrix. |
| `tests/test_command_construction.py` | NEW — 4 tests pinning `--single-turn` / no-`--no-conversation` / `--kl-divergence-base` / `errors='replace'` | Would have caught the v0.1.1 (`--no-conversation` regression) and v0.1.2 (Unicode crash) bugs pre-matrix. |
| `tests/test_tokenize_to_ids.py` | NEW — 5 tests pinning the `[1, 2, 3]` parser, empty-input short-circuit, and the v0.1.3 raise-on-error contract | Would have caught a future regression of the v0.1.3 fail-loud contract. |
| `tests/test_kld_regex.py` | NEW — 6 tests pinning the perplexity regexes, including the gemma-missing-Same-top-p edge case | Surfaced and fixed the `Same top p` (space) parsing bug while writing the test. |
| `tests/test_report_json_layout.py` | NEW — 5 tests pinning the JSON schema (composite is scalar, schema string, axes block) | Would have caught the v0.1 nested-composite serialisation bug pre-matrix. |
| `tests/test_corpus_identity.py` | NEW — 5 tests for the new sidecar machinery | Guards the corpus-mismatch check. |
| `tests/test_validation.py` | Strengthened with explicit GTM assertions: `full_match_rate < 1.0`, `mean_prefix > 0`, ≥3 distinct ref texts | Even the integration test would have flagged the v0.1.1 banner bug if it had these; previously it asserted only on `composite.band`. |
| `README.md` | Refresh to reflect v0.1.2 model-token tokenization + v0.1.3 candidate-length normalization, point to `LIMITATIONS.md`, list new test files | Docs honesty. |
| `axes/gtm.py` (module docstring) | Same docs cleanup | Docs honesty. |
| `LIMITATIONS.md` | NEW — documents the limitations we're shipping with: detokenize→retokenize gap, corpus-anchored KLD, provisional bands, single platform tested, GTM/EOS conflation, whitespace fallback removal | Codex review surfaced these explicitly; documenting them is more honest than burying TODO comments. |

### Test count

| Version | Total tests | Notes |
|---|---|---|
| v0.1 – v0.1.2 | 14 | Math + parsers only; subprocess wrappers and runner had zero coverage. |
| v0.1.3 | **44** unit + 1 integration | 30 new regression tests across 6 new files; integration test strengthened with 4 new GTM assertions. |

v0.1.3 is the first version with regression test coverage of the
subprocess and runner layers. Everything that broke in v0.1.x now has
a fast, no-subprocess test that would have caught the bug before the
matrix run.

### Open questions (deliberately deferred to v0.2)

- **Should the corpus-identity check be a hard error or a warning?**
  Current behaviour is hard error (RuntimeError). Argument for warning:
  in pipeline use, a user might intentionally re-use a base file across
  truncations of the same corpus. Argument for hard error: the cost of
  a meaningless KLD result that gets believed is much higher than the
  cost of a misfire.
- **Bands re-calibration once trajectory KLD lands** — current 95/80/60
  thresholds are calibrated against corpus-KLD and will need updating
  when the absolute KLD scale changes.

---

## v0.1.2 — 2026-04-29 (matrix-2 → matrix-3 patch round)

### Context

v0.1.1's matrix run surfaced two more bugs:

1. **Mistral-Small-24B failed in the KLD subprocess with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 5204: invalid continuation byte`**. `subprocess.run(..., text=True)` in `runner.py` strict-decodes stderr as utf-8. llama-perplexity occasionally emits non-utf-8 bytes (likely from a model name or progress string), which crashes the entire score before any KLD numbers are returned. Mistral failed in v0.1 too (we thought it was timeout) but the actual failure mode this time was the decode crash, not timeout.
2. **GTM unit mismatch**: `mean_prefix_agreement_length` is in *whitespace-tokens* while `n_predict` is in *model tokens*. Whitespace tokenization can produce more tokens than the model decoded (or fewer, depending on the text shape), so the score `100 * mean_prefix / n_predict` could exceed 1.0 and hit the clip. v0.1.1 matrix showed this on gemma-4 31B, where `mean_prefix = 59 (whitespace) > n_predict = 48 (model)` produced a clipped GTM = 100 even though KLD was 49.

### v0.1.1 matrix results (recorded for reference)

7 models on `q8/turbo4 OFF` vs `fp16-KV`, 30 prompts, n_predict=48, KLD chunks=32, ctx=512.

| Model | composite | band | GTM | KLD@D | mean_kld | prefix/n |
|---|---|---|---|---|---|---|
| gemma-4 31B Q8 | 65.98 | DEGRADED | **100.00** ⚠ | 49.23 | 0.7086 | 1.234 ⚠ |
| phi-4 Q8 | 65.13 | DEGRADED | 48.40 | 99.55 | 0.0046 | 0.484 |
| qwen3.5-2B Q8 | 63.11 | DEGRADED | 46.46 | 98.35 | 0.0167 | 0.465 |
| gemma-4 E2B Q4_K_L | 59.99 | FAIL | 44.17 | 93.50 | 0.0672 | 0.442 |
| qwen2.5-7B Q8 | 43.78 | FAIL | 28.12 | 98.75 | 0.0126 | 0.281 |
| gemma-4 26B-A4B Q8 | 25.01 | FAIL | 43.26 | 17.59 | 1.7381 | 0.433 |
| Mistral-Small-24B Q4 | — | — | 55.83 | UnicodeDecodeError | — | — |

What was real in v0.1.1:
- KLD@D rankings are unchanged from v0.1, still match paper §4.3 perfectly. KLD axis is the trustworthy half.
- Composite ranking correctly identifies gemma-4 26B-A4B as worst (composite 25, FAIL) and shows the gemma artifact as the dominant signal.

What was fake in v0.1.1:
- gemma-4 31B GTM = 100. Whitespace-token over-count clipped to 100. Real model-token prefix-agreement is much lower; the score is artificially inflating.
- Composite ordering of healthy models. phi-4, qwen3.5-2B, gemma-31B all land 63–66 because GTM dominates the composite and GTM is in wrong units. Once GTM is fixed, composite ordering for the healthy tail should track KLD@D.

### v0.1.2 changes

| File | Change | Why |
|---|---|---|
| `runner.py` | Add `errors="replace"` to all 3 `subprocess.run(text=True, ...)` calls | Fixes Mistral's UnicodeDecodeError. llama-perplexity stderr can contain non-utf-8 bytes from model loading; strict utf-8 decode crashes. Replace policy keeps the stream usable. |
| `runner.py` | New `tokenize_to_ids(model, text)` helper that shells to `llama-tokenize --ids --no-bos --no-parse-special --log-disable --stdin` | Enables true model-token diffing for GTM. Returns a `list[int]` of token IDs. Inherits `errors="replace"`. |
| `axes/gtm.py` | Replace `_tokenize_words` with `tokenize_to_ids` in the diff path | Fixes the unit mismatch. `mean_prefix_agreement_length` is now in model tokens, comparable to `n_predict`. The whitespace-tokenizer is kept as a fallback in case `llama-tokenize` errors. |
| (no test changes) | | `_tokenize_words` is still in unit tests as a stable utility; the replacement is in the diff path only. |

### v0.1.2 smoke test (Qwen3.5-2B Q8, 5 prompts, n_predict=32, chunks=8)

```
GTM     diagnostics
  full match rate         : 60.0 %
  median first divergence : token 20.0
  mean prefix agreement   : 30.0 tokens     (model tokens now, was 17.4 whitespace tokens)
KLD     diagnostics
  mean KLD (nats)         : 0.0158
```

GTM score: `100 * 30/32` = **93.75** (was 54.4 with whitespace tokenization). 30 of 32 model tokens matched on average — meaningful, not clipped, comparable across models.

### v0.1.2 matrix results

*(Pending; will append when the v0.1.2 7-model run completes.)*

---

## v0.1.1 — 2026-04-29 (matrix-1 → matrix-2 patch round)

### Context

v0.1's first 7-model matrix run surfaced four bugs that together made the
output unusable:

1. **Composite scalar serialised as a nested dict.** `d['composite']` returned
   `{"composite": 60.23, "band": "DEGRADED", ...}` instead of the scalar.
   Aggregator scripts that did `f"{d['composite']:.2f}"` printed
   `{'compos…` and pretended that was a number.
2. **GTM axis was comparing llama-cli help banners, not generations.** The
   runner passed `--no-conversation` to llama-cli. This fork rejects that
   flag with a custom error (`please use llama-completion instead`) and
   falls through to printing help. The runner captured the help text and
   compared two help banners against each other across all 30 prompts. Match
   rates of 33–57% were entirely artifacts of how often two help banners
   happened to match — not of model faithfulness.
3. **Backspace control characters in llama-cli output broke the noise filter.**
   This fork's llama-cli emits the loading spinner and the generation prefix
   with `\x08` (backspace) chars: `|\x08 \x08[Start thinking]`. The regex
   `^\|\s.*` did not match because `\x08` is not whitespace. Even after the
   `--no-conversation` fix, the noise filter dropped real generations.
4. **Per-model timeout was 30 minutes.** Mistral-Small-24B's full GTM + KLD
   run on a 24B Q4 model needs longer than 30 min on M5 Max. Mistral failed
   with a Python `TimeoutExpired` exception in the KLD-base subprocess.
   (Re-classified in v0.1.2: the actual root cause was UnicodeDecodeError
   on llama-perplexity stderr, not the timeout. The 30-min timeout was a
   contributing factor on the v0.1 run but not the proximate cause.)

### v0.1 matrix results (recorded for reference; numbers are partly invalid)

7 models on `q8/turbo4 OFF` vs `fp16-KV` reference, 30 prompts, n_predict=48,
KLD chunks=32, ctx=512.

| Model | GTM (binary, banner-noise) | KLD@D | mean KLD nats | Hand-computed composite | Notes |
|---|---|---|---|---|---|
| phi-4 Q8 | 56.67 | 99.55 | 0.0046 | 72.21 (DEGRADED) | KLD valid; GTM is banner |
| qwen2.5-7B Q8 | 43.33 | 98.75 | 0.0126 | 60.23 (DEGRADED) | KLD valid; GTM is banner |
| qwen3.5-2B Q8 | 33.33 | 98.35 | 0.0167 | 49.78 (FAIL) | KLD valid; GTM is banner |
| gemma-4 E2B Q4_K_L | 30.00 | 93.50 | 0.0672 | 45.42 (FAIL) | KLD valid; GTM is banner |
| gemma-4 31B Q8 | 23.33 | 49.23 | 0.7086 | 31.66 (FAIL) | KLD valid; GTM is banner |
| gemma-4 26B-A4B Q8 | 36.67 | 17.59 | 1.7381 | 23.78 (FAIL) | KLD valid; GTM is banner |
| Mistral-Small-24B Q4 | 43.33 | — | — | — | killed by 30-min outer timeout |

What was real in v0.1:
- KLD@D rankings are correct and match the paper §4.3 exactly. exp(-KLD)·100
  reproduces the full distribution-drift story (gemma-26B-A4B at 17.59,
  Qwen2.5-7B at 98.75, etc.).
- The composite ranking would have been correct *for the right reasons* if
  GTM were real, because KLD already separates the regimes.

What was fake in v0.1:
- Every GTM number. The "match rate" was the rate at which two captures of
  llama-cli's help text happened to be byte-identical (apparently ~33–57%
  depending on stochastic content like timestamps inside the help text).
- The composite score field in the JSON.

### v0.1.1 changes

| File | Change | Why |
|---|---|---|
| `runner.py` | Drop `--no-conversation` from llama-cli args | This fork rejects the flag and falls through to help-mode. `--single-turn` alone is enough to disable interactive mode. |
| `runner.py` | Strip `\x08` (backspace) from captured stdout before noise filtering | llama-cli puts backspaces inside the spinner AND the `|<BS> <BS>[…]` generation prefix. Without stripping them, the gen-line regex never matches. |
| `runner.py` | Add `Loading model…` and `> prompt-echo` to `_NOISE_PATTERNS` | Cleans up output so the gen-line regex finds the right entry point. |
| `runner.py` | Add gen-line extractor: keep only text from first `^\|\s` line, strip the `| ` prefix | Removes the build-info / available-commands banner that prints on every run on this fork. |
| `runner.py` | Add ASCII-art block-character stripper | The fork's logo uses Unicode block chars; previous filter left them. |
| `axes/gtm.py` | Switch GTM score from `100 * full_match_rate` (binary per prompt) to `100 * mean_prefix_agreement_length / n_predict` (continuous) | Binary was too strict: a single-token divergence at position 5 of 48 zeroed the prompt's contribution. Continuous version distinguishes "matched 5 tokens" from "matched 47 tokens", which is what we actually care about for ranking near-faithful quantizations. `full_match_rate` is still reported as a diagnostic. |
| `report.py` | Flatten `composite` to a scalar at the JSON top level; full breakdown moves to `composite_detail` | Aggregators expected `d['composite']` to be a number per the spec mockups. v0.1 nested it as a dict. Bumped JSON schema to `refract.report.v0.1.1`. |
| `/tmp/refract-matrix.sh` | Per-model `timeout 1800` → `timeout 7200` | Mistral-Small-24B's KLD axis needed >30 min. With real generation now (not banner echo) GTM also takes longer per call. 2 hours per model is the new safety margin. |

### v0.1.1 smoke test (Qwen3.5-2B Q8, 5 prompts, n_predict=32, chunks=8)

```
GTM     diagnostics
  full match rate         : 60.0 %     (now meaningful — different prompts give different outputs)
  median first divergence : token 10
  mean prefix agreement   : 17.4 tokens   (continuous score = 100 * 17.4/32 = 54)
KLD     diagnostics
  mean KLD (nats)         : 0.0158     (matches v0.1 KLD; KLD axis was always correct)
```

Captured ref text post-fix (truncated):
```
prompt: 'The capital of France is'
  ref: '[Start thinking]\nThinking Process:\n\n1. **Analyze the Request:** The user is asking a simple factual question: "The capital of France is".'
prompt: 'The largest ocean on Earth is the'
  ref: '[Start thinking]\nThinking Process:\n\n1. **Analyze the Request:** The user is asking a simple factual question: "The largest ocean on Earth is the".'
```

Different prompts → different outputs ✓. Banner stripped ✓. JSON `d['composite']`
is now a scalar ✓.

---

## v0.1 — 2026-04-29 (initial implementation)

First reference implementation. Two of four planned axes (GTM and KLD@D);
R-NIAH and PLAD deferred to v0.2.

Spec source: post-paper conversation in this repo. Goals: replace PPL with a
reference-anchored multi-axis composite, ship a CLI usable on any GGUF model,
verify Metal-determinism via a noise-floor measurement.

Initial files:
- `cli.py` — argparse front-end
- `runner.py` — subprocess wrappers around `llama-cli` and `llama-perplexity`
- `score.py` — harmonic-mean composite + EXCELLENT/PASS/DEGRADED/FAIL bands
- `report.py` — text + JSON report formatter
- `axes/gtm.py` — Greedy Trajectory Match (binary full-match-rate)
- `axes/kld.py` — KL Divergence at Decode (corpus-proxy via llama-perplexity)
- `prompts/v0.1.jsonl` — 30 CC0 prompts (factual / arithmetic / code / reasoning / instruction / dialogue)
- `tests/test_unit.py` — 14 unit tests
- `tests/test_validation.py` — gated integration test against gemma-4 26B-A4B

Initial bands (calibrated against v0.1.1 matrix; will be revisited in v0.2):
- 95–100 EXCELLENT
- 80–94  PASS
- 60–79  DEGRADED
- <60    FAIL
