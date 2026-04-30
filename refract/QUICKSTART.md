# REFRACT QUICKSTART

> **⚠️ ALPHA — for initial testing and feedback only.**
>
> The framework works end-to-end and produces real, useful numbers
> today, but:
> - **Setup is manual** — clone the repo, build llama.cpp / install
>   mlx-lm yourself, fetch the corpus + prompts, edit paths in flags.
>   No `pip install refract` yet.
> - **Proper packaging is coming** (PyPI, a `setup.sh` that builds
>   llama.cpp from a pinned commit, bundled prompts/corpus, an installable
>   CLI entry point). For now run via `python3 -m refract.cli`.
> - **vLLM backend is a skeleton.** llama.cpp + MLX backends work.
> - **Confidence guards exist but aren't exhaustive** — you may find
>   edge cases. Please open an issue with the JSON.
> - **Score interpretation is calibrated on one matrix run** of 7
>   models. Bands (90/80/60) are provisional and may shift in v0.4.
>
> Goals for this alpha: real users on real models exposing real friction
> we can fix. If you hit a wall, open an issue with your `selftest`
> output and the JSON of the failing run.

Goal: get from "git clone" to a real REFRACT score in under 30 minutes.

## What REFRACT does (one paragraph)

REFRACT scores how faithful a quantized KV-cache config is to the same
model's fp16-KV reference. Score 0–100, higher is better. It's a
multi-axis composite (Trajectory + KLD + R-NIAH + PLAD), bit-exact on
Metal, fail-loud (any single broken axis tanks the composite). Replaces
"lower PPL = better" because PPL inverts sign on instruct-tuned models.

## Prereqs

You need at minimum:

  - Python 3.10+
  - One of:
    - **llama.cpp build** with `--jinja` support and the REFRACT v0.1.4
      patch in `tools/completion/completion.cpp`. (Patch emits per-token
      JSONL when `REFRACT_TRAJECTORY` env var is set.)
    - **mlx-lm** (`pip install mlx mlx-lm`). MLX backend is native
      Python; no patches needed.
    - **vllm** (`pip install vllm`). Backend is skeleton in v0.3.1;
      check the source for plug-in pointers.
  - A model in the right format for your backend (.gguf for llama.cpp;
    a directory with `config.json + model.safetensors` for mlx).
  - A natural-text corpus for KLD@D (wikitext-2's `wiki.test.raw` is
    the standard). Get it from
    [llama.cpp's wikitext download](https://github.com/ggerganov/llama.cpp/tree/master/scripts).
  - The prompts JSONL (ships at `refract/prompts/v0.1.jsonl`).
  - For R-NIAH (full mode): a long-text haystack file (wikitext-2's
    `wiki.train.raw` works for cells up to 16K tokens).

## Step 0 — preflight

```
python3 -m refract.cli selftest --backend auto --model /path/to/your.model
```

Verifies binaries, flags, env vars, and a tiny generation. If it bails,
fix the reported issue before going further. Don't burn 30 minutes of a
real run finding out your llama.cpp lacks turbo.

## Step 1 — first quick score (5–7 min on a 7B Q8)

```
python3 -m refract.cli score \
    --model /path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus /path/to/wiki.test.raw \
    --json-out my-first-report.json
```

This runs Trajectory + KLD@D — the two cheap axes. You'll get a
composite score and a band (EXCELLENT/PASS/DEGRADED/FAIL) plus a
plain-English diagnosis of what the per-axis pattern means.

## Step 2 — full audit (25–30 min on a 7B Q8)

Add `--full` plus the R-NIAH haystack flag:

```
python3 -m refract.cli score \
    --model /path/to/model.gguf \
    --candidate "ctk=q8_0,ctv=q8_0" \
    --prompts refract/prompts/v0.1.jsonl \
    --corpus /path/to/wiki.test.raw \
    --full \
    --rniah-haystack /path/to/wiki.train.raw \
    --rniah-ctx-max 16384 \
    --json-out my-full-report.json
```

`--rniah-ctx-max` should match (or be below) your model's max context.
Cells above this are skipped — better to know upfront.

## Step 3 — interpret the result

Quick table:

| Composite | Band      | What it means                                  |
|-----------|-----------|------------------------------------------------|
| 90–100    | EXCELLENT | Indistinguishable from fp16. Safe to deploy.   |
| 80–90     | PASS      | Minor drift; safe to deploy in most uses.      |
| 60–80     | DEGRADED  | Visible drift; audit on your workload first.   |
| 0–60      | FAIL      | Material quality loss; treat as broken.        |

If the composite is below 90, look at the per-axis breakdown and the
**Diagnosis** block in the report. It will tell you in plain English
which surface broke (e.g., "decode distribution drift detected;
candidate generates different tokens than fp16 on short-context
prompts") and a suggested next move.

For deeper interpretation see [`INTERPRETATION.md`](INTERPRETATION.md).

## Step 4 — compare candidates side by side

```
python3 -m refract.cli compare \
    report-q8q8.json report-q8turbo4.json report-q4q4.json
```

Prints a comparison table. Useful for finding the breaking point of a
model under increasingly aggressive quants.

## Backends

| Backend  | Status   | Use for                                         |
|----------|----------|-------------------------------------------------|
| llamacpp | shipping | .gguf models, all four axes, TurboQuant configs |
| mlx      | shipping | MLX models (directory layout); Trajectory + R-NIAH + PLAD work; KLD has limitations on RotatingKVCache models |
| vllm     | skeleton | Plug-in path defined; CUDA/ROCm focus           |

Override default with `--backend mlx` (or `REFRACT_BACKEND=mlx`).

## Common pitfalls (also see [PITFALLS.md](PITFALLS.md))

- **Don't use the v0.1.x `gtm` axis** — it has a known
  detokenize→retokenize unit-mismatch bug. v0.3.1 default is
  `--axis-a trajectory` (the proper fix).
- **Instruct models need chat-template handling** — REFRACT v0.3.0+
  applies it automatically via `--jinja`. If you see all-zero
  retrieval (R-NIAH `base_acc = 0` everywhere), your llama.cpp build
  may be too old.
- **Thinking-mode models** — auto-detected at run start; reasoning
  disabled via `-rea off`. The detection line in the banner says
  whether your model triggered it.
- **R-NIAH with `base_acc < 0.2` averaged across cells** flags
  `confidence: low` in the JSON — the model isn't engaging the task
  and the score is noise-floor.
- **PLAD `paraphrase = NaN`** means no synonym matches in your prompts
  set. Other perturbations (typo/case/punct) still produce valid
  numbers; the cell is recorded as `skipped_perturbations` in JSON.

## Reproducibility

Reports embed:
  - `framework_version` (REFRACT version)
  - `environment.backend` (llamacpp / mlx / vllm)
  - `environment.llama_cpp_commit` (when llamacpp)
  - `environment.mlx_lm_version` (when mlx)
  - `score_direction` and `score_range` (so machine consumers can't
    accidentally invert the comparison)

When sharing scores ("I got 87 on Mistral-7B"), include the JSON. The
number alone is not reproducible without the version stamp.
