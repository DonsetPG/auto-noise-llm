# autoresearch-ifim

This repo adapts the original **autoresearch** training loop into a white-box long-context analysis harness for the paper in `main.tex`.

Instead of optimizing `val_bpb` on a tiny GPT, the repo now runs autonomous experiments that estimate:

- the **Information Profile** (position-wise input sensitivity),
- the **Expected Information Profile (EIP)**,
- empirical **Needle in a Haystack (NIAH)** retrieval,
- the correlation between positional sensitivity and retrieval,
- sensitivity sweeps over Monte Carlo sample count, context length, prompt wording, generation length, seeds, and model family.

The repo is still deliberately small. The workflow is:

- `prepare.py` sets up the corpus cache and repo bookkeeping.
- `train.py` is the main experiment driver the agent can edit and iterate on.
- `program.md` is the autonomous-research operating manual.
- `paper.md` is the living draft the agent updates as evidence accumulates.

## What this reproduces from the paper

Operationally, the repo implements the paper's computational core:

- **Information Profile** via Monte Carlo estimation of the expected squared norm of gradients with respect to input embeddings.
- **EIP** by averaging resampled profiles across multiple prompts.
- **NIAH evaluation** by placing a synthetic needle at controlled positions in a long haystack and testing recall.
- **Profile vs. retrieval correlation** by scanning multiple needle positions and comparing the resulting sensitivity curve against NIAH accuracy.
- **Sensitivity analysis** by varying estimator and prompt parameters.

The estimator uses **ancestral sampling** by default for the Monte Carlo profile estimate. Greedy decoding is used by default for the empirical NIAH task. That split matches the paper's methodology.

## Model support

The repo is designed for **white-box** autoregressive language models that expose gradients and input embeddings. In practice, that means:

- Hugging Face `AutoModelForCausalLM` models from the hub,
- local Hugging Face-compatible model directories,
- any open-source causal LM that can be loaded through the same interface.

Typical examples are Qwen, Llama, and Mistral families. The code auto-detects CUDA, then MPS, then CPU.

## Quick start

1. Merge the dependency notes from `pyproject_additions.md` into your `pyproject.toml` if those packages are not already present.
2. Prepare the corpus cache and initialize bookkeeping:

```bash
uv run prepare.py --num-shards 4 --init-repo
```

3. Run a baseline reproduction on a small open model:

```bash
uv run train.py --model-id Qwen/Qwen2.5-0.5B-Instruct --experiment-group baseline
```

4. Run a sensitivity sweep:

```bash
uv run train.py --model-id Qwen/Qwen2.5-0.5B-Instruct --experiment-group sensitivity
```

5. Define the current multi-model sweep rosters:

```bash
LONG_CONTEXT_MODEL_IDS="ibm-granite/granite-3.3-2b-instruct,microsoft/Phi-4-mini-instruct,meta-llama/Llama-3.2-3B-Instruct,meta-llama/Llama-3.2-1B-Instruct,google/gemma-3-4b-it,HuggingFaceTB/SmolLM3-3B"
PILOT_MODEL_IDS="google/gemma-3-1b-it,HuggingFaceTB/SmolLM2-1.7B-Instruct-16k,allenai/OLMo-2-0425-1B-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct"
```

6. Run a long-context matrix sweep across context size and MC sample count for the primary roster:

```bash
uv run train.py \
  --experiment-group matrix \
  --model-ids "${LONG_CONTEXT_MODEL_IDS}" \
  --matrix-contexts 2048,8192,16384,32768,65536,98304,128000 \
  --matrix-mc-samples 4,8,16 \
  --needle-positions 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --eip-inputs 0 \
  --trust-remote-code
```

7. Run the cheaper pilot/debug matrix:

```bash
uv run train.py \
  --experiment-group matrix \
  --model-ids "${PILOT_MODEL_IDS}" \
  --matrix-contexts 1024,2048,4096,8192,16384,32768 \
  --matrix-mc-samples 4,8,16 \
  --needle-positions 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --eip-inputs 0
```

Gemma models are gated on Hugging Face and require authenticated access. `microsoft/Phi-4-mini-instruct` should be run with `--trust-remote-code`. `HuggingFaceTB/SmolLM3-3B` is 65,536 tokens by default and should only be pushed to 128K when YaRN/rope settings are explicitly confirmed.

## Default behavior

The defaults are intentionally conservative so the first run is more likely to finish:

- `context_tokens = 2048`
- `mc_samples = 8`
- `eip_inputs = 4`
- `max_new_tokens = 8`
- `needle_positions = 0.1,0.3,0.5,0.7,0.9`

These are **not** the paper's end-state settings. They are a practical starting point for autonomous iteration. The agent should scale them up once the pipeline is stable.

## Output files

Each run creates an artifact directory under `artifacts/<run_id>/` with:

- `config.json` — exact run configuration,
- `summary.json` — machine-readable metrics,
- `run_summary.md` — human-readable summary,
- `position_scan.json` — per-position profile / NIAH details,
- `eip.json` — averaged EIP curve and derived metrics,
- optional PNG plots if matplotlib is available.

Matrix sweeps also create a sweep-level artifact directory with:

- `matrix_config.json` — requested grid and resolved model-limit clamp,
- `matrix_summary.json` — aggregated cell metrics and convergence report,
- `matrix_summary.md` — human-readable matrix tables,
- heatmap PNGs for raw metrics and, when useful, log-scale heatmaps.

Cross-run summary generation (`uv run python make_summary_figures.py`) is expected to produce:

- `needle_sensitivity_position_matrices.png` for all models,
- `niah_accuracy_position_matrices.png` for all models,
- `long_context_limit_summary.png` for all models,
- `context_vs_mean_needle_info.png` with x-axis `context_tokens` and y-axis `mean_needle_info`.

The repo root also keeps:

- `results.tsv` — one row per run,
- `paper.md` — living paper draft / evidence log.

## Results schema

`results.tsv` has one summary row per run with these columns:

```text
run_id	commit	model_id	backend	experiment_group	seed	context_tokens	num_positions	mc_samples	eip_inputs	max_new_tokens	temperature	top_p	mean_niah_accuracy	spearman_profile_vs_niah	mean_needle_info	eip_lost_middle_ratio	sink_info	peak_vram_gb	total_seconds	status	description	artifact_dir
```

This keeps the loop grep-friendly and simple for autonomous agents.

## Design choices

- **White-box only.** The paper's estimator requires gradients with respect to input embeddings, so black-box APIs are out of scope.
- **Single editable driver.** The agent still mainly works through `train.py`, preserving the spirit of the original repo.
- **Downloadable or fallback haystacks.** If you download the climbmix shards, the repo samples real long-form haystacks. If not, it falls back to built-in text snippets so the pipeline still boots.
- **Operational, not maximal.** The code aims to be a working research harness, not the most optimized profiler for every model family.
- **Evidence first.** The agent is expected to update `paper.md` only when a claim is backed by artifacts already present in `artifacts/`.
- **Matrix-first long-context sweeps.** Large-context work should vary context length and MC samples together, then inspect convergence against the largest successful MC count for each context.

## Project structure

```text
README.md              — repo overview
prepare.py             — setup, corpus sampling, bookkeeping
train.py               — experiment driver and model adapter
program.md             — autonomous research instructions
paper.md               — living draft / claim tracker
results.tsv            — run ledger
pyproject_additions.md — dependencies to merge into pyproject.toml
```

## Recommended next steps for the agent

The intended autonomous loop is:

1. reproduce the baseline profile / NIAH correlation on one small model,
2. increase estimator stability,
3. compare architectures and sizes,
4. run sensitivity sweeps,
5. update `paper.md`,
6. iterate until the paper draft is evidence-backed.
