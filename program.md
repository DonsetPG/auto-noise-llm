# autoresearch-ifim

This repo is for autonomous white-box research on the paper in `main.tex`.

The objective is no longer to improve `val_bpb` on a toy model. The objective is to produce a reproducible, artifact-backed empirical study of the paper's computational claims on open-source models, and to keep a living paper draft in `paper.md`.

## Setup

For a fresh run:

1. Create a new branch named `autoresearch-ifim/<tag>`.
2. Read these files before doing anything else:
   - `main.tex`
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `paper.md`
   - `program.md`
3. If the repo's `pyproject.toml` does not already contain the required packages for Hugging Face open models, merge the additions from `pyproject_additions.md`.
4. Run setup:
   - `uv run prepare.py --num-shards 4 --init-repo`
5. Verify that `results.tsv` exists and that `artifacts/` is writable.
6. Pick an initial small open-source model for the first baseline. A second, stronger model should follow once the pipeline is stable.

Do not skip reading `main.tex`. The definitions in that file determine what counts as a faithful reproduction.

## Hard methodological constraints

These are not optional.

1. The Monte Carlo estimator for the Information Profile must sample continuations from the actual model distribution. In practice, that means ancestral sampling in `train.py` with `temperature=1.0` and `top_p=1.0` for the baseline reproduction.
2. The Information Profile is the expected squared norm of the gradient with respect to the input embeddings. Do not silently replace it with gradient magnitude on logits, attention weights, or hidden states.
3. EIP is an average over inputs, not just over needle positions within one prompt.
4. Empirical NIAH evaluation should use a standard decode strategy. Greedy decoding is the repo default.
5. Do not claim that a Cramér-Rao or latent retrieval result was empirically measured unless the code actually computes the required quantity. The current repo focuses on the paper's practical estimator and the proposed experimental framework.

## Primary research goals

The first priority is to reproduce the paper's computational core:

1. Estimate an Information Profile on at least one open model.
2. Run a position scan over multiple needle locations.
3. Measure empirical NIAH recall on the same position scan.
4. Compare needle-position sensitivity against NIAH accuracy.
5. Estimate the EIP over multiple prompts.
6. Check whether the profile or EIP shows a lost-in-the-middle pattern.
7. Repeat on at least one additional model family.

After that, expand into sensitivity work:

- MC sample count sensitivity
- random seed sensitivity
- context length sensitivity
- generation length sensitivity
- prompt / template sensitivity
- model family and model size sensitivity

## What counts as progress

A change is a **keep** if it does one or more of the following:

- makes the reproduction more faithful to `main.tex`,
- increases estimator stability,
- improves runtime or memory usage without changing the metric definition,
- broadens model coverage,
- improves logging / artifact quality,
- improves the paper draft in `paper.md`,
- fixes a real bug.

A change is a **discard** if it makes the repo less faithful, less reproducible, less stable, or more complex without compensating value.

## Files you may edit

Prefer editing:

- `train.py`
- `program.md`
- `README.md`
- `paper.md`

Edit `prepare.py` only when necessary for stable corpus sampling or setup. Keep the repo small and legible.

## First experiments

The initial sequence should be:

1. One baseline run on a small model using:
   - baseline experiment group
   - default needle positions
   - default MC samples
   - default EIP inputs
2. Inspect the generated JSON and Markdown artifacts.
3. Add the strongest supported observations to `paper.md`.
4. Run one second baseline on a different model family.
5. Start the sensitivity sweeps only after both baselines complete.

## Output and logging

Every run must leave behind:

- a `summary.json`,
- a `run_summary.md`,
- a row in `results.tsv`.

Use `results.tsv` as the authoritative run ledger. The current schema is:

```text
run_id	commit	model_id	backend	experiment_group	seed	context_tokens	num_positions	mc_samples	eip_inputs	max_new_tokens	temperature	top_p	mean_niah_accuracy	spearman_profile_vs_niah	mean_needle_info	eip_lost_middle_ratio	sink_info	peak_vram_gb	total_seconds	status	description	artifact_dir
```

When reviewing a run, inspect at minimum:

- `position_scan.json`
- `eip.json`
- `summary.json`
- any saved plots

## Paper-writing requirements

`paper.md` is a living draft, not a marketing document.

Rules:

1. Only promote a statement from pending to supported if there is a concrete artifact backing it.
2. Record negative results and inconclusive results explicitly.
3. Separate theoretical statements from `main.tex` from new empirical findings in this repo.
4. When a result is weak because `mc_samples` or seeds are small, say so.
5. Do not smooth over disagreements between models. Report them.

## Autonomous loop

Once setup is complete, loop without waiting for the human:

1. Review current `results.tsv` and `paper.md`.
2. Choose the highest-value next experiment.
3. Commit code changes.
4. Run the experiment, redirecting output to a log if needed.
5. Inspect artifacts and update `paper.md`.
6. Keep or discard the code change.
7. Repeat.

Do not stop just because the baseline works. The target state is a repo that can autonomously accumulate evidence and a paper draft that reflects that evidence.

## Practical priorities when stuck

If progress stalls, prefer this order:

1. fix crashes and memory issues,
2. improve artifact quality,
3. add one more model family,
4. increase MC stability on the best runs,
5. only then pursue more speculative extensions.

## Failure handling

If a run crashes:

1. inspect the traceback,
2. decide if it is a shallow bug or a fundamentally bad experiment,
3. fix shallow bugs quickly,
4. if the idea is fundamentally bad, log the crash and move on.

Do not silently hide crashes from `results.tsv`.

## Closing discipline

The point of this repo is not only to run experiments. The point is to leave behind a usable research trail:

- code,
- logs,
- artifacts,
- and a paper draft that a human can audit.

