# pyproject additions

The original `pyproject.toml` was not attached, so this file lists the extra dependencies you should merge into it.

The repo can run without every optional package, but for the full open-model workflow you should ensure the following are present:

```toml
[project]
dependencies = [
  # existing repo deps stay as-is
  "transformers>=4.45",
  "accelerate>=0.34",
  "safetensors>=0.4",
  "sentencepiece>=0.2",
  "matplotlib>=3.8",
]
```

Notes:

- `transformers`, `accelerate`, `safetensors`, and `sentencepiece` are needed for Hugging Face open-source causal LMs.
- `matplotlib` is only used for saving plots. If it is absent, the run still completes and writes JSON / Markdown artifacts.
- `requests` and `pyarrow` are still useful if you want `prepare.py` to download and sample the climbmix haystack corpus, but the repo has a fallback text path if you skip that step.

