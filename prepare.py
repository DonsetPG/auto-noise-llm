"""
Setup and stable utilities for context-window information-geometry experiments.

This is the repo-stable companion to train.py. It handles:
- haystack corpus download and sampling,
- device / seed helpers,
- artifact and result bookkeeping,
- prompt helper utilities shared across experiments.

Usage:
    uv run prepare.py --num-shards 4 --init-repo
    uv run prepare.py --skip-download --init-repo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path.home() / ".cache" / "autoresearch_ifim"
DATA_DIR = CACHE_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_TSV = ROOT_DIR / "results.tsv"
PAPER_FILE = ROOT_DIR / "paper.md"

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

DEFAULT_NUM_SHARDS = 4
DEFAULT_CONTEXT_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_MC_SAMPLES = 8
DEFAULT_EIP_INPUTS = 4
DEFAULT_EIP_BINS = 64
DEFAULT_NEEDLE_POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_MODEL_IDS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

RESULTS_HEADER = [
    "run_id",
    "commit",
    "model_id",
    "backend",
    "experiment_group",
    "seed",
    "context_tokens",
    "num_positions",
    "mc_samples",
    "eip_inputs",
    "max_new_tokens",
    "temperature",
    "top_p",
    "mean_niah_accuracy",
    "spearman_profile_vs_niah",
    "mean_needle_info",
    "eip_lost_middle_ratio",
    "sink_info",
    "peak_vram_gb",
    "total_seconds",
    "status",
    "description",
    "artifact_dir",
]

FALLBACK_DOCUMENTS = [
    (
        "Long context evaluation is difficult because the model must preserve a weak signal across a large number "
        "of irrelevant tokens. Empirical benchmarks often report retrieval success, but they do not directly explain "
        "how much of the original signal remains available in the model's conditional distribution."
    ),
    (
        "Information geometry offers a way to reason about sensitivity. If a small perturbation of an input embedding "
        "causes a meaningful change in the output distribution, then the model has retained some information about "
        "that input position. If the perturbation hardly matters, the position is effectively forgotten."
    ),
    (
        "A useful operational probe is to insert a synthetic fact into a long haystack and ask a narrow question "
        "that targets only that fact. One can then compare the output behavior with a white-box estimate of how much "
        "the model's logits depend on the token embeddings at the location of the inserted fact."
    ),
    (
        "Some architectures appear to remember the beginning and end of the prompt better than the middle. The "
        "resulting U-shaped curve has often been described as lost in the middle. A positional sensitivity estimate "
        "can reveal whether the behavior is visible analytically, rather than only through end-to-end task success."
    ),
    (
        "Because open-source models expose their parameters and input embeddings, they are convenient for experiments "
        "that require gradient access. Closed systems may produce strong retrieval behavior, but without gradients "
        "they are outside the scope of this methodology."
    ),
]

NEEDLE_TEMPLATES = (
    "Important fact: the secret code is {secret}.",
    "Memorize this exactly: {secret}.",
    "The hidden identifier you will be asked about is {secret}.",
    "For later reference, the access code is {secret}.",
)

QUESTION_TEMPLATES = (
    "\n\nQuestion: What is the secret code mentioned in the context? Answer with only the code.\nAnswer:",
    "\n\nQuestion: Which exact access code appeared in the context? Return only the code.\nAnswer:",
    "\n\nQuestion: What hidden identifier was stated above? Respond with only the identifier.\nAnswer:",
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slugify(value: str, max_len: int = 80) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        value = "item"
    return value[:max_len]


def sanitize_tsv_value(value: object) -> str:
    text = str(value)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return text.strip()


def write_json(path: Path | str, payload: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def select_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred != "auto":
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def parse_positions(spec: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(spec, (tuple, list)):
        values = [float(x) for x in spec]
    else:
        values = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not values:
        raise ValueError("Needle positions cannot be empty.")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Needle position must be in [0, 1], got {value}")
    return tuple(values)


def make_secret_code(seed: int) -> str:
    adjectives = [
        "amber", "ivory", "silent", "cobalt", "scarlet", "lunar", "verdant", "atomic", "silver", "hidden"
    ]
    nouns = [
        "river", "signal", "harbor", "falcon", "orbit", "forest", "vector", "archive", "anchor", "cipher"
    ]
    rng = random.Random(seed)
    return f"{rng.choice(adjectives)}-{rng.choice(nouns)}-{rng.randint(1000, 9999)}"


def make_needle_text(secret: str, seed: int, template_index: Optional[int] = None) -> str:
    if template_index is None:
        template_index = seed % len(NEEDLE_TEMPLATES)
    template = NEEDLE_TEMPLATES[template_index % len(NEEDLE_TEMPLATES)]
    return template.format(secret=secret)


def make_question_text(seed: int, template_index: Optional[int] = None) -> str:
    if template_index is None:
        template_index = seed % len(QUESTION_TEMPLATES)
    return QUESTION_TEMPLATES[template_index % len(QUESTION_TEMPLATES)]


def append_tsv_row(path: Path | str, row: Dict[str, object], header: Sequence[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header), delimiter="\t")
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        cleaned = {key: sanitize_tsv_value(row.get(key, "")) for key in header}
        writer.writerow(cleaned)


def init_repo_files() -> None:
    ensure_dir(CACHE_DIR)
    ensure_dir(DATA_DIR)
    ensure_dir(ARTIFACTS_DIR)
    if not RESULTS_TSV.exists():
        append_tsv_row(RESULTS_TSV, {}, RESULTS_HEADER)
        # remove the empty data row that DictWriter would otherwise add
        text = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
        RESULTS_TSV.write_text(text[0] + "\n", encoding="utf-8")
    if not PAPER_FILE.exists():
        PAPER_FILE.write_text(
            "# Working paper draft\n\nSee the distributed paper.md template in the packaged repo.\n",
            encoding="utf-8",
        )


def make_run_id(model_id: str, experiment_group: str, seed: int) -> str:
    return f"{timestamp_string()}-{slugify(model_id, 40)}-{slugify(experiment_group, 20)}-s{seed}"


def make_artifact_dir(run_id: str) -> Path:
    artifact_dir = ARTIFACTS_DIR / run_id
    ensure_dir(artifact_dir)
    return artifact_dir


# ---------------------------------------------------------------------------
# Corpus download and sampling
# ---------------------------------------------------------------------------

def download_single_shard(index: int) -> bool:
    import requests

    filename = f"shard_{index:05d}.parquet"
    filepath = DATA_DIR / filename
    if filepath.exists():
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath.with_suffix(".parquet.tmp")
            with temp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            temp_path.replace(filepath)
            print(f"  Downloaded {filename}")
            return True
        except Exception as exc:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {exc}")
            for path in [filepath, filepath.with_suffix(".parquet.tmp")]:
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards: int, download_workers: int = 4) -> None:
    from multiprocessing import Pool

    ensure_dir(DATA_DIR)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(1 for idx in ids if (DATA_DIR / f"shard_{idx:05d}.parquet").exists())
    if existing == len(ids):
        print(f"Data: all {len(ids)} requested shards already exist at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)
    ok = sum(1 for result in results if result)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")


def list_parquet_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    files = sorted(
        path for path in DATA_DIR.iterdir()
        if path.name.endswith(".parquet") and not path.name.endswith(".tmp")
    )
    return files


def iter_documents(split: str = "train", max_documents: Optional[int] = None) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        yield from FALLBACK_DOCUMENTS
        return

    parquet_paths = list_parquet_files()
    if not parquet_paths:
        yield from FALLBACK_DOCUMENTS
        return

    val_path = DATA_DIR / VAL_FILENAME
    if split == "train":
        parquet_paths = [path for path in parquet_paths if path != val_path]
        if not parquet_paths:
            parquet_paths = [path for path in list_parquet_files() if path != val_path] or list_parquet_files()
    elif split == "val":
        parquet_paths = [val_path] if val_path.exists() else parquet_paths[:1]
    else:
        raise ValueError(f"Unknown split: {split}")

    count = 0
    for filepath in parquet_paths:
        try:
            pf = pq.ParquetFile(filepath)
        except Exception:
            continue
        for rg_idx in range(pf.num_row_groups):
            try:
                rg = pf.read_row_group(rg_idx)
                column = rg.column("text").to_pylist()
            except Exception:
                continue
            for text in column:
                if not isinstance(text, str):
                    continue
                cleaned = text.replace("\x00", " ").strip()
                if cleaned:
                    yield cleaned
                    count += 1
                    if max_documents is not None and count >= max_documents:
                        return

    if count == 0:
        yield from FALLBACK_DOCUMENTS


_DOCUMENT_POOL_CACHE: Dict[Tuple[str, int], List[str]] = {}


def load_document_pool(split: str = "train", max_documents: int = 256) -> List[str]:
    key = (split, max_documents)
    if key in _DOCUMENT_POOL_CACHE:
        return _DOCUMENT_POOL_CACHE[key]

    pool: List[str] = []
    for doc in iter_documents(split=split, max_documents=max_documents * 2):
        if len(doc) >= 128:
            pool.append(doc)
        if len(pool) >= max_documents:
            break

    if not pool:
        pool = list(FALLBACK_DOCUMENTS)

    _DOCUMENT_POOL_CACHE[key] = pool
    return pool


def sample_haystack_text(
    target_chars: int,
    seed: int,
    split: str = "train",
    max_documents: int = 256,
) -> str:
    docs = load_document_pool(split=split, max_documents=max_documents)
    rng = random.Random(seed)
    parts: List[str] = []
    total = 0

    while total < target_chars:
        doc = rng.choice(docs)
        if len(doc) > 400:
            start = rng.randint(0, max(0, len(doc) - 300))
            fragment = doc[start:start + min(len(doc) - start, max(300, target_chars // 2))]
        else:
            fragment = doc
        fragment = fragment.strip()
        if not fragment:
            continue
        parts.append(fragment)
        total += len(fragment) + 2
        if len(parts) > 64:
            break

    if not parts:
        parts = list(FALLBACK_DOCUMENTS)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare corpus cache and repo bookkeeping for IFIM autoresearch")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS, help="Number of training shards to cache")
    parser.add_argument("--download-workers", type=int, default=4, help="Parallel shard download workers")
    parser.add_argument("--skip-download", action="store_true", help="Skip corpus download and use fallback text")
    parser.add_argument("--init-repo", action="store_true", help="Create results.tsv and artifacts/ if missing")
    args = parser.parse_args()

    print(f"Root directory:   {ROOT_DIR}")
    print(f"Cache directory:  {CACHE_DIR}")
    print(f"Artifacts:        {ARTIFACTS_DIR}")
    print()

    if args.init_repo:
        init_repo_files()
        print("Repo bookkeeping initialized.")
        print()

    if not args.skip_download:
        download_data(args.num_shards, download_workers=args.download_workers)
        print()

    doc_count = 0
    total_chars = 0
    for doc in iter_documents(split="train", max_documents=16):
        doc_count += 1
        total_chars += len(doc)
    if doc_count == 0:
        print("Corpus status: using fallback text.")
    else:
        print(f"Corpus status: sampled {doc_count} documents / {total_chars:,} chars from cache or fallback.")

    print("Done.")


if __name__ == "__main__":
    main()

