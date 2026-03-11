"""
Autoresearch experiment driver for reproducing the Information Geometry paper.

Typical usage:
    uv run train.py --model-id Qwen/Qwen2.5-0.5B-Instruct --experiment-group baseline
    uv run train.py --model-id Qwen/Qwen2.5-0.5B-Instruct --experiment-group sensitivity
    uv run train.py --experiment-group compare --model-ids model_a,model_b,model_c
"""

from __future__ import annotations

import argparse
import gc
import math
import re
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from prepare import (
    ARTIFACTS_DIR,
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_EIP_BINS,
    DEFAULT_EIP_INPUTS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MC_SAMPLES,
    DEFAULT_MODEL_IDS,
    DEFAULT_NEEDLE_POSITIONS,
    RESULTS_HEADER,
    RESULTS_TSV,
    append_tsv_row,
    default_dtype_for_device,
    get_git_commit,
    init_repo_files,
    make_artifact_dir,
    make_needle_text,
    make_question_text,
    make_run_id,
    make_secret_code,
    parse_positions,
    sample_haystack_text,
    seed_everything,
    select_device,
    slugify,
    write_json,
    write_text,
)

ROOT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Small math / formatting helpers
# ---------------------------------------------------------------------------

def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_contains(prediction: str, secret: str) -> bool:
    return normalize_answer(secret) in normalize_answer(prediction)


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return float(num / (den_x * den_y))


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    return pearson_corr(rankdata(xs), rankdata(ys))


def resample_profile(profile: Sequence[float], bins: int) -> List[float]:
    tensor = torch.tensor(list(profile), dtype=torch.float32)
    if tensor.numel() == 0:
        return [0.0 for _ in range(bins)]
    if tensor.numel() == 1:
        return [float(tensor.item()) for _ in range(bins)]
    view = tensor.view(1, 1, -1)
    resized = F.interpolate(view, size=bins, mode="linear", align_corners=True)
    return [float(x) for x in resized.view(-1).tolist()]


def edge_middle_ratio(curve: Sequence[float], edge_frac: float = 0.15, middle_frac: float = 0.20) -> float:
    if not curve:
        return float("nan")
    n = len(curve)
    edge_n = max(1, int(round(n * edge_frac)))
    middle_n = max(1, int(round(n * middle_frac)))
    left = list(curve[:edge_n])
    right = list(curve[-edge_n:])
    middle_start = max(0, (n - middle_n) // 2)
    middle = list(curve[middle_start:middle_start + middle_n])
    edge_mean = safe_mean(left + right)
    middle_mean = safe_mean(middle)
    if edge_mean == 0:
        return float("nan")
    return float(middle_mean / edge_mean)


def format_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def parse_int_csv(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_model_id_list(model_id: str, model_ids: Optional[str]) -> List[str]:
    if model_ids:
        return [x.strip() for x in model_ids.split(",") if x.strip()]
    return [model_id]


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

def import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for open-model support. Merge pyproject_additions.md into pyproject.toml."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def infer_max_context_tokens(model, tokenizer) -> int:
    candidates: List[int] = []
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "sliding_window"):
        value = getattr(getattr(model, "config", object()), attr, None)
        if isinstance(value, int) and 16 <= value <= 10_000_000:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 16 <= tokenizer_limit <= 10_000_000:
        candidates.append(tokenizer_limit)
    if not candidates:
        return DEFAULT_CONTEXT_TOKENS
    return min(candidates)


class HFAdapter:
    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype, trust_remote_code: bool = False):
        AutoModelForCausalLM, AutoTokenizer = import_transformers()
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        model_kwargs = dict(trust_remote_code=trust_remote_code)
        if device.type != "cpu":
            model_kwargs["torch_dtype"] = dtype
        try:
            model_kwargs["low_cpu_mem_usage"] = True
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.embedding_layer = self.model.get_input_embeddings()
        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.max_context_tokens = infer_max_context_tokens(self.model, self.tokenizer)

    def autocast_context(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def encode_text(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode_tokens(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def greedy_completion(self, prompt_ids: Sequence[int], max_new_tokens: int) -> List[int]:
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            with self.autocast_context():
                outputs = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=self.device)], dim=1)
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break
        return ids[0, len(prompt_ids):].tolist()

    @torch.no_grad()
    def ancestral_completion(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> List[int]:
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(seed)

        for _ in range(max_new_tokens):
            with self.autocast_context():
                outputs = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = outputs.logits[:, -1, :].float().cpu().squeeze(0)

            if temperature <= 0:
                next_id = int(logits.argmax().item())
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                if 0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumulative <= top_p
                    keep[0] = True
                    filtered = torch.zeros_like(probs)
                    filtered[sorted_indices[keep]] = probs[sorted_indices[keep]]
                    norm = filtered.sum()
                    probs = filtered / norm if norm > 0 else probs
                next_id = int(torch.multinomial(probs, num_samples=1, generator=cpu_generator).item())

            next_token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            ids = torch.cat([ids, next_token], dim=1)
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break

        return ids[0, len(prompt_ids):].tolist()

    def score_sample_and_grad(self, prompt_ids: Sequence[int], sample_ids: Sequence[int]) -> Tuple[float, List[float]]:
        full_ids = list(prompt_ids) + list(sample_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            base_embeds = self.embedding_layer(input_ids)
        inputs_embeds = base_embeds.detach().clone().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        with self.autocast_context():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        prompt_len = len(prompt_ids)
        sampled_log_probs = token_log_probs[:, prompt_len - 1:] if prompt_len > 0 else token_log_probs
        log_likelihood = sampled_log_probs.sum()
        log_likelihood.backward()

        grad = inputs_embeds.grad[0, :prompt_len].detach().float()
        grad_sq_norm = grad.square().sum(dim=-1).cpu().tolist()
        return float(log_likelihood.item()), [float(x) for x in grad_sq_norm]


# ---------------------------------------------------------------------------
# Experiment configuration and prompt construction
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    model_id: str = DEFAULT_MODEL_IDS[0]
    backend: str = "hf"
    experiment_group: str = "baseline"
    seed: int = 0
    context_tokens: int = DEFAULT_CONTEXT_TOKENS
    needle_positions: Tuple[float, ...] = DEFAULT_NEEDLE_POSITIONS
    mc_samples: int = DEFAULT_MC_SAMPLES
    eip_inputs: int = DEFAULT_EIP_INPUTS
    eip_bins: int = DEFAULT_EIP_BINS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    corpus_split: str = "train"
    trust_remote_code: bool = False
    save_prompts: bool = False
    description: str = "baseline"
    sensitivity_mc_samples: Tuple[int, ...] = ()
    sensitivity_contexts: Tuple[int, ...] = ()
    sensitivity_max_new: Tuple[int, ...] = ()
    sensitivity_seeds: Tuple[int, ...] = ()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        return cls(
            model_id=args.model_id,
            backend="hf",
            experiment_group=args.experiment_group,
            seed=args.seed,
            context_tokens=args.context_tokens,
            needle_positions=parse_positions(args.needle_positions),
            mc_samples=args.mc_samples,
            eip_inputs=args.eip_inputs,
            eip_bins=args.eip_bins,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            corpus_split=args.corpus_split,
            trust_remote_code=args.trust_remote_code,
            save_prompts=args.save_prompts,
            description=args.description,
            sensitivity_mc_samples=tuple(parse_int_csv(args.sensitivity_mc_samples)),
            sensitivity_contexts=tuple(parse_int_csv(args.sensitivity_contexts)),
            sensitivity_max_new=tuple(parse_int_csv(args.sensitivity_max_new)),
            sensitivity_seeds=tuple(parse_int_csv(args.sensitivity_seeds)),
        )


@dataclass
class PromptExample:
    prompt_ids: List[int]
    secret: str
    needle_start: int
    needle_end: int
    prompt_tokens: int
    requested_position: float
    actual_position: float
    prompt_preview: str


def resolve_context_tokens(requested: int, adapter: HFAdapter, max_new_tokens: int) -> int:
    reserve = max_new_tokens + 32
    effective = min(requested, max(64, adapter.max_context_tokens - reserve))
    return effective


def build_niah_prompt(
    adapter: HFAdapter,
    context_tokens: int,
    needle_position: float,
    seed: int,
    corpus_split: str,
    save_prompts: bool,
) -> PromptExample:
    secret = make_secret_code(seed)
    needle_text = "\n\n" + make_needle_text(secret, seed) + "\n\n"
    question_text = make_question_text(seed)
    prefix_text = "Read the following context carefully.\n\n"

    bos_ids = [adapter.bos_token_id] if adapter.bos_token_id is not None else []
    prefix_ids = adapter.encode_text(prefix_text, add_special_tokens=False)
    needle_ids = adapter.encode_text(needle_text, add_special_tokens=False)
    question_ids = adapter.encode_text(question_text, add_special_tokens=False)

    reserved = len(bos_ids) + len(prefix_ids) + len(needle_ids) + len(question_ids)
    if context_tokens <= reserved + 32:
        raise ValueError(
            f"context_tokens={context_tokens} is too small once prompt scaffolding is reserved ({reserved} tokens)."
        )

    haystack_target = context_tokens - reserved
    haystack_ids: List[int] = []
    text_seed = seed
    while len(haystack_ids) < haystack_target:
        target_chars = max(2000, 6 * (haystack_target - len(haystack_ids)))
        haystack_text = sample_haystack_text(target_chars=target_chars, seed=text_seed, split=corpus_split)
        haystack_ids.extend(adapter.encode_text(haystack_text, add_special_tokens=False))
        text_seed += 1
        if text_seed > seed + 32:
            break
    haystack_ids = haystack_ids[:haystack_target]

    if not haystack_ids:
        raise RuntimeError("Could not construct a haystack prompt.")

    insertion_idx = int(round(needle_position * len(haystack_ids)))
    insertion_idx = max(0, min(insertion_idx, len(haystack_ids)))

    prompt_ids = (
        bos_ids
        + prefix_ids
        + haystack_ids[:insertion_idx]
        + needle_ids
        + haystack_ids[insertion_idx:]
        + question_ids
    )

    needle_start = len(bos_ids) + len(prefix_ids) + insertion_idx
    needle_end = needle_start + len(needle_ids)
    actual_position = ((needle_start + needle_end) / 2.0) / max(1, len(prompt_ids))

    prompt_preview = ""
    if save_prompts:
        preview_ids = prompt_ids[: min(len(prompt_ids), 512)]
        prompt_preview = adapter.decode_tokens(preview_ids, skip_special_tokens=True)

    return PromptExample(
        prompt_ids=list(prompt_ids),
        secret=secret,
        needle_start=needle_start,
        needle_end=needle_end,
        prompt_tokens=len(prompt_ids),
        requested_position=needle_position,
        actual_position=actual_position,
        prompt_preview=prompt_preview,
    )


# ---------------------------------------------------------------------------
# Core experiments
# ---------------------------------------------------------------------------

def estimate_information_profile(
    adapter: HFAdapter,
    prompt_ids: Sequence[int],
    mc_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> Dict[str, object]:
    prompt_len = len(prompt_ids)
    accum = [0.0 for _ in range(prompt_len)]
    sample_summaries: List[Dict[str, object]] = []

    for sample_idx in range(mc_samples):
        sample_ids = adapter.ancestral_completion(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + 1000 + sample_idx,
        )
        log_likelihood, grad_sq_norm = adapter.score_sample_and_grad(prompt_ids, sample_ids)
        for i, value in enumerate(grad_sq_norm):
            accum[i] += value
        sample_summaries.append(
            {
                "sample_index": sample_idx,
                "generated_tokens": len(sample_ids),
                "log_likelihood": log_likelihood,
                "continuation_text": adapter.decode_tokens(sample_ids, skip_special_tokens=True).strip(),
            }
        )

    if mc_samples > 0:
        profile = [value / mc_samples for value in accum]
    else:
        profile = accum

    return {
        "profile": profile,
        "samples": sample_summaries,
    }


def summarize_position_run(
    adapter: HFAdapter,
    config: ExperimentConfig,
    position: float,
    seed: int,
    effective_context_tokens: int,
) -> Dict[str, object]:
    example = build_niah_prompt(
        adapter=adapter,
        context_tokens=effective_context_tokens,
        needle_position=position,
        seed=seed,
        corpus_split=config.corpus_split,
        save_prompts=config.save_prompts,
    )

    mc = estimate_information_profile(
        adapter=adapter,
        prompt_ids=example.prompt_ids,
        mc_samples=config.mc_samples,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=seed,
    )
    profile = list(mc["profile"])
    profile_resampled = resample_profile(profile, bins=config.eip_bins)
    greedy_ids = adapter.greedy_completion(example.prompt_ids, max_new_tokens=config.max_new_tokens)
    prediction = adapter.decode_tokens(greedy_ids, skip_special_tokens=True).strip()
    niah_accuracy = 1.0 if exact_match_contains(prediction, example.secret) else 0.0

    needle_span = profile[example.needle_start:example.needle_end] or [0.0]
    needle_info = safe_mean(needle_span)
    sink_info = float(profile[0]) if profile else float("nan")

    return {
        "requested_position": position,
        "actual_position": example.actual_position,
        "seed": seed,
        "secret": example.secret,
        "prompt_tokens": example.prompt_tokens,
        "needle_start": example.needle_start,
        "needle_end": example.needle_end,
        "needle_info": needle_info,
        "sink_info": sink_info,
        "profile_mean": safe_mean(profile),
        "profile_curve": profile,
        "profile_curve_resampled": profile_resampled,
        "niah_prediction": prediction,
        "niah_accuracy": niah_accuracy,
        "sample_summaries": mc["samples"],
        "prompt_preview": example.prompt_preview,
    }


def run_position_scan(
    adapter: HFAdapter,
    config: ExperimentConfig,
    effective_context_tokens: int,
) -> Dict[str, object]:
    results: List[Dict[str, object]] = []

    for idx, position in enumerate(config.needle_positions):
        seed = config.seed + idx * 97
        result = summarize_position_run(
            adapter=adapter,
            config=config,
            position=position,
            seed=seed,
            effective_context_tokens=effective_context_tokens,
        )
        results.append(result)

    accuracies = [float(item["niah_accuracy"]) for item in results]
    needle_infos = [float(item["needle_info"]) for item in results]
    sink_infos = [float(item["sink_info"]) for item in results]

    return {
        "positions": results,
        "mean_niah_accuracy": safe_mean(accuracies),
        "mean_needle_info": safe_mean(needle_infos),
        "mean_sink_info": safe_mean(sink_infos),
        "spearman_profile_vs_niah": spearman_corr(needle_infos, accuracies),
        "pearson_profile_vs_niah": pearson_corr(needle_infos, accuracies),
    }


def run_eip(
    adapter: HFAdapter,
    config: ExperimentConfig,
    effective_context_tokens: int,
) -> Dict[str, object]:
    curves: List[List[float]] = []
    inputs: List[Dict[str, object]] = []

    if config.eip_inputs <= 0:
        return {
            "curve": [],
            "lost_middle_ratio": float("nan"),
            "sink_info": float("nan"),
            "inputs": [],
        }

    for idx in range(config.eip_inputs):
        position = config.needle_positions[idx % len(config.needle_positions)]
        seed = config.seed + 10_000 + idx
        result = summarize_position_run(
            adapter=adapter,
            config=config,
            position=position,
            seed=seed,
            effective_context_tokens=effective_context_tokens,
        )
        curve = list(result["profile_curve_resampled"])
        curves.append(curve)
        inputs.append(
            {
                "seed": seed,
                "requested_position": position,
                "actual_position": result["actual_position"],
                "needle_info": result["needle_info"],
                "niah_accuracy": result["niah_accuracy"],
            }
        )

    eip_curve = [
        safe_mean([curve[i] for curve in curves])
        for i in range(config.eip_bins)
    ]
    sink_info = eip_curve[0] if eip_curve else float("nan")
    return {
        "curve": eip_curve,
        "lost_middle_ratio": edge_middle_ratio(eip_curve),
        "sink_info": sink_info,
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def maybe_make_plots(position_scan: Dict[str, object], eip: Dict[str, object], artifact_dir: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    saved: List[str] = []

    positions = [float(item["requested_position"]) for item in position_scan["positions"]]
    needle_infos = [float(item["needle_info"]) for item in position_scan["positions"]]
    accuracies = [float(item["niah_accuracy"]) for item in position_scan["positions"]]

    fig = plt.figure()
    plt.plot(positions, needle_infos, marker="o")
    plt.xlabel("needle position")
    plt.ylabel("needle sensitivity")
    plt.title("Needle sensitivity by position")
    path = artifact_dir / "position_scan_profile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(path.name)

    fig = plt.figure()
    plt.plot(positions, accuracies, marker="o")
    plt.xlabel("needle position")
    plt.ylabel("NIAH accuracy")
    plt.title("NIAH accuracy by position")
    path = artifact_dir / "position_scan_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(path.name)

    if eip.get("curve"):
        xs = list(range(len(eip["curve"])))
        fig = plt.figure()
        plt.plot(xs, eip["curve"])
        plt.xlabel("normalized context bin")
        plt.ylabel("EIP")
        plt.title("Expected Information Profile")
        path = artifact_dir / "eip_curve.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path.name)

    return saved


def write_run_summary_markdown(summary: Dict[str, object], artifact_dir: Path) -> None:
    position_rows = []
    for item in summary["position_scan"]["positions"]:
        position_rows.append(
            "| {requested:.2f} | {actual:.3f} | {needle_info} | {acc} | {tokens} |".format(
                requested=float(item["requested_position"]),
                actual=float(item["actual_position"]),
                needle_info=format_float(item["needle_info"]),
                acc=format_float(item["niah_accuracy"]),
                tokens=int(item["prompt_tokens"]),
            )
        )

    eip = summary.get("eip", {})
    text = f"""# Run {summary['run_id']}

- Model: `{summary['model_id']}`
- Experiment group: `{summary['experiment_group']}`
- Status: `{summary['status']}`
- Commit: `{summary['commit']}`
- Context tokens: `{summary['context_tokens']}`
- MC samples: `{summary['mc_samples']}`
- EIP inputs: `{summary['eip_inputs']}`
- Mean NIAH accuracy: `{format_float(summary['mean_niah_accuracy'])}`
- Spearman(profile, NIAH): `{format_float(summary['spearman_profile_vs_niah'])}`
- Mean needle sensitivity: `{format_float(summary['mean_needle_info'])}`
- EIP lost-middle ratio: `{format_float(summary['eip_lost_middle_ratio'])}`
- Sink sensitivity: `{format_float(summary['sink_info'])}`
- Peak VRAM (GB): `{format_float(summary['peak_vram_gb'], digits=3)}`
- Total seconds: `{format_float(summary['total_seconds'], digits=2)}`

## Position scan

| requested position | actual position | needle sensitivity | NIAH accuracy | prompt tokens |
| --- | --- | --- | --- | --- |
{chr(10).join(position_rows)}

## EIP

- Lost-middle ratio: `{format_float(eip.get('lost_middle_ratio'))}`
- Sink sensitivity: `{format_float(eip.get('sink_info'))}`

## Notes

{summary['description']}
"""
    write_text(artifact_dir / "run_summary.md", text)


def append_summary_row(summary: Dict[str, object]) -> None:
    row = {
        "run_id": summary["run_id"],
        "commit": summary["commit"],
        "model_id": summary["model_id"],
        "backend": summary["backend"],
        "experiment_group": summary["experiment_group"],
        "seed": summary["seed"],
        "context_tokens": summary["context_tokens"],
        "num_positions": summary["num_positions"],
        "mc_samples": summary["mc_samples"],
        "eip_inputs": summary["eip_inputs"],
        "max_new_tokens": summary["max_new_tokens"],
        "temperature": summary["temperature"],
        "top_p": summary["top_p"],
        "mean_niah_accuracy": format_float(summary["mean_niah_accuracy"]),
        "spearman_profile_vs_niah": format_float(summary["spearman_profile_vs_niah"]),
        "mean_needle_info": format_float(summary["mean_needle_info"]),
        "eip_lost_middle_ratio": format_float(summary["eip_lost_middle_ratio"]),
        "sink_info": format_float(summary["sink_info"]),
        "peak_vram_gb": format_float(summary["peak_vram_gb"], digits=3),
        "total_seconds": format_float(summary["total_seconds"], digits=2),
        "status": summary["status"],
        "description": summary["description"],
        "artifact_dir": summary["artifact_dir"],
    }
    append_tsv_row(RESULTS_TSV, row, RESULTS_HEADER)


def print_summary(summary: Dict[str, object]) -> None:
    print("---")
    print(f"status:                    {summary['status']}")
    print(f"model_id:                  {summary['model_id']}")
    print(f"experiment_group:          {summary['experiment_group']}")
    print(f"mean_niah_accuracy:        {format_float(summary['mean_niah_accuracy'])}")
    print(f"spearman_profile_vs_niah:  {format_float(summary['spearman_profile_vs_niah'])}")
    print(f"mean_needle_info:          {format_float(summary['mean_needle_info'])}")
    print(f"eip_lost_middle_ratio:     {format_float(summary['eip_lost_middle_ratio'])}")
    print(f"sink_info:                 {format_float(summary['sink_info'])}")
    print(f"peak_vram_gb:              {format_float(summary['peak_vram_gb'], digits=3)}")
    print(f"total_seconds:             {format_float(summary['total_seconds'], digits=2)}")
    print(f"artifact_dir:              {summary['artifact_dir']}")


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

def execute_single_run(
    adapter: HFAdapter,
    config: ExperimentConfig,
    run_id: Optional[str] = None,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, object]:
    seed_everything(config.seed)
    commit = get_git_commit()
    effective_context = resolve_context_tokens(config.context_tokens, adapter, config.max_new_tokens)
    run_id = run_id or make_run_id(config.model_id, config.experiment_group, config.seed)
    artifact_dir = artifact_dir or make_artifact_dir(run_id)
    start = time.time()

    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter.device)

    write_json(artifact_dir / "config.json", asdict(config))

    position_scan = run_position_scan(adapter=adapter, config=config, effective_context_tokens=effective_context)
    eip = run_eip(adapter=adapter, config=config, effective_context_tokens=effective_context) if config.eip_inputs > 0 else {
        "curve": [],
        "lost_middle_ratio": float("nan"),
        "sink_info": float("nan"),
        "inputs": [],
    }

    total_seconds = time.time() - start
    peak_vram_gb = 0.0
    if adapter.device.type == "cuda":
        peak_vram_gb = float(torch.cuda.max_memory_allocated(adapter.device) / 1024 / 1024 / 1024)

    summary = {
        "run_id": run_id,
        "commit": commit,
        "model_id": config.model_id,
        "backend": config.backend,
        "experiment_group": config.experiment_group,
        "seed": config.seed,
        "context_tokens": effective_context,
        "num_positions": len(config.needle_positions),
        "mc_samples": config.mc_samples,
        "eip_inputs": config.eip_inputs,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "mean_niah_accuracy": position_scan["mean_niah_accuracy"],
        "spearman_profile_vs_niah": position_scan["spearman_profile_vs_niah"],
        "mean_needle_info": position_scan["mean_needle_info"],
        "eip_lost_middle_ratio": eip["lost_middle_ratio"],
        "sink_info": eip["sink_info"] if eip.get("curve") else position_scan["mean_sink_info"],
        "peak_vram_gb": peak_vram_gb,
        "total_seconds": total_seconds,
        "status": "ok",
        "description": config.description,
        "artifact_dir": str(artifact_dir.relative_to(ROOT_DIR)),
        "position_scan": position_scan,
        "eip": eip,
    }

    plots = maybe_make_plots(position_scan=position_scan, eip=eip, artifact_dir=artifact_dir)
    summary["plots"] = plots

    write_json(artifact_dir / "position_scan.json", position_scan)
    write_json(artifact_dir / "eip.json", eip)
    write_json(artifact_dir / "summary.json", summary)
    write_run_summary_markdown(summary, artifact_dir)
    append_summary_row(summary)
    print_summary(summary)
    return summary


def crash_summary(config: ExperimentConfig, artifact_dir: Path, error: BaseException, start: float, device: torch.device) -> Dict[str, object]:
    peak_vram_gb = 0.0
    if device.type == "cuda":
        try:
            peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024)
        except Exception:
            peak_vram_gb = 0.0

    summary = {
        "run_id": artifact_dir.name,
        "commit": get_git_commit(),
        "model_id": config.model_id,
        "backend": config.backend,
        "experiment_group": config.experiment_group,
        "seed": config.seed,
        "context_tokens": config.context_tokens,
        "num_positions": len(config.needle_positions),
        "mc_samples": config.mc_samples,
        "eip_inputs": config.eip_inputs,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "mean_niah_accuracy": float("nan"),
        "spearman_profile_vs_niah": float("nan"),
        "mean_needle_info": float("nan"),
        "eip_lost_middle_ratio": float("nan"),
        "sink_info": float("nan"),
        "peak_vram_gb": peak_vram_gb,
        "total_seconds": time.time() - start,
        "status": "crash",
        "description": f"{config.description} | {type(error).__name__}: {error}",
        "artifact_dir": str(artifact_dir.relative_to(ROOT_DIR)),
        "traceback": traceback.format_exc(),
    }
    write_json(artifact_dir / "summary.json", summary)
    write_text(artifact_dir / "traceback.txt", summary["traceback"])
    append_summary_row(summary)
    print_summary(summary)
    return summary


def run_one_model(config: ExperimentConfig, device: torch.device, dtype: torch.dtype) -> List[Dict[str, object]]:
    init_repo_files()
    seed_everything(config.seed)

    adapter = HFAdapter(
        model_id=config.model_id,
        device=device,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )

    summaries: List[Dict[str, object]] = []

    def guarded_execute(run_config: ExperimentConfig) -> Dict[str, object]:
        run_id = make_run_id(run_config.model_id, run_config.experiment_group, run_config.seed)
        artifact_dir = make_artifact_dir(run_id)
        start = time.time()
        try:
            return execute_single_run(adapter, run_config, run_id=run_id, artifact_dir=artifact_dir)
        except BaseException as exc:
            return crash_summary(run_config, artifact_dir, exc, start, device)

    if config.experiment_group == "profile":
        profile_config = replace(
            config,
            needle_positions=(config.needle_positions[0],),
            eip_inputs=0,
            description=config.description or "single-profile debug run",
        )
        summaries.append(guarded_execute(profile_config))

    elif config.experiment_group == "baseline":
        summaries.append(guarded_execute(config))

    elif config.experiment_group == "sensitivity":
        baseline = replace(config, experiment_group="sensitivity-baseline", description="sensitivity baseline")
        summaries.append(guarded_execute(baseline))

        mc_values = [value for value in config.sensitivity_mc_samples if value != config.mc_samples]
        context_values = [value for value in config.sensitivity_contexts if value != config.context_tokens]
        max_new_values = [value for value in config.sensitivity_max_new if value != config.max_new_tokens]
        seed_values = [value for value in config.sensitivity_seeds if value != config.seed]

        for value in mc_values:
            summaries.append(
                guarded_execute(
                    replace(
                        config,
                        experiment_group="sensitivity-mc-samples",
                        mc_samples=value,
                        description=f"vary mc_samples={value}",
                    )
                )
            )
        for value in context_values:
            summaries.append(
                guarded_execute(
                    replace(
                        config,
                        experiment_group="sensitivity-context",
                        context_tokens=value,
                        description=f"vary context_tokens={value}",
                    )
                )
            )
        for value in max_new_values:
            summaries.append(
                guarded_execute(
                    replace(
                        config,
                        experiment_group="sensitivity-max-new",
                        max_new_tokens=value,
                        description=f"vary max_new_tokens={value}",
                    )
                )
            )
        for value in seed_values:
            summaries.append(
                guarded_execute(
                    replace(
                        config,
                        experiment_group="sensitivity-seed",
                        seed=value,
                        description=f"vary seed={value}",
                    )
                )
            )
    else:
        raise ValueError(f"Unsupported experiment_group for single-model execution: {config.experiment_group}")

    del adapter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summaries


def run_compare(base_config: ExperimentConfig, model_ids: Sequence[str], device: torch.device, dtype: torch.dtype) -> List[Dict[str, object]]:
    all_summaries: List[Dict[str, object]] = []
    for idx, model_id in enumerate(model_ids):
        compare_config = replace(
            base_config,
            model_id=model_id,
            experiment_group="compare-baseline",
            seed=base_config.seed + idx,
            description=f"compare baseline for {model_id}",
        )
        all_summaries.extend(run_one_model(compare_config, device=device, dtype=dtype))
    return all_summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run IFIM / NIAH autoresearch experiments on open-source models")
parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_IDS[0], help="Single model id or local HF path")
parser.add_argument("--model-ids", type=str, default=None, help="Comma-separated model ids for compare mode")
parser.add_argument(
    "--experiment-group",
    type=str,
    choices=["baseline", "profile", "sensitivity", "compare"],
    default="baseline",
)
parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
parser.add_argument("--needle-positions", type=str, default="0.1,0.3,0.5,0.7,0.9")
parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
parser.add_argument("--eip-inputs", type=int, default=DEFAULT_EIP_INPUTS)
parser.add_argument("--eip-bins", type=int, default=DEFAULT_EIP_BINS)
parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
parser.add_argument("--temperature", type=float, default=1.0, help="Use 1.0 for unbiased ancestral sampling")
parser.add_argument("--top-p", type=float, default=1.0, help="Use 1.0 for unbiased ancestral sampling")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--trust-remote-code", action="store_true")
parser.add_argument("--save-prompts", action="store_true")
parser.add_argument("--corpus-split", type=str, choices=["train", "val"], default="train")
parser.add_argument("--description", type=str, default="baseline")

# sensitivity controls
parser.add_argument("--sensitivity-mc-samples", type=str, default="4,8,16")
parser.add_argument("--sensitivity-contexts", type=str, default="1024,2048,4096")
parser.add_argument("--sensitivity-max-new", type=str, default="4,8,16")
parser.add_argument("--sensitivity-seeds", type=str, default="0,1,2")


def main(args: argparse.Namespace) -> None:
    init_repo_files()

    if args.temperature != 1.0 or args.top_p != 1.0:
        print("WARNING: temperature != 1.0 or top_p != 1.0 biases the Monte Carlo estimator away from the true model distribution.")

    device = select_device(args.device)
    dtype = default_dtype_for_device(device)
    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    print()

    base_config = ExperimentConfig.from_args(args)

    if args.experiment_group == "compare":
        model_ids = parse_model_id_list(args.model_id, args.model_ids)
        summaries = run_compare(base_config, model_ids=model_ids, device=device, dtype=dtype)
        print(f"Completed {len(summaries)} compare runs.")
    else:
        summaries = run_one_model(base_config, device=device, dtype=dtype)
        print(f"Completed {len(summaries)} run(s).")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

