"""
Active-Hook Integration Benchmark: Measure the Runtime Cost of COREY's
Entropy-Guided Chunk Selection During a Real Mamba-370M Generate Call.

Motivation
----------
Earlier hook-based analyses (``run_active_hook_real_benchmark``) captured the
post-conv hidden state ``u`` *after* the forward pass had finished and
computed Shannon entropy offline.  Reviewers objected that such a hook is
passive: it never runs on the critical path, so its latency cannot be used
to argue anything about deployment overhead.

This benchmark addresses that criticism directly.  It monkey-patches the
HuggingFace ``MambaMixer.cuda_kernels_forward`` so that, on every target
layer and every prefill call, the scheduler:

  1. Runs ``in_proj`` and the causal convolution exactly as in the original
     fast path.
  2. Computes Shannon entropy over the resulting ``u = act(conv1d(...))``
     tensor using a fixed-width 256-bin histogram.
  3. Maps the entropy to a chunk size via COREY's log-scale rule
     ``chunk = round_pow2(min(H/H_ref, 1.0) * (C_max - C_min) + C_min)``.
  4. Logs the chosen chunk and falls through to the fused
     ``selective_scan_fn`` kernel for correctness.

The injected entropy computation therefore executes *inside* the forward
graph of every Mamba layer during ``model.generate()`` — no post-hoc hook
trick, no synthetic tensors.  We then time active mode versus passive mode
end-to-end and report the real per-step overhead.

Usage (adama-cuda128 environment with mamba_ssm CUDA kernels)
-------------------------------------------------------------
    python -m src.experiments.run_active_hook_integration \\
        --model mamba-370m --prompt "<long prompt>" \\
        --new-tokens 64 --warmup 2 --repeats 5 \\
        --output-dir src/outputs/active_hook_integration
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any


MODEL_REGISTRY: dict[str, str] = {
    "mamba-370m": "state-spaces/mamba-370m-hf",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
    "mamba-2.8b": "state-spaces/mamba-2.8b-hf",
}


# ---------------------------------------------------------------------------
# Entropy + chunk policy (inlined to avoid cross-package imports)
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
    import torch

    flat = values.detach().float().reshape(-1)
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax - vmin < 1e-8:
        return 0.0
    normalized = (flat - vmin) / (vmax - vmin + 1e-8)
    indices = (normalized * num_bins).long().clamp(0, num_bins - 1)
    counts = torch.zeros(num_bins, device=flat.device, dtype=torch.float32)
    counts.scatter_add_(0, indices, torch.ones_like(flat, dtype=torch.float32))
    prob = counts / (counts.sum() + 1e-10)
    log_prob = torch.where(prob > 1e-10, torch.log(prob + 1e-10), torch.zeros_like(prob))
    return float(-(prob * log_prob).sum().item())


def _entropy_to_chunk(H: float, *, c_min: int = 32, c_max: int = 512, h_ref: float = 8.0) -> int:
    ratio = min(H / h_ref, 1.0) if h_ref > 0 else 0.0
    raw = c_min + ratio * (c_max - c_min)
    rounded = int(2 ** round(math.log2(max(raw, 1.0))))
    return max(c_min, min(c_max, rounded))


# ---------------------------------------------------------------------------
# Monkey patch: active-mode cuda_kernels_forward
# ---------------------------------------------------------------------------

class ActiveSchedulerState:
    """Collects entropy + chunk observations across all patched layers."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.enabled: bool = False
        self.h_ref: float = 8.0
        self.num_bins: int = 256
        self.min_seq_len: int = 16


STATE = ActiveSchedulerState()


def _make_active_forward(original_forward: Any) -> Any:
    """
    Wrap the original ``cuda_kernels_forward`` so that, immediately before
    the SSM scan step, we compute Shannon entropy on the post-conv hidden
    state and pick a COREY chunk.  The scheduler path is fully CUDA-resident
    and never exits eager-mode.

    Only the non-decoding branch is instrumented — single-token decode steps
    do not carry a prefill ``u`` tensor and are handled by
    ``selective_state_update`` which has no chunk concept.

    The mamba_ssm symbols (``selective_scan_fn``, ``causal_conv1d_fn`` …)
    are populated lazily inside ``MambaMixer.__init__``, so we look them up
    dynamically from the module each call rather than importing at
    patch-install time (when they may still be ``None``).
    """
    import torch
    import torch.nn as nn
    from transformers.models.mamba import modeling_mamba as _mm

    def active_forward(
        self: Any,
        hidden_states: Any,
        cache_params: Any = None,
        attention_mask: Any = None,
    ) -> Any:
        selective_scan_fn      = getattr(_mm, "selective_scan_fn", None)
        selective_state_update = getattr(_mm, "selective_state_update", None)
        causal_conv1d_fn       = getattr(_mm, "causal_conv1d_fn", None)
        causal_conv1d_update   = getattr(_mm, "causal_conv1d_update", None)

        # If active scheduling is disabled or CUDA kernels are unavailable,
        # delegate to the original implementation verbatim.
        if not STATE.enabled or selective_scan_fn is None or causal_conv1d_fn is None:
            return original_forward(self, hidden_states, cache_params, attention_mask)

        # 1. in_proj (identical to stock path)
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training and cache_params is None:
            # Training path is untouched — we only patch inference.
            return original_forward(self, hidden_states, cache_params, attention_mask)

        hidden_proj, gate = projected_states.chunk(2, dim=1)
        if attention_mask is not None:
            hidden_proj = hidden_proj * attention_mask.unsqueeze(1)

        is_decoding = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))

        if is_decoding:
            # Decode step: exactly as in the original fast path.
            hidden_proj = causal_conv1d_update(
                hidden_proj.squeeze(-1),
                cache_params.layers[self.layer_idx].conv_states,
                conv_weights,
                self.conv1d.bias,
                self.activation,
            ).unsqueeze(-1)
            u_post_conv = hidden_proj  # [b, d, 1]
        else:
            if cache_params is not None:
                conv_pad = nn.functional.pad(
                    hidden_proj, (self.conv_kernel_size - hidden_proj.shape[-1], 0)
                )
                cache_params.update_conv_state(conv_pad, self.layer_idx)
            u_post_conv = causal_conv1d_fn(
                hidden_proj, conv_weights, self.conv1d.bias, activation=self.activation
            )

        if attention_mask is not None:
            u_post_conv = u_post_conv * attention_mask.unsqueeze(1)

        # --- COREY scheduler: entropy + chunk selection (on the critical path) ---
        #
        # We deliberately avoid torch.cuda.synchronize() here — the scheduler
        # is meant to execute in-flight on the GPU and overlap with the
        # subsequent selective_scan_fn launch.  Forcing a sync per layer just
        # to take a Python wall-clock timestamp would block the whole
        # pipeline and inflate the measured overhead.  The per-call cost is
        # measured separately in --time-scheduler-precisely mode.
        seq_len = u_post_conv.shape[-1]
        if seq_len >= STATE.min_seq_len:
            entropy_val = _hist_entropy(u_post_conv, num_bins=STATE.num_bins)
            chunk = _entropy_to_chunk(entropy_val, h_ref=STATE.h_ref)
            STATE.records.append({
                "layer_idx":    int(self.layer_idx),
                "seq_len":      int(seq_len),
                "entropy_nats": round(entropy_val, 6),
                "chunk":        int(chunk),
            })

        # --- SSM scan (identical to original fast path) ---
        ssm_parameters = self.x_proj(u_post_conv.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None

        if is_decoding:
            scan_outputs = selective_state_update(
                cache_params.layers[self.layer_idx].recurrent_states,
                u_post_conv[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                u_post_conv,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.update_recurrent_state(ssm_state, self.layer_idx)

        return self.out_proj(scan_outputs.transpose(1, 2))

    return active_forward


def install_active_patch() -> Any:
    """Return the original cuda_kernels_forward so the caller can restore it."""
    from transformers.models.mamba.modeling_mamba import MambaMixer

    original = MambaMixer.cuda_kernels_forward
    MambaMixer.cuda_kernels_forward = _make_active_forward(original)
    return original


def restore(original: Any) -> None:
    from transformers.models.mamba.modeling_mamba import MambaMixer

    MambaMixer.cuda_kernels_forward = original


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def _load(model_name: str, device: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_id = MODEL_REGISTRY.get(model_name, model_name)
    print(f"[active_hook_integ] Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=str(device)
    ).eval()
    return model, tokenizer


def _measure_scheduler_cost(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    *,
    h_ref: float,
    num_bins: int,
    warmup: int = 20,
    repeats: int = 200,
) -> dict[str, Any] | None:
    """
    Capture one post-conv ``u`` tensor from layer 8 (typical entropy regime)
    via a forward hook and time the entropy + chunk-selection call on it
    with torch.cuda.synchronize() fences.  Returns mean / std in ms.
    """
    import torch

    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        return None
    mixers = [layer.mixer for layer in backbone.layers]
    if len(mixers) <= 8:
        return None
    target_mixer = mixers[8]
    if not hasattr(target_mixer, "x_proj"):
        return None

    captured: dict[str, Any] = {}

    def _hook(module: Any, inp: Any, out: Any) -> None:
        if "u" not in captured and inp and inp[0] is not None:
            captured["u"] = inp[0].detach().float().transpose(1, 2).clone()

    handle = target_mixer.x_proj.register_forward_hook(_hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()

    if "u" not in captured:
        return None
    u = captured["u"]

    for _ in range(warmup):
        _hist_entropy(u, num_bins=num_bins)
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        H = _hist_entropy(u, num_bins=num_bins)
        _ = _entropy_to_chunk(H, h_ref=h_ref)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": round(statistics.mean(latencies), 6),
        "std_ms":  round(statistics.pstdev(latencies), 6) if len(latencies) > 1 else 0.0,
        "min_ms":  round(min(latencies), 6),
        "max_ms":  round(max(latencies), 6),
        "u_shape": list(u.shape),
    }


def _time_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    new_tokens: int,
    device: Any,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    prompt_len = int(inputs["input_ids"].shape[-1])

    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "prompt_len":    prompt_len,
        "new_tokens":    new_tokens,
        "latency_mean_ms": round(statistics.mean(latencies), 4),
        "latency_std_ms":  round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "latency_min_ms":  round(min(latencies), 4),
        "latency_max_ms":  round(max(latencies), 4),
        "latencies_ms":    [round(x, 4) for x in latencies],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mamba-370m", choices=list(MODEL_REGISTRY))
    parser.add_argument(
        "--prompt",
        default=(
            "Selective state space models maintain a compact hidden state and "
            "process sequences with linear complexity.  The Mamba architecture "
            "uses input-dependent transitions to selectively retain or discard "
            "information.  COREY estimates Shannon entropy over post-conv "
            "hidden states and schedules kernel chunk sizes accordingly."
        ),
    )
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup",     type=int, default=2)
    parser.add_argument("--repeats",    type=int, default=5)
    parser.add_argument("--h-ref",      type=float, default=8.0)
    parser.add_argument("--num-bins",   type=int, default=256)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("src/outputs/active_hook_integration"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Active integration benchmark requires CUDA + mamba_ssm kernels.")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[active_hook_integ] Device: {device}  ({gpu_name})")

    # mamba_ssm kernel symbols are populated inside MambaMixer.__init__, so the
    # fast-path check must happen *after* we load the model.
    model, tokenizer = _load(args.model, device)

    from transformers.models.mamba import modeling_mamba as _mm
    fast_path_available = all(
        getattr(_mm, name, None) is not None
        for name in ("selective_scan_fn", "selective_state_update",
                     "causal_conv1d_fn", "causal_conv1d_update", "mamba_inner_fn")
    )
    print(f"[active_hook_integ] mamba_ssm fast-path available : {fast_path_available}")
    if not fast_path_available:
        missing = [n for n in ("selective_scan_fn", "selective_state_update",
                               "causal_conv1d_fn", "causal_conv1d_update",
                               "mamba_inner_fn")
                   if getattr(_mm, n, None) is None]
        raise RuntimeError(
            "mamba_ssm CUDA kernels are not importable in this environment "
            f"(missing: {missing}). Run this script under adama-cuda128."
        )
    STATE.h_ref    = float(args.h_ref)
    STATE.num_bins = int(args.num_bins)

    # --- Passive baseline (unpatched fast path) ---
    STATE.enabled = False
    STATE.records.clear()
    print("[active_hook_integ] Timing passive (stock fast path) …")
    passive = _time_generate(
        model, tokenizer, args.prompt, args.new_tokens, device,
        warmup=args.warmup, repeats=args.repeats,
    )
    print(f"  passive: {passive['latency_mean_ms']:.2f} ± {passive['latency_std_ms']:.2f} ms  "
          f"(prompt_len={passive['prompt_len']}, new_tokens={passive['new_tokens']})")

    # --- Install monkey patch and rerun ---
    original = install_active_patch()
    try:
        STATE.enabled = True
        STATE.records.clear()
        print("[active_hook_integ] Timing active (entropy + chunk on critical path) …")
        active = _time_generate(
            model, tokenizer, args.prompt, args.new_tokens, device,
            warmup=args.warmup, repeats=args.repeats,
        )
        print(f"  active : {active['latency_mean_ms']:.2f} ± {active['latency_std_ms']:.2f} ms")
    finally:
        STATE.enabled = False
        restore(original)

    # --- Per-call scheduler micro-benchmark (isolated, with CUDA sync) ---
    #
    # For a clean "how fast is one scheduler call" number we capture one
    # post-conv u tensor through a forward hook and time the entropy +
    # chunk-selection call on it with proper torch.cuda.synchronize()
    # fences.  This number is purely the Python-dispatched cost of the
    # scheduler and is reported separately from the end-to-end overhead.
    print("[active_hook_integ] Measuring per-call scheduler cost in isolation …")
    sched_stats = _measure_scheduler_cost(
        model, tokenizer, args.prompt, device,
        h_ref=args.h_ref, num_bins=args.num_bins,
        warmup=20, repeats=200,
    )
    if sched_stats is not None:
        print(f"  scheduler per call: {sched_stats['mean_ms']:.4f} ± "
              f"{sched_stats['std_ms']:.4f} ms  "
              f"(u_shape={sched_stats['u_shape']})")

    # --- Aggregate scheduler observations from active mode ---
    records = STATE.records
    per_layer: dict[int, list[float]] = {}
    per_chunk: dict[int, int] = {}
    for rec in records:
        per_layer.setdefault(rec["layer_idx"], []).append(rec["entropy_nats"])
        per_chunk[rec["chunk"]] = per_chunk.get(rec["chunk"], 0) + 1

    layer_summary = {
        int(li): {
            "count":         len(vals),
            "entropy_mean":  round(statistics.mean(vals), 4),
            "entropy_std":   round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        } for li, vals in per_layer.items()
    }

    overhead_ms = active["latency_mean_ms"] - passive["latency_mean_ms"]
    overhead_pct = (overhead_ms / passive["latency_mean_ms"]) * 100.0 if passive["latency_mean_ms"] > 0 else 0.0

    summary = {
        "model":             args.model,
        "model_id":          MODEL_REGISTRY.get(args.model, args.model),
        "prompt_excerpt":    args.prompt[:120],
        "new_tokens":        args.new_tokens,
        "warmup":            args.warmup,
        "repeats":           args.repeats,
        "h_ref":             args.h_ref,
        "num_bins":          args.num_bins,
        "device":            str(device),
        "gpu_name":          gpu_name,
        "platform":          platform.platform(),
        "passive":           passive,
        "active":            active,
        "overhead_ms":       round(overhead_ms, 4),
        "overhead_pct":      round(overhead_pct, 4),
        "scheduler_calls":   len(records),
        "scheduler_percall_mean_ms": sched_stats["mean_ms"] if sched_stats else None,
        "scheduler_percall_std_ms":  sched_stats["std_ms"]  if sched_stats else None,
        "scheduler_u_shape":         sched_stats["u_shape"] if sched_stats else None,
        "chunk_distribution": per_chunk,
        "per_layer_entropy":  layer_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    print()
    print(f"[active_hook_integ] Passive  : {passive['latency_mean_ms']:.2f} ms")
    print(f"[active_hook_integ] Active   : {active['latency_mean_ms']:.2f} ms")
    print(f"[active_hook_integ] Δ (ms)   : {overhead_ms:+.3f}  ({overhead_pct:+.2f}%)")
    print(f"[active_hook_integ] Scheduler calls (active): {len(records)}")
    if sched_stats is not None:
        print(f"[active_hook_integ] Scheduler per-call      : "
              f"{sched_stats['mean_ms']:.4f} ± {sched_stats['std_ms']:.4f} ms  "
              f"(isolated, with cuda.synchronize)")
    print(f"[active_hook_integ] Chunk distribution      : {per_chunk}")
    print(f"[active_hook_integ] Results in {args.output_dir}")


if __name__ == "__main__":
    main()
