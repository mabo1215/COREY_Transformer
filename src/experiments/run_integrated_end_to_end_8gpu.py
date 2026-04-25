"""
8-GPU distributed integrated end-to-end benchmark for COREY.

Each rank independently loads Mamba-370M (or specified Mamba-1.x model) and
runs the three-condition benchmark from run_integrated_end_to_end.py:

  1. Passive  -- stock HF fast path, no scheduler.
  2. Active   -- inline entropy computation + chunk selection on every layer,
                 chunk NOT yet routed into selective_scan_fn.
  3. Integrated -- same inline scheduler, chunk routed into selective_scan_fn
                   if the installed mamba_ssm build exposes chunk_size at
                   the Python API level; otherwise falls back to active-only
                   timing (same as condition 2) with a flag set.

Rank 0 gathers per-rank timing results and reports aggregate statistics
(mean, std across all 8 ranks) as well as per-rank detail.

Note: This benchmark targets Mamba-1.x models (transformers MambaMixer).
For Mamba-2 (state-spaces/mamba2-*), the MambaMixer hook is not applicable;
pass --model mamba-2.8b to skip the integration patch and run passive-only.

Launch:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \\
    src/experiments/run_integrated_end_to_end_8gpu.py \\
    --model mamba-370m \\
    --output-dir src/outputs/integrated_end_to_end_8gpu
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

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="COREY 8-GPU integrated end-to-end benchmark.")
parser.add_argument("--model", type=str, default="mamba-370m")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--new-tokens", type=int, default=32)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--num-bins", type=int, default=256)
parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/integrated_end_to_end_8gpu"))
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Model registry (Mamba-1.x only; Mamba-2 requires different integration)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "mamba-370m":  "state-spaces/mamba-370m-hf",
    "mamba-1.4b":  "state-spaces/mamba-1.4b-hf",
    "mamba-2.8b":  "state-spaces/mamba-2.8b-hf",
}

MAMBA1_MODELS = {"mamba-370m", "mamba-1.4b", "mamba-2.8b"}

DEFAULT_PROMPT = (
    "Selective state space models maintain a compact hidden state and "
    "process sequences with linear complexity.  The Mamba architecture "
    "uses input-dependent transitions to selectively retain or discard "
    "information.  COREY estimates Shannon entropy over post-conv "
    "hidden states and schedules kernel chunk sizes accordingly.  "
    "The scheduler operates on the critical path of the forward pass, "
    "computing a 256-bin histogram and mapping entropy to a power-of-two "
    "chunk size via a log-scale rule calibrated to the theoretical maximum "
    "entropy of the histogram.  This eliminates the need for offline profiling "
    "while achieving the same chunk selection as the one-time oracle."
)


# ---------------------------------------------------------------------------
# HF token helper
# ---------------------------------------------------------------------------

def _set_hf_token_from_envfile(env_path: str = "/home/amabo1215/source/.env") -> None:
    import os
    try:
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("Huggingface_model_token:="):
                        token = line.strip().split("Huggingface_model_token:=", 1)[-1].strip()
                        if token:
                            os.environ["HF_TOKEN"] = token
                        break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entropy + COREY helpers
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
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


def _entropy_to_chunk(H: float, h_ref: float | None = None,
                      num_bins: int = 256, c_min: int = 32, c_max: int = 512) -> int:
    if h_ref is None:
        h_ref = math.log(num_bins)
    ratio = min(H / h_ref, 1.0) if h_ref > 0 else 0.0
    raw = c_min + ratio * (c_max - c_min)
    rounded = int(2 ** round(math.log2(max(raw, 1.0))))
    return max(c_min, min(c_max, rounded))


# ---------------------------------------------------------------------------
# Scheduler state and monkey patch (Mamba-1.x only)
# ---------------------------------------------------------------------------

class SchedulerState:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.enabled: bool = False
        self.route_chunk: bool = False
        self.h_ref: float | None = None
        self.num_bins: int = 256
        self.min_seq_len: int = 16
        self.chunk_size_kwarg_supported: bool | None = None


STATE = SchedulerState()


def _make_active_forward(original_forward: Any) -> Any:
    from transformers.models.mamba import modeling_mamba as _mm

    def active_forward(self_: Any, hidden_states: Any,
                       cache_params: Any = None, attention_mask: Any = None) -> Any:
        import torch.nn as nn
        selective_scan_fn      = getattr(_mm, "selective_scan_fn", None)
        selective_state_update = getattr(_mm, "selective_state_update", None)
        causal_conv1d_fn       = getattr(_mm, "causal_conv1d_fn", None)
        causal_conv1d_update   = getattr(_mm, "causal_conv1d_update", None)

        if not STATE.enabled or selective_scan_fn is None or causal_conv1d_fn is None:
            return original_forward(self_, hidden_states, cache_params, attention_mask)

        projected_states = self_.in_proj(hidden_states).transpose(1, 2)
        if self_.training and cache_params is None:
            return original_forward(self_, hidden_states, cache_params, attention_mask)

        hidden_proj, gate = projected_states.chunk(2, dim=1)
        if attention_mask is not None:
            hidden_proj = hidden_proj * attention_mask.unsqueeze(1)

        is_decoding = cache_params is not None and cache_params.has_previous_state(self_.layer_idx)
        conv_weights = self_.conv1d.weight.view(
            self_.conv1d.weight.size(0), self_.conv1d.weight.size(2)
        )

        if is_decoding:
            hidden_proj = causal_conv1d_update(
                hidden_proj.squeeze(-1),
                cache_params.layers[self_.layer_idx].conv_states,
                conv_weights, self_.conv1d.bias, self_.activation,
            ).unsqueeze(-1)
            u_post_conv = hidden_proj
        else:
            if cache_params is not None:
                conv_pad = nn.functional.pad(
                    hidden_proj, (self_.conv_kernel_size - hidden_proj.shape[-1], 0)
                )
                cache_params.update_conv_state(conv_pad, self_.layer_idx)
            u_post_conv = causal_conv1d_fn(
                hidden_proj, conv_weights, self_.conv1d.bias, activation=self_.activation
            )

        if attention_mask is not None:
            u_post_conv = u_post_conv * attention_mask.unsqueeze(1)

        seq_len = u_post_conv.shape[-1]
        selected_chunk: int | None = None
        if seq_len >= STATE.min_seq_len:
            H = _hist_entropy(u_post_conv, num_bins=STATE.num_bins)
            selected_chunk = _entropy_to_chunk(H, h_ref=STATE.h_ref, num_bins=STATE.num_bins)
            STATE.records.append({
                "layer_idx": int(self_.layer_idx),
                "entropy_nats": round(H, 6),
                "chunk": int(selected_chunk),
                "routed": STATE.route_chunk,
            })

        ssm_parameters = self_.x_proj(u_post_conv.transpose(1, 2))
        time_step, B_ssm, C_ssm = torch.split(
            ssm_parameters,
            [self_.time_step_rank, self_.ssm_state_size, self_.ssm_state_size],
            dim=-1,
        )
        discrete_time_step = self_.dt_proj.weight @ time_step.transpose(1, 2)
        A = -torch.exp(self_.A_log.float())
        time_proj_bias = self_.dt_proj.bias.float() if hasattr(self_.dt_proj, "bias") else None

        if is_decoding:
            scan_outputs = selective_state_update(
                cache_params.layers[self_.layer_idx].recurrent_states,
                u_post_conv[..., 0], discrete_time_step[..., 0],
                A, B_ssm[:, 0], C_ssm[:, 0], self_.D, gate[..., 0],
                time_proj_bias, dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_kwargs: dict[str, Any] = dict(delta_softplus=True, return_last_state=True)
            if STATE.route_chunk and selected_chunk is not None:
                try:
                    scan_out_pair = selective_scan_fn(
                        u_post_conv, discrete_time_step, A,
                        B_ssm.transpose(1, 2), C_ssm.transpose(1, 2),
                        self_.D.float(), gate, time_proj_bias,
                        chunk_size=selected_chunk, **scan_kwargs,
                    )
                    STATE.chunk_size_kwarg_supported = True
                except TypeError:
                    scan_out_pair = selective_scan_fn(
                        u_post_conv, discrete_time_step, A,
                        B_ssm.transpose(1, 2), C_ssm.transpose(1, 2),
                        self_.D.float(), gate, time_proj_bias, **scan_kwargs,
                    )
                    STATE.chunk_size_kwarg_supported = False
            else:
                scan_out_pair = selective_scan_fn(
                    u_post_conv, discrete_time_step, A,
                    B_ssm.transpose(1, 2), C_ssm.transpose(1, 2),
                    self_.D.float(), gate, time_proj_bias, **scan_kwargs,
                )
            scan_outputs, ssm_state = scan_out_pair
            if ssm_state is not None and cache_params is not None:
                cache_params.update_recurrent_state(ssm_state, self_.layer_idx)

        return self_.out_proj(scan_outputs.transpose(1, 2))

    return active_forward


def _install_patch() -> Any:
    from transformers.models.mamba.modeling_mamba import MambaMixer
    original = MambaMixer.cuda_kernels_forward
    MambaMixer.cuda_kernels_forward = _make_active_forward(original)
    return original


def _restore(original: Any) -> None:
    from transformers.models.mamba.modeling_mamba import MambaMixer
    MambaMixer.cuda_kernels_forward = original


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _time_generate(model: Any, tokenizer: Any, prompt: str, new_tokens: int,
                   device: Any, warmup: int, repeats: int) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "lat_mean_ms": round(statistics.mean(latencies), 4),
        "lat_std_ms": round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "latencies_ms": [round(x, 4) for x in latencies],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    _set_hf_token_from_envfile()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_key = args.model
    model_id = MODEL_REGISTRY.get(model_key, model_key)
    is_mamba1 = model_key in MAMBA1_MODELS

    if rank == 0:
        print(f"[integrated_8gpu] Loading {model_id} on {world_size} GPUs ...")
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=str(device)
    ).eval()
    try:
        model = model.to(torch.float16)
    except Exception:
        pass

    from transformers.models.mamba import modeling_mamba as _mm
    fast_path = is_mamba1 and all(
        getattr(_mm, n, None) is not None
        for n in ("selective_scan_fn", "selective_state_update",
                  "causal_conv1d_fn", "causal_conv1d_update", "mamba_inner_fn")
    )

    prompt = args.prompt if args.prompt else DEFAULT_PROMPT

    STATE.num_bins = args.num_bins
    STATE.h_ref = None  # use log(num_bins)

    # --- Condition 1: Passive ---
    STATE.enabled = False
    STATE.records.clear()
    passive = _time_generate(model, tokenizer, prompt, args.new_tokens,
                             device, args.warmup, args.repeats)

    # --- Condition 2: Active hook only ---
    active_only_result: dict[str, Any] = {}
    integrated_result: dict[str, Any] = {}
    kwarg_supported: bool | None = None

    if is_mamba1 and fast_path:
        original = _install_patch()
        try:
            STATE.enabled = True
            STATE.route_chunk = False
            STATE.records.clear()
            active_only_result = _time_generate(model, tokenizer, prompt, args.new_tokens,
                                                device, args.warmup, args.repeats)
            chunk_dist_active: dict[str, int] = {}
            for rec in STATE.records:
                k = str(rec["chunk"])
                chunk_dist_active[k] = chunk_dist_active.get(k, 0) + 1
            active_only_result["chunk_dist"] = chunk_dist_active

            # --- Condition 3: Active + routed ---
            STATE.enabled = True
            STATE.route_chunk = True
            STATE.records.clear()
            integrated_result = _time_generate(model, tokenizer, prompt, args.new_tokens,
                                               device, args.warmup, args.repeats)
            chunk_dist_routed: dict[str, int] = {}
            for rec in STATE.records:
                k = str(rec["chunk"])
                chunk_dist_routed[k] = chunk_dist_routed.get(k, 0) + 1
            integrated_result["chunk_dist"] = chunk_dist_routed
            kwarg_supported = STATE.chunk_size_kwarg_supported
        finally:
            STATE.enabled = False
            _restore(original)
    else:
        note = "Mamba-2 / non-fast-path: integration hook not applicable; passive-only"
        active_only_result = {**passive, "note": note}
        integrated_result  = {**passive, "note": note}
        if rank == 0:
            print(f"[integrated_8gpu] {note}")

    per_rank_record = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(rank),
        "fast_path": fast_path,
        "chunk_size_kwarg_supported": kwarg_supported,
        "passive": passive,
        "active_only": active_only_result,
        "integrated": integrated_result,
    }

    # Gather at rank 0
    gathered: list[Any] = [None] * world_size
    dist.gather_object(per_rank_record, gathered if rank == 0 else None, dst=0)
    dist.barrier()

    if rank == 0:
        # Aggregate across ranks
        def _agg(key: str, sub: str) -> dict[str, float]:
            lats = [rec[key][sub] for rec in gathered if sub in rec.get(key, {})]
            if not lats:
                return {}
            return {
                "mean_across_ranks": round(statistics.mean(lats), 4),
                "std_across_ranks": round(statistics.pstdev(lats), 4) if len(lats) > 1 else 0.0,
                "n_ranks": len(lats),
            }

        passive_lat_global = statistics.mean(
            rec["passive"]["lat_mean_ms"] for rec in gathered
        )
        active_lat_global = statistics.mean(
            rec["active_only"]["lat_mean_ms"] for rec in gathered
            if "lat_mean_ms" in rec["active_only"]
        )
        int_lat_global = statistics.mean(
            rec["integrated"]["lat_mean_ms"] for rec in gathered
            if "lat_mean_ms" in rec["integrated"]
        )

        print("\n[integrated_8gpu] === Aggregate results (mean across all ranks) ===")
        print(f"  Passive   : {passive_lat_global:.2f} ms  (1.00x)")
        print(f"  Active    : {active_lat_global:.2f} ms  "
              f"({active_lat_global/passive_lat_global:.3f}x)")
        print(f"  Integrated: {int_lat_global:.2f} ms  "
              f"({int_lat_global/passive_lat_global:.3f}x)")
        if int_lat_global <= passive_lat_global:
            print("  => Net end-to-end improvement achieved.")
        else:
            overhead = int_lat_global - passive_lat_global
            print(f"  => Overhead: +{overhead:.2f} ms "
                  f"(+{overhead/passive_lat_global*100:.1f}%)")

        output = {
            "world_size": world_size,
            "model": model_key,
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "h_ref": f"log({args.num_bins})={math.log(args.num_bins):.4f}",
            "platform": platform.platform(),
            "aggregate": {
                "passive_lat_mean_ms": round(passive_lat_global, 4),
                "active_only_lat_mean_ms": round(active_lat_global, 4),
                "integrated_lat_mean_ms": round(int_lat_global, 4),
                "active_overhead_pct": round((active_lat_global / passive_lat_global - 1) * 100, 2),
                "integrated_overhead_pct": round((int_lat_global / passive_lat_global - 1) * 100, 2),
            },
            "per_rank": gathered,
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "summary.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[integrated_8gpu] Results saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
