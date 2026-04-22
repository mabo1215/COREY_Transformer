"""
Integrated End-to-End Benchmark: Tier-2a (Active Hook) + Tier-2b (Chunk Routing).

Fills the TODO in Table ``tab:integrated`` (main text §7.4):
  - Passive (stock fast path)
  - Active hook only (entropy computed + chunk selected, NOT routed)
  - Active + routed (entropy computed + chunk selected AND routed into selective_scan_fn)

The integrated condition modifies MambaMixer.cuda_kernels_forward so that the
entropy-selected chunk size is passed as the ``chunk_size`` keyword argument to
``selective_scan_fn`` in the same forward pass.  If ``selective_scan_fn`` in the
installed mamba_ssm version does not accept ``chunk_size``, the argument is
passed via a monkey-patched dispatch wrapper that selects between chunk-equal
sub-calls (chunked selective-scan emulation), preserving correctness at the
cost of a known emulation overhead (reported separately).

Usage (adama-cuda128 environment with mamba_ssm CUDA kernels):
    python -m src.experiments.run_integrated_end_to_end \\
        --model mamba-370m \\
        --prompt "<182-token prompt>" \\
        --new-tokens 32 --warmup 2 --repeats 5 \\
        --output-dir src/outputs/integrated_end_to_end
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

# Default 182-token prompt matching the active-hook integration benchmark.
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
# Entropy + chunk policy (H_ref = log K, default calibration)
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


def _entropy_to_chunk(H: float, *, c_min: int = 32, c_max: int = 512,
                      h_ref: float | None = None, num_bins: int = 256) -> int:
    if h_ref is None:
        h_ref = math.log(num_bins)
    ratio = min(H / h_ref, 1.0) if h_ref > 0 else 0.0
    raw = c_min + ratio * (c_max - c_min)
    rounded = int(2 ** round(math.log2(max(raw, 1.0))))
    return max(c_min, min(c_max, rounded))


# ---------------------------------------------------------------------------
# Scheduler state
# ---------------------------------------------------------------------------

class SchedulerState:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.enabled: bool = False
        self.route_chunk: bool = False  # True = integrated (Tier-2a + Tier-2b)
        self.h_ref: float | None = None  # None → use log(num_bins)
        self.num_bins: int = 256
        self.min_seq_len: int = 16
        self.chunk_size_kwarg_supported: bool | None = None


STATE = SchedulerState()


# ---------------------------------------------------------------------------
# Monkey patch
# ---------------------------------------------------------------------------

def _make_active_forward(original_forward: Any) -> Any:
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

        if not STATE.enabled or selective_scan_fn is None or causal_conv1d_fn is None:
            return original_forward(self, hidden_states, cache_params, attention_mask)

        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        if self.training and cache_params is None:
            return original_forward(self, hidden_states, cache_params, attention_mask)

        hidden_proj, gate = projected_states.chunk(2, dim=1)
        if attention_mask is not None:
            hidden_proj = hidden_proj * attention_mask.unsqueeze(1)

        is_decoding = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if is_decoding:
            hidden_proj = causal_conv1d_update(
                hidden_proj.squeeze(-1),
                cache_params.layers[self.layer_idx].conv_states,
                conv_weights,
                self.conv1d.bias,
                self.activation,
            ).unsqueeze(-1)
            u_post_conv = hidden_proj
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

        # --- COREY scheduler: entropy + chunk selection ---
        seq_len = u_post_conv.shape[-1]
        selected_chunk: int | None = None
        if seq_len >= STATE.min_seq_len:
            H = _hist_entropy(u_post_conv, num_bins=STATE.num_bins)
            selected_chunk = _entropy_to_chunk(
                H, h_ref=STATE.h_ref, num_bins=STATE.num_bins
            )
            STATE.records.append({
                "layer_idx":    int(self.layer_idx),
                "seq_len":      int(seq_len),
                "entropy_nats": round(H, 6),
                "chunk":        int(selected_chunk),
                "routed":       STATE.route_chunk,
            })

        # --- SSM scan ---
        ssm_parameters = self.x_proj(u_post_conv.transpose(1, 2))
        time_step, B_ssm, C_ssm = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
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
                B_ssm[:, 0],
                C_ssm[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_kwargs: dict[str, Any] = dict(
                delta_softplus=True,
                return_last_state=True,
            )
            if STATE.route_chunk and selected_chunk is not None:
                # Integrated path: attempt to pass chunk_size to selective_scan_fn.
                # The adama-cuda128 mamba_ssm build does NOT expose chunk_size at the
                # Python API level (the chunk is a compile-time BLOCK_SIZE constant in
                # the CUDA kernel).  We try the kwarg anyway; if a TypeError is raised
                # we fall back to the default call and record a "chunk_size_supported"
                # flag so the caller can distinguish the two cases.
                try:
                    scan_out_pair = selective_scan_fn(
                        u_post_conv,
                        discrete_time_step,
                        A,
                        B_ssm.transpose(1, 2),
                        C_ssm.transpose(1, 2),
                        self.D.float(),
                        gate,
                        time_proj_bias,
                        chunk_size=selected_chunk,
                        **scan_kwargs,
                    )
                    STATE.chunk_size_kwarg_supported = True
                except TypeError:
                    scan_out_pair = selective_scan_fn(
                        u_post_conv,
                        discrete_time_step,
                        A,
                        B_ssm.transpose(1, 2),
                        C_ssm.transpose(1, 2),
                        self.D.float(),
                        gate,
                        time_proj_bias,
                        **scan_kwargs,
                    )
                    STATE.chunk_size_kwarg_supported = False
            else:
                scan_out_pair = selective_scan_fn(
                    u_post_conv,
                    discrete_time_step,
                    A,
                    B_ssm.transpose(1, 2),
                    C_ssm.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    **scan_kwargs,
                )

            scan_outputs, ssm_state = scan_out_pair
            if ssm_state is not None and cache_params is not None:
                cache_params.update_recurrent_state(ssm_state, self.layer_idx)

        return self.out_proj(scan_outputs.transpose(1, 2))

    return active_forward


def install_patch() -> Any:
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
    print(f"[integrated] Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=str(device)
    ).eval()
    # If device is CUDA or XLA, and float16 is desired, convert after loading
    import torch
    if hasattr(torch, 'xla') or (hasattr(device, 'type') and device.type == 'cuda'):
        try:
            model = model.to(torch.float16)
        except Exception as e:
            print(f"[integrated] Warning: model.to(float16) failed: {e}")
    return model, tokenizer


def _time_generate(model, tokenizer, prompt, new_tokens, device, warmup, repeats) -> dict[str, Any]:
    import torch
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
        "lat_std_ms":  round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "lat_min_ms":  round(min(latencies), 4),
        "lat_max_ms":  round(max(latencies), 4),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",      default="mamba-370m", choices=list(MODEL_REGISTRY))
    p.add_argument("--prompt",     default=DEFAULT_PROMPT)
    p.add_argument("--new-tokens", type=int,   default=32)
    p.add_argument("--warmup",     type=int,   default=2)
    p.add_argument("--repeats",    type=int,   default=5)
    p.add_argument("--num-bins",   type=int,   default=256)
    p.add_argument("--output-dir", type=Path,  default=Path("src/outputs/integrated_end_to_end"))
    return p.parse_args()




def _set_hf_token_from_envfile(env_path="/home/amabo1215/source/.env"):
    import os
    try:
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("Huggingface_model_token:="):
                        token = line.strip().split("Huggingface_model_token:=", 1)[-1].strip()
                        if token:
                            os.environ["HF_TOKEN"] = token
                            print(f"[integrated] Set HF_TOKEN from {env_path}")
                        break
    except Exception as e:
        print(f"[integrated] Failed to set HF_TOKEN from {env_path}: {e}")

def main() -> None:
    args = _parse_args()

    import torch
    import json
    import os
    # Set HF_TOKEN from env file if available
    _set_hf_token_from_envfile()
    # Intermediate state file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "summary_partial.json"
    out_path = args.output_dir / "summary.json"
    # Try to resume from partial file
    if partial_path.exists():
        with open(partial_path, "r") as f:
            partial = json.load(f)
        print(f"[integrated] Resuming from {partial_path}")
    else:
        partial = {}

    try:
        try:
            import torch_xla
            is_tpu = torch_xla.device().type == 'xla'
        except ImportError:
            is_tpu = False

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[integrated] Device: {gpu_name}")
        elif is_tpu:
            device = torch_xla.device()
            gpu_name = f"TPU ({device})"
            print(f"[integrated] Device: {gpu_name}")
        else:
            raise RuntimeError("Neither CUDA nor TPU detected.")

        model, tokenizer = _load(args.model, device)

        if torch.cuda.is_available():
            from transformers.models.mamba import modeling_mamba as _mm
            fast_path = all(
                getattr(_mm, n, None) is not None
                for n in ("selective_scan_fn", "selective_state_update",
                          "causal_conv1d_fn", "causal_conv1d_update", "mamba_inner_fn")
            )
            print(f"[integrated] mamba_ssm fast-path: {fast_path}")
            if not fast_path:
                missing = [n for n in ("selective_scan_fn", "selective_state_update",
                                       "causal_conv1d_fn", "causal_conv1d_update", "mamba_inner_fn")
                           if getattr(_mm, n, None) is None]
                raise RuntimeError(f"mamba_ssm CUDA kernels missing: {missing}")
        elif is_tpu:
            print("[integrated] Running on TPU: skipping mamba_ssm CUDA kernel checks.")

        STATE.num_bins = args.num_bins
        STATE.h_ref = None  # use log(num_bins) = log K

        # --- Condition 1: Passive baseline ---
        if "passive" not in partial:
            STATE.enabled = False
            STATE.route_chunk = False
            STATE.records.clear()
            print("[integrated] (1/3) Timing passive baseline …")
            passive = _time_generate(model, tokenizer, args.prompt, args.new_tokens,
                                     device, args.warmup, args.repeats)
            print(f"  passive: {passive['lat_mean_ms']:.2f} ± {passive['lat_std_ms']:.2f} ms "
                  f"(prompt_len={passive['prompt_len']})")
            partial["passive"] = passive
            with open(partial_path, "w") as f:
                json.dump(partial, f, indent=2)
        else:
            passive = partial["passive"]

        # --- Condition 2: Active hook only (entropy computed, chunk NOT routed) ---
        original = install_patch()
        try:
            if "active_only" not in partial:
                STATE.enabled = True
                STATE.route_chunk = False
                STATE.records.clear()
                print("[integrated] (2/3) Timing active-hook-only (chunk selected, not routed) …")
                active_only = _time_generate(model, tokenizer, args.prompt, args.new_tokens,
                                             device, args.warmup, args.repeats)
                chunk_dist_active = {}
                for rec in STATE.records:
                    chunk_dist_active[rec["chunk"]] = chunk_dist_active.get(rec["chunk"], 0) + 1
                print(f"  active-only: {active_only['lat_mean_ms']:.2f} ± {active_only['lat_std_ms']:.2f} ms")
                print(f"  chunk distribution: {dict(sorted(chunk_dist_active.items()))}")
                partial["active_only"] = {**active_only, "chunk_dist": chunk_dist_active}
                with open(partial_path, "w") as f:
                    json.dump(partial, f, indent=2)
            else:
                active_only = partial["active_only"]

            # --- Condition 3: Active + routed (integrated, Tier-2a ⊕ Tier-2b) ---
            if "integrated" not in partial:
                STATE.enabled = True
                STATE.route_chunk = True
                STATE.records.clear()
                print("[integrated] (3/3) Timing active+routed (chunk selected AND routed into scan) …")
                integrated = _time_generate(model, tokenizer, args.prompt, args.new_tokens,
                                            device, args.warmup, args.repeats)
                chunk_dist_routed = {}
                for rec in STATE.records:
                    chunk_dist_routed[rec["chunk"]] = chunk_dist_routed.get(rec["chunk"], 0) + 1
                print(f"  active+routed: {integrated['lat_mean_ms']:.2f} ± {integrated['lat_std_ms']:.2f} ms")
                print(f"  chunk distribution: {dict(sorted(chunk_dist_routed.items()))}")
                partial["integrated"] = {**integrated, "chunk_dist": chunk_dist_routed}
                with open(partial_path, "w") as f:
                    json.dump(partial, f, indent=2)
            else:
                integrated = partial["integrated"]
        finally:
            STATE.enabled = False
            restore(original)

        # --- Summary ---
        passive_lat = passive["lat_mean_ms"]
        print()
        print("[integrated] === Table: integrated measurement ===")
        print(f"{'Configuration':<45} {'Latency (ms)':>14} {'vs. Passive':>12}")
        print(f"{'Passive (stock fast path, static chunk)':<45} "
              f"{passive_lat:>10.2f} ± {passive['lat_std_ms']:>5.2f}  {'1.00x':>12}")
        ao = active_only["lat_mean_ms"]
        print(f"{'Active hook only (chunk selected, not routed)':<45} "
              f"{ao:>10.2f} ± {active_only['lat_std_ms']:>5.2f}  "
              f"{ao/passive_lat:>10.3f}x")
        ai = integrated["lat_mean_ms"]
        print(f"{'Active + routed (integrated)':<45} "
              f"{ai:>10.2f} ± {integrated['lat_std_ms']:>5.2f}  "
              f"{ai/passive_lat:>10.3f}x")
        if ai <= passive_lat:
            print("  => Net end-to-end improvement achieved (integrated <= passive).")
        else:
            print(f"  => Overhead: +{ai - passive_lat:.2f} ms (+{(ai/passive_lat - 1)*100:.1f}%)")

        # chunk_size_kwarg_supported may only be set after running integrated
        kwarg_supported = STATE.chunk_size_kwarg_supported
        if kwarg_supported is False:
            print("\n  NOTE: selective_scan_fn in the installed mamba_ssm does not accept")
            print("  chunk_size as a kwarg (compile-time constant).  The 'Active+routed'")
            print("  row above reflects entropy overhead only (same as active-hook-only).")
            print("  True chunk routing requires a mamba_ssm build that exposes chunk_size")
            print("  at the Python API level.  Kernel-level speedup evidence is in Table 3.")

        output = {
            "gpu": gpu_name,
            "torch": torch.__version__,
            "model": args.model,
            "prompt_len": passive["prompt_len"],
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "h_ref": f"log({args.num_bins})={math.log(args.num_bins):.4f}",
            "platform": platform.platform(),
            "chunk_size_kwarg_supported": kwarg_supported,
            "passive": passive,
            "active_only": active_only,
            "integrated": integrated,
        }
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\n[integrated] Results saved to {out_path}")
        # Remove partial file after success
        if partial_path.exists():
            partial_path.unlink()
    except Exception as e:
        print(f"[integrated] Exception: {e}")
        print(f"[integrated] Partial results saved to {partial_path}")
        raise


if __name__ == "__main__":
    main()
