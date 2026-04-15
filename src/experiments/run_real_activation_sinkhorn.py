"""
W2 Real-Activation Sinkhorn Proxy Analysis.

Loads a real Mamba checkpoint, extracts intermediate activations via forward
hooks, applies a simulated Hadamard rotation, and computes the Sinkhorn proxy
residual that validates Theorem 1 on real (not synthetic) data.

The analysis mirrors the synthetic Sinkhorn proxy in run_entropy_guided_experiments.py
but operates on live checkpoint activations drawn from mixer.in_proj outputs
of the Hugging Face Mamba family.

Design choices:
  - "before rotation": activations at the in_proj output (x branch)
  - "after rotation" : fast-Hadamard-transform applied to the same activations
                       (simulates what COREY's fused Hadamard layer would see)
  - Sinkhorn proxy    : project a positive kernel onto the Birkhoff polytope
                        via row/column normalization (Sinkhorn–Knopp iterations)
                        and measure the l1 residual ||q - Bp||_1

Usage (WSL2 with GPU and HF checkpoints available):
  python -m src.experiments.run_real_activation_sinkhorn \
      --model mamba-370m \
      --layers 0 1 2 3 \
      --num-samples 20 \
      --seq-len 64 \
      --output-dir src/outputs/real_activation_sinkhorn
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


# ---------------------------------------------------------------------------
# Sinkhorn proxy (inline, mirrors _fit_sinkhorn_proxy in run_entropy_guided)
# ---------------------------------------------------------------------------


def _hist_entropy_np(arr: Any, num_bins: int = 64) -> float:
    """Compute histogram entropy in nats."""
    import numpy as np

    flat = arr.flatten().astype(float)
    vmin, vmax = flat.min(), flat.max()
    if vmax - vmin < 1e-8:
        return 0.0
    counts, _ = np.histogram(flat, bins=num_bins)
    prob = counts / (counts.sum() + 1e-12)
    log_prob = np.where(prob > 1e-12, np.log(prob + 1e-12), 0.0)
    return float(-(prob * log_prob).sum())


def _sinkhorn_normalize(K: Any, max_iters: int = 100, tol: float = 1e-6) -> Any:
    """Sinkhorn–Knopp normalization; returns doubly-stochastic B from kernel K."""
    import numpy as np

    B = K.copy()
    n = B.shape[0]
    for _ in range(max_iters):
        # Row normalize
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        B = B / row_sums
        # Column normalize
        col_sums = B.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
        B = B / col_sums
        # Convergence check
        row_err = float(np.abs(B.sum(axis=1) - 1.0).max())
        col_err = float(np.abs(B.sum(axis=0) - 1.0).max())
        if max(row_err, col_err) < tol:
            break
    return B


def _fit_sinkhorn_proxy(before: Any, after: Any, num_bins: int = 64) -> dict[str, float]:
    """
    Fit a doubly-stochastic proxy B such that q ≈ Bp (histogram mass vectors).

    Returns:
      entropy_before, entropy_after, entropy_gain,
      sinkhorn_residual_l1, sinkhorn_residual_l2,
      sinkhorn_row_error, sinkhorn_col_error
    """
    import numpy as np

    flat_before = before.flatten().astype(float)
    flat_after = after.flatten().astype(float)

    entropy_before = _hist_entropy_np(flat_before, num_bins)
    entropy_after = _hist_entropy_np(flat_after, num_bins)

    # Build normalized histogram vectors p, q
    vmin = min(flat_before.min(), flat_after.min())
    vmax = max(flat_before.max(), flat_after.max())
    if vmax - vmin < 1e-8:
        return {
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "entropy_gain": 0.0,
            "sinkhorn_residual_l1": float("nan"),
            "sinkhorn_residual_l2": float("nan"),
            "sinkhorn_row_error": float("nan"),
            "sinkhorn_col_error": float("nan"),
        }

    bins_b, _ = np.histogram(flat_before, bins=num_bins, range=(vmin, vmax))
    bins_a, _ = np.histogram(flat_after, bins=num_bins, range=(vmin, vmax))
    p = bins_b / (bins_b.sum() + 1e-12)
    q = bins_a / (bins_a.sum() + 1e-12)

    # Build positive kernel (outer product of empirical densities)
    K = np.outer(q, p) + 1e-6
    B = _sinkhorn_normalize(K)

    # Residual: ||q - Bp||_1
    q_approx = B @ p
    residual_l1 = float(np.abs(q - q_approx).sum())
    residual_l2 = float(np.sqrt(((q - q_approx) ** 2).sum()))
    row_error = float(np.abs(B.sum(axis=1) - 1.0).max())
    col_error = float(np.abs(B.sum(axis=0) - 1.0).max())

    return {
        "entropy_before": round(entropy_before, 6),
        "entropy_after": round(entropy_after, 6),
        "entropy_gain": round(entropy_after - entropy_before, 6),
        "sinkhorn_residual_l1": round(residual_l1, 6),
        "sinkhorn_residual_l2": round(residual_l2, 6),
        "sinkhorn_row_error": round(row_error, 8),
        "sinkhorn_col_error": round(col_error, 8),
    }


# ---------------------------------------------------------------------------
# Hadamard rotation (inline, mirrors fast_hadamard_transform)
# ---------------------------------------------------------------------------


def _fast_hadamard_transform_np(x: Any) -> Any:
    """Walsh–Hadamard transform on the last dimension (in-place, numpy)."""
    import numpy as np

    n = x.shape[-1]
    # Pad to next power of two
    log2_n = int(math.ceil(math.log2(max(n, 1))))
    n_padded = 2 ** log2_n
    if n_padded != n:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, n_padded - n)]
        x = np.pad(x, pad_width)
    h = x.copy()
    step = 1
    while step < n_padded:
        half = step
        for i in range(0, n_padded, step * 2):
            a = h[..., i : i + half]
            b = h[..., i + half : i + 2 * half]
            h[..., i : i + half] = (a + b) / math.sqrt(2)
            h[..., i + half : i + 2 * half] = (a - b) / math.sqrt(2)
        step *= 2
    return h[..., :n]


# ---------------------------------------------------------------------------
# Model loading and activation extraction
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mamba-370m": "state-spaces/mamba-370m-hf",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
    "mamba-2.8b": "state-spaces/mamba-2.8b-hf",
}


def _load_model_and_tokenizer(model_name: str, device: Any) -> tuple[Any, Any]:
    """Load a Mamba HuggingFace checkpoint and tokenizer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required.") from exc

    model_id = MODEL_REGISTRY.get(model_name, model_name)
    print(f"[W2] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=str(device),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _find_mixer_layers(model: Any) -> list[Any]:
    """Return the list of mixer modules from a HuggingFace Mamba model."""
    # MambaForCausalLM structure: model.backbone.layers[i].mixer (transformers)
    # or model.model.layers[i].mixer depending on version.
    for attr in ["backbone", "model"]:
        backbone = getattr(model, attr, None)
        if backbone is not None and hasattr(backbone, "layers"):
            return [layer.mixer for layer in backbone.layers]
    raise AttributeError(
        "Cannot locate mixer layers. Inspect model.named_modules() manually."
    )


def _get_in_proj_activations(mixer: Any, model: Any, tokenizer: Any,
                               prompt: str, device: Any) -> Any | None:
    """
    Extract in_proj output activations from a single mixer layer.

    Returns the output tensor (detached, CPU numpy) or None if hook fails.
    """
    import torch

    store: dict[str, Any] = {}

    def _hook(module: Any, inp: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            store["act"] = output.detach().cpu().float().numpy()
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            store["act"] = output[0].detach().cpu().float().numpy()

    # Find in_proj (various naming conventions across transformers versions)
    proj_module = None
    for name in ["in_proj", "x_proj", "in_out_proj"]:
        proj_module = getattr(mixer, name, None)
        if proj_module is not None:
            break
    if proj_module is None:
        return None

    handle = proj_module.register_forward_hook(_hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()

    return store.get("act")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import numpy as np
    except ImportError as exc:
        raise ImportError("torch and numpy are required.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[W2] Device: {device}")

    model, tokenizer = _load_model_and_tokenizer(args.model, device)

    mixer_layers = _find_mixer_layers(model)
    n_layers = len(mixer_layers)
    print(f"[W2] Model has {n_layers} mixer layers. Analyzing: {args.layers}")

    # Generate a small set of prompts from different domains
    base_prompts = [
        "The state space model processes sequences by maintaining a hidden state.",
        "Efficient long-context inference is critical for modern NLP applications.",
        "Hadamard transforms redistribute activation energy across all channels.",
        "Operator fusion reduces memory bandwidth requirements in GPU kernels.",
        "Selective state space models provide linear complexity for sequence modeling.",
    ]
    # Extend to num_samples by cycling
    prompts = [base_prompts[i % len(base_prompts)] for i in range(args.num_samples)]

    rows: list[dict[str, Any]] = []

    for layer_idx in args.layers:
        if layer_idx >= n_layers:
            print(f"[W2] Layer {layer_idx} out of range (n_layers={n_layers}), skipping.")
            continue
        mixer = mixer_layers[layer_idx]
        print(f"[W2] Processing layer {layer_idx} ...")

        for sample_idx, prompt in enumerate(prompts):
            act = _get_in_proj_activations(mixer, model, tokenizer, prompt, device)
            if act is None:
                print(f"  [W2] Layer {layer_idx}, sample {sample_idx}: no activation captured, skipping.")
                continue

            # Take the x-branch (first half of in_proj output if combined xz)
            # act shape: [B, L, 2*D] or [B, L, D]
            flat_act = act.reshape(-1, act.shape[-1])
            if flat_act.shape[-1] > 1 and flat_act.shape[-1] % 2 == 0:
                # Split x/z branches; use x branch for entropy analysis
                half = flat_act.shape[-1] // 2
                x_branch = flat_act[:, :half]
            else:
                x_branch = flat_act

            # Apply Hadamard rotation (simulates COREY's fused Hadamard layer)
            x_rotated = _fast_hadamard_transform_np(x_branch)

            # Compute Sinkhorn proxy
            proxy = _fit_sinkhorn_proxy(x_branch, x_rotated, num_bins=args.sinkhorn_bins)

            # Outlier statistics
            x_std = float(np.std(x_branch)) + 1e-8
            outlier_before = float((np.abs(x_branch) > 3.0 * x_std).mean())
            x_rot_std = float(np.std(x_rotated)) + 1e-8
            outlier_after = float((np.abs(x_rotated) > 3.0 * x_rot_std).mean())

            row = {
                "model": args.model,
                "layer_idx": layer_idx,
                "sample_idx": sample_idx,
                "act_shape": str(act.shape),
                "x_branch_size": x_branch.size,
                "outlier_before": round(outlier_before, 6),
                "outlier_after": round(outlier_after, 6),
                **proxy,
            }
            rows.append(row)

            status = "✓" if proxy["entropy_gain"] > 0 else "✗"
            print(f"  layer={layer_idx} sample={sample_idx}: "
                  f"H_before={proxy['entropy_before']:.4f}  H_after={proxy['entropy_after']:.4f}  "
                  f"gain={proxy['entropy_gain']:+.4f} {status}  "
                  f"sinkhorn_l1={proxy['sinkhorn_residual_l1']:.4f}")

    if not rows:
        raise RuntimeError("No activation rows collected. Check model loading and layer indices.")

    # Aggregate statistics
    l1_vals = [r["sinkhorn_residual_l1"] for r in rows if math.isfinite(r["sinkhorn_residual_l1"])]
    gain_vals = [r["entropy_gain"] for r in rows if math.isfinite(r["entropy_gain"])]
    n_positive_gain = sum(1 for v in gain_vals if v > 0)
    n_below_01 = sum(1 for v in l1_vals if v < 0.10)

    summary = {
        "model": args.model,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "layers_analyzed": args.layers,
        "num_samples_per_layer": args.num_samples,
        "sinkhorn_bins": args.sinkhorn_bins,
        "n_rows": len(rows),
        "entropy_gain_mean": round(mean(gain_vals), 6) if gain_vals else None,
        "entropy_gain_std": round(pstdev(gain_vals), 6) if len(gain_vals) > 1 else 0.0,
        "entropy_gain_positive_fraction": round(n_positive_gain / len(gain_vals), 4) if gain_vals else None,
        "sinkhorn_l1_mean": round(mean(l1_vals), 6) if l1_vals else None,
        "sinkhorn_l1_std": round(pstdev(l1_vals), 6) if len(l1_vals) > 1 else 0.0,
        "sinkhorn_l1_min": round(min(l1_vals), 6) if l1_vals else None,
        "sinkhorn_l1_max": round(max(l1_vals), 6) if l1_vals else None,
        "sinkhorn_l1_below_0.10_fraction": round(n_below_01 / len(l1_vals), 4) if l1_vals else None,
    }

    print(f"\n[W2] Aggregate results over {len(rows)} (layer, sample) pairs:")
    print(f"  Entropy gain > 0: {n_positive_gain}/{len(gain_vals)} ({100*n_positive_gain/max(len(gain_vals),1):.1f}%)")
    if l1_vals:
        print(f"  Sinkhorn l1 residual: {mean(l1_vals):.4f} ± {pstdev(l1_vals):.4f}  "
              f"(min={min(l1_vals):.4f}, max={max(l1_vals):.4f})")
        print(f"  Below 0.10: {n_below_01}/{len(l1_vals)}")

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with (output_dir / "per_layer_results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"[W2] Outputs written to: {output_dir}")
    return {"output_dir": str(output_dir), "summary": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W2: Sinkhorn proxy validation on real Mamba checkpoint activations."
    )
    parser.add_argument("--model", default="mamba-370m",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which Mamba checkpoint to analyze.")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Layer indices to analyze.")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of prompt samples per layer.")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="(Unused; prompt length is determined by tokenizer.)")
    parser.add_argument("--sinkhorn-bins", type=int, default=64,
                        help="Number of histogram bins for Sinkhorn proxy.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("src/outputs/real_activation_sinkhorn"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_analysis(args)
    print(json.dumps({"status": "ok", "output_dir": result["output_dir"],
                      "summary": result["summary"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
