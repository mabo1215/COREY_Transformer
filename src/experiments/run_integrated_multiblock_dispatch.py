from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated benchmark only when a real multi-BLOCK selective-scan "
            "dispatch is available. The dispatch module must expose selective_scan_fn "
            "with a chunk_size kwarg and preserve recurrence semantics."
        )
    )
    parser.add_argument("--dispatch-module", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/h800_multiblock_dispatch_probe"))
    parser.add_argument("--model", default="mamba-370m")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=256)
    return parser.parse_args()


def _probe_dispatch(dispatch_fn: Any) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA unavailable"}
    device = torch.device("cuda")
    dtype = torch.float16
    batch, dim, seq, d_state = 1, 64, 128, 16
    u = torch.randn(batch, dim, seq, device=device, dtype=dtype)
    delta = torch.rand(batch, dim, seq, device=device, dtype=dtype)
    A = torch.randn(dim, d_state, device=device, dtype=torch.float32)
    B = torch.randn(batch, d_state, seq, device=device, dtype=dtype)
    C = torch.randn(batch, d_state, seq, device=device, dtype=dtype)
    D = torch.randn(dim, device=device, dtype=torch.float32)
    try:
        out = dispatch_fn(u, delta, A, B, C, D=D, delta_softplus=True, chunk_size=64)
        if isinstance(out, tuple):
            out = out[0]
        torch.cuda.synchronize()
        return {
            "ok": tuple(out.shape) == (batch, dim, seq),
            "shape": list(out.shape),
            "finite_ratio": float(torch.isfinite(out.float()).float().mean().item()),
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "benchmark": "integrated_multiblock_dispatch",
        "status": "blocked",
        "dispatch_module": args.dispatch_module,
        "notes": (
            "A true run requires a real dispatch module that compiles or exposes "
            "multiple recurrence-preserving selective-scan BLOCK_SIZE variants. "
            "Chunked sequence emulation is intentionally not used because it resets "
            "scan state at chunk boundaries unless the kernel accepts an initial state."
        ),
    }

    if args.dispatch_module is None:
        result["error"] = "No --dispatch-module supplied."
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    try:
        module = importlib.import_module(args.dispatch_module)
        dispatch_fn = getattr(module, "selective_scan_fn")
    except Exception as exc:
        result["error"] = f"Unable to import dispatch module: {exc!r}"
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    probe = _probe_dispatch(dispatch_fn)
    result["probe"] = probe
    if not probe.get("ok"):
        result["error"] = "Dispatch probe failed."
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    result["status"] = "ready"
    result["next_command"] = (
        "Run src/experiments/run_integrated_end_to_end.py after wiring this dispatch "
        "module into transformers.models.mamba.modeling_mamba.selective_scan_fn on the H800 image."
    )
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
