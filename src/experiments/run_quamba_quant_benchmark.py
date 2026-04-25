#!/usr/bin/env python
"""
run_quamba_quant_benchmark.py

Benchmark Mamba-specific quantized inference via the Quamba INT4 path
(Chiang et al., 2024).  Falls back to FP16 baseline when Quamba CUDA
extensions are unavailable and documents the limitation.

Requirements for real Quamba execution (RTX 3090 / sm_89 recommended):
  - quamba==2.0.0a1 compiled from source against CUDA 12.8
  - mamba_ssm==2.2.2, causal_conv1d==1.6.1, fast_hadamard_transform==1.0.4.post1
  - See src/usage.md for remote 4x3090 setup instructions.

Outputs results to src/outputs/quamba_quant_benchmark/
"""
import argparse
import json
import os
import time
from pathlib import Path


def _quamba_available() -> bool:
    try:
        import quamba  # noqa: F401
        return True
    except ImportError:
        return False


def _run_quamba_int4(model_id: str, prompts: list, bits: int, group_size: int, device: str):
    from algorithms.mamba_integration import (
        GenerationRequest,
        HuggingFaceMambaBackend,
        ModelSpec,
        QuantizationConfig,
        RuntimeConfig,
    )

    quant_cfg = QuantizationConfig(mode="int4", backend="awq", bits=bits, group_size=group_size)
    runtime_cfg = RuntimeConfig(device=device, quantization=quant_cfg)
    model_spec = ModelSpec(name="quamba-int4", model_id=model_id)
    backend = HuggingFaceMambaBackend(model_spec, runtime_cfg)
    try:
        backend.load()
    except RuntimeError as exc:
        return None, str(exc)

    results = []
    for i, prompt in enumerate(prompts):
        req = GenerationRequest(prompt=prompt, max_new_tokens=32)
        t0 = time.perf_counter()
        out = backend.generate(req)
        latency = (time.perf_counter() - t0) * 1000
        results.append({
            "run": i,
            "latency_ms": latency,
            "output": out.text,
            "prompt": prompt[:80],
            "telemetry": out.telemetry.__dict__,
            "quantization": {"mode": "int4", "bits": bits, "group_size": group_size},
        })
        print(f"[INFO] quamba-int4 [{i}]: {latency:.1f} ms")
    return results, None


def _run_fp16_baseline(model_id: str, prompts: list, device: str):
    from algorithms.mamba_integration import (
        GenerationRequest,
        HuggingFaceMambaBackend,
        ModelSpec,
        RuntimeConfig,
    )

    runtime_cfg = RuntimeConfig(device=device)
    model_spec = ModelSpec(name="mamba-fp16", model_id=model_id)
    backend = HuggingFaceMambaBackend(model_spec, runtime_cfg)
    try:
        backend.load()
    except Exception as exc:
        return None, str(exc)

    results = []
    for i, prompt in enumerate(prompts):
        req = GenerationRequest(prompt=prompt, max_new_tokens=32)
        t0 = time.perf_counter()
        out = backend.generate(req)
        latency = (time.perf_counter() - t0) * 1000
        results.append({
            "run": i,
            "latency_ms": latency,
            "output": out.text,
            "prompt": prompt[:80],
            "telemetry": out.telemetry.__dict__,
            "quantization": {"mode": "fp16", "bits": 16, "group_size": None},
        })
        print(f"[INFO] fp16 [{i}]: {latency:.1f} ms")
    return results, None


def main():
    parser = argparse.ArgumentParser(description="Benchmark Quamba quantized inference.")
    parser.add_argument(
        "--model-id", type=str, default="benchang1110/mamba2-2.7b-hf", help="Model ID."
    )
    parser.add_argument("--quant-backend", type=str, default="awq",
                        help="Quantization backend (awq/gptq/none).")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits.")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt override.")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to JSONL file with prompts (LongBench test set).")
    parser.add_argument("--max-prompts", type=int, default=20,
                        help="Maximum number of prompts to process.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--output-dir", type=str, default="src/outputs/quamba_quant_benchmark",
        help="Output directory.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    prompts = []
    if args.data_file:
        with open(args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("input") or item.get("question") or str(item)
                prompts.append(prompt)
        prompts = prompts[: args.max_prompts]
        print(f"[INFO] Loaded {len(prompts)} prompts from {args.data_file}")
    elif args.prompt is not None:
        prompts = [args.prompt]
    else:
        raise ValueError("Either --prompt or --data-file must be provided.")

    status = {
        "model_id": args.model_id,
        "quant_backend": args.quant_backend,
        "bits": args.bits,
        "group_size": args.group_size,
        "device": args.device,
        "quamba_available": _quamba_available(),
    }

    if _quamba_available() and args.quant_backend == "awq":
        print("[INFO] Quamba extensions detected, attempting INT4 quantized run ...")
        int4_results, err = _run_quamba_int4(
            args.model_id, prompts, args.bits, args.group_size, args.device
        )
        if int4_results is not None:
            status["mode"] = "quamba_int4"
            status["results"] = int4_results
        else:
            print(f"[WARN] INT4 run failed: {err}")
            print("[INFO] Falling back to FP16 baseline ...")
            fp16_results, fp16_err = _run_fp16_baseline(args.model_id, prompts, args.device)
            status["mode"] = "fp16_fallback"
            status["int4_error"] = err
            status["results"] = fp16_results or []
            if fp16_err:
                status["fp16_error"] = fp16_err
    else:
        # AWQ not supported for Mamba in current stack; run FP16 baseline.
        note = (
            "AutoAWQ does not support Mamba checkpoints (verified: autoawq 0.2.9 returns "
            "'mamba isn't supported yet').  INT4 quantized benchmarking requires the "
            "Quamba-specific path: quamba==2.0.0a1 compiled against CUDA 12.8 on sm_89.  "
            "See src/usage.md for remote 4x3090 GPU setup.  Running FP16 baseline instead."
        )
        print(f"[WARN] {note}")
        status["mode"] = "fp16_baseline_only"
        status["awq_note"] = note
        fp16_results, fp16_err = _run_fp16_baseline(args.model_id, prompts, args.device)
        status["results"] = fp16_results or []
        if fp16_err:
            status["fp16_error"] = fp16_err
            print(f"[WARN] FP16 baseline also failed: {fp16_err}")

    out_path = Path(args.output_dir) / "quamba_quant_benchmark_result.json"
    with open(out_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"[DONE] Results saved to {out_path}")


if __name__ == "__main__":
    main()
