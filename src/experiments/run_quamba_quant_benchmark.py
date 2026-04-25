#!/usr/bin/env python
"""
run_quamba_quant_benchmark.py

Benchmark Quamba quantized inference using HuggingFaceMambaBackend.
Outputs results to src/outputs/quamba_quant_benchmark/
"""
import os
import argparse
from pathlib import Path
import time
from algorithms.mamba_integration import ModelSpec, RuntimeConfig, QuantizationConfig, HuggingFaceMambaBackend, GenerationRequest

def main():
    parser = argparse.ArgumentParser(description="Benchmark Quamba quantized inference.")
    parser.add_argument("--model-id", type=str, default="benchang1110/mamba2-2.7b-hf", help="Model ID.")
    parser.add_argument("--quant-backend", type=str, default="awq", help="Quantization backend (awq/gptq/none).")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits.")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size.")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt text.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--output-dir", type=str, default="src/outputs/quamba_quant_benchmark", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    quant_cfg = QuantizationConfig(mode="int4", backend=args.quant_backend, bits=args.bits, group_size=args.group_size)
    runtime_cfg = RuntimeConfig(device=args.device, quantization=quant_cfg)
    model_spec = ModelSpec(name="quamba", model_id=args.model_id)
    backend = HuggingFaceMambaBackend(model_spec, runtime_cfg)
    backend.load()
    print(f"[INFO] Model loaded: {args.model_id} (quant: {args.quant_backend}, bits: {args.bits})")
    req = GenerationRequest(prompt=args.prompt, max_new_tokens=32)
    start = time.perf_counter()
    out = backend.generate(req)
    latency = time.perf_counter() - start
    result = {
        "latency_ms": latency * 1000,
        "output": out.text,
        "telemetry": out.telemetry.__dict__,
        "quantization": quant_cfg.__dict__,
    }
    print(f"[RESULT] {result}")
    out_path = Path(args.output_dir) / "quamba_quant_benchmark_result.json"
    import json
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DONE] Results saved to {out_path}")

if __name__ == "__main__":
    main()
