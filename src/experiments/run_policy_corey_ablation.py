#!/usr/bin/env python
"""
run_policy_corey_ablation.py

Ablation sweep for policy_corey/policy_static on Mamba-2.8B, n>=20.
Outputs results to src/outputs/policy_corey_ablation/
"""
import os
import argparse
from pathlib import Path
import time
from algorithms.mamba_integration import ModelSpec, RuntimeConfig, HuggingFaceMambaBackend, GenerationRequest

def main():
    parser = argparse.ArgumentParser(description="Ablation sweep for policy_corey/policy_static on Mamba-2.8B.")
    parser.add_argument("--model-id", type=str, default="benchang1110/mamba2-2.7b-hf", help="Model ID.")
    parser.add_argument("--n", type=int, default=20, help="Number of ablation runs (only used if --prompt is given).")
    parser.add_argument("--policy", type=str, choices=["corey", "static"], default="corey", help="Policy to test.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (overridden by --data-file if provided).")
    parser.add_argument("--data-file", type=str, default=None, help="Path to JSONL file with prompts (LongBench test set).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--output-dir", type=str, default="src/outputs/policy_corey_ablation", help="Output directory.")
    args = parser.parse_args()

    import json
    os.makedirs(args.output_dir, exist_ok=True)
    model_spec = ModelSpec(name="mamba-2.8b", model_id=args.model_id)
    runtime_cfg = RuntimeConfig(device=args.device)
    if args.policy == "corey":
        from algorithms.mamba_integration import EntropyGuidedSchedulerHook
        scheduler_hook = EntropyGuidedSchedulerHook()
    else:
        from algorithms.mamba_integration import StaticTileSchedulerHook
        scheduler_hook = StaticTileSchedulerHook()
    backend = HuggingFaceMambaBackend(model_spec, runtime_cfg, scheduler_hook)
    backend.load()
    print(f"[INFO] Model loaded: {args.model_id} (policy: {args.policy})")
    results = []

    prompts = []
    if args.data_file:
        # Load prompts from JSONL file (LongBench test set)
        with open(args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Try to extract the prompt field (LongBench format)
                if "input" in item:
                    prompt = item["input"]
                elif "question" in item:
                    prompt = item["question"]
                else:
                    prompt = str(item)
                prompts.append(prompt)
        print(f"[INFO] Loaded {len(prompts)} prompts from {args.data_file}")
    elif args.prompt is not None:
        prompts = [args.prompt] * args.n
    else:
        raise ValueError("Either --prompt or --data-file must be provided.")

    for i, prompt in enumerate(prompts):
        req = GenerationRequest(prompt=prompt, max_new_tokens=32)
        start = time.perf_counter()
        out = backend.generate(req)
        latency = time.perf_counter() - start
        result = {
            "run": i,
            "latency_ms": latency * 1000,
            "output": out.text,
            "telemetry": out.telemetry.__dict__,
            "policy": args.policy,
            "prompt": prompt,
        }
        print(f"[RESULT] {result}")
        results.append(result)
    out_path = Path(args.output_dir) / f"policy_corey_ablation_{args.policy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Results saved to {out_path}")

if __name__ == "__main__":
    main()
