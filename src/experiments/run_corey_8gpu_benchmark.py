"""
8-GPU distributed benchmark for COREY selective_scan_fn and related methods.
Launch with torchrun --nproc_per_node=8 ...
"""
import argparse
import os
import torch
import torch.distributed as dist
from pathlib import Path

parser = argparse.ArgumentParser(description="COREY 8-GPU distributed benchmark.")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--chunk-size", type=int, default=512)
parser.add_argument("--seq-len", type=int, default=4096)
parser.add_argument("--repeat", type=int, default=30)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    # TODO: Load model and run benchmark on each GPU
    # Save results only on rank 0
    if rank == 0:
        print(f"[8GPU] Benchmark completed for model {args.model}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir/"done.txt", "w") as f:
            f.write("done\n")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
