"""E28 - chunk choice of the Triton Mamba-1 scan under two-GPU tensor parallelism with collectives (review M7b).

Under tensor parallelism of degree t every rank runs the scan at width d/t, and the output projection of the
layer ends in an all-reduce of the layer output (Megatron-style row-parallel linear). The study's shard-width
sweep (E12) timed the shard shapes on one GPU with no communication. This experiment runs two ranks on two
NVLink-connected GPUs (torchrun, NCCL) and times, on both ranks at once, the Triton Mamba-1 scan at the
per-rank widths of the 370M, 1.4B and 2.8B checkpoints at t = 2 (d_inner / 2 = 1024, 2048, 2560), for every
chunk, under two conditions:

  alone    both ranks scan their shard with no communication (the E12 condition, now on both GPUs at once)
  overlap  a NCCL all-reduce of the layer output (batch x length x d_model, fp16; d_model = d_inner / 2) is
           issued on a separate stream before every scan call, so communication overlaps the scan, as in
           tensor-parallel layers that overlap the reduction of one layer with the compute of the next

Synthetic inputs (the kernel is data-oblivious), lengths ``--lengths``, batch ``--batches``, ``--rounds`` rounds
with the chunk order shuffled per shape and round (the same order on both ranks, so the collectives match),
device time (GPU-side pre-spin), warm L2, exclusive-GPU guard on the two physical GPUs.

``--analyze`` (no GPU) reports, per condition and rank: the per-shape oracle and tie set (round means and 95%
CIs), the regret of the alone-optimal chunk under overlap, the share of shapes whose optimum moves beyond the
tie set under overlap, the agreement of the two ranks' optima (consensus), the slowdown of the scan under
overlap, and the regret of the frozen cost model (RTX 3090 parameters) in both conditions.

Launch (two GPUs):  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 -m src.experiments.tpds.e28_tp_collectives
Output: ``e28_tp_<tag>.jsonl`` (rank 0 writes one row per shape, condition and round with both ranks' timings)
and ``e28_tp_<tag>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from src.experiments.tpds.common import (MAMBA1_CHUNKS, RESULTS_DIR, JsonlWriter, Logger, host_name, read_jsonl,
                                         summarize, write_json)

WIDTHS = {"370m": 1024, "1p4b": 2048, "2p8b": 2560}     # d_inner / 2 at t = 2
CONDITIONS = ("alone", "overlap")
TIE = 0.02


# --------------------------------------------------------------------------- run (torchrun, two ranks)
def physical_gpu(local_rank: int) -> int | None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in vis.split(",") if x.strip()]
    return int(ids[local_rank]) if local_rank < len(ids) and ids[local_rank].isdigit() else local_rank


def time_cell(scan, comm, buf, cond, warmup, repeats, pre_spin_ms):
    """Per-repeat scan latency (ms, CUDA events on the compute stream); with ``overlap`` an all-reduce of
    ``buf`` is issued on ``comm`` before every scan call."""
    import torch
    import torch.distributed as dist

    from src.experiments.tpds.common import _spin_cycles_per_ms

    def issue(start_event):
        # the all-reduce waits for the start of its scan, so the two overlap even when the host runs ahead
        if cond == "overlap":
            comm.wait_event(start_event)
            with torch.cuda.stream(comm):
                dist.all_reduce(buf, async_op=False)

    for _ in range(warmup):
        ev = torch.cuda.Event()
        ev.record()
        issue(ev)
        scan()
    torch.cuda.synchronize()
    dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    if pre_spin_ms > 0:
        torch.cuda._sleep(int(pre_spin_ms * _spin_cycles_per_ms()))
    for i in range(repeats):
        starts[i].record()
        issue(starts[i])
        scan()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def run(args):
    import torch
    import torch.distributed as dist

    from src.experiments.tpds import kernels as K
    from src.experiments.tpds.common import (all_gpu_pids, env_metadata, foreign_gpu_pids, register_own_gpu_pids,
                                             wait_for_exclusive_gpu)

    from datetime import timedelta

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    # long timeout: one rank may pause for the exclusive-GPU guard while the other waits at a barrier
    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "run with torchrun --nproc_per_node 2"
    phys = physical_gpu(local)
    tag = args.tag or host_name()
    log = Logger(f"e28_tp_{tag}_rank{rank}")
    baseline = all_gpu_pids(phys)
    torch.zeros(1, device="cuda")
    dist.barrier()
    if args.exclusive_guard:
        register_own_gpu_pids(baseline, index=phys, log=log)
    w = None
    if rank == 0:
        meta = env_metadata()
        meta.update({"experiment": "E28", "world_size": world, "physical_gpus": [physical_gpu(i) for i in range(world)],
                     "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                     "args": {k: str(v) for k, v in vars(args).items()}})
        w = JsonlWriter(args.out_dir / f"e28_tp_{tag}.jsonl", meta=meta)
    done = set(w.done) if w else set()
    obj = [sorted(done)]
    dist.broadcast_object_list(obj, src=0)
    done = set(obj[0])
    comm = torch.cuda.Stream()
    chunks = [int(c) for c in args.chunks.split(",")]
    for model in [m for m in args.models.split(",") if m]:
        d = WIDTHS[model]
        for L in [int(x) for x in args.lengths.split(",")]:
            for B in [int(x) for x in args.batches.split(",")]:
                inp = K.make_mamba1_inputs(B, d, 16, L, dtype=torch.float16, seed=1000 * rank + L + B)
                out = torch.empty_like(inp["u"])
                buf = torch.randn(B * L * d, device="cuda", dtype=torch.float16)   # layer output, d_model = d
                for cond in CONDITIONS:
                    for r in range(args.rounds):
                        key = f"e28|{model}|L{L}|b{B}|{cond}|r{r}"
                        if key in done:
                            continue
                        order = list(chunks)
                        random.Random(key).shuffle(order)
                        while True:
                            if args.exclusive_guard and wait_for_exclusive_gpu(args.guard_until, log, index=phys) is None:
                                raise SystemExit(3)
                            dist.barrier()          # both ranks start the cell together
                            busy0 = foreign_gpu_pids(phys) if args.exclusive_guard else None
                            cells = {}
                            for c in order:
                                fn = (lambda c=c: K.mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"],
                                                                inp["D"], inp["z"], None, True, chunk=c, out=out))
                                cells[str(c)] = summarize(time_cell(fn, comm, buf, cond, args.warmup, args.repeats,
                                                                    args.pre_spin_ms))
                            busy1 = foreign_gpu_pids(phys) if args.exclusive_guard else None
                            busy = [bool(busy0 or busy1)]
                            allbusy = [None] * world
                            dist.all_gather_object(allbusy, busy)
                            if not any(b[0] for b in allbusy):
                                break
                            log(f"{key}: a foreign process used one of the GPUs; re-timing")
                        gathered = [None] * world
                        dist.all_gather_object(gathered, cells)
                        if rank == 0:
                            w.write({"kind": "timing", "key": key, "model": model, "d": d, "d_model": d, "L": L,
                                     "batch": B, "condition": cond, "round": r, "order": order,
                                     "allreduce_bytes": int(buf.numel() * 2),
                                     "ranks": {str(i): g for i, g in enumerate(gathered)}})
                            log(f"{key}: " + " ".join(f"c{c}={gathered[0][str(c)]['mean_ms']:.3f}" for c in chunks))
                del inp, out, buf
                torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        w.close()
    dist.destroy_process_group()
    if rank == 0:
        analyze(args)


# --------------------------------------------------------------------------- analysis (CPU)
def frame(rows, cond, rank):
    import pandas as pd

    recs = []
    for r in rows:
        if r["condition"] != cond:
            continue
        for c, cell in r["ranks"][str(rank)].items():
            recs.append({"family": "mamba1", "d": r["d"], "L": r["L"], "batch": r["batch"], "round": r["round"],
                         "chunk": int(c), "mean_ms": cell["mean_ms"]})
    return pd.DataFrame(recs)


def analyze(args):
    from src.experiments.tpds.e12_analysis import regret, shape_table, triton_model_choice

    tag = args.tag or host_name()
    rows = [r for r in read_jsonl(args.out_dir / f"e28_tp_{tag}.jsonl") if r.get("kind") == "timing"]
    chunks = sorted({int(c) for r in rows for c in r["ranks"]["0"]})
    tabs = {(cond, rk): shape_table(frame(rows, cond, rk), chunks) for cond in CONDITIONS for rk in (0, 1)}
    res = {"experiment": "E28-analysis", "tag": tag, "chunks": chunks, "tie_margin": TIE, "per_rank": {}}
    for rk in (0, 1):
        a, o = tabs[("alone", rk)], tabs[("overlap", rk)]
        if a.empty or o.empty:
            continue
        alone_choice = {(r.d, r.L, r.batch): r.oracle for r in a.itertuples()}
        om = o.set_index(["d", "L", "batch"])
        moved = []
        slow = []
        for r in a.itertuples():
            k = (r.d, r.L, r.batch)
            if k not in om.index:
                continue
            ro = om.loc[k]
            moved.append(r.oracle not in ro["tied"])
            slow.append(float(np.median(ro["lat"] / r.lat)))
        rec = {"alone_oracle_under_overlap": regret(o, chunks, alone_choice),
               "share_optimum_moves": float(np.mean(moved)) if moved else None,
               "median_scan_slowdown_overlap": float(np.median(slow)) if slow else None,
               "max_scan_slowdown_overlap": float(np.max(slow)) if slow else None,
               "n_shapes": int(len(a))}
        try:
            in_dir = args.out_dir
            mc = triton_model_choice(a, chunks, args.model_host, in_dir)
            rec["cost_model_alone"] = regret(a, chunks, mc)
            rec["cost_model_overlap"] = regret(o, chunks, mc)
        except Exception as e:  # noqa: BLE001 - the model row is optional (needs the fitted parameters)
            rec["cost_model_error"] = f"{type(e).__name__}: {e}"[:200]
        res["per_rank"][str(rk)] = rec
    for cond in CONDITIONS:
        t0, t1 = tabs[(cond, 0)], tabs[(cond, 1)]
        if t0.empty or t1.empty:
            continue
        m = t0.merge(t1, on=["d", "L", "batch"], suffixes=("_0", "_1"))
        same = (m["oracle_0"] == m["oracle_1"]).mean()
        tied = np.mean([r.oracle_0 in r.tied_1 and r.oracle_1 in r.tied_0 for r in m.itertuples()])
        res[f"rank_agreement_{cond}"] = {"same_oracle": float(same), "mutually_tied": float(tied), "n": int(len(m))}
    write_json(args.out_dir / f"e28_tp_{tag}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "chunks"}, indent=1)[:3000])
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="370m,1p4b,2p8b")
    p.add_argument("--lengths", default="512,2048,8192")
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--chunks", default=",".join(str(c) for c in MAMBA1_CHUNKS))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--pre-spin-ms", type=float, default=20.0)
    p.add_argument("--exclusive-guard", action="store_true")
    p.add_argument("--guard-until", type=float, default=None)
    p.add_argument("--model-host", default="vgpu3090", help="cost-model parameters used in the analysis")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
