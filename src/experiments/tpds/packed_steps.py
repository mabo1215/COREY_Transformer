"""Packed variable-length prefill steps sampled from public serving traces (input of e27_varlen.py).

A continuous-batching engine with chunked prefill issues one packed variable-length kernel call per
scheduler step: the prompt chunks of several requests are concatenated along the sequence axis, and
the kernel resets the recurrent state at each sequence boundary. This script replays the request
arrivals and prompt lengths of the Azure LLM inference traces 2023 (code and conversation, first
hour) through the chunked-prefill scheduler of ``serving_key_churn.py`` (FCFS, token budget T per
step, decode colocated, offered load rho = 0.5) and records, for every prefill step, the token count
of each packed sequence and whether it continues a prompt begun in an earlier step (its scan then
starts from a carried state). Steps are sampled stratified by budget and number of packed sequences.

Output: ``packed_steps_<tag>.json`` with the sampled steps and the sampling summary.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from src.experiments.tpds.common import RESULTS_DIR, write_json
from src.experiments.tpds.serving_key_churn import (DATA_DIR, RHO_PRIMARY, T_PF_S, TRACE_FILES, load_trace,
                                                     select_window, simulate_chunked)

TRACES = ("azure2023_code", "azure2023_conv")
SEQ_CLASSES = ((1, 1), (2, 3), (4, 7), (8, 10**9))     # packed sequences per step


def ensure_trace(name: str, data_dir: Path) -> None:
    spec = TRACE_FILES[name]
    path = data_dir / spec["file"]
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(spec["url"], path)


def sample_steps(args) -> dict:
    rng = np.random.default_rng(args.seed)
    out, summary = [], []
    for name in TRACES:
        ensure_trace(name, args.data_dir)
        win, info = select_window(load_trace(name, args.data_dir), "1h")
        s = float(win["ctx"].sum()) * T_PF_S / (info["duration_s"] * RHO_PRIMARY)
        for T in args.budgets:
            sim = simulate_chunked(win["t"] * s, win["ctx"], win["gen"], T, colocate=True, record_lens=True)
            steps = sim["steps"]
            nseq = np.array([len(st) for st in steps])
            for lo, hi in SEQ_CLASSES:
                idx = np.flatnonzero((nseq >= lo) & (nseq <= hi))
                pick = rng.choice(idx, size=min(args.per_class, len(idx)), replace=False) if len(idx) else []
                for i in sorted(int(x) for x in pick):
                    lens = [int(c) for c, _ in steps[i]]
                    out.append({"trace": name, "budget": T, "step": i, "seq_class": f"{lo}-{hi}",
                                "lens": lens, "continued": [bool(k) for _, k in steps[i]],
                                "tokens": int(sum(lens)), "nseq": len(lens)})
                summary.append({"trace": name, "budget": T, "seq_class": f"{lo}-{hi}", "steps": int(len(idx)),
                                "sampled": int(len(pick)), "share": float(len(idx) / max(len(steps), 1))})
    return {"experiment": "packed-steps", "traces": list(TRACES), "window": "1h", "rho": RHO_PRIMARY,
            "budgets": list(args.budgets), "per_class": args.per_class, "seed": args.seed,
            "n_steps": len(out), "strata": summary, "steps": out}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[2048, 8192])
    p.add_argument("--per-class", type=int, default=12, help="steps sampled per (trace, budget, class)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--tag", default="taco")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    res = sample_steps(args)
    write_json(args.out_dir / f"packed_steps_{args.tag}.json", res)
    print(json.dumps(res["strata"], indent=1))
    print(f"{res['n_steps']} steps written")


if __name__ == "__main__":
    main()
