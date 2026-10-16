"""taco-r2 item M7b (offline part): where the 32-bit indexing of the chunk-state tensor faults, from the stored
result files. CPU only; reads stored result files and times nothing.

The Triton SSD kernels of mamba_ssm (2.3.2) index the chunk-state tensor (batch x ceil(L / C) x heads x head dim x
state) with 32-bit offsets and fault with an illegal memory access above 2^31 elements; the sweeps skip such
cells and record them as unavailable. TFLA's mLSTM kernels do the same for their saved chunk states (batch x heads
x (L / C + 1) x d_qk x d_v), at 2^31 elements or more. This script

  1. lists every unavailable (shape, chunk) cell of every per-call table (``e2_*.csv``: NaN latency at a chunk) and
     of the synthetic sweeps (E17 ``skipped_chunks_int32``, E24 ``index_limit_chunks``), with the state-tensor
     size that the 2^31 rule predicts, and checks that the rule predicts exactly the recorded cells;
  2. derives the candidate-set rule of the recipe: per kernel instance and chunk, the largest batch x length
     product (and the largest length at batch 1, 4, 16) whose state tensor stays below 2^31 elements.

Output: ``taco_r2_baselines_m7b.json``.

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_index_limit
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.tpds import taco_r2_baselines_common as C
from src.experiments.tpds.analysis import chunk_columns, load_e2
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl

LIMIT = 2 ** 31
# SSD instances of the Mamba-2 checkpoints (mamba_ssm configs): d_inner = 2 d_model, head dim 64, state 128
SSD = {"Mamba-2 370M": {"d_inner": 2048, "headdim": 64, "dstate": 128},
       "Mamba-2 2.7B": {"d_inner": 5120, "headdim": 64, "dstate": 128},
       "Mamba-2 synthetic d4096": {"d_inner": 4096, "headdim": 64, "dstate": 128}}
TFLA = {"370m": (4, 256, 512), "1.4b": (8, 256, 512), "2.8b": (10, 256, 512)}   # heads, d_qk, d_v (E24 layouts)
CHUNKS_SSD = (32, 64, 128, 256, 512)
CHUNKS_TFLA = (32, 64, 128, 256, 512, 1024, 2048)


def ssd_elems(b, L, inst: dict, C_) -> int:
    h = inst["d_inner"] // inst["headdim"]
    return int(b) * math.ceil(int(L) / int(C_)) * h * inst["headdim"] * inst["dstate"]


def tfla_elems(b, L, layout, C_) -> int:
    nh, dqk, dv = TFLA[layout]
    return int(b) * nh * (int(L) // int(C_) + 1) * dqk * dv


def instance_of_csv(name: str) -> str | None:
    if not name.startswith("e2_mamba2"):
        return None
    return "Mamba-2 2.7B" if "2p7b" in name else "Mamba-2 370M"


def per_call_tables(in_dir: Path) -> dict:
    out = {}
    for p in sorted(in_dir.glob("e2_*.csv")):
        df = load_e2([p])
        if df.empty:
            continue
        ch = chunk_columns(df)
        lat = df[[f"lat_c{c}_mean" for c in ch]].to_numpy(float)
        bad = ~np.isfinite(lat)
        inst = instance_of_csv(p.name)
        rec = {"n_calls": int(len(df)), "chunks": ch, "n_unavailable_cells": int(bad.sum())}
        if inst is not None:
            pred = np.array([[ssd_elems(b, L, SSD[inst], c) > LIMIT for c in ch] for b, L in zip(df["batch"], df["L"])])
            rec["instance"] = inst
            rec["n_cells_predicted_by_2p31_rule"] = int(pred.sum())
            rec["rule_matches_recorded_cells_exactly"] = bool((pred == bad).all())
        if bad.any():
            cells = []
            for j, c in enumerate(ch):
                m = bad[:, j]
                if not m.any():
                    continue
                g = df[m].groupby(["batch", "L"]).size()
                cells.append({"chunk": c, "n_calls": int(m.sum()),
                              "shapes_batch_L": [[int(b), int(L), int(n)] for (b, L), n in g.items()],
                              "min_L": int(df.loc[m, "L"].min()), "batches": sorted({int(b) for b in df.loc[m, "batch"]})})
                if inst is not None:
                    cells[-1]["state_elems_range"] = [min(ssd_elems(b, L, SSD[inst], c) for b, L in zip(df.loc[m, "batch"], df.loc[m, "L"])),
                                                      max(ssd_elems(b, L, SSD[inst], c) for b, L in zip(df.loc[m, "batch"], df.loc[m, "L"]))]
                    ok = ~m & (df["batch"].to_numpy() == 16)
                    if ok.any():
                        rec.setdefault("largest_available_L_at_batch16", {})[str(c)] = int(df.loc[ok, "L"].max())
            rec["unavailable"] = cells
        out[p.name] = rec
    return out


def synthetic(in_dir: Path) -> dict:
    out = {}
    for p in sorted(in_dir.glob("e17_workbound_*.jsonl")):
        rows = [r for r in read_jsonl(p) if r.get("kind") == "timing"]
        cells = {}
        for r in rows:
            for c in r.get("skipped_chunks_int32", []) or []:
                k = (r["family"], r["shape"], int(r["L"]), int(r["batch"]), int(c))
                cells.setdefault(k, 0)
                cells[k] += 1
        pred = []
        for r in rows:
            if r["family"] != "ssd":
                continue
            d = int(re.sub(r"\D", "", r["shape"]))
            inst = {"d_inner": d, "headdim": 64, "dstate": 128}
            for c in CHUNKS_SSD:
                if ssd_elems(r["batch"], r["L"], inst, c) > LIMIT:
                    pred.append((r["family"], r["shape"], int(r["L"]), int(r["batch"]), int(c)))
        out[p.name] = {"n_rows": len(rows), "unavailable": [{"family": k[0], "shape": k[1], "L": k[2], "batch": k[3], "chunk": k[4],
                                                             "n_rounds": n} for k, n in sorted(cells.items())],
                       "ssd_rule_matches_recorded_cells_exactly": sorted(set(pred)) == sorted(cells)}
    for p in sorted(in_dir.glob("e24_tfla_*.jsonl")):
        rows = [r for r in read_jsonl(p) if r.get("kind") == "timing"]
        if not rows:
            continue
        cells, pred = {}, set()
        for r in rows:
            for c in r.get("index_limit_chunks", []) or []:
                k = (r["layout"], int(r["L"]), int(r["batch"]), int(c))
                cells[k] = cells.get(k, 0) + 1
            for c in CHUNKS_TFLA:
                if int(r["L"]) % c == 0 and tfla_elems(r["batch"], r["L"], r["layout"], c) >= LIMIT:
                    pred.add((r["layout"], int(r["L"]), int(r["batch"]), int(c)))
        out[p.name] = {"n_rows": len(rows),
                       "unavailable": [{"layout": k[0], "L": k[1], "batch": k[2], "chunk": k[3], "n_rounds": n,
                                        "state_elems": tfla_elems(k[2], k[1], k[0], k[3])} for k, n in sorted(cells.items())],
                       "n_distinct_cells": len(cells),
                       "rule_matches_recorded_cells_exactly": sorted(pred) == sorted(cells)}
    return out


def limits() -> dict:
    """Largest batch x number-of-chunks product and the implied largest length per batch, per instance and chunk."""
    out = {"ssd": {}, "tfla": {}}
    for name, inst in SSD.items():
        per_tile = (inst["d_inner"] // inst["headdim"]) * inst["headdim"] * inst["dstate"]
        n_tiles = LIMIT // per_tile                               # b * ceil(L / C) <= n_tiles
        out["ssd"][name] = {"elements_per_chunk_and_sequence": per_tile, "max_batch_x_chunks": int(n_tiles),
                            "max_L_by_chunk_and_batch": {str(c): {str(b): int((n_tiles // b) * c) for b in (1, 4, 16, 64)}
                                                         for c in CHUNKS_SSD}}
    for lay, (nh, dqk, dv) in TFLA.items():
        per = nh * dqk * dv
        n = (LIMIT - 1) // per                                    # b * (L / C + 1) < 2^31 / per
        out["tfla"][lay] = {"elements_per_chunk_and_sequence": per, "max_batch_x_chunks_plus_one": int(n),
                            "max_L_by_chunk_and_batch": {str(c): {str(b): int(max(0, n // b - 1) * c) for b in (1, 4, 16)}
                                                         for c in CHUNKS_TFLA}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    checks = C.Checks()
    tabs = per_call_tables(args.in_dir)
    syn = synthetic(args.in_dir)
    lim = limits()
    for n, r in tabs.items():
        if "rule_matches_recorded_cells_exactly" in r:
            checks.add(f"{n}: 2^31 rule predicts the unavailable cells", int(r["rule_matches_recorded_cells_exactly"]), 1, n, 0)
    for n, r in syn.items():
        k = "ssd_rule_matches_recorded_cells_exactly" if "ssd_rule_matches_recorded_cells_exactly" in r else "rule_matches_recorded_cells_exactly"
        checks.add(f"{n}: 2^31 rule predicts the unavailable cells", int(r[k]), 1, n, 0)
    checks.add("SSD 2.7B: chunk 32 at batch 16 allowed up to L = 6528", lim["ssd"]["Mamba-2 2.7B"]["max_L_by_chunk_and_batch"]["32"]["16"],
               6528, "Supplementary S6", 0)
    res = {"experiment": "taco-r2 M7b 32-bit index limit", "definitions": __doc__, "limit_elements": LIMIT,
           "per_call_tables_with_unavailable_cells": {n: r for n, r in tabs.items() if r["n_unavailable_cells"]},
           "per_call_tables_checked": len(tabs),
           "per_call_tables_without_unavailable_cells": sorted(n for n, r in tabs.items() if not r["n_unavailable_cells"]),
           "synthetic_sweeps": syn, "limits": lim, "checks": checks.summary()}
    path = C.save("m7b", res, args.in_dir)
    print(f"wrote {path}; checks {res['checks']['n_match']}/{res['checks']['n']}")
    for m in res["checks"]["mismatches"]:
        print("  MISMATCH", m["check"], m["computed"], m["reference"])
    for n, r in res["per_call_tables_with_unavailable_cells"].items():
        print(n, r.get("instance"), [(c["chunk"], c["n_calls"], c["min_L"], c["batches"]) for c in r["unavailable"]])
    for n, r in syn.items():
        print(n, [(x.get("family", x.get("layout")), x.get("shape"), x["L"], x["batch"], x["chunk"]) for x in r["unavailable"]])


if __name__ == "__main__":
    main()
