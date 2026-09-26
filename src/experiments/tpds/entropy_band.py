"""How wide is the entropy band of real scan inputs, and how does Eq. (2) map it? (items B9, C7)

Reads the E2 per-call tables and reports, per model and GPU host: the range and the
5th-95th percentile band of the scan-input entropy (bits, K=256) over batch-1 calls,
the median width of that band within one layer, and the distribution of the chunks
that Eq. (2) selects in the kernel-level range [32, 512] and in the routed range
[128, 2048].

Output: ``entropy_band_<host>.json``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from src.experiments.tpds.analysis import load_e2
from src.experiments.tpds.common import RESULTS_DIR, entropy_rule_chunk, write_json

BITS_MAX = 8.0  # log2(256)


def band(h: np.ndarray) -> dict:
    return {"min": float(h.min()), "max": float(h.max()), "p5": float(np.percentile(h, 5)),
            "p95": float(np.percentile(h, 95))}


def summarize(df) -> dict:
    b1 = df[df["batch"] == 1]
    h = b1["entropy_bits"].to_numpy(dtype=float)
    per_layer = {int(l): band(s["entropy_bits"].to_numpy(dtype=float)) for l, s in b1.groupby("layer")}
    widths = [v["p95"] - v["p5"] for v in per_layer.values()]
    all_h = df["entropy_bits"].to_numpy(dtype=float)
    kernel = Counter(entropy_rule_chunk(x / BITS_MAX, 32, 512) for x in all_h)
    routed = Counter(entropy_rule_chunk(x / BITS_MAX, 128, 2048) for x in all_h)
    return {"n_calls": int(len(df)), "n_batch1": int(len(b1)), "batch1": band(h),
            "per_layer_batch1": per_layer, "median_within_layer_p5_p95_width": float(np.median(widths)),
            "rule_chunks_kernel_range": {str(k): int(v) for k, v in sorted(kernel.items())},
            "rule_chunks_routed_range": {str(k): int(v) for k, v in sorted(routed.items())}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--host", required=True)
    args = p.parse_args(argv)
    res = {"experiment": "entropy_band", "host": args.host, "unit": "bits (K=256)", "models": {}}
    for m in ("mamba1", "mamba2"):
        df = load_e2([args.in_dir / f"e2_{m}_{args.host}.csv"])
        if df.empty:
            continue
        res["models"][m] = s = summarize(df)
        print(f"{m}: batch-1 entropy {s['batch1']['min']:.2f}-{s['batch1']['max']:.2f} bits "
              f"(p5-p95 {s['batch1']['p5']:.2f}-{s['batch1']['p95']:.2f}); within-layer p5-p95 width "
              f"{s['median_within_layer_p5_p95_width']:.2f} bits; Eq.(2) [32,512] {s['rule_chunks_kernel_range']}; "
              f"[128,2048] {s['rule_chunks_routed_range']}")
    write_json(args.in_dir / f"entropy_band_{args.host}.json", res)


if __name__ == "__main__":
    main()
