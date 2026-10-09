"""Oracle headroom over a chunk table keyed on length bucket and batch size, per data set.

The no-gain bound (Proposition 1 of the manuscript) compares a data-dependent policy with a
shape-keyed table. For the coarse table keyed on (length bucket, batch), chosen on held-out
prompt folds exactly as ``e5_breakeven._cv_shape_table`` chooses it, this script reports per
data set (captured Mamba-1 calls of each GPU, both timing conventions where available):

  headroom_us     mean over calls of [latency(table chunk) - latency(per-call oracle)], microseconds
  headroom_pct    the same as a share of the table's mean latency (latency-weighted share of the
                  table's time an oracle would recover; the convention of the manuscript)

CPU only; reads ``e2_mamba1_<tag>.csv``. Output: ``bucket_headroom.json``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.tpds.analysis import chunk_columns, latency_matrix, load_e2
from src.experiments.tpds.common import RESULTS_DIR, write_json
from src.experiments.tpds.e5_breakeven import _cv_shape_table

# tag -> (GPU, convention); device-time tags of the RTX PRO 6000 and RTX 3090 hold round medians
DEFAULT_TAGS = {
    "pro6000": ("RTX PRO 6000", "launch-inclusive, first session"),
    "pro6000_r4plain": ("RTX PRO 6000", "launch-inclusive, second session (round medians)"),
    "pro6000_r4spin": ("RTX PRO 6000", "device time (round medians)"),
    "vgpu3090": ("RTX 3090", "launch-inclusive"),
    "vgpu3090_r4spin": ("RTX 3090", "device time (round medians)"),
    "h800": ("H800", "launch-inclusive"),
    "h800_r3spin": ("H800", "device time"),
    "vgpu3090_1p4b_r3": ("RTX 3090, 1.4B", "device time"),
    "vgpu3090_2p8b_r3": ("RTX 3090, 2.8B", "device time"),
}


def bucket_headroom(csv_path: Path) -> dict:
    df = load_e2([csv_path])
    chunks = chunk_columns(df)
    lat = latency_matrix(df, chunks)
    oracle = float(lat.min(1).mean())
    table = _cv_shape_table(df, lat, chunks)
    return {"n_calls": int(len(df)), "oracle_ms": oracle, "table_ms": table,
            "headroom_us": (table - oracle) * 1e3, "headroom_pct": 100.0 * (table - oracle) / table}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "bucket_headroom.json")
    args = p.parse_args(argv)
    res = {"experiment": "bucket_headroom", "table": "length bucket x batch, chosen on held-out prompt folds "
           "(e5_breakeven._cv_shape_table)", "data_sets": {}}
    for tag, (gpu, conv) in DEFAULT_TAGS.items():
        path = args.in_dir / f"e2_mamba1_{tag}.csv"
        if not path.exists():
            continue
        r = {"gpu": gpu, "convention": conv, **bucket_headroom(path)}
        res["data_sets"][tag] = r
        print(f"{tag:20s} {gpu:16s} {conv:45s} {r['headroom_us']:7.2f} us  {r['headroom_pct']:5.2f} %")
    write_json(args.out, res)


if __name__ == "__main__":
    main()
