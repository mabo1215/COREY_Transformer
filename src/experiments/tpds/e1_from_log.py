"""Rebuild E1 timing rows from the run log when the JSONL output was lost (item C1).

E1c (2026-09-26) timed all of its cells, but its JSONL writer kept appending to a file
that a concurrent git checkout had replaced, so the rows on disk stop before the first
timing. Every timed cell is also logged as one line

    [<time>] <family> L=<L> c=<chunk> <input> <mode> r<round>: <mean> +- <std> ms (H=<bits> bits)

This script parses those lines back into timing rows (mean, std, n and the 95% t-interval
of the mean, which is what the E1 analysis uses), takes the meta and validation rows from
the partial JSONL, and writes ``e1_<tag>.jsonl`` and ``e1_<tag>.json`` in the usual format.
Fields that are not in the log (median, min, GPU state around each timing) are absent;
every row carries ``"source": "log"``.

Usage:
    python -m src.experiments.tpds.e1_from_log --log <run log> --partial <partial jsonl> --tag pro6000_e1c
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from src.experiments.tpds.common import RESULTS_DIR, t_crit, write_json
from src.experiments.tpds.e1_value_invariance import summarize_rows

LINE = re.compile(r"^\[(?P<t>[\d-]+ [\d:]+)\] (?P<fam>mamba1|mamba2|gdn) L=(?P<L>\d+) c=(?P<c>\d+) "
                  r"(?P<inp>\S+)\s+(?P<mode>act|all) r(?P<r>\d+): (?P<mean>[\d.]+) \+- (?P<std>[\d.]+) ms "
                  r"\(H=(?P<H>[\d.]+) bits\)\s*$")


def parse_log(path: Path, n: int, batch: int) -> list[dict]:
    rows: dict[str, dict] = {}
    half_t = t_crit(n - 1) / math.sqrt(n)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.match(line)
        if not m:
            continue
        fam, L, c, inp, mode, r = m["fam"], int(m["L"]), int(m["c"]), m["inp"], m["mode"], int(m["r"])
        mean, std, H = float(m["mean"]), float(m["std"]), float(m["H"])
        key = f"{fam}|L{L}|b{batch}|c{c}|{inp}|{mode}|r{r}"
        rows[key] = {"kind": "timing", "key": key, "family": fam, "L": L, "batch": batch, "chunk": c,
                     "input": inp, "mode": mode, "round": r, "entropy_bits_k256": H, "entropy_norm": H / 8.0,
                     "n": n, "mean_ms": mean, "std_ms": std, "ci95_lo_ms": mean - half_t * std,
                     "ci95_hi_ms": mean + half_t * std, "logged_at": m["t"], "source": "log"}
    return list(rows.values())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True, help="run log with one line per timed cell")
    p.add_argument("--partial", type=Path, required=True, help="partial JSONL with the meta and validation rows")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--expect", type=int, default=None, help="expected number of timing rows (sanity check)")
    args = p.parse_args(argv)

    partial = [json.loads(x) for x in args.partial.read_text().splitlines() if x.strip()]
    metas = [r for r in partial if r.get("kind") == "meta"]
    meta = dict(metas[-1]) if metas else {}
    run_args = meta.get("args", {})
    n, batch = int(run_args.get("repeats", 100)), int(run_args.get("batch", 1))
    val = [r for r in partial if r.get("kind") == "validation"]
    timing = parse_log(args.log, n, batch)
    if args.expect is not None and len(timing) != args.expect:
        raise SystemExit(f"parsed {len(timing)} timing rows, expected {args.expect}")
    meta["reconstructed_from_log"] = {
        "log": args.log.name, "partial_jsonl": args.partial.name, "n_timing_rows": len(timing),
        "reason": "JSONL rows were written to a file replaced by a concurrent git checkout; the per-cell log "
                  "lines are the surviving record. Median, min and GPU state per timing are not recoverable."}

    out_jsonl = args.out_dir / f"e1_{args.tag}.jsonl"
    with out_jsonl.open("w") as f:
        for r in [meta] + val + timing:
            f.write(json.dumps(r) + "\n")
    write_json(args.out_dir / f"e1_{args.tag}.json", {
        "experiment": "E1", "meta": meta, "validation": val,
        "all_validation_passed": all(v["passed"] for v in val) if val else None,
        "summary": summarize_rows(timing), "rows": timing})
    print(f"wrote {out_jsonl} and e1_{args.tag}.json: {len(timing)} timing rows, {len(val)} validation rows")


if __name__ == "__main__":
    main()
