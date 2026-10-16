"""Whole-step CUDA-graph results (E32) weighted by the prompt lengths of public serving traces (taco-r2 F1; CPU only).

E32 times whole prefill steps at batch 1 for L = 256, 512, 1000, 1024, 2048, 3000, 4096, 6000 and 8192 tokens. A
request-level serving workload runs one prefill per prompt, so its step mix follows the trace's prompt-length
distribution. Each power-of-two prompt-length bucket of a trace window (as stored by the serving replay) is
represented by the E32 batch-1 shapes whose length falls in that bucket (prompts shorter than 256 tokens by the
256-token step, longer than 8192 by the 8192-token step; the share of such remapped requests is reported). For each
model and session the script reports the gain of the held-out shape table and of the in-session best chunk over the
library default, (a) latency-weighted over the trace's requests and (b) as the mean over requests of the per-step
gain.

Output: ``taco_r2_e32_weighted.json`` in the results directory.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict

from src.experiments.tpds.common import RESULTS_DIR

LABELS = ("ssd370m", "ssd2p7b", "gdn1p4b")
SESSIONS = ("4c_e32s1", "4c_e32s2")


def bucket(L: int) -> int:
    return 1 << int(math.floor(math.log2(L)))


def main():
    churn = json.loads((RESULTS_DIR / "serving_key_churn.json").read_text())
    windows = {f"{tr}|{w}": v["prompt_len_pow2_bucket_hist"] for tr, t in churn["traces"].items()
               for w, v in t["windows"].items() if v.get("prompt_len_pow2_bucket_hist")}
    res = {"windows": list(windows), "models": {}}
    for lab in LABELS:
        d = json.loads((RESULTS_DIR / f"e32_model_graph_{SESSIONS[0]}_{lab}_{SESSIONS[1]}_{lab}.json").read_text())
        out = {}
        for ses in SESSIONS:
            cells = [c for c in d["cells"] if c["mode"] == "graph" and c["batch"] == 1 and c["session"] == f"{ses}_{lab}"]
            by_b = defaultdict(list)
            for c in cells:
                dflt = c["median_ms"][c["default"]]
                tab = c["median_ms"].get(f"c{c['table_chunk']}", dflt)
                best = c["median_ms"][c["best"]]
                by_b[bucket(c["L"])].append((dflt, tab, best))
            rep = {b: tuple(statistics.fmean(x[i] for x in v) for i in range(3)) for b, v in by_b.items()}
            lo, hi = min(rep), max(rep)
            per_w = {}
            for wname, hist in windows.items():
                num_t = num_b = den = 0.0
                mean_t = mean_b = n = 0.0
                remapped = 0
                for bs, cnt in hist.items():
                    b = int(bs)
                    tgt = min(max(b, lo), hi)
                    if tgt not in rep:
                        tgt = min(rep, key=lambda x: abs(math.log2(x) - math.log2(tgt)))
                    if tgt != b:
                        remapped += cnt
                    dflt, tab, best = rep[tgt]
                    num_t += cnt * (dflt - tab)
                    num_b += cnt * (dflt - best)
                    den += cnt * dflt
                    mean_t += cnt * (dflt - tab) / dflt
                    mean_b += cnt * (dflt - best) / dflt
                    n += cnt
                per_w[wname] = {"table_gain_lw_pct": round(100 * num_t / den, 3),
                                "best_gain_lw_pct": round(100 * num_b / den, 3),
                                "table_gain_mean_pct": round(100 * mean_t / n, 3),
                                "best_gain_mean_pct": round(100 * mean_b / n, 3),
                                "share_requests_remapped": round(remapped / n, 3)}
            vals_t = [v["table_gain_lw_pct"] for v in per_w.values()]
            vals_m = [v["table_gain_mean_pct"] for v in per_w.values()]
            vals_b = [v["best_gain_mean_pct"] for v in per_w.values()]
            out[ses] = {"per_window": per_w,
                        "table_gain_lw_range_pct": [min(vals_t), max(vals_t)],
                        "table_gain_mean_range_pct": [min(vals_m), max(vals_m)],
                        "best_gain_mean_range_pct": [min(vals_b), max(vals_b)]}
            print(lab, ses, "table LW", out[ses]["table_gain_lw_range_pct"], "table mean", out[ses]["table_gain_mean_range_pct"],
                  "best mean", out[ses]["best_gain_mean_range_pct"])
        res["models"][lab] = out
    (RESULTS_DIR / "taco_r2_e32_weighted.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
