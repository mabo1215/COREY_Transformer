"""taco-r2 items M4a, M4b and M4c: a-priori selection protocols, calibration-only selectors and fit documentation
of the Triton Mamba-1 cost model (Eq. (3)). CPU only; reads stored result files and times nothing.

Fits and selectors are produced by the functions of ``taco_model_baselines`` (``m4c_tag``: the 13 candidate forms
with their leave-one-calibration-shape-out (LOSO) refits; ``m5a_tag``: every calibration-only ranker including
the learned regressors over seeds 0-4 and the zero-parameter compile-facts rule; ``m3c_tag``: fit variants,
multi-starts and the 12-key bootstrap), which reproduce the stored round-4 analyses. This script adds what the
taco-r2 review asks for and the stored analyses do not give:

M4a  the six a-priori protocols (candidate sets ablation12 and ablation12+work_only x criteria LOGO on the other
     GPUs' captured calls, LOGO on the other GPUs' calibration LOSO, target-calibration LOSO; stored tie rule:
     fewest free parameters among forms within 0.1 points of the best criterion, then the lowest criterion, then
     list order), each with per-call AND latency-weighted regret, p95, maximum, per-batch and aligned/unaligned
     values and 95% prompt-cluster intervals, evaluated (i) on the data set of the other Table-5 rows (RTX PRO 6000
     and RTX 3090: round medians of three rounds; H800: one block) and (ii) on the calibration tag's own table
     (the stored analyses: RTX PRO 6000 pooled round means, RTX 3090 single-block session); the candidate forms
     with their three criterion values and target regrets per fold (target GPU); the exact-tie alternatives; a
     deterministic a-priori tie-break by the LOSO held-out log-RMSE; and one pre-specified protocol.
M4b  every calibration-only selector on the same data set per GPU (top-1, top-2, top-3; per call and
     latency-weighted; per batch; intervals), MLPs as five-seed mean, s.d. and range, GPU mean and worst, and
     paired differences of the eight-parameter forms against the three-parameter restriction.
M4c  fit documentation: refits on full-protocol timings only, inverse-variance weighted, without the first-key
     search timings, OLS, the best of 50 random starts, the 72 static re-measurements only and all timings of
     the six default-table configurations only; for each the per-call and latency-weighted regret, R^2 of log
     latency and R^2 of the minimized objective; the search-versus-full-protocol discrepancy; the timing sources;
     the 12-key bootstrap and leave-one-key-out regrets; ordering and guard metadata of the calibration sweep.

Outputs: ``taco_r2_baselines_m4.json`` (all sections plus the per-call choices of every form and selector, read
by ``taco_r2_untuned_baselines``).

Usage:  PYTHONPATH=. python3 -m src.experiments.tpds.taco_r2_cost_model_protocols [--jobs 6] [--n-boot 2000]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from src.experiments.tpds import cost_model as cm  # noqa: E402
from src.experiments.tpds import cost_model_baselines_r4 as r4  # noqa: E402
from src.experiments.tpds import taco_model_baselines as TMB  # noqa: E402
from src.experiments.tpds import taco_r2_baselines_common as C  # noqa: E402
from src.experiments.tpds.common import RESULTS_DIR, read_jsonl  # noqa: E402

TAGS = TMB.TAGS
# evaluation data sets per convention and GPU: first = data set of the other Table-5 rows, second = the
# calibration tag's own table (the data set of the stored cost-model analyses), when different
EVAL_SETS = {"device_time": {"pro6000": ("pro6000_device", "pro6000_device_pooled"),
                             "vgpu3090": ("vgpu3090_device", "vgpu3090_device_one_block"),
                             "h800": ("h800_device",)},
             "launch_inclusive": {"pro6000": ("pro6000_launch",), "vgpu3090": ("vgpu3090_launch",),
                                  "h800": ("h800_launch",)}}
CRITERIA = ("logo_captured_calls", "logo_calibration_loso", "target_calibration_loso")
PRESPECIFIED = ("ablation12+work_only", "target_calibration_loso", "rmse_tiebreak")
PAPER_PROTOCOL = ("ablation12+work_only", "logo_captured_calls", "stored")
RANKERS = ("cost_model", "work_only", "work_term_model", "nearest_calibrated_key", "loglinear_interp_logL_logB",
           "length_tree_calibration", "shape_tree_calibration", "gbr_basic", "gbr_structural", "compile_facts_rule",
           "tfla_runtime")
MLP = ("mlp_basic", "mlp_structural")
QUOTED = {  # review values (taco-r2 file, M4a/M4b/M4c), checked below
    "M4a_device_pro6000": [2.24, 0.38, 0.56, 4.16, 0.38, 0.56], "M4a_device_vgpu3090": [6.23, 2.49, 1.74, 6.23, 2.49, 0.88],
    "M4c_full_protocol_only": {"pro6000": 0.73, "vgpu3090": 4.28, "h800": 6.05},
    "M4c_objective_r2": {"pro6000": 0.89, "vgpu3090": 0.93, "h800": 0.94},
    "M4c_static72": {"pro6000": 0.56, "vgpu3090": 1.8, "h800": 6.1},
    "M4c_leave_one_key_out": {"pro6000": 0.8, "vgpu3090": 7.4, "h800": 3.7},
    "M4b_apriori": {"pro6000": 4.16, "vgpu3090": 6.23, "h800": 3.01},
    "M4b_work_only": {"pro6000": 4.20, "vgpu3090": 0.85, "h800": 3.53},
    "M4b_mlp_basic_5seeds": {"pro6000": 1.59, "vgpu3090": 1.31, "h800": 7.13},
    "M4b_mlp_structural_5seeds": {"pro6000": 3.75, "vgpu3090": 4.90, "h800": 1.11},
}


# =========================================================================== per-tag worker
def _theta(params: dict) -> np.ndarray:
    th = np.array([cm.DEFAULT_FIXED.get(i, 0.0) for i in range(len(cm.PARAMS))], float)
    for i, k in enumerate(cm.PARAMS):
        if k in params:
            th[i] = float(params[k])
    return th


def tag_worker(job) -> dict:
    tag, in_dir, opts = job
    t0 = time.time()
    ctx = r4.Ctx(tag, Path(in_dir))
    F, Fc = TMB.farr(ctx.f), TMB.farr(ctx.calls.fc)
    stored = json.loads((Path(in_dir) / f"cost_model_baselines_r4_{tag}.json").read_text())
    S0 = ctx.calls.matrix(TMB.predict_np(ctx.theta, Fc))
    stored_choice = S0.argmin(1)
    out = {"tag": tag, "keys": [str(k) for k in ctx.df["key"]], "chunks": [int(c) for c in ctx.chunks]}
    # --- M4a: the 13 forms with LOSO criteria
    m4c, choices = TMB.m4c_tag(ctx, F, Fc, stored)
    form_scores = {}
    for n, rec in m4c["forms"].items():
        th = np.array([rec["params"][k] for k in cm.PARAMS], float)
        form_scores[n] = ctx.calls.matrix(TMB.predict_np(th, Fc))
        if not np.array_equal(form_scores[n].argmin(1), choices[n]):
            raise RuntimeError(f"{tag}: form {n} scores and choice disagree")
    out["forms"] = {n: {k: v for k, v in r.items() if k != "params"} | {"params": r["params"]} for n, r in m4c["forms"].items()}
    out["form_checks"] = m4c["checks"]
    out["form_scores"] = form_scores
    # --- M4b: every calibration-only ranker (learned over seeds 0..mlp_seeds-1)
    m5a, mats, rankers = TMB.m5a_tag(ctx, stored, opts)
    out["ranker_scores"] = mats
    out["m5a_checks"] = m5a["checks"]
    out["m5a_learned"] = m5a["learned"]
    out["compile_facts_rule_definition"] = m5a["compile_facts_rule_definition"]
    # --- M4c: fit documentation
    m3c = TMB.m3c_tag(ctx, F, Fc, opts, stored_choice)
    src = ctx.points["src"].to_numpy()
    BD, nw, ns, Cc = (ctx.points[k].to_numpy() for k in ("BD", "nw", "ns", "C"))
    dflt = np.array([(int(b), int(w)) == cm.DEFAULT_LAUNCH[int(c)] and int(s) == 1 for b, w, s, c in zip(BD, nw, ns, Cc)])
    masks = {"stored_soft_l1": np.ones(len(src), bool), "full_protocol_only": np.isin(src, ["e4_static", "e1"]),
             "static72_only": src == "e4_static", "default_table_configurations_only": dflt}
    first = read_jsonl(Path(in_dir) / f"e4_{tag}.jsonl", kind="m1tune")[0]["key"]
    masks["exclude_first_key_search"] = ~((src == "e4_search") & (ctx.points["key"].to_numpy() == first))
    variants = {}
    for n, m in masks.items():
        if n == "stored_soft_l1":
            th, info = ctx.theta.copy(), {"cost": None}
        else:
            th, info = TMB.fit_fast(TMB.sub(F, m), ctx.y[m])
        variants[n] = {"theta": th, "mask": m, "loss": "soft_l1", "weights": None}
    w, winfo = TMB.inverse_variance_weights(ctx)
    th, _ = TMB.fit_fast(F, ctx.y, weights=w)
    variants["inverse_variance_weighted"] = {"theta": th, "mask": masks["stored_soft_l1"], "loss": "soft_l1",
                                             "weights": w, "weights_info": winfo}
    th, _ = TMB.fit_fast(F, ctx.y, loss="linear")
    variants["ols_all"] = {"theta": th, "mask": masks["stored_soft_l1"], "loss": "linear", "weights": None}
    variants["multistart_best_of_50"] = {"theta": _theta(m3c["multistart"]["best_start_params"]),
                                         "mask": masks["stored_soft_l1"], "loss": "soft_l1", "weights": None}
    # random multi-starts of the small refits (same start distribution as taco_model_baselines.m3c_tag, seed 0):
    # the 72- to 162-point fits are poorly determined, so the three-restart protocol can stop in different optima
    rng = np.random.default_rng(TMB.SEED)
    free = [i for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED]
    starts = []
    for _ in range(opts.n_multistart):
        x = cm.X0.copy()
        for i in free:
            if i < cm.N_SCALE:
                x[i] = cm.X0[i] * 10.0 ** rng.uniform(-2, 2)
            elif cm.PARAMS[i] == "W0":
                x[i] = rng.uniform(1.0, 64.0)
            else:
                x[i] = rng.uniform(1.0, 256.0)
        starts.append(np.clip(x, cm.LO, cm.HI))
    small_ms = {}
    for n in ("full_protocol_only", "static72_only", "default_table_configurations_only"):
        m = variants[n]["mask"]
        Fm, ym = TMB.sub(F, m), ctx.y[m]
        base_cost = TMB.loss_cost(np.log(TMB.predict_np(variants[n]["theta"], Fm)) - np.log(ym))
        runs = []
        for x in starts:
            th, cost = TMB.fit_single(Fm, ym, x)
            ch = ctx.calls.matrix(TMB.predict_np(th, Fc)).argmin(1)
            runs.append((cost, float(100 * TMB.call_regret(ctx.lat, ch).mean()), th))
        costs = np.array([c for c, _, _ in runs])
        regs = np.array([g for _, g, _ in runs])
        b = int(np.argmin(costs))
        near = costs <= costs[b] * 1.01
        small_ms[n] = {"n_starts": len(runs), "three_restart_protocol_cost": base_cost,
                       "best_cost": float(costs[b]),
                       "best_cost_rel_improvement_over_protocol": float(1 - costs[b] / base_cost),
                       "best_start_own_table_mean_regret_pct": float(regs[b]),
                       "own_table_regret_among_starts_within_1pct_of_best_cost": {
                           "n": int(near.sum()), "min": float(regs[near].min()), "median": float(np.median(regs[near])),
                           "max": float(regs[near].max())}}
        if costs[b] < base_cost * (1 - 1e-9):
            variants[n + "_best_of_50_starts"] = {"theta": runs[b][2], "mask": m, "loss": "soft_l1", "weights": None}
    out["small_refit_multistart"] = small_ms
    var_out, var_scores = {}, {}
    for n, v in variants.items():
        S = ctx.calls.matrix(TMB.predict_np(v["theta"], Fc))
        var_scores[n] = S
        m = v["mask"]
        Fm = TMB.sub(F, m)
        wm = None if v["weights"] is None else v["weights"][m]
        var_out[n] = {"n_fit_points": int(m.sum()),
                      "n_fit_points_by_source": {s: int(((src == s) & m).sum()) for s in ("e4_search", "e4_static", "e1")},
                      "params": {cm.PARAMS[i]: float(v["theta"][i]) for i in range(len(cm.PARAMS)) if i not in cm.DEFAULT_FIXED},
                      "loss": v["loss"], "weighted": v["weights"] is not None,
                      "r2_log_on_fit_points": cm.fit_quality(TMB.predict_np(v["theta"], Fm), ctx.y[m])["r2_log"],
                      "r2_log_on_all_2298": cm.fit_quality(TMB.predict_np(v["theta"], F), ctx.y)["r2_log"],
                      "objective_r2_on_fit_points": TMB.objective_r2(v["theta"], Fm, ctx.y[m], v["loss"], wm)["r2_objective"],
                      "share_calls_same_choice_as_stored_fit": float((S.argmin(1) == stored_choice).mean())}
        if "weights_info" in v:
            var_out[n]["weights_info"] = v["weights_info"]
    out["fit_variants"] = var_out
    out["fit_variant_scores"] = var_scores
    out["m3c"] = {k: m3c[k] for k in ("multistart", "cluster_bootstrap", "objective_r2_stored_fit",
                                      "search_vs_full_protocol", "summary_captured_mean_regret_pct")}
    out["m3c"]["cluster_bootstrap"] = {k: v for k, v in m3c["cluster_bootstrap"].items() if k != "replicate_mean_regret_pct"}
    out["m3c"]["multistart"] = {k: v for k, v in m3c["multistart"].items()}
    out["m3c_variant_regret_pct"] = {k: m3c[k]["captured_calls"]["mean_regret_pct"]
                                     for k in ("stored_soft_l1", "full_protocol_only", "inverse_variance_weighted",
                                               "exclude_first_key_search", "ols_all")}
    meta_args = {}
    for r in read_jsonl(Path(in_dir) / f"e4_{tag}.jsonl", kind="meta"):
        meta_args.update(r.get("args", {}))
    m1 = read_jsonl(Path(in_dir) / f"e4_{tag}.jsonl", kind="m1tune")
    orders = {tuple((s["block_t"], s["block_d"], s["num_warps"], s["num_stages"]) for s in r["search"]) for r in m1}
    out["calibration_sweep_metadata"] = {
        "key_order_in_file": [r["key"].split("|", 1)[1] for r in m1],
        "identical_configuration_order_for_every_key": len(orders) == 1,
        "search_warmup": meta_args.get("search_warmup"), "search_repeats": meta_args.get("search_repeats"),
        "static_warmup": meta_args.get("warmup"), "static_repeats": meta_args.get("repeats"),
        "exclusive_guard": bool(meta_args.get("exclusive_guard", False)),
        "shuffle_or_seed_argument_present": any(("shuffle" in k or "seed" in k) for k in meta_args)}
    out["n_points_by_source"] = {s: int((src == s).sum()) for s in ("e4_search", "e4_static", "e1")}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =========================================================================== protocols (M4a)
def select(cands, crit, nfree, rmse, rule: str):
    """Stored rule (``taco_model_baselines.select_tie_rule``) or the same with exact ties broken by the LOSO
    held-out log-RMSE (``rmse_tiebreak``)."""
    sel, alts, within = TMB.select_tie_rule(list(cands), crit, nfree)
    if rule == "rmse_tiebreak" and alts:
        pool = [sel, *alts]
        sel = min(pool, key=lambda n: (rmse[n], list(cands).index(n)))
        alts = [a for a in pool if a != sel]
    return sel, alts, within


def protocol_tables(per_tag: dict, conv: str) -> dict:
    tags = TAGS[conv]
    forms = {g: per_tag[t]["forms"] for g, t in tags.items()}
    nfree = {n: r["n_free_params"] for n, r in next(iter(forms.values())).items()}
    out = {}
    for gpu in tags:
        srcs = [g for g in tags if g != gpu]
        for set_name, cands in r4.CANDIDATE_SETS.items():
            crits = {"logo_captured_calls": {n: float(np.mean([forms[s][n]["captured_calls"]["mean_regret_pct"] for s in srcs]))
                                             for n in cands},
                     "logo_calibration_loso": {n: float(np.mean([forms[s][n]["loso"]["mean_chunk_regret_pct"] for s in srcs]))
                                               for n in cands},
                     "target_calibration_loso": {n: forms[gpu][n]["loso"]["mean_chunk_regret_pct"] for n in cands}}
            rmse = {"logo_captured_calls": {n: float(np.mean([forms[s][n]["loso"]["heldout_rmse_log_mean"] for s in srcs]))
                                            for n in cands},
                    "logo_calibration_loso": {n: float(np.mean([forms[s][n]["loso"]["heldout_rmse_log_mean"] for s in srcs]))
                                              for n in cands},
                    "target_calibration_loso": {n: forms[gpu][n]["loso"]["heldout_rmse_log_mean"] for n in cands}}
            for cname in CRITERIA:
                for rule in ("stored", "rmse_tiebreak"):
                    sel, alts, within = select(cands, crits[cname], nfree, rmse[cname], rule)
                    out[(gpu, set_name, cname, rule)] = {
                        "selected_form": sel, "n_free_params": nfree[sel], "criterion_value": crits[cname][sel],
                        "forms_within_0p1": within, "exact_tie_alternatives": alts,
                        "candidate_table": {n: {"n_free_params": nfree[n], **{c: crits[c][n] for c in CRITERIA},
                                                "loso_heldout_rmse_log": rmse[cname][n],
                                                "target_captured_mean_regret_pct": forms[gpu][n]["captured_calls"]["mean_regret_pct"]}
                                            for n in cands}}
    return out


def eval_tables(conv: str, in_dir: Path) -> dict:
    return {gpu: {nm: C.triton_table(nm, in_dir) for nm in names} for gpu, names in EVAL_SETS[conv].items()}


def remap_scores(T: C.CallTable, keys: list[str], S: np.ndarray) -> np.ndarray:
    pos = {k: i for i, k in enumerate(keys)}
    return S[np.array([pos[k] for k in T.keys()])]


def split_records(T: C.CallTable, pick: np.ndarray) -> dict:
    out = {}
    for nm, m in (("aligned", T.aligned == 1), ("unaligned", T.aligned == 0)):
        s = C.stats_of(T, pick, sel=m)
        out[nm] = {k: s[k] for k in ("n", "mean_regret_pct", "p95_regret_pct", "lw_share_of_oracle_pct")}
    return out


def m4a(per_tag: dict, convs: list[str], in_dir: Path, n_boot: int, checks: C.Checks) -> tuple[dict, dict]:
    res, choices = {}, {}
    for conv in convs:
        tabs = protocol_tables(per_tag, conv)
        evs = eval_tables(conv, in_dir)
        conv_out = {}
        for gpu, tag in TAGS[conv].items():
            pt = per_tag[tag]
            gpu_out = {"calibration_tag": tag, "protocols": {}, "frozen_full_form": {}}
            picks_by_set = {nm: {} for nm in evs[gpu]}
            labels = {}
            for (g, set_name, cname, rule), sel in tabs.items():
                if g != gpu:
                    continue
                pname = f"{set_name}|{cname}|{rule}"
                labels[pname] = sel["selected_form"]
                rec = {k: v for k, v in sel.items() if k != "candidate_table"}
                rec["candidate_table"] = sel["candidate_table"]
                rec["exact_tie_alternative_target_regret_pct"] = {
                    a: pt["forms"][a]["captured_calls"]["mean_regret_pct"] for a in sel["exact_tie_alternatives"]}
                gpu_out["protocols"][pname] = rec
            for nm, T in evs[gpu].items():
                pk = {f"form:{n}": remap_scores(T, pt["keys"], pt["form_scores"][n]).argmin(1) for n in pt["forms"]}
                picks_by_set[nm] = pk
            # evaluate every form once per data set (protocols share forms)
            for nm, T in evs[gpu].items():
                used = sorted(set(labels.values()) | {"full", "work_only"} |
                              {a for p in gpu_out["protocols"].values() for a in p["exact_tie_alternatives"]})
                picks = {f"form:{n}": picks_by_set[nm][f"form:{n}"] for n in used}
                pairs = [("form:full", "form:work_only")] + [(f"form:{n}", "form:work_only") for n in used
                                                               if n not in ("full", "work_only")]
                recs, paired, kcl = C.evaluate(T, picks, nm, n_boot, pairs)
                for pname, n in labels.items():
                    gpu_out["protocols"][pname].setdefault("evaluation", {})[nm] = {
                        **C.compact(recs[f"form:{n}"]), "aligned_unaligned": split_records(T, picks[f"form:{n}"])}
                    for a in gpu_out["protocols"][pname]["exact_tie_alternatives"]:
                        gpu_out["protocols"][pname].setdefault("exact_tie_alternatives_evaluation", {}).setdefault(nm, {})[a] = \
                            C.compact(recs[f"form:{a}"], by_batch=False)
                gpu_out["frozen_full_form"][nm] = {**C.compact(recs["form:full"]),
                                                   "aligned_unaligned": split_records(T, picks["form:full"])}
                gpu_out.setdefault("paired_vs_work_only", {})[nm] = paired["all_calls"]
                gpu_out.setdefault("n_prompt_clusters", {})[nm] = kcl
                choices[(conv, gpu, nm)] = {n: [T.chunks[j] for j in picks_by_set[nm][f"form:{n}"]] for n in pt["forms"]}
                choices[(conv, gpu, nm, "keys")] = T.keys()
            conv_out[gpu] = gpu_out
        # protocol range per GPU and summary table (primary data set)
        summ = {}
        for gpu in TAGS[conv]:
            prim = EVAL_SETS[conv][gpu][0]
            own = EVAL_SETS[conv][gpu][-1]
            rows = {}
            for pname, p in conv_out[gpu]["protocols"].items():
                ev, evo = p["evaluation"][prim], p["evaluation"][own]
                rows[pname] = {"form": p["selected_form"], "n_free_params": p["n_free_params"],
                               "per_call_mean_pct": ev["mean_regret_pct"], "per_call_mean_ci95": ev["mean_regret_pct_ci95"],
                               "lw_share_of_oracle_pct": ev["lw_share_of_oracle_pct"],
                               "lw_ci95": ev["lw_share_of_oracle_pct_ci95"], "p95_pct": ev["p95_regret_pct"],
                               "max_pct": ev["max_regret_pct"],
                               "per_batch_mean_pct": {b: v["mean_regret_pct"] for b, v in ev["by_batch"].items()},
                               "per_batch_lw_pct": {b: v["lw_share_of_oracle_pct"] for b, v in ev["by_batch"].items()},
                               "aligned_unaligned_mean_pct": [ev["aligned_unaligned"]["aligned"]["mean_regret_pct"],
                                                              ev["aligned_unaligned"]["unaligned"]["mean_regret_pct"]],
                               "own_table_per_call_mean_pct": evo["mean_regret_pct"],
                               "own_table_lw_share_of_oracle_pct": evo["lw_share_of_oracle_pct"],
                               "exact_tie_alternatives": {a: {"own_table_per_call_mean_pct": p["exact_tie_alternative_target_regret_pct"][a],
                                                              "primary_per_call_mean_pct": p["exact_tie_alternatives_evaluation"][prim][a]["mean_regret_pct"],
                                                              "primary_lw_pct": p["exact_tie_alternatives_evaluation"][prim][a]["lw_share_of_oracle_pct"]}
                                                          for a in p["exact_tie_alternatives"]}}
            st = [rows[f"{s}|{c}|stored"] for s in r4.CANDIDATE_SETS for c in CRITERIA]
            summ[gpu] = {"primary_data_set": prim, "own_table": own, "rows": rows,
                         "six_protocols_stored_tie_rule_order": [f"{s}|{c}" for s in r4.CANDIDATE_SETS for c in CRITERIA],
                         "six_protocols_per_call_mean_pct_primary": [x["per_call_mean_pct"] for x in st],
                         "six_protocols_lw_pct_primary": [x["lw_share_of_oracle_pct"] for x in st],
                         "six_protocols_per_call_mean_pct_own_table": [x["own_table_per_call_mean_pct"] for x in st],
                         "six_protocols_lw_pct_own_table": [x["own_table_lw_share_of_oracle_pct"] for x in st],
                         "range_per_call_primary": [min(x["per_call_mean_pct"] for x in st), max(x["per_call_mean_pct"] for x in st)],
                         "range_lw_primary": [min(x["lw_share_of_oracle_pct"] for x in st), max(x["lw_share_of_oracle_pct"] for x in st)],
                         "frozen_full_form_primary": {k: conv_out[gpu]["frozen_full_form"][prim][k]
                                                      for k in ("mean_regret_pct", "mean_regret_pct_ci95", "lw_share_of_oracle_pct",
                                                                "lw_share_of_oracle_pct_ci95", "p95_regret_pct", "max_regret_pct")},
                         "frozen_full_form_primary_by_batch": {b: {"mean": v["mean_regret_pct"], "lw": v["lw_share_of_oracle_pct"]}
                                                                for b, v in conv_out[gpu]["frozen_full_form"][prim]["by_batch"].items()},
                         "frozen_full_form_primary_aligned_unaligned": conv_out[gpu]["frozen_full_form"][prim]["aligned_unaligned"],
                         "frozen_full_form_own_table": {k: conv_out[gpu]["frozen_full_form"][own][k]
                                                        for k in ("mean_regret_pct", "lw_share_of_oracle_pct")},
                         "frozen_full_form_own_table_by_batch": {b: v["mean_regret_pct"]
                                                                 for b, v in conv_out[gpu]["frozen_full_form"][own]["by_batch"].items()},
                         "frozen_full_form_own_table_aligned_unaligned": conv_out[gpu]["frozen_full_form"][own]["aligned_unaligned"]}
            if conv == "device_time" and f"M4a_device_{gpu}" in QUOTED:
                for i, (q, v) in enumerate(zip(QUOTED[f"M4a_device_{gpu}"], summ[gpu]["six_protocols_per_call_mean_pct_own_table"])):
                    checks.add(f"M4a device {gpu} protocol {i} (own table)", v, q, "review M4a", 0.006)
        res[conv] = {"per_gpu": conv_out, "summary": summ}
    return res, choices


# =========================================================================== M4b
def m4b(per_tag: dict, in_dir: Path, n_boot: int, a_priori: dict, checks: C.Checks, mlp_seeds: int) -> dict:
    conv = "device_time"
    evs = eval_tables(conv, in_dir)
    out = {"per_gpu": {}, "gpu_mean_worst": {}}
    top1_primary = {}
    for gpu, tag in TAGS[conv].items():
        pt = per_tag[tag]
        gout = {}
        for nm, T in evs[gpu].items():
            mats = {n: remap_scores(T, pt["keys"], pt["ranker_scores"][n]) for n in pt["ranker_scores"]}
            for key, form in a_priori[gpu].items():
                mats[key] = remap_scores(T, pt["keys"], pt["form_scores"][form])
            names = [n for n in mats if not n.startswith("tfla_runtime_")]
            picks = {n: mats[n].argmin(1) for n in names}
            pairs = [("cost_model", "work_only")] + [(k, "work_only") for k in a_priori[gpu]] + \
                    [("cost_model", "mlp_basic"), ("cost_model", "mlp_structural"), ("cost_model", "compile_facts_rule")]
            recs, paired, _ = C.evaluate(T, picks, nm, n_boot, pairs)
            rows = {}
            for n in names:
                rows[n] = {"top1": C.compact(recs[n])}
                for k in (2, 3):
                    pk = r4.topk_picks(T.df, T.chunks, mats[n], k)
                    s = C.stats_of(T, pk)
                    rows[n][f"top{k}"] = {f: s[f] for f in ("mean_regret_pct", "p95_regret_pct", "max_regret_pct",
                                                            "within_2pct_share", "lw_share_of_oracle_pct", "us_per_call")}
            # MLP seed summaries
            for base in MLP:
                seeds = [base] + [f"{base}_seed{s}" for s in range(1, mlp_seeds)]
                summ = {}
                for k in ("top1", "top2", "top3"):
                    for f in ("mean_regret_pct", "lw_share_of_oracle_pct", "p95_regret_pct", "max_regret_pct"):
                        v = np.array([rows[s][k][f] for s in seeds])
                        summ[f"{k}_{f}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                                            "max": float(v.max()), "per_seed": [float(x) for x in v]}
                rows[f"{base}_over_{mlp_seeds}_seeds"] = summ
            gout[nm] = {"rows": rows, "paired": paired["all_calls"]}
            if nm == EVAL_SETS[conv][gpu][0]:
                top1_primary[gpu] = rows
        out["per_gpu"][gpu] = gout
    # GPU mean and worst (primary data sets, top-1)
    sel_names = [*RANKERS, *a_priori[next(iter(a_priori))].keys()]
    tab = {}
    for n in sel_names:
        pc = [top1_primary[g][n]["top1"]["mean_regret_pct"] for g in TAGS[conv]]
        lw = [top1_primary[g][n]["top1"]["lw_share_of_oracle_pct"] for g in TAGS[conv]]
        tab[n] = {"per_call": pc, "per_call_gpu_mean": float(np.mean(pc)), "per_call_worst": float(np.max(pc)),
                  "lw": lw, "lw_gpu_mean": float(np.mean(lw)), "lw_worst": float(np.max(lw)),
                  "top2_per_call": [top1_primary[g][n]["top2"]["mean_regret_pct"] for g in TAGS[conv]],
                  "top3_per_call": [top1_primary[g][n]["top3"]["mean_regret_pct"] for g in TAGS[conv]]}
    for base in MLP:
        k = f"{base}_over_{mlp_seeds}_seeds"
        pc = [top1_primary[g][k]["top1_mean_regret_pct"]["mean"] for g in TAGS[conv]]
        lw = [top1_primary[g][k]["top1_lw_share_of_oracle_pct"]["mean"] for g in TAGS[conv]]
        tab[k] = {"per_call": pc, "per_call_gpu_mean": float(np.mean(pc)), "per_call_worst": float(np.max(pc)),
                  "per_call_sd_over_seeds": [top1_primary[g][k]["top1_mean_regret_pct"]["sd"] for g in TAGS[conv]],
                  "per_call_seed_range": [[top1_primary[g][k]["top1_mean_regret_pct"]["min"],
                                           top1_primary[g][k]["top1_mean_regret_pct"]["max"]] for g in TAGS[conv]],
                  "lw": lw, "lw_gpu_mean": float(np.mean(lw)), "lw_worst": float(np.max(lw)),
                  "top2_per_call": [top1_primary[g][k]["top2_mean_regret_pct"]["mean"] for g in TAGS[conv]],
                  "top3_per_call": [top1_primary[g][k]["top3_mean_regret_pct"]["mean"] for g in TAGS[conv]]}
        tab[f"{base}_seed0"] = {"per_call": [top1_primary[g][base]["top1"]["mean_regret_pct"] for g in TAGS[conv]]}
    out["gpu_mean_worst"] = {"gpus": list(TAGS[conv]), "data_sets": [EVAL_SETS[conv][g][0] for g in TAGS[conv]],
                             "table": tab}
    # own-table (stored-analysis) view of the same rows, for the reproduction of quoted values
    own = {}
    for gpu in TAGS[conv]:
        nm = EVAL_SETS[conv][gpu][-1]
        rows = out["per_gpu"][gpu][nm]["rows"]
        own[gpu] = {n: rows[n]["top1"]["mean_regret_pct"] for n in sel_names}
        for base in MLP:
            own[gpu][f"{base}_over_{mlp_seeds}_seeds"] = rows[f"{base}_over_{mlp_seeds}_seeds"]["top1_mean_regret_pct"]["mean"]
    out["own_table_top1_per_call"] = own
    for gpu in TAGS[conv]:
        prim = top1_primary[gpu]
        checks.add(f"M4b work_only {gpu} (primary)", prim["work_only"]["top1"]["mean_regret_pct"], QUOTED["M4b_work_only"][gpu],
                   "review M4b / Table 5", 0.006)
        checks.add(f"M4b a priori (paper protocol) {gpu} (own table)", own[gpu]["a_priori_paper_protocol"],
                   QUOTED["M4b_apriori"][gpu], "review M4b / Table 5", 0.006)
        for base in MLP:
            qk = "M4b_mlp_basic_5seeds" if base == "mlp_basic" else "M4b_mlp_structural_5seeds"
            v = own[gpu][f"{base}_over_{mlp_seeds}_seeds"]
            checks.add(f"M4b {base} five-seed mean {gpu} (own table)", v, QUOTED[qk][gpu], "review M4b", 0.006)
    return out


# =========================================================================== M4c
def m4c(per_tag: dict, convs: list[str], in_dir: Path, n_boot: int, checks: C.Checks) -> dict:
    out = {}
    stored_prev = json.loads((in_dir / "taco_model_baselines.json").read_text())["per_tag"]
    for conv in convs:
        evs = eval_tables(conv, in_dir)
        cout = {}
        for gpu, tag in TAGS[conv].items():
            pt = per_tag[tag]
            prim = EVAL_SETS[conv][gpu][0]
            own = EVAL_SETS[conv][gpu][-1]
            rec = {"calibration_tag": tag, "n_points_by_source": pt["n_points_by_source"],
                   "calibration_sweep_metadata": pt["calibration_sweep_metadata"], "variants": {}}
            for nm in (prim, own):
                T = evs[gpu][nm]
                picks = {n: remap_scores(T, pt["keys"], S).argmin(1) for n, S in pt["fit_variant_scores"].items()}
                recs, _, _ = C.evaluate(T, picks, f"{nm}|m4c", n_boot)
                for n in picks:
                    rec["variants"].setdefault(n, dict(pt["fit_variants"][n]))[f"eval_{nm}"] = C.compact(recs[n])
            rec["m3c_rerun_regret_pct"] = pt["m3c_variant_regret_pct"]
            rec["small_refit_multistart"] = pt["small_refit_multistart"]
            prev = stored_prev.get(tag, {})
            rec["taco_r1_stored_values"] = {
                "M3c_summary_captured_mean_regret_pct": prev.get("M3c", {}).get("summary_captured_mean_regret_pct"),
                "M5b_static72_only_mean_regret_pct": prev.get("M5b", {}).get("static72_only_fit", {}).get("captured_calls", {})
                .get("mean_regret_pct"),
                "note": "values of taco_model_baselines.json (taco-r1 run); differences come from optima of the small "
                        "refits that the three-restart protocol reaches differently (see small_refit_multistart)"}
            rec["multistart"] = {k: v for k, v in pt["m3c"]["multistart"].items() if k != "best_start_params"}
            rec["cluster_bootstrap_12_keys"] = {k: pt["m3c"]["cluster_bootstrap"][k]
                                                for k in ("n_replicates", "captured_mean_regret_pct", "captured_p95_regret_pct",
                                                          "share_same_choice_as_stored_fit")}
            rec["objective_r2_stored_fit"] = pt["m3c"]["objective_r2_stored_fit"]
            rec["search_vs_full_protocol"] = pt["m3c"]["search_vs_full_protocol"]
            rec["leave_one_key_out_full_form"] = {k: pt["forms"]["full"]["loso"][k]
                                                  for k in ("mean_chunk_regret_pct", "max_chunk_regret_pct", "se_chunk_regret_pct",
                                                            "heldout_rmse_log_mean")}
            cout[gpu] = rec
            if conv == "device_time":
                v = rec["variants"]
                checks.add(f"M4c full-protocol-only refit {gpu} (own table)", v["full_protocol_only"][f"eval_{own}"]["mean_regret_pct"],
                           QUOTED["M4c_full_protocol_only"][gpu], "review M4c (three-restart protocol; see small_refit_multistart)",
                           0.006)
                checks.add(f"M4c objective R2 {gpu}", rec["objective_r2_stored_fit"]["r2_objective"],
                           QUOTED["M4c_objective_r2"][gpu], "review M4c", 0.006)
                checks.add(f"M4c static72-only {gpu} (own table)", v["static72_only"][f"eval_{own}"]["mean_regret_pct"],
                           QUOTED["M4c_static72"][gpu], "review M4c", 0.06)
                checks.add(f"M4c leave-one-key-out {gpu}", rec["leave_one_key_out_full_form"]["mean_chunk_regret_pct"],
                           QUOTED["M4c_leave_one_key_out"][gpu], "review M4c / main text", 0.06)
        out[conv] = cout
    # the OLS value printed in the appendix (RTX 3090, device time, single-block session)
    if "device_time" in out:
        v = out["device_time"]["vgpu3090"]["variants"]["ols_all"]["eval_vgpu3090_device_one_block"]["mean_regret_pct"]
        out["appendix_ols_check"] = {"printed": "2.1, 1.8 and 3.0% (RTX PRO 6000, RTX 3090, H800)",
                                     "recomputed_pct": [out["device_time"][g]["variants"]["ols_all"][f"eval_{EVAL_SETS['device_time'][g][-1]}"]["mean_regret_pct"]
                                                        for g in ("pro6000", "vgpu3090", "h800")],
                                     "rtx3090_value": v, "rtx3090_rounded_1dp": round(v, 1),
                                     "verdict": "the RTX 3090 value is %.4f%%, which rounds to %.1f%%, not 1.8%%" % (v, round(v, 1))}
    return out


# =========================================================================== main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=2000, help="prompt-cluster bootstrap replicates of the evaluations")
    p.add_argument("--fit-boot", type=int, default=200, help="12-key bootstrap replicates of the fit (M4c)")
    p.add_argument("--n-multistart", type=int, default=50)
    p.add_argument("--mlp-seeds", type=int, default=5)
    p.add_argument("--mc-draws", type=int, default=2000)
    p.add_argument("--convs", default="device_time,launch_inclusive")
    args = p.parse_args(argv)
    t0 = time.time()
    convs = [c for c in args.convs.split(",") if c]
    opts = SimpleNamespace(n_boot=args.fit_boot, n_multistart=args.n_multistart, mlp_seeds=args.mlp_seeds,
                           mc_draws=args.mc_draws)
    tags = [t for c in convs for t in TAGS[c].values()]
    per_tag = {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(tags)))) as ex:
        for r in ex.map(tag_worker, [(t, str(args.in_dir), opts) for t in tags]):
            per_tag[r["tag"]] = r
            print(f"{r['tag']}: fitted in {r['seconds']} s ({time.time() - t0:.0f} s elapsed)", flush=True)
    checks = C.Checks()
    # reproduction of the stored round-4 / taco-r1 analyses
    stored = json.loads((args.in_dir / "taco_model_baselines.json").read_text())
    for tag, r in per_tag.items():
        bad = [k for k, v in {**r["form_checks"], **{k: v for k, v in r["m5a_checks"].items() if isinstance(v, dict)}}.items()
               if isinstance(v, dict) and v.get("pass") is False]
        checks.add(f"{tag}: stored round-4 form and ranker reproductions failing", len(bad), 0, "taco_model_baselines checks", 0)
    res = {"experiment": "taco-r2 M4a/M4b/M4c cost-model protocols", "definitions": __doc__,
           "conventions": {"lw_share_of_oracle_pct": "100 x sum(excess latency) / sum(oracle latency)",
                           "intervals": f"95% prompt-cluster bootstrap, {args.n_boot} replicates",
                           "eval_sets": {c: EVAL_SETS[c] for c in convs}},
           "prespecified_protocol": {"candidate_set": PRESPECIFIED[0], "criterion": PRESPECIFIED[1],
                                     "tie_rule": "fewest free parameters among forms within 0.1 points of the lowest "
                                                 "criterion value, then the lowest criterion value, then the lowest LOSO "
                                                 "held-out log-RMSE",
                                     "rationale": "uses only the target GPU's own calibration timings (no captured call "
                                                  "of any GPU and no other GPU), so it is the procedure a practitioner "
                                                  "can run on a new GPU and it does not depend on which GPUs are pooled; "
                                                  "it matches the 'What transfers' recommendation (choose the terms "
                                                  "without the target calls); the three-parameter restriction is a "
                                                  "legitimate candidate; the RMSE tie-break makes it deterministic"},
           "paper_protocol": {"candidate_set": PAPER_PROTOCOL[0], "criterion": PAPER_PROTOCOL[1]}}
    res["M4a"], choices = m4a(per_tag, convs, args.in_dir, args.n_boot, checks)
    # stored selections must be reproduced
    for conv in convs:
        st = stored["M4c"]["selections"][conv]
        for key, v in st.items():
            gpu, set_name, cname = key.split("|")
            mine = res["M4a"][conv]["per_gpu"][gpu]["protocols"][f"{set_name}|{cname}|stored"]
            checks.add(f"M4a {conv} {key} selected form equals stored", int(mine["selected_form"] == v["selected_form"]), 1,
                       "taco_model_baselines.json M4c.selections", 0)
    a_priori = {}
    for gpu in TAGS["device_time"]:
        pr = res["M4a"]["device_time"]["per_gpu"][gpu]["protocols"]
        a_priori[gpu] = {"a_priori_paper_protocol": pr[f"{PAPER_PROTOCOL[0]}|{PAPER_PROTOCOL[1]}|stored"]["selected_form"],
                         "a_priori_prespecified": pr["|".join(PRESPECIFIED)]["selected_form"]}
    res["a_priori_forms_device_time"] = a_priori
    if "device_time" in convs:
        res["M4b"] = m4b(per_tag, args.in_dir, args.n_boot, a_priori, checks, args.mlp_seeds)
        h = per_tag[TAGS["device_time"]["h800"]]["forms"]
        res["M4b"]["h800_add_l2_miss_equals_frozen_form"] = {
            "add_l2_miss_mean_regret_pct": h["add_l2_miss"]["captured_calls"]["mean_regret_pct"],
            "full_mean_regret_pct": h["full"]["captured_calls"]["mean_regret_pct"],
            "identical": abs(h["add_l2_miss"]["captured_calls"]["mean_regret_pct"] - h["full"]["captured_calls"]["mean_regret_pct"]) < 1e-9,
            "fitted_tau_d": h["add_l2_miss"]["params"]["tau_d"]}
        res["M4b"]["compile_facts_rule_definition"] = per_tag[TAGS["device_time"]["h800"]]["compile_facts_rule_definition"]
        res["M4b"]["learned_hyperparameters"] = {g: per_tag[t]["m5a_learned"] for g, t in TAGS["device_time"].items()}
    res["M4c"] = m4c(per_tag, convs, args.in_dir, args.n_boot, checks)
    # per-call choices for the untuned-baseline analysis (keys of the primary data sets)
    pc = {}
    for gpu in TAGS["device_time"]:
        for nm in EVAL_SETS["device_time"][gpu]:
            ch = choices[("device_time", gpu, nm)]
            pc[nm] = {"keys": choices[("device_time", gpu, nm, "keys")],
                      "frozen_full_form": ch["full"],
                      "a_priori_paper_protocol": ch[a_priori[gpu]["a_priori_paper_protocol"]],
                      "a_priori_prespecified": ch[a_priori[gpu]["a_priori_prespecified"]],
                      "work_only": ch["work_only"]}
    res["per_call_choices_device_time"] = pc
    res["checks"] = checks.summary()
    res["runtime_s"] = round(time.time() - t0, 1)
    path = C.save("m4", res, args.in_dir)
    print(f"wrote {path}")
    print(f"checks: {res['checks']['n_match']}/{res['checks']['n']} match")
    for m in res["checks"]["mismatches"]:
        print("  MISMATCH", m["check"], m["computed"], m["reference"])
    for conv in convs:
        for gpu, s in res["M4a"][conv]["summary"].items():
            print(f"{conv:16s} {gpu:9s} six protocols per call (primary) {[round(x, 2) for x in s['six_protocols_per_call_mean_pct_primary']]}"
                  f" LW {[round(x, 2) for x in s['six_protocols_lw_pct_primary']]}")
            print(f"{'':16s} {'':9s} six protocols per call (own table) {[round(x, 2) for x in s['six_protocols_per_call_mean_pct_own_table']]}"
                  f" LW {[round(x, 2) for x in s['six_protocols_lw_pct_own_table']]}")


if __name__ == "__main__":
    main()
