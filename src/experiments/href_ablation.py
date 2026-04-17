"""
H_ref Ablation: Analytical chunk-selection sensitivity analysis.

COREY selects chunk size via:
    r = min(H / H_ref, 1.0)
    C = clip(2^round(log2(C_min + r*(C_max - C_min))), C_min, C_max)

where C_min=32, C_max=512.

This script sweeps H_ref over {4.16, 5.0, 6.0, 8.0} for two measured
entropy values:
  - H=4.60 nats  (synthetic randn W1 benchmark, seq_len=4096, dim=1024, FP16)
  - H=4.02 nats  (mean over 80 real Mamba checkpoint prompts)

Latency values are taken directly from the measured w1_chunk_sweep outputs
(RTX 3070, CUDA 12.8, seq_len=4096, batch=1, dim=1024, d_state=16, FP16).

No GPU is required; the script is fully analytical.

Outputs:
    src/outputs/href_ablation/href_ablation.csv
    src/outputs/href_ablation/href_ablation_summary.txt

Usage:
    python -m src.experiments.href_ablation
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Measured chunk-to-latency table (RTX 3070, w1_chunk_sweep)
# chunk_size -> latency_mean_ms
# ---------------------------------------------------------------------------
CHUNK_LATENCY_MS: dict[int, float] = {
    32:  6.3151,
    64:  3.2992,
    128: 1.8670,
    256: 1.1477,
    512: 0.7481,
}

# COREY formula parameters
C_MIN: int = 32
C_MAX: int = 512


def entropy_to_chunk(
    entropy_nats: float,
    h_ref: float,
    c_min: int = C_MIN,
    c_max: int = C_MAX,
) -> int:
    """
    Map activation entropy (nats) to chunk size under the given H_ref.

    r = min(H / H_ref, 1.0)
    chunk_raw = C_min + r * (C_max - C_min)
    chunk = clip(2^round(log2(chunk_raw)), C_min, C_max)
    """
    r = min(entropy_nats / h_ref, 1.0)
    chunk_raw = c_min + r * (c_max - c_min)
    log2_val = math.log2(max(chunk_raw, 1.0))
    chunk_pow2 = 2 ** round(log2_val)
    return max(c_min, min(c_max, int(chunk_pow2)))


def latency_for_chunk(chunk: int) -> float:
    """Return the measured latency for the given chunk size (ms)."""
    if chunk in CHUNK_LATENCY_MS:
        return CHUNK_LATENCY_MS[chunk]
    # Nearest neighbor fallback (should not be needed for valid power-of-2 chunks)
    nearest = min(CHUNK_LATENCY_MS.keys(), key=lambda c: abs(c - chunk))
    return CHUNK_LATENCY_MS[nearest]


def run_ablation() -> list[dict]:
    """Compute the H_ref ablation table and return as a list of row dicts."""

    # H_ref values requested by the reviewer
    h_ref_values = [
        ("log2(K=64)", math.log(64),   "theoretical max for K=64 histogram bins"),
        ("5.0",         5.0,             "intermediate value"),
        ("6.0",         6.0,             "intermediate value"),
        ("8.0",         8.0,             "current default"),
    ]

    # Entropy scenarios
    entropy_scenarios = [
        ("randn_w1",   4.604287, "synthetic randn input (W1 benchmark, seed=42)"),
        ("real_ckpt",  4.02,     "mean over 80 real Mamba checkpoint prompts"),
    ]

    rows: list[dict] = []
    for entropy_label, h, entropy_desc in entropy_scenarios:
        for h_ref_label, h_ref, h_ref_desc in h_ref_values:
            r = min(h / h_ref, 1.0)
            chunk = entropy_to_chunk(h, h_ref)
            lat = latency_for_chunk(chunk)
            # Speedup relative to static baseline (chunk=64, 3.2992 ms)
            static_lat = CHUNK_LATENCY_MS[64]
            speedup_vs_static = round(static_lat / lat, 4) if lat > 0 else None
            rows.append({
                "entropy_scenario":    entropy_label,
                "entropy_nats":        round(h, 6),
                "entropy_description": entropy_desc,
                "h_ref":               round(h_ref, 6),
                "h_ref_label":         h_ref_label,
                "h_ref_description":   h_ref_desc,
                "r_value":             round(r, 6),
                "selected_chunk":      chunk,
                "latency_ms":          lat,
                "speedup_vs_static64": speedup_vs_static,
            })
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[href_ablation] CSV written: {output_path}")


def write_summary(rows: list[dict], output_path: Path) -> None:
    """Write a human-readable summary of the ablation."""
    lines: list[str] = []
    lines.append("H_ref Ablation: Chunk Selection and Latency Sensitivity")
    lines.append("=" * 60)
    lines.append(
        f"COREY formula:  r = min(H/H_ref, 1)  |  "
        f"C = clip(2^round(log2(C_min + r*(C_max-C_min))), C_min={C_MIN}, C_max={C_MAX})"
    )
    lines.append(
        "Latencies from: RTX 3070, CUDA 12.8, "
        "seq_len=4096, batch=1, dim=1024, d_state=16, FP16"
    )
    lines.append(
        f"Static baseline (chunk=64): {CHUNK_LATENCY_MS[64]:.4f} ms"
    )
    lines.append("")

    # Group by entropy scenario
    scenarios_seen: list[str] = []
    scenario_order: list[str] = []
    for row in rows:
        if row["entropy_scenario"] not in scenario_order:
            scenario_order.append(row["entropy_scenario"])

    for scenario in scenario_order:
        scenario_rows = [r for r in rows if r["entropy_scenario"] == scenario]
        h = scenario_rows[0]["entropy_nats"]
        desc = scenario_rows[0]["entropy_description"]
        lines.append(f"Scenario: {scenario}  (H = {h:.4f} nats, {desc})")
        lines.append(
            f"  {'H_ref':>10}  {'r':>6}  {'chunk':>7}  {'latency_ms':>11}  {'speedup_vs_64':>14}"
        )
        lines.append("  " + "-" * 56)
        for row in scenario_rows:
            marker = " <-- default" if abs(row["h_ref"] - 8.0) < 1e-9 else ""
            lines.append(
                f"  {row['h_ref_label']:>10}  {row['r_value']:>6.4f}  "
                f"{row['selected_chunk']:>7}  {row['latency_ms']:>11.4f}  "
                f"{row['speedup_vs_static64']:>14.4f}x{marker}"
            )
        lines.append("")

    lines.append("Key findings:")
    lines.append(
        "  - For the randn W1 benchmark (H=4.60), all H_ref < 4.60 fully saturate r=1.0"
    )
    lines.append(
        "    and select C=512 (0.75 ms), while H_ref=8.0 gives r=0.575 -> C=256 (1.15 ms)."
    )
    lines.append(
        "  - For real checkpoint activations (H=4.02), H_ref=4.16 gives r=0.966 -> C=512,"
    )
    lines.append(
        "    while H_ref=8.0 gives r=0.503 -> C=256, a 0.40 ms (35%) latency penalty."
    )
    lines.append(
        "  - H_ref should be calibrated to the histogram bin count K;"
    )
    lines.append(
        "    H_ref=log(K) is the principled upper bound for the entropy estimator."
    )

    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"[href_ablation] Summary written: {output_path}")
    print()
    print(text)


def main() -> None:
    output_dir = Path("src/outputs/href_ablation")
    rows = run_ablation()
    write_csv(rows, output_dir / "href_ablation.csv")
    write_summary(rows, output_dir / "href_ablation_summary.txt")


if __name__ == "__main__":
    main()
