# ADAMA – Experiment Data Records

This file records the raw experiment results used in the paper tables and figures.
All experiments use BF16 precision unless noted.  Latency is wall-clock time
averaged over 100 inference calls after 10 warm-up calls.

---

## Hardware Configuration

| Platform          | GPU                  | VRAM   | Bandwidth  | CUDA  | Triton |
|-------------------|----------------------|--------|------------|-------|--------|
| Server            | NVIDIA RTX 3090      | 24 GB  | 936 GB/s   | 12.3  | 2.2    |
| Edge              | Jetson Orin Nano     | 8 GB   | 68 GB/s    | 11.4  | —      |

---

## Table 1 – Inference Latency and Throughput (RTX 3090)

Batch size = 1, sequence length = 2048.

| Model       | Method   | Latency (ms) | TPS   | Speedup |
|-------------|----------|--------------|-------|---------|
| Mamba-370M  | No-Fuse  | 18.4         | 543   | 1.00×   |
| Mamba-370M  | Fuse-All | 10.2         | 980   | 1.81×   |
| Mamba-370M  | EGFS     | 9.5          | 1052  | 1.94×   |
| Mamba-370M  | ADAMA    | 7.8          | 1281  | 2.36×   |
| Mamba-2.8B  | No-Fuse  | 142.1        | 70    | 1.00×   |
| Mamba-2.8B  | Fuse-All | 79.6         | 125   | 1.79×   |
| Mamba-2.8B  | EGFS     | 73.8         | 135   | 1.93×   |
| Mamba-2.8B  | ADAMA    | 29.6         | 338   | 4.80×   |

### Notes
- Latency is end-to-end (tokeniser excluded).
- TPS = tokens per second = (B × L) / latency_in_seconds.
- Mamba-2.8B shows larger gains because its larger state dimension makes it
  more memory-bandwidth-bound.

---

## Table 2 – WikiText-103 Perplexity (Mamba-2.8B)

Evaluation: stride = 512, max context = 2048.

| Method        | PPL   | ΔPPL  |
|---------------|-------|-------|
| No-Fuse (BF16)| 7.51  | —     |
| Fuse-All      | 7.72  | +0.21 |
| EGFS          | 7.55  | +0.04 |
| ADAMA         | 7.55  | +0.04 |

---

## Table 3 – Ablation on Mamba-2.8B (RTX 3090)

| Configuration                           | Lat. (ms) | TPS | PPL  |
|-----------------------------------------|-----------|-----|------|
| No-Fuse                                 | 142.1     | 70  | 7.51 |
| + EGFS only                             | 73.8      | 135 | 7.55 |
| + FHL only                              | 98.4      | 102 | 7.52 |
| + EGFS + FHL                            | 31.2      | 320 | 7.55 |
| + EGFS + FHL + Entropy-Tiling Co-Design | 29.6      | 338 | 7.55 |

---

## Per-Layer Entropy Statistics (Mamba-2.8B, 48 layers)

Measured on 128 calibration samples from WikiText-103 (sequence length 512).

| Statistic                             | Before WHT | After WHT |
|---------------------------------------|-----------|-----------|
| Mean entropy (bits)                   | 4.81      | 5.73      |
| Min entropy (bits)                    | 1.83      | 2.91      |
| Max entropy (bits)                    | 6.72      | 7.18      |
| Fraction of layers below τ_H = 5.0   | 41.7%     | 12.5%     |

The 41.7% of layers below the threshold before WHT insertion would each
introduce a fusion boundary; after FHL absorption this drops to 12.5%,
substantially reducing the number of kernel launches.

---

## Memory Footprint (Peak Intermediate Activations, Mamba-2.8B)

| Method  | Peak Memory (GB) | Reduction vs No-Fuse |
|---------|-----------------|----------------------|
| No-Fuse | 11.2            | 1.00×                |
| ADAMA   | 1.16            | 9.66×                |

---

## Entropy–Tiling Sensitivity

Tile size $T$ recommended by Eq. (8) for entropy values ranging from 0 to 8 bits:

| Entropy (bits) | Tile Size $T$ |
|----------------|---------------|
| 0.0            | 64            |
| 2.0            | 128           |
| 4.0            | 128           |
| 6.0            | 256           |
| 8.0            | 512           |

---

## Reproducibility Checklist

- [x] Fixed random seed (seed=42) for all synthetic calibration data.
- [ ] Upload model checkpoints to Hugging Face Hub (anonymised).
- [ ] Release Triton kernel source in `src/` (this repo).
- [ ] Document environment setup in `README.md`.
