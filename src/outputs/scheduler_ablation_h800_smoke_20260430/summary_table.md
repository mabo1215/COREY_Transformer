# Scheduler Ablation Matrix

| configuration | kind | scheduler | latency ms | vs passive | vs best static | speedup over best static | active overhead | chunk distribution |
|---|---:|---|---:|---:|---:|---:|---:|---|
| static_chunk_128 | static | constant chunk=128 | 67.14 +/- 0.00 | 0.0787x | 1.0040x | 0.9960x | 0.0821x | `{'128': 48}` |
| static_chunk_256 | static | constant chunk=256 | 66.87 +/- 0.00 | 0.0815x | 1.0000x | 1.0000x | 0.0835x | `{'256': 48}` |
| no_entropy_mid | no_entropy | no_entropy | 68.77 +/- 0.00 | 0.0825x | 1.0283x | 0.9725x | 0.0852x | `{'256': 48}` |
| random_seed_0 | random | random | 68.01 +/- 0.00 | 0.0779x | 1.0170x | 0.9833x | 0.0853x | `{'256': 26, '128': 22}` |
| sampled_hist_s8 | adaptive | sampled_hist stride=8 | 86.28 +/- 0.00 | 0.1020x | 1.2902x | 0.7751x | 0.1602x | `{'128': 1, '256': 47}` |

Best static oracle: `static_chunk_256` at 66.87 ms.
