# Static Oracle vs Adaptive Summary

| configuration | kind | chunk | scheduler | latency ms | vs passive | vs best static | speedup over best static | chunk distribution |
|---|---:|---:|---|---:|---:|---:|---:|---|
| static_chunk_128 | static | 128 | constant | 113.91 +/- 0.00 | 0.0459x | 1.1808x | 0.8469x | `{'128': 48}` |
| static_chunk_256 | static | 256 | constant | 96.47 +/- 0.00 | 0.0404x | 1.0000x | 1.0000x | `{'256': 48}` |
| adaptive_sampled_hist_s8 | adaptive |  | sampled_hist stride=8 | 117.25 +/- 0.00 | 0.0489x | 1.2154x | 0.8228x | `{'128': 1, '256': 47}` |

Best static oracle: `static_chunk_256` at 96.47 ms.
Adaptive vs best static: 1.2154x latency ratio, 0.8228x speedup.
