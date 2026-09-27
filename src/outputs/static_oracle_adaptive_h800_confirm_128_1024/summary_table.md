# Static Oracle vs Adaptive Summary

| configuration | kind | chunk | scheduler | latency ms | vs passive | vs best static | speedup over best static | chunk distribution |
|---|---:|---:|---|---:|---:|---:|---:|---|
| static_chunk_128 | static | 128 | constant | 911.70 +/- 24.58 | 1.0054x | 1.0108x | 0.9893x | `{'128': 2544}` |
| static_chunk_1024 | static | 1024 | constant | 901.98 +/- 33.12 | 1.0319x | 1.0000x | 1.0000x | `{'1024': 2544}` |
| adaptive_sampled_hist_s8 | adaptive |  | sampled_hist stride=8 | 886.95 +/- 15.77 | 1.0319x | 0.9833x | 1.0170x | `{'512': 2544}` |

Best static oracle: `static_chunk_1024` at 901.98 ms.
Adaptive vs best static: 0.9833x latency ratio, 1.0170x speedup.
