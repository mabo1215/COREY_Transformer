# Static Oracle vs Adaptive Summary

| configuration | kind | chunk | scheduler | latency ms | vs passive | vs best static | speedup over best static | chunk distribution |
|---|---:|---:|---|---:|---:|---:|---:|---|
| static_chunk_128 | static | 128 | constant | 876.55 +/- 26.55 | 0.9967x | 1.0000x | 1.0000x | `{'128': 2544}` |
| static_chunk_256 | static | 256 | constant | 914.34 +/- 22.64 | 0.9835x | 1.0431x | 0.9587x | `{'256': 2544}` |
| static_chunk_512 | static | 512 | constant | 881.78 +/- 36.80 | 1.0070x | 1.0060x | 0.9941x | `{'512': 2544}` |
| static_chunk_1024 | static | 1024 | constant | 899.81 +/- 21.74 | 0.9891x | 1.0265x | 0.9741x | `{'1024': 2544}` |
| static_chunk_2048 | static | 2048 | constant | 910.09 +/- 28.69 | 1.0214x | 1.0383x | 0.9631x | `{'2048': 2544}` |
| adaptive_sampled_hist_s8 | adaptive |  | sampled_hist stride=8 | 905.55 +/- 18.24 | 1.0510x | 1.0331x | 0.9680x | `{'1024': 2544}` |

Best static oracle: `static_chunk_128` at 876.55 ms.
Adaptive vs best static: 1.0331x latency ratio, 0.9680x speedup.
