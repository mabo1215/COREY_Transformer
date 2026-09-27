# Scheduler Ablation Matrix

| configuration | kind | scheduler | latency ms | vs passive | vs best static | speedup over best static | active overhead | chunk distribution |
|---|---:|---|---:|---:|---:|---:|---:|---|
| static_chunk_128 | static | constant chunk=128 | 880.14 +/- 4.34 | 0.9928x | 1.0099x | 0.9902x | 0.9996x | `{}` |
| static_chunk_256 | static | constant chunk=256 | 871.55 +/- 4.92 | 1.0061x | 1.0000x | 1.0000x | 1.0098x | `{}` |
| static_chunk_512 | static | constant chunk=512 | 892.83 +/- 3.86 | 0.9757x | 1.0244x | 0.9762x | 0.9766x | `{}` |
| guarded_sampled_hist_s8 | guarded | guarded_sampled_hist stride=8 | 934.68 +/- 5.17 | 1.0118x | 1.0724x | 0.9325x | 1.0184x | `{}` |
| learned_table | learned | learned_table stride=8 | 880.29 +/- 3.90 | 1.0043x | 1.0100x | 0.9901x | 1.0073x | `{}` |

Best static oracle: `static_chunk_256` at 871.55 ms.
