# Scheduler Ablation Matrix

| configuration | kind | scheduler | latency ms | vs passive | vs best static | speedup over best static | active overhead | chunk distribution |
|---|---:|---|---:|---:|---:|---:|---:|---|
| static_chunk_128 | static | constant chunk=128 | 900.00 +/- 14.96 | 0.9831x | 1.0000x | 1.0000x | 0.9871x | `{'128': 1104}` |
| static_chunk_256 | static | constant chunk=256 | 936.66 +/- 15.34 | 1.0524x | 1.0407x | 0.9609x | 0.9996x | `{'256': 1104}` |
| static_chunk_512 | static | constant chunk=512 | 905.97 +/- 21.93 | 0.9713x | 1.0066x | 0.9934x | 0.9890x | `{'512': 1104}` |
| guarded_sampled_hist_s8 | guarded | guarded_sampled_hist stride=8 | 932.86 +/- 16.12 | 1.0503x | 1.0365x | 0.9648x | 1.0572x | `{'512': 1104}` |
| learned_table | learned | learned_table stride=8 | 924.20 +/- 17.23 | 1.0393x | 1.0269x | 0.9738x | 1.0699x | `{'512': 1104}` |

Best static oracle: `static_chunk_128` at 900.00 ms.
