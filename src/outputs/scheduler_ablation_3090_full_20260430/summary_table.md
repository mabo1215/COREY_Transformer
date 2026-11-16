# Scheduler Ablation Matrix

| configuration | kind | scheduler | latency ms | vs passive | vs best static | speedup over best static | active overhead | chunk distribution |
|---|---:|---|---:|---:|---:|---:|---:|---|
| static_chunk_128 | static | constant chunk=128 | 1393.36 +/- 5.02 | 1.0038x | 1.0000x | 1.0000x | 1.0010x | `{'128': 2544}` |
| static_chunk_256 | static | constant chunk=256 | 1424.89 +/- 4.54 | 1.0097x | 1.0226x | 0.9779x | 1.0079x | `{'256': 2544}` |
| static_chunk_512 | static | constant chunk=512 | 1406.22 +/- 5.74 | 1.0118x | 1.0092x | 0.9909x | 1.0097x | `{'512': 2544}` |
| static_chunk_1024 | static | constant chunk=1024 | 1421.14 +/- 5.53 | 1.0106x | 1.0199x | 0.9804x | 1.0098x | `{'1024': 2544}` |
| static_chunk_2048 | static | constant chunk=2048 | 1406.47 +/- 4.54 | 1.0095x | 1.0094x | 0.9907x | 1.0076x | `{'2048': 2544}` |
| no_entropy_mid | no_entropy | no_entropy | 1421.87 +/- 5.11 | 1.0120x | 1.0205x | 0.9799x | 1.0110x | `{'1024': 2544}` |
| random_seed_0 | random | random | 1385.10 +/- 5.15 | 1.0102x | 0.9941x | 1.0060x | 1.0088x | `{'1024': 528, '128': 521, '512': 502, '2048': 486, '256': 507}` |
| hist | adaptive | hist | 1471.22 +/- 6.27 | 1.0498x | 1.0559x | 0.9471x | 1.0478x | `{'1024': 2544}` |
| sampled_hist_s8 | adaptive | sampled_hist stride=8 | 1449.39 +/- 5.20 | 1.0217x | 1.0402x | 0.9613x | 1.0200x | `{'1024': 2544}` |
| cheap_proxy | proxy | cheap_proxy | 1423.16 +/- 4.12 | 1.0162x | 1.0214x | 0.9791x | 1.0138x | `{'1024': 2544}` |
| variance_proxy | adaptive | variance_proxy | 1438.37 +/- 5.02 | 1.0225x | 1.0323x | 0.9687x | 1.0197x | `{'1024': 2332, '2048': 212}` |
| kurtosis_proxy | adaptive | kurtosis_proxy | 1429.20 +/- 4.78 | 1.0190x | 1.0257x | 0.9749x | 1.0172x | `{'2048': 2544}` |
| token_hist_s8 | adaptive | token_hist stride=8 | 1442.94 +/- 4.54 | 1.0351x | 1.0356x | 0.9656x | 1.0335x | `{'1024': 2173, '2048': 371}` |

Best static oracle: `static_chunk_128` at 1393.36 ms.
