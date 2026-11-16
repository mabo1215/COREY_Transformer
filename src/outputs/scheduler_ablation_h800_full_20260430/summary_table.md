# Scheduler Ablation Matrix

| configuration | kind | scheduler | latency ms | vs passive | vs best static | speedup over best static | active overhead | chunk distribution |
|---|---:|---|---:|---:|---:|---:|---:|---|
| static_chunk_128 | static | constant chunk=128 | 892.71 +/- 12.11 | 1.0032x | 1.0013x | 0.9987x | 1.0058x | `{'128': 2544}` |
| static_chunk_256 | static | constant chunk=256 | 914.43 +/- 14.58 | 1.0045x | 1.0257x | 0.9749x | 1.0019x | `{'256': 2544}` |
| static_chunk_512 | static | constant chunk=512 | 891.51 +/- 10.17 | 1.0048x | 1.0000x | 1.0000x | 1.0034x | `{'512': 2544}` |
| static_chunk_1024 | static | constant chunk=1024 | 897.66 +/- 16.59 | 1.0061x | 1.0069x | 0.9931x | 1.0116x | `{'1024': 2544}` |
| static_chunk_2048 | static | constant chunk=2048 | 907.93 +/- 10.60 | 1.0153x | 1.0184x | 0.9819x | 1.0041x | `{'2048': 2544}` |
| no_entropy_mid | no_entropy | no_entropy | 903.77 +/- 29.67 | 1.0116x | 1.0137x | 0.9864x | 1.0028x | `{'1024': 2544}` |
| random_seed_0 | random | random | 899.17 +/- 10.32 | 1.0132x | 1.0086x | 0.9915x | 1.0169x | `{'1024': 528, '128': 521, '512': 502, '2048': 486, '256': 507}` |
| hist | adaptive | hist | 970.26 +/- 27.36 | 1.0791x | 1.0883x | 0.9188x | 1.0645x | `{'1024': 2544}` |
| sampled_hist_s8 | adaptive | sampled_hist stride=8 | 932.69 +/- 20.70 | 1.0447x | 1.0462x | 0.9559x | 1.0373x | `{'1024': 2544}` |
| cheap_proxy | proxy | cheap_proxy | 921.81 +/- 20.86 | 1.0202x | 1.0340x | 0.9671x | 1.0155x | `{'1024': 2544}` |
| variance_proxy | adaptive | variance_proxy | 934.80 +/- 26.04 | 1.0456x | 1.0486x | 0.9537x | 1.0205x | `{'1024': 2332, '2048': 212}` |
| kurtosis_proxy | adaptive | kurtosis_proxy | 922.96 +/- 14.96 | 1.0282x | 1.0353x | 0.9659x | 1.0285x | `{'2048': 2544}` |
| token_hist_s8 | adaptive | token_hist stride=8 | 999.68 +/- 24.35 | 1.0504x | 1.1213x | 0.8918x | 1.0548x | `{'1024': 2173, '2048': 371}` |

Best static oracle: `static_chunk_512` at 891.51 ms.
