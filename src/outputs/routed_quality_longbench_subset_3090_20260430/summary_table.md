# Routed Quality Check

| task | sample | exact generated tokens | mismatches | passive metric | routed metric | metric delta | passive ppl | routed ppl | ppl ratio | routed chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| narrativeqa | 0 | True | 0 | 0.000000 | 0.000000 | +0.000000 | 20.5190 | 20.5190 | 1.000000x | `{256: 350, 512: 34}` |
| qasper | 0 | True | 0 | 0.153846 | 0.153846 | +0.000000 | 19.5829 | 19.5829 | 1.000000x | `{256: 350, 512: 34}` |
| multifieldqa_en | 0 | True | 0 | 0.000000 | 0.000000 | +0.000000 | 12.0490 | 12.0490 | 1.000000x | `{256: 350, 512: 34}` |
| gov_report | 0 | True | 0 | 0.011429 | 0.011429 | +0.000000 | 13.3218 | 13.3218 | 1.000000x | `{256: 350, 512: 34}` |

## Task Summary

| task | samples | exact matches | mismatches | passive metric | routed metric | delta | mean ppl ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| gov_report | 1 | 1 | 0 | 0.011429 | 0.011429 | +0.000000 | 1.000000x |
| multifieldqa_en | 1 | 1 | 0 | 0.000000 | 0.000000 | +0.000000 | 1.000000x |
| narrativeqa | 1 | 1 | 0 | 0.000000 | 0.000000 | +0.000000 | 1.000000x |
| qasper | 1 | 1 | 0 | 0.153846 | 0.153846 | +0.000000 | 1.000000x |
