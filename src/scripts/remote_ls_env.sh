#!/usr/bin/env bash
ls /home1/mabo1215/.adama-micromamba/envs/quamba-py310/lib/python3.10/site-packages/ 2>/dev/null | grep -E '^torch|^transformers|^datasets' | head -15
