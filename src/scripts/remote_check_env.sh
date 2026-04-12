#!/usr/bin/env bash
cd /home1/mabo1215/COREY_Transformer
export MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba
MM=/home1/mabo1215/.corey-wsl-tools/bin/micromamba
$MM run -n quamba-py310 python -c 'import torch, transformers, datasets; print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__, "| cuda", torch.cuda.is_available())'
