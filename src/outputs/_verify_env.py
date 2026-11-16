import torch
import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("torch:", torch.__version__)
print("mamba_ssm:", mamba_ssm.__version__)
print("selective_scan ok")
