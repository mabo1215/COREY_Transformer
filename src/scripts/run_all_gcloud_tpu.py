#!/usr/bin/env python3
"""
Automated GCP TPU experiment runner for COREY.
- Creates TPU VM (if needed)
- Installs dependencies
- Runs all key experiments (integrated, heterogeneous, calibrated, tpu benchmark)
- Downloads results
- Cleans up resources

Usage:
  python src/scripts/run_all_gcloud_tpu.py --zone us-east1-d --tpu-type v6e-8 --model mamba-370m --seq-len 4096 --chunk-size 512

Requirements:
- gcloud CLI installed and authenticated
- This script runs locally and uses gcloud ssh/scp to control the TPU VM
"""

import argparse
import subprocess
import os
import json
from pathlib import Path


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    else:
        config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to config.json')
parser.add_argument('--project-id', type=str, default=None)
parser.add_argument('--zone', type=str, default=None)
parser.add_argument('--tpu-type', type=str, default=None)
parser.add_argument('--tpu-name', type=str, default=None)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--seq-len', type=int, default=None)
parser.add_argument('--chunk-size', type=int, default=None)
parser.add_argument('--repeat', type=int, default=None)
parser.add_argument('--output-dir', type=str, default=None)
# 添加下面这一行：
parser.add_argument('--version', type=str, default=None, help='TPU VM Runtime version')
args = parser.parse_args()

config = load_config(args.config)

def get_arg(name, default=None):
    v = getattr(args, name)
    if v is not None:
        return v
    if name in config:
        return config[name]
    return default


project_id = get_arg('project_id')
zone = get_arg('zone')
tpu_type = get_arg('tpu_type', 'v6e-8')
tpu_name = get_arg('tpu_name', 'corey-tpu-exp')
model = get_arg('model', 'mamba-370m')
seq_len = get_arg('seq_len', 4096)
chunk_size = get_arg('chunk_size', 512)
repeat = get_arg('repeat', 30)
output_dir = get_arg('output_dir', 'src/outputs/gcloud_tpu_all')
version = get_arg('version', None)


# 参数健壮性检查
missing = []
for k in ["zone", "tpu_type", "tpu_name", "model", "seq_len", "chunk_size", "repeat", "output_dir", "version"]:
    if eval(k) is None:
        missing.append(k)
if missing:
    raise ValueError(f"Missing required config parameters: {', '.join(missing)}. Please check config.json or provide them as command-line arguments.")

# 类型转换
seq_len = int(seq_len)
chunk_size = int(chunk_size)
repeat = int(repeat)

# Set gcloud project if specified
if project_id:
    print(f"[INFO] Setting gcloud project to {project_id}")
    subprocess.run(['gcloud', 'config', 'set', 'project', str(project_id)], check=True)




# 1. Create TPU VM（支持资源池轮询）
import sys
resource_pool = config.get("resource_pool", None)
create_success = False
if resource_pool:
    for res in resource_pool:
        zone = res["zone"]
        tpu_type = res["tpu_type"]
        version = res["version"]
        spot = res.get("spot", False)
        print(f"[INFO] Trying TPU VM {tpu_name} in {zone} ({tpu_type}), version={version} ({'spot' if spot else 'on-demand'}) ...")
        cmd = [
            'gcloud', 'compute', 'tpus', 'tpu-vm', 'create', tpu_name,
            f'--zone={zone}', f'--accelerator-type={tpu_type}',
            f'--version={version}'
        ]
        if spot:
            cmd.append('--spot')
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            create_success = True
            break
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode() if hasattr(e.stderr, 'decode') else str(e)
            if 'ALREADY_EXISTS' in err:
                print(f"[INFO] TPU VM already exists, skipping creation.")
                create_success = True
                break
            if 'Insufficient capacity' in err:
                print(f"[WARN] Insufficient capacity for {zone} {tpu_type}, trying next...")
                continue
            else:
                print(f"[ERROR] Failed to create TPU VM: {err}")
                sys.exit(1)
    if not create_success:
        print("[FATAL] All resource pool options exhausted. No available TPU capacity.")
        sys.exit(1)
else:
    print(f"[INFO] Creating TPU VM {tpu_name} in {zone} ({tpu_type}), version={version} (spot) ...")
    cmd = [
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'create', tpu_name,
        f'--zone={zone}', f'--accelerator-type={tpu_type}',
        f'--version={version}',
        '--spot'
    ]
    subprocess.run(cmd, check=True)


# 2. Upload code (sync src/ and requirements.txt)

print("[INFO] Uploading code to TPU VM...")
# 递归上传 src 目录（仅目录加 --recurse），排除无效文件


# 仅在 TPU VM 上下载代码
gcs_bucket = config.get('gcs_bucket', None)
if not gcs_bucket:
    raise ValueError('gcs_bucket must be set in config.json')
print("[INFO] Downloading code from GCS bucket on TPU VM ...")
subprocess.run([
    'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
    '--command', f'mkdir -p ~/src && gsutil -m rsync -r gs://{gcs_bucket}/code/src ~/src && gsutil cp gs://{gcs_bucket}/code/requirements.txt ~/'
], check=True)


# 3. Install dependencies


# 先检测TPU VM上已安装的numpy等包版本，只有缺失或不兼容时才安装requirements.txt
print("[INFO] Checking existing Python packages on TPU VM...")
check_pkgs = subprocess.run([
    'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
    '--command', 'python3 -c "import numpy; print(numpy.__version__)"'
], capture_output=True, text=True)
need_install = False
if check_pkgs.returncode != 0:
    print("[WARN] numpy not found, will install requirements.txt...")
    need_install = True
else:
    numpy_version = check_pkgs.stdout.strip().splitlines()[-1]
    print(f"[INFO] Detected numpy version on TPU VM: {numpy_version}")
    if numpy_version.startswith('2.'):
        print("[WARN] numpy>=2 detected, will downgrade...")
        need_install = True
    else:
        print("[INFO] numpy version is compatible, skipping install.")

if need_install:
    print("[INFO] Installing dependencies on TPU VM via requirements.txt ...")
    subprocess.run([
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
        '--command', 'python3 -m pip install -r ~/requirements.txt'
    ], check=True)
    print("[INFO] Ensuring numpy<2.0 on TPU VM...")
    subprocess.run([
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
        '--command', 'python3 -m pip install \"numpy<2\"'
    ], check=True)
else:
    print("[INFO] Skipped requirements.txt install; system packages are compatible.")

# 检查 torch_xla 是否可用，否则提示用户重建 TPU VM
print("[INFO] Checking torch_xla availability on TPU VM...")
check_xla = subprocess.run([
    'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
    '--command', 'python3 -c "import torch_xla"'
], capture_output=True)
if check_xla.returncode != 0:
    print("[FATAL] torch_xla not found in TPU runtime.")
    sys.exit(1)

# 4. Run experiments

exp_cmds = [
    f"python3 ~/src/experiments/run_corey_tpu_benchmark.py --device tpu --model {model} --chunk-size {chunk_size} --seq-len {seq_len} --repeat {repeat} --output-dir ~/src/outputs/corey_tpu_benchmark",
    f"python3 ~/src/experiments/run_integrated_end_to_end.py --model {model} --output-dir ~/src/outputs/integrated_end_to_end",
    f"python3 ~/src/experiments/run_heterogeneous_corpus.py --model {model} --output-dir ~/src/outputs/heterogeneous_corpus",
]
for cmd in exp_cmds:
    print(f"[INFO] Running on TPU VM: {cmd}")
    subprocess.run([
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
        '--command', cmd
    ], check=True)

# 5. Download results

print("[INFO] Downloading results from TPU VM...")
for remote_dir in [
    '~/src/outputs/corey_tpu_benchmark',
    '~/src/outputs/integrated_end_to_end',
    '~/src/outputs/heterogeneous_corpus',
]:
    subprocess.run([
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'scp', '--recurse',
        f'{tpu_name}:{remote_dir}', output_dir, f'--zone={zone}'
    ], check=True)

# 6. Delete TPU VM

print(f"[INFO] Deleting TPU VM {tpu_name}...")
subprocess.run([
    'gcloud', 'compute', 'tpus', 'tpu-vm', 'delete', tpu_name, f'--zone={zone}', '--quiet'
], check=True)


# 7. Upload results to GCS if configured
gcs_bucket = config.get("gcs_bucket")
gcs_results_prefix = config.get("gcs_results_prefix", "results/")
if gcs_bucket:
    print(f"[INFO] Uploading results to GCS bucket: {gcs_bucket}/{gcs_results_prefix}")
    # Recursively upload all files in output_dir to the GCS bucket
    subprocess.run([
        'gsutil', '-m', 'cp', '-r', str(output_dir), f"gs://{gcs_bucket}/{gcs_results_prefix}"
    ], check=True)
    print(f"[ALL DONE] Results uploaded to: gs://{gcs_bucket}/{gcs_results_prefix}")
else:
    print("[ALL DONE] Results downloaded to:", args.output_dir)
