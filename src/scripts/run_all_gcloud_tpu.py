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
import time
from pathlib import Path
from shutil import which


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    else:
        config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def resolve_cli(primary_name, windows_names=(), override=None):
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return str(override_path)
        raise FileNotFoundError(f"Configured path for '{primary_name}' does not exist: {override}")

    candidate_names = [primary_name, *windows_names]
    for candidate in candidate_names:
        resolved = which(candidate)
        if resolved:
            return resolved

    if os.name == "nt":
        candidate_paths = []
        sdk_home = os.environ.get("GOOGLE_CLOUD_SDK_HOME")
        if sdk_home:
            sdk_bin = Path(sdk_home) / "bin"
            candidate_paths.extend(sdk_bin / name for name in candidate_names)

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            sdk_bin = Path(local_app_data) / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin"
            candidate_paths.extend(sdk_bin / name for name in candidate_names)

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return str(candidate_path)

    raise FileNotFoundError(
        f"Could not find '{primary_name}'. Install Google Cloud SDK or add it to PATH. "
        f"Tried: {', '.join(candidate_names)}"
    )


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
parser.add_argument('--gcloud-bin', type=str, default=None, help='Path to gcloud executable')
parser.add_argument('--gsutil-bin', type=str, default=None, help='Path to gsutil executable')
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


def get_cli_override(name):
    arg_name = name.replace('-', '_')
    value = getattr(args, arg_name, None)
    if value:
        return value
    return config.get(arg_name)


def is_capacity_error(err_text):
    err_lower = err_text.lower()
    capacity_markers = [
        'insufficient capacity',
        'no more capacity',
        'resource exhausted',
        'unavailable in the zone',
        'code": 8',
    ]
    return any(marker in err_lower for marker in capacity_markers)


def is_retryable_create_error(err_text):
    err_lower = err_text.lower()
    retryable_markers = [
        '"code": 13',
        'internal error has occurred',
        'internal error',
        'deadline exceeded',
        'temporarily unavailable',
        'service unavailable',
        'unavailable',
        'connection reset',
    ]
    return any(marker in err_lower for marker in retryable_markers)


gcloud_bin = resolve_cli(
    'gcloud',
    windows_names=('gcloud.cmd', 'gcloud.exe', 'gcloud.bat'),
    override=get_cli_override('gcloud-bin'),
)


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
    subprocess.run([gcloud_bin, 'config', 'set', 'project', str(project_id)], check=True)




# 1. Create TPU VM（支持资源池轮询）
import sys
resource_pool = config.get("resource_pool", None)
create_success = False
if resource_pool:
    create_retry_rounds = int(config.get("create_retry_rounds", 3))
    create_retry_sleep_sec = int(config.get("create_retry_sleep_sec", 20))
    total_candidates = len(resource_pool)
    for round_idx in range(create_retry_rounds):
        print(f"[INFO] Create round {round_idx + 1}/{create_retry_rounds} across {total_candidates} candidates")
        for res in resource_pool:
            zone = res["zone"]
            tpu_type = res["tpu_type"]
            version = res["version"]
            spot = res.get("spot", False)
            print(f"[INFO] Trying TPU VM {tpu_name} in {zone} ({tpu_type}), version={version} ({'spot' if spot else 'on-demand'}) ...")
            cmd = [
                gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'create', tpu_name,
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
                if is_capacity_error(err):
                    print(f"[WARN] Capacity unavailable for {zone} {tpu_type} ({'spot' if spot else 'on-demand'}), trying next...")
                    continue
                if is_retryable_create_error(err):
                    print(f"[WARN] Transient create error for {zone} {tpu_type} ({'spot' if spot else 'on-demand'}), trying next...")
                    continue
                print(f"[ERROR] Failed to create TPU VM: {err}")
                sys.exit(1)

        if create_success:
            break

        if round_idx < create_retry_rounds - 1:
            print(f"[WARN] No capacity found in this round. Waiting {create_retry_sleep_sec}s before next round...")
            time.sleep(create_retry_sleep_sec)

    if not create_success:
        print("[FATAL] All resource pool options exhausted after retry rounds. No available TPU capacity.")
        sys.exit(1)
else:
    print(f"[INFO] Creating TPU VM {tpu_name} in {zone} ({tpu_type}), version={version} (spot) ...")
    cmd = [
        gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'create', tpu_name,
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
    gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
    '--command', f'mkdir -p ~/src && gsutil -m rsync -r gs://{gcs_bucket}/code/src ~/src && gsutil cp gs://{gcs_bucket}/code/requirements.txt ~/'
], check=True)


# 3. Install dependencies


# 先检测TPU VM上已安装的numpy等包版本，只有缺失或不兼容时才安装requirements.txt
print("[INFO] Checking existing Python packages on TPU VM...")
check_pkgs = subprocess.run([
    gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
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
        gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
        '--command', 'python3 -m pip install -r ~/requirements.txt'
    ], check=True)
    print("[INFO] Ensuring numpy<2.0 on TPU VM...")
    subprocess.run([
        gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
        '--command', 'python3 -m pip install \"numpy<2\"'
    ], check=True)
else:
    print("[INFO] Skipped requirements.txt install; system packages are compatible.")

# 检查 torch_xla 是否可用，否则提示用户重建 TPU VM
print("[INFO] Checking torch_xla and torch version/device on TPU VM...")
check_xla = subprocess.run([
    gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
    '--command', 'PJRT_DEVICE=TPU python3 -c "import torch; import torch_xla; print(\'Torch Version:\', torch.__version__); print(\'XLA Device:\', torch_xla.device())"'
], capture_output=True, text=True)
if check_xla.returncode != 0:
    print("[FATAL] torch_xla not found or not working in TPU runtime.")
    sys.exit(1)
else:
    print(check_xla.stdout)
    if "xla:0" not in check_xla.stdout.lower():
        print("[WARNING] torch_xla is installed but TPU hardware (xla:0) is NOT detected. Please check PJRT_DEVICE and runtime compatibility.")

# 4. Run experiments

exp_cmds = [
    f"PJRT_DEVICE=TPU python3 ~/src/experiments/run_corey_tpu_benchmark.py --device tpu --model {model} --chunk-size {chunk_size} --seq-len {seq_len} --repeat {repeat} --output-dir ~/src/outputs/corey_tpu_benchmark",
    f"PJRT_DEVICE=TPU python3 ~/src/experiments/run_integrated_end_to_end.py --model {model} --output-dir ~/src/outputs/integrated_end_to_end",
    f"PJRT_DEVICE=TPU python3 ~/src/experiments/run_heterogeneous_corpus.py --model {model} --output-dir ~/src/outputs/heterogeneous_corpus",
]
for cmd in exp_cmds:
    print(f"[INFO] Running on TPU VM: {cmd}")
    subprocess.run([
        gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_name, f'--zone={zone}',
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
        gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'scp', '--recurse',
        f'{tpu_name}:{remote_dir}', output_dir, f'--zone={zone}'
    ], check=True)

# 6. Delete TPU VM

print(f"[INFO] Deleting TPU VM {tpu_name}...")
subprocess.run([
    gcloud_bin, 'compute', 'tpus', 'tpu-vm', 'delete', tpu_name, f'--zone={zone}', '--quiet'
], check=True)


# 7. Upload results to GCS if configured
gcs_bucket = config.get("gcs_bucket")
gcs_results_prefix = config.get("gcs_results_prefix", "results/")
if gcs_bucket:
    gsutil_bin = resolve_cli(
        'gsutil',
        windows_names=('gsutil.cmd', 'gsutil.exe', 'gsutil.bat'),
        override=get_cli_override('gsutil-bin'),
    )
    print(f"[INFO] Uploading results to GCS bucket: {gcs_bucket}/{gcs_results_prefix}")
    # Recursively upload all files in output_dir to the GCS bucket
    subprocess.run([
        gsutil_bin, '-m', 'cp', '-r', str(output_dir), f"gs://{gcs_bucket}/{gcs_results_prefix}"
    ], check=True)
    print(f"[ALL DONE] Results uploaded to: gs://{gcs_bucket}/{gcs_results_prefix}")
else:
    print("[ALL DONE] Results downloaded to:", args.output_dir)
