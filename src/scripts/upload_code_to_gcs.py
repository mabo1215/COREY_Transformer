#!/usr/bin/env python3
"""
Upload code and requirements.txt to GCS bucket for TPU experiments.
Usage:
  python src/scripts/upload_code_to_gcs.py --gcs-bucket corey-transformer-paper-results
"""

import json
import os
from pathlib import Path
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='src/scripts/config.json', help='Path to config.json')
parser.add_argument('--gcs-bucket', type=str, default=None, help='GCS bucket name (overrides config)')
parser.add_argument('--file', type=str, default=None, help='Only upload the specified file (relative to ./src/)')
args = parser.parse_args()

# Prefer command-line arguments; otherwise read from config.json
def load_config(config_path):
  config_path = Path(config_path)
  if config_path.exists():
    with open(config_path, 'r') as f:
      return json.load(f)
  return {}

config = load_config(args.config)
gcs_bucket = args.gcs_bucket or config.get('gcs_bucket', None)



if args.file:
  # Upload only the specified file
  file_path = os.path.join('src', args.file)
  if not os.path.isfile(file_path):
    print(f"[ERROR] File not found: {file_path}")
    exit(1)
  print(f"[INFO] Uploading {file_path} to GCS bucket {gcs_bucket} ...")
  subprocess.run(['gsutil', 'cp', file_path, f'gs://{gcs_bucket}/code/src/{args.file}'], check=True)
  print(f"[INFO] {args.file} uploaded to GCS bucket.")
else:
  print(f"[INFO] Syncing src/ to GCS bucket {gcs_bucket} (excluding src/outputs/) ...")
  # Use rsync --exclude to skip src/outputs/, src/__pycache__/, and src/AGENTS.md
  exclude_regex = r'^(outputs/.*|__pycache__/.*|AGENTS\.md)$'
  subprocess.run([
    'gsutil', '-m', 'rsync', '-r', '-x', exclude_regex, './src', f'gs://{gcs_bucket}/code/src'
  ], check=True)
  print(f"[INFO] Uploading requirements.txt to GCS bucket {gcs_bucket} ...")
  subprocess.run(['gsutil', 'cp', 'requirements.txt', f'gs://{gcs_bucket}/code/requirements.txt'], check=True)
  print("[INFO] Code and requirements.txt uploaded to GCS bucket.")

# 1. Verify file completeness on the GCS side
import os
import filecmp
import tempfile
import glob
def list_gcs_files(prefix):
  result = subprocess.run(['gsutil', 'ls', '-r', prefix], capture_output=True, text=True)
  return [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(prefix)]

def check_gcs_file_exists(local_path, gcs_path):
  result = subprocess.run(['gsutil', 'ls', gcs_path], capture_output=True, text=True)
  return gcs_path in result.stdout


# Verification logic: run only for full uploads
if not args.file:
  print("[INFO] Verifying GCS code/src/ file list matches local src/ ...")
  def is_excluded(path):
    return (
      path.startswith('outputs/') or
      path.startswith('__pycache__/') or
      path == 'AGENTS.md'
    )

  local_files = [os.path.relpath(f, './src') for f in glob.glob('./src/**/*', recursive=True) if os.path.isfile(f)]
  local_files = [f for f in local_files if not is_excluded(f)]
  gcs_files = list_gcs_files(f'gs://{gcs_bucket}/code/src/')
  gcs_files_rel = [f[len(f'gs://{gcs_bucket}/code/src/'):] for f in gcs_files if not f.endswith('/')]
  missing = set(local_files) - set(gcs_files_rel)
  if missing:
    print(f"[ERROR] Missing files in GCS: {missing}")
    exit(1)
  else:
    print("[INFO] All local src/ files found in GCS (excluding outputs/, __pycache__/, AGENTS.md).")

  if not check_gcs_file_exists('requirements.txt', f'gs://{gcs_bucket}/code/requirements.txt'):
    print("[ERROR] requirements.txt not found in GCS!")
    exit(1)
  else:
    print("[INFO] requirements.txt found in GCS.")

# 2. Check the bucket lifecycle policy
print("[INFO] Checking GCS bucket lifecycle policy ...")
result = subprocess.run(['gsutil', 'lifecycle', 'get', f'gs://{gcs_bucket}'], capture_output=True, text=True)
if 'No lifecycle configuration' in result.stdout:
  print("[WARN] No lifecycle policy set for this bucket. Consider adding one to control storage cost.")
else:
  print(f"[INFO] Bucket lifecycle policy: {result.stdout.strip()}")
