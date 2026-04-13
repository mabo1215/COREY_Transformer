#!/usr/bin/env bash
set -euo pipefail

MM="${MM:-$HOME/.corey-wsl-tools/bin/micromamba}"
ROOT="${MAMBA_ROOT_PREFIX:-$HOME/.corey-micromamba}"
ENV_NAME="${ENV_NAME:-corey-cuda128}"
HF_HOME="${HF_HOME:-/mnt/c/Users/295461/.cache/huggingface}"
MAX_JOBS="${MAX_JOBS:-1}"
MAMBA_REF="${MAMBA_REF:-v2.3.1}"
MAMBA_GENCODE="${MAMBA_GENCODE:--gencode arch=compute_89,code=sm_89}"

if [[ ! -x "$MM" ]]; then
  printf '[error] micromamba not found at %s\n' "$MM" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

git clone --depth 1 --branch "$MAMBA_REF" https://github.com/state-spaces/mamba.git "$tmp_dir"
cd "$tmp_dir"

python - <<'PY'
from pathlib import Path

path = Path("setup.py")
text = path.read_text()
start = '        if bare_metal_version <= Version("12.9"):\n            cc_flag.append("-gencode")\n            cc_flag.append("arch=compute_53,code=sm_53")'
end = '    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as'
start_index = text.index(start)
end_index = text.index(end)
replacement = '''        custom_cc_flags = os.environ.get("MAMBA_CUDA_GENCODE", "").strip().split()
        if custom_cc_flags:
            cc_flag.extend(custom_cc_flags)
        else:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_89,code=sm_89")

'''
path.write_text(text[:start_index] + replacement + text[end_index:])
PY

export HF_HOME
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDAARCHS="89"

MAMBA_FORCE_BUILD=TRUE MAMBA_CUDA_GENCODE="$MAMBA_GENCODE" \
  "$MM" run -r "$ROOT" -n "$ENV_NAME" \
  python -m pip install -v --no-build-isolation --no-deps --no-cache-dir --force-reinstall .