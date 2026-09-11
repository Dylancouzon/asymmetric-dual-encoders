#!/usr/bin/env bash
# Run AFTER mounting persistent storage at /home/dylan and restoring the checkout.
# Requires python3 with pip if pinned uv is absent. uv downloads CPython 3.12.14
# if necessary. No experiment is run.
# Usage: bash scripts/m13_cloud_bootstrap.sh [repository]
set -euo pipefail

repo=$(realpath "${1:-/home/dylan/asymetric-dual-encoders}")
requirements="$repo/m13/cloud_requirements.txt"
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'This environment is pinned for Linux x86_64.' >&2; exit 2;
}
[[ "$repo" == /home/dylan/* && -f "$requirements" ]] || {
  echo 'Expected a restored M13 checkout below persistent /home/dylan.' >&2; exit 2;
}
command -v mountpoint >/dev/null || { echo 'mountpoint is required.' >&2; exit 2; }
mountpoint -q /home/dylan || {
  echo '/home/dylan must be the persistent storage mount, not container disk.' >&2; exit 2;
}

export UV_CACHE_DIR=/home/dylan/.cache/uv
export UV_PYTHON_INSTALL_DIR=/home/dylan/.local/share/uv/python
export HF_HOME=/home/dylan/.cache/huggingface
export XDG_CACHE_HOME=/home/dylan/.cache
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"
uv_command=$(command -v uv || true)
if [[ -z "$uv_command" || $("$uv_command" --version) != 'uv 0.12.5 '* ]]; then
  uv_tools=/home/dylan/.local/share/m13-uv-0.12.5
  if [[ ! -e "$uv_tools" ]]; then
    python3 -m pip install --disable-pip-version-check --index-url https://pypi.org/simple \
      --target "$uv_tools" 'uv==0.12.5'
  fi
  uv_command="$uv_tools/bin/uv"
fi
[[ -x "$uv_command" && $("$uv_command" --version) == 'uv 0.12.5 '* ]] || {
  echo 'Pinned uv 0.12.5 installation is missing or incomplete.' >&2; exit 2;
}
venv="$repo/.venv"
# Do not follow a local workstation symlink or replace an existing environment.
[[ ! -L "$venv" ]] || { echo 'Refusing an existing .venv symlink.' >&2; exit 2; }
if [[ -e "$venv" ]]; then
  [[ -x "$venv/bin/python" && -f "$venv/pyvenv.cfg" ]] || {
    echo 'Existing .venv is not a usable virtual environment.' >&2; exit 2;
  }
  "$venv/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 12, 14), sys.version'
else
  "$uv_command" venv --python 3.12.14 "$venv"
fi

# Install CUDA torch first from its official wheel index. The second operation
# retains this exact installed build while resolving the remaining PyPI pins.
"$uv_command" pip install --python "$venv/bin/python" --index-url https://download.pytorch.org/whl/cu126 'torch==2.8.0+cu126'
"$uv_command" pip install --python "$venv/bin/python" --index-url https://pypi.org/simple -r "$requirements"
"$uv_command" pip check --python "$venv/bin/python"

# Offline import/hardware check only: no model or protected-data reader is used.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$venv/bin/python" - "$requirements" <<'PY'
import importlib
import importlib.metadata as metadata
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line and not line.startswith('#'):
        name, version = line.split('==')
        assert metadata.version(name) == version, (name, metadata.version(name), version)
for module in ('transformers', 'sentence_transformers', 'datasets', 'numpy',
               'scipy', 'sklearn', 'onnx', 'onnxruntime', 'fastembed',
               'pytest', 'pytrec_eval', 'qdrant_client'):
    importlib.import_module(module)
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: check pod GPU and driver'
assert torch.cuda.device_count() == 1, 'Expected exactly one visible GPU'
gpu = torch.cuda.get_device_properties(0)
assert 'A100' in gpu.name, f'Expected A100, got {gpu.name}'
assert 79 * 1024**3 <= gpu.total_memory <= 81 * 1024**3, (
    'Expected approximately 80 GiB GPU memory', gpu.total_memory)
assert torch.version.cuda == '12.6', torch.version.cuda
assert torch.cuda.is_bf16_supported(), 'Registered training requires CUDA bf16'
x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
assert (x @ x).float().mean().item() == 32
torch.cuda.synchronize()
print('Python:', sys.version.split()[0], 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), 'bytes:', torch.cuda.get_device_properties(0).total_memory)
print('Pinned packages, imports and CUDA bf16 allocation passed.')
PY
echo 'Environment ready. Restore admitted artifacts and HF fixtures before run_checks.sh.'
echo "Use HF_HOME=$HF_HOME and $venv/bin/python for subsequent commands."
