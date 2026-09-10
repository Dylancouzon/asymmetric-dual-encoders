#!/usr/bin/env bash
# Active CPU checks. Historical/cache-dependent suites are listed in HARNESS.md.
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "Missing $PY; see HARNESS.md for environment prerequisites." >&2
  exit 2
fi
export PYTHONPATH="$PWD:$PWD/m7src"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

failed=0
run() {
  local label="$1"
  shift
  echo "Running $label"
  if "$PY" "$@"; then
    echo "PASS: $label"
  else
    echo "FAIL: $label" >&2
    failed=1
  fi
}

run "M10 data, training and decision checks" -m pytest -q -ra m10src
run "M9 registered statistics" m9src/test_final_stats.py
run "M12 fusion parity" m12src/test_qfusion.py

if [ "$failed" -eq 0 ]; then
  echo "All active CPU checks passed."
else
  echo "Active checks failed; see the errors above and HARNESS.md prerequisites." >&2
fi
exit "$failed"
