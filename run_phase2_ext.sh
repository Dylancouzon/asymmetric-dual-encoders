#!/usr/bin/env bash
# Learning-rate turnover: the screen's best arm was at the top of the published range and still
# rising, so this extends to 3e-4 / 1e-3 / 3e-3 with random negatives only. See
# program.phase2_screen_ext. Same allocator setting as the screen, one process, arms in sequence.
set -euo pipefail
cd "$(dirname "$0")/m7src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ../.venv/bin/python -u -c "
import program
program.save('phase2_screen_ext', program.phase2_screen_ext(program.BASE))
"
