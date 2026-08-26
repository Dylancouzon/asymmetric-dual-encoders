#!/usr/bin/env bash
# The phase-2 contrastive screen: six arms built around the learning rate, the decisive test of
# whether contrastive training collapses for a diagnosable reason (lr 30-300x above every published
# frozen-tower recipe) or for an unknown one. See program.phase2_screen for the arm design and
# program.may_invoke_contrastive_kill for the criterion the arms must satisfy before a kill.
#
# SMOKE FIRST. The screen's four hard-negative arms had never executed, and all four crashed on the
# same KL shape error after paying a mining pass each. sweep.smoke runs one representative arm at 90
# steps (~3 min) so a code fault costs minutes, not hours; sweep.grid then stops at the first arm
# that raises instead of repeating it.
#
# One process, arms in sequence: the GPU must not be shared (m7/LEDGER.md OOM incident).
set -euo pipefail
cd "$(dirname "$0")/m7src"
# 10 GB of VRAM, and this screen's A phase allocates a 3 GB negative bank plus score tiles of
# several different shapes plus eval buffers. Without expandable segments the caching allocator
# fragments, WSL's driver starts failing to make allocations resident (dmesg: dxgkio_make_resident
# -12), allocations spill to host memory, and the run degrades to 100% GPU utilisation at ~0%
# memory bandwidth -- it looks busy and makes no progress. Numerics are unaffected; every arm runs
# with it, so arms stay comparable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ../.venv/bin/python -u -c "
import program, sweep
# The representative arm is a hard-negative one: that is the path with no execution history.
sweep.smoke(program.BASE, {'objective': 'C', 'hard_neg_k': 16, 'hard_neg_source': 'teacher',
                           'fn_margin': 0.05, 'lr': 5e-5, 'lr_weights': 5e-4,
                           'warmup_steps': 500, 'lr_schedule': 'warmup_linear'})
program.save('phase2_screen', program.phase2_screen(program.BASE))
"
