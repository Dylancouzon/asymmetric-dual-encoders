#!/bin/bash
# W8 band 1, the remainder after family F. Registered order (registry `order`): F -> A -> G -> B
# -> E -> C -> D, with C CUT and E-bs128 CLOUD_ONLY (the A100, Dylan 2026-09-05) -- run_arm
# refuses E-bs128 on the box, so it is simply absent here.
#
# 11 arms x 5,000,000 examples (screen dose) at ~624 ex/s = ~2.2 h each, ~24 h total. Doc token
# caches are keyed on the row draw, so arms sharing n_docs=1,250,000 pay the ~4 min build once.
# Launched after 12/12 on the 90-step CUDA shape smoke with the current code
# (results/m10_arm_smoke.json, all_shapes_pass true) -- a code change is a new path.
cd /home/dylan/asymetric-dual-encoders
mkdir -p work/m10arms
for ARM in ANCHOR A1 A2 A3 G-384 G-1536 G-MLP B-100/0 B-50/50 D-NORM D-COV; do
  SAFE=$(echo "$ARM" | tr '/' '_')
  T0=$(date +%s)
  echo "=== $ARM start $(date)" >> work/m10arms/rest_chain.log
  .venv/bin/python -u m10src/run_arm.py "$ARM" --device cuda > "work/m10arms/${SAFE}.log" 2>&1
  RC=$?
  DUR=$(( $(date +%s) - T0 ))
  echo "=== $ARM exit $RC after ${DUR}s $(date)" >> work/m10arms/rest_chain.log
  # A per-arm result continues the chain; a SHARED PRECONDITION stops it. An OOM or a shape fault
  # two hours in is this arm's own outcome and the other ten still carry their contrasts. But a
  # non-zero exit within 120 s is a REFUSAL -- a missing verdict, a stale registry, an absent
  # artifact -- which every remaining arm will hit identically. The first version of this script
  # lacked the distinction and burned all 11 arms in 14 seconds on one missing F verdict.
  if [ $RC -ne 0 ]; then
    if [ $DUR -lt 120 ]; then
      echo "=== $ARM REFUSED in ${DUR}s -- a shared precondition, STOPPING the chain" >> work/m10arms/rest_chain.log
      tail -3 "work/m10arms/${SAFE}.log" >> work/m10arms/rest_chain.log
      exit $RC
    fi
    echo "=== $ARM FAILED rc=$RC after ${DUR}s, continuing" >> work/m10arms/rest_chain.log
  fi
done
echo "=== rest chain COMPLETE $(date)" >> work/m10arms/rest_chain.log
