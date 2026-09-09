#!/bin/bash
# The two DESCRIPTIVE runs ruled by gpt-6-astra on 2026-09-09, under Dylan's delegation
# ("let's allow it to make the decision on those 2 points"). Neither selects anything and neither
# can re-decide a verdict -- see `screen_registry.json` `descriptive_runs` and each arm's entry.
#
#   ANCHOR-seed1   5,000,000 examples, seed 1   ~2.7 h   decision 2: anchor sensitivity
#   A3-20M        20,000,000 examples, seed 0  ~10.9 h   decision 1: A4's corpus vs A3's at 20M,
#                                                        read against the COMPLETED F-bge-small
#
# Short arm first: its number is a reporting obligation for every interval in the paper, and it
# lands in under three hours. Launched after both passed a 90-step CUDA smoke on the current code
# with the right seeds (`work/m10arms/newarms_smoke.log`).
cd /home/dylan/asymetric-dual-encoders
mkdir -p work/m10arms
for ARM in ANCHOR-seed1 A3-20M; do
  T0=$(date +%s)
  echo "=== $ARM start $(date)" >> work/m10arms/descriptive_chain.log
  .venv/bin/python -u m10src/run_arm.py "$ARM" --device cuda > "work/m10arms/${ARM}.log" 2>&1
  RC=$?
  DUR=$(( $(date +%s) - T0 ))
  echo "=== $ARM exit $RC after ${DUR}s $(date)" >> work/m10arms/descriptive_chain.log
  # Same rule as `run_rest.sh`: a non-zero exit inside 120 s is a REFUSAL (a shared precondition
  # every later arm hits identically) and stops the chain; a later one is this arm's own outcome.
  if [ $RC -ne 0 ]; then
    if [ $DUR -lt 120 ]; then
      echo "=== $ARM REFUSED in ${DUR}s -- a shared precondition, STOPPING" >> work/m10arms/descriptive_chain.log
      tail -3 "work/m10arms/${ARM}.log" >> work/m10arms/descriptive_chain.log
      exit $RC
    fi
    echo "=== $ARM FAILED rc=$RC after ${DUR}s, continuing" >> work/m10arms/descriptive_chain.log
  fi
done
echo "=== descriptive chain complete $(date)" >> work/m10arms/descriptive_chain.log
