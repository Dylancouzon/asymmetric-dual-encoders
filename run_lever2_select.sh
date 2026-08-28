#!/usr/bin/env bash
# Lever #2 selection: best-step re-run (proxy picked step 1500) + the pre-registered full-suite
# comparison vs the winner. Selection decisions per the LEDGER protocol.
set -u
cd /home/dylan/asymetric-dual-encoders
exec 9> /tmp/run_lever2_select.lock
flock -n 9 || { echo "another instance holds the lock; refusing"; exit 1; }
export M7_ENCODER=stella-400M-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=========== p35w-500k-s1500 (best-step re-run) $(date -Is) ==========="
.venv/bin/python -u - <<'PY' || exit 1
import json, sys
sys.path.insert(0, "m7src"); sys.path.insert(0, "bench")
from dataclasses import fields
from train import Cfg
import sweep
stored = json.load(open("work/runs/p1-objB.json"))["cfg"]; stored.pop("run_id")
valid = {k: v for k, v in stored.items() if k in {f.name for f in fields(Cfg)}}
base = Cfg(**valid)
a = sweep.one("p35w-500k-s1500", base, objective="A", init="run:p35b-500k", steps_b=0,
              steps_a=1500, eval_every=500, hard_neg_k=0, lr=1e-3, lr_weights=1e-2,
              warmup_steps=200, lr_schedule="warmup_linear", b_pseudo_queries=0)
if a is None:
    raise SystemExit("re-run failed")
print(f"BEST-STEP RERUN DONE a={a}")
PY
echo "=========== compare_release p35w-500k-s1500 vs s2w-1e3-s1000 $(date -Is) ==========="
(cd m7src && ../.venv/bin/python -u compare_release.py p35w-500k-s1500 s2w-1e3-s1000) || exit 1
echo "LEVER2 SELECT COMPLETE $(date -Is)"
