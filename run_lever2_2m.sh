#!/usr/bin/env bash
# Lever #2, the 2m arm (runs because 500k resolved ABOVE the winner — LEDGER ordering rule).
# B 16k with the ~924K-span pool, then the A phase mirroring the winner arm. Best-step +
# compare stay manual.
set -u
cd /home/dylan/asymetric-dual-encoders
exec 9> /tmp/run_lever2_2m.lock
flock -n 9 || { echo "another instance holds the lock; refusing"; exit 1; }
export M7_ENCODER=stella-400M-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=========== p35b-2m (B 16k + 924K pseudo mix) then the A arm $(date -Is) ==========="
.venv/bin/python -u - <<'PY' || exit 1
import json, sys
sys.path.insert(0, "m7src"); sys.path.insert(0, "bench")
from dataclasses import fields
from train import Cfg
import sweep
stored = json.load(open("work/runs/p1-objB.json"))["cfg"]; stored.pop("run_id")
valid = {k: v for k, v in stored.items() if k in {f.name for f in fields(Cfg)}}
base = Cfg(**valid)
b = sweep.one("p35b-2m", base, objective="B", steps_b=16000, steps_a=0,
              b_pseudo_queries=2_000_000)
if b is None:
    raise SystemExit("B arm failed")
a = sweep.one("p35a-2m-1e3", base, objective="A", init="run:p35b-2m", steps_b=0,
              steps_a=2000, eval_every=500, hard_neg_k=0, lr=1e-3, lr_weights=1e-2,
              warmup_steps=200, lr_schedule="warmup_linear", b_pseudo_queries=0)
print(f"LEVER2-2M ARMS DONE b={b} a={a}")
PY
echo "LEVER2-2M COMPLETE $(date -Is)"
