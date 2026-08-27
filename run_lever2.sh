#!/usr/bin/env bash
# Capacity lever #2 (pseudo-query coverage), per the LEDGER 2026-08-27 pre-registration.
# Waits for the doc2query probe to release the GPU, smokes the B-with-pseudo path (never
# executed before), then runs the 500k arm: B 8k with the decontaminated pseudo mix, then the
# A phase exactly mirroring the winner arm. Best-step re-run + compare_release stay manual:
# they are selection decisions, not compute.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "waiting for the doc2query probe to finish $(date -Is)"
until grep -qE "d2q_minus_orig|Traceback" logs/doc2query_probe.log; do sleep 60; done
echo "waiting for the pseudo-query decontam pass $(date -Is)"
until [ -f work/pseudoq/kept-pseudoq-500000-0.json ]; do sleep 30; done
echo "=========== smoke: B path with pseudo-queries $(date -Is) ==========="
.venv/bin/python -u - <<'PY' || exit 1
import json, sys
sys.path.insert(0, "m7src"); sys.path.insert(0, "bench")
from dataclasses import fields
from train import Cfg
import sweep
stored = json.load(open("work/runs/p1-objB.json"))["cfg"]; stored.pop("run_id")
valid = {k: v for k, v in stored.items() if k in {f.name for f in fields(Cfg)}}
base = Cfg(**valid)
sweep.smoke(base, {"objective": "B", "b_pseudo_queries": 500_000})
PY
echo "=========== p35b-500k (B 8k + pseudo mix) then the A arm $(date -Is) ==========="
.venv/bin/python -u - <<'PY' || exit 1
import json, sys
sys.path.insert(0, "m7src"); sys.path.insert(0, "bench")
from dataclasses import fields
from train import Cfg
import sweep
stored = json.load(open("work/runs/p1-objB.json"))["cfg"]; stored.pop("run_id")
valid = {k: v for k, v in stored.items() if k in {f.name for f in fields(Cfg)}}
base = Cfg(**valid)
b = sweep.one("p35b-500k", base, objective="B", steps_b=8000, steps_a=0,
              b_pseudo_queries=500_000)
if b is None:
    raise SystemExit("B arm failed")
a = sweep.one("p35a-500k-1e3", base, objective="A", init="run:p35b-500k", steps_b=0,
              steps_a=2000, eval_every=500, hard_neg_k=0, lr=1e-3, lr_weights=1e-2,
              warmup_steps=200, lr_schedule="warmup_linear", b_pseudo_queries=0)
print(f"LEVER2 ARMS DONE b={b} a={a}")
PY
echo "LEVER2 COMPLETE $(date -Is)"
