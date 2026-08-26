#!/usr/bin/env bash
# The stella confirmation chain. Waits for the swap encode to finish, smokes the training code
# path (new: the B2 banned-rows mask, the refreshed kept.json, the stella pool), then runs the
# B checkpoint and the pre-registered lr-band confirmation arms. Gate + CIs + the winner's
# best-step re-run stay manual: they are selection decisions, not compute.
set -u
cd /home/dylan/asymetric-dual-encoders
export M7_ENCODER=stella-400M-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "waiting for the swap to complete $(date -Is)"
until grep -q "STELLA SWAP COMPLETE" logs/stella_swap.log; do sleep 60; done
echo "=========== smoke: B + A paths under stella + mask $(date -Is) ==========="
.venv/bin/python -u - <<'PY' || exit 1
import json, sys
sys.path.insert(0, "m7src"); sys.path.insert(0, "bench")
from dataclasses import fields
from train import Cfg
import sweep
stored = json.load(open("work/runs/p1-objB.json"))["cfg"]; stored.pop("run_id")
valid = {k: v for k, v in stored.items() if k in {f.name for f in fields(Cfg)}}
base = Cfg(**valid)
sweep.smoke(base, {"objective": "C", "hard_neg_k": 0})   # C exercises both B and A steps
PY
echo "=========== s1-objB + confirmation arms $(date -Is) ==========="
.venv/bin/python -u - <<'PY' || exit 1
import json, sys
sys.path.insert(0, "m7src"); sys.path.insert(0, "bench")
from dataclasses import fields
from train import Cfg
import program
stored = json.load(open("work/runs/p1-objB.json"))["cfg"]; stored.pop("run_id")
valid = {k: v for k, v in stored.items() if k in {f.name for f in fields(Cfg)}}
program.stella_confirm(Cfg(**valid))
PY
echo "STELLA CONFIRM COMPLETE $(date -Is)"
