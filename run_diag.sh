#!/usr/bin/env bash
# The two written-but-never-run cheap jobs, strictly sequential (one GPU job at a time).
#   0.1-full : the load-bearing Stage 0.1 ridge result on the FULL pinned dev suite
#              -> results/m7_stage0_ridge.json (does not exist; claim unverifiable from GitHub)
#   diag     : the score geometry behind the contrastive collapse, incl. the fn_margin mask rate
#              that ran UNOBSERVED through the whole phase-1 grid -> results/m7_diag_scores.json
set -u
cd /home/dylan/asymetric-dual-encoders/m7src
PY=../.venv/bin/python
L=../logs
export HF_HUB_ENABLE_HF_TRANSFER=1
for job in "ridge_full_eval.py teacher noprefix" "diag_scores.py 2000"; do
  set -- $job
  echo "=========== $* $(date -Is) ===========" | tee -a $L/diag.log
  $PY "$@" >> $L/diag.log 2>&1
  echo "--- exit $? $(date -Is) mem: $(free -m | awk '/Mem:/{print $3"/"$2" MB"}')" | tee -a $L/diag.log
done
echo "DIAG COMPLETE $(date -Is)" | tee -a $L/diag.log
