#!/usr/bin/env bash
# Every committed test suite, in one command, with a nonzero exit if any fails.
#
# Exists because `test_freeze_guard.py` had been FAILING since the 2026-08-26 teacher swap and
# nobody noticed: its fixtures named arctic-embed-l and stella's own `post_dense` value, both of
# which stopped being drift once stella became the active encoder. The ledger recorded
# "conformance 42/42" and said nothing about the other four suites, because nothing ran them
# together. A suite no one runs is documentation, not a test.
set -u
cd "$(dirname "$0")"
export M7_ENCODER=${M7_ENCODER:-stella-400M-v5} PYTHONPATH=m7src
PY=.venv/bin/python
rc=0
for t in test_conformance test_encoders test_dep_stats test_freeze_guard test_fusion_paths \
         test_init_rows test_signflip_calibration test_signflip_weaknull; do
  f="m7src/${t}.py"
  [ -f "$f" ] || { echo "SKIP  $t (missing)"; continue; }
  out=$($PY -u "$f" 2>&1); code=$?
  tail=$(echo "$out" | grep -iE "failure|FAIL|OK:|passed" | tail -1)
  if [ $code -eq 0 ]; then echo "PASS  $t   ${tail}"; else echo "FAIL  $t (exit $code)   ${tail}"; rc=1; fi
done
echo
[ $rc -eq 0 ] && echo "ALL SUITES PASS" || echo "SOME SUITES FAILED"
exit $rc
