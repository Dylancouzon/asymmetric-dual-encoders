#!/usr/bin/env bash
# Every committed M8 suite. Nonzero exit if any fails.
#
# "A suite nobody runs is documentation, which is how test_freeze_guard.py stayed broken for two
# days after the teacher swap" (m7/CODEMAP.md). This is the only thing that runs them.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
fail=0
run() {
  echo "=== $1 ==="
  if $PY "$1"; then echo "--- ok"; else echo "--- FAILED"; fail=1; fi
  echo
}

run m8src/test_guards.py          # G1 + G2: the refusals that must happen
run m8src/decide.py               # the ship rule's own end-to-end self-test
run m8src/test_decide.py          # its reductions and, more importantly, its refusals

# NOT YET PORTED, and named here so the gap is visible rather than silent. Each becomes a `run`
# line when it lands; until then this block is the checklist (LEDGER section 6).
cat <<'PENDING'
=== NOT YET PORTED (LEDGER section 4.4 gap list) ===
  rule_audit.py            every mechanically-checkable rule against every arm family it binds
  test_freeze_binding.py   the refusals freeze.write must make on M8 paths
  test_final_guard.py      the one-shot access guard: peeled tag, spent receipt, pid lock,
                           infra-retry arity, corpus-only loading, BM25 package/config check
  test_dep_stats.py        already covered by m7src; re-point at M8's endpoints when they exist
  (test_decide.py has landed and now runs above.)
Running M8's confirmatory access before these exist is a LEDGER G3 violation.
PENDING
echo

exit $fail
