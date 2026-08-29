#!/usr/bin/env bash
# Move the rows `sweep.one` appends to m7/RESULTS.md into m8/RESULTS.md, and revert M7's file.
#
# WHY THIS EXISTS. `sweep.one` is frozen M7 code and appends a row for every run it completes to
# `m7/RESULTS.md` -- M7's experiment ledger, which G3 puts off limits to M8. Every M8 training
# batch therefore dirties it, and this has been fixed by hand three times now (the noise-floor
# arms, the B-leg chains, the B3 arms). Doing it by hand is how a batch eventually gets committed
# into M7's ledger by an absent-minded `git add -A`.
#
# Usage: m8src/reclaim_results_rows.sh "<heading for this batch>"
# Refuses if m7/RESULTS.md has changes that are NOT pure appends, because that would mean
# something edited M7's existing rows and reverting would destroy work rather than restore it.
set -euo pipefail
cd "$(dirname "$0")/.."
HEADING="${1:?usage: reclaim_results_rows.sh \"<heading>\"}"

if git diff --quiet m7/RESULTS.md; then
  echo "m7/RESULTS.md is clean; nothing to reclaim."
  exit 0
fi

# A pure append shows only '+' lines in the diff body. Any '-' line means an existing row moved.
if git diff -U0 m7/RESULTS.md | grep -qE '^-[^-]'; then
  echo "REFUSING: m7/RESULTS.md has REMOVED or CHANGED lines, not just appended ones." >&2
  echo "Reverting would discard someone's edit. Inspect it by hand:" >&2
  git diff --stat m7/RESULTS.md >&2
  exit 1
fi

ROWS="$(git diff m7/RESULTS.md | grep '^+' | grep -v '^+++' | sed 's/^+//')"
N="$(printf '%s\n' "$ROWS" | grep -c . || true)"
git checkout -- m7/RESULTS.md
{ printf '\n### %s\n\n' "$HEADING"; printf '%s\n' "$ROWS"; } >> m8/RESULTS.md
echo "reclaimed $N row(s) into m8/RESULTS.md; m7/RESULTS.md reverted (G3)."
