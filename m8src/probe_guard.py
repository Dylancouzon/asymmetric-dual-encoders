"""LEDGER G1 -- no probe runs before its bar is committed and pushed.

Why this is code. A pre-registered rule a session can re-read in its own favour is not a
pre-registration (m7/CODEMAP.md, "Decision executors"). M7's step-selection rule was found
unapplied *by accident*, after it had already governed four arms and a promoted adoption; the fix
was `rule_audit.py`, and the lesson generalises: the ledger is a text file, so anything that
depends on the ledger saying something must be a program that reads it.

What it enforces, for a probe id:
  1. The id has a row in `m8/LEDGER.md` section 9 with a non-empty bar, endpoint, comparator,
     multiplicity and no-survivor outcome. A row whose bar is still `TBD-noise-floor` is REFUSED:
     that placeholder exists precisely so that running before the noise floor is measured is
     impossible (LEDGER 4.7 / G4).
  2. `m8/LEDGER.md` has no uncommitted modification -- the registration must be the version in
     git, not the version on the disk of the session about to read the result.
  3. HEAD is pushed to the remote. A bar that exists only locally can be rewritten by the same
     session that saw the number.

It deliberately does NOT try to check that the probe's code actually measures what the row says;
that is not mechanically checkable, and LEDGER 4.4 requires such rules to be listed as
unverifiable rather than reported as passes.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "m8" / "LEDGER.md"

REQUIRED = ("bar", "endpoint", "comparator", "multiplicity", "no survivor")
# "none" is NOT empty: for a diagnostic with a single arm, "multiplicity: none" is a real
# registration and refusing it would push the registration into prose, where it stops being
# checkable. Only a blank, a dash or a placeholder is a refusal.
_EMPTY = {"", "-", "--", "tbd", "todo"}


class ProbeNotRegistered(RuntimeError):
    """Raised instead of running an unregistered or under-registered probe."""


def _section9(text):
    m = re.search(r"^## 9\. Probe registrations\s*$", text, re.M)
    if not m:
        raise ProbeNotRegistered("m8/LEDGER.md has no section 9 (Probe registrations)")
    nxt = re.search(r"^## 10\.", text[m.end():], re.M)
    return text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]


def _rows(section):
    """-> {probe_id: {column_header: cell}} for every markdown table row in section 9."""
    out, headers = {}, None
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            headers = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if headers is None:
            headers = [c.lower().strip("* ") for c in cells]
            continue
        if all(set(c) <= set("-: ") for c in cells):      # the |---|---| separator
            continue
        if len(cells) != len(headers):
            continue
        pid = cells[0].strip("*` ")
        if not pid:
            continue
        out[pid] = dict(zip(headers, cells))
    return out


def registrations(text=None):
    text = text if text is not None else LEDGER.read_text()
    return _rows(_section9(text))


def _git(*args):
    return subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True,
                          text=True, check=False).stdout.strip()


def check_ledger_committed():
    """The registration must be the committed one, and the commit must be pushed."""
    problems = []
    if _git("status", "--porcelain", "--", "m8/LEDGER.md"):
        problems.append("m8/LEDGER.md has uncommitted changes -- commit and push the "
                        "registration before reading any number it binds (LEDGER G1)")
    head = _git("rev-parse", "HEAD")
    remotes = _git("branch", "-r", "--contains", head)
    if not remotes:
        problems.append(f"HEAD {head[:8]} is not on any remote branch -- push before running "
                        "(a bar that exists only locally can be rewritten by the session that "
                        "saw the number)")
    return problems, head


def assert_registered(probe_id, *, strict_commit=True):
    """Refuse unless `probe_id` is fully registered at the current commit. Returns its row."""
    regs = registrations()
    row = None
    for pid, r in regs.items():
        if pid.lower() == probe_id.lower():
            row = r
            break
    if row is None:
        raise ProbeNotRegistered(
            f"probe {probe_id!r} has no row in m8/LEDGER.md section 9. Registered: "
            f"{sorted(regs)}. Registering it is a LEDGER 15 amendment, made BEFORE the run.")
    missing = [f for f in REQUIRED
               if not any(f in h for h in row) or
               _cell(row, f).lower() in _EMPTY]
    if missing:
        raise ProbeNotRegistered(
            f"probe {probe_id!r} is registered but incomplete: missing/empty {missing}. "
            f"LEDGER G1 requires bar, endpoint, comparator, multiplicity and no-survivor outcome.")
    bar = _cell(row, "bar")
    if "tbd" in bar.lower():
        raise ProbeNotRegistered(
            f"probe {probe_id!r} has bar {bar!r}. A TBD bar is a refusal, not a placeholder to run "
            f"through: measure the noise floor first (LEDGER 4.7 / G4), then freeze the bar at "
            f">=2x the floor by a LEDGER 15 amendment, then run.")
    if strict_commit:
        problems, head = check_ledger_committed()
        if problems:
            raise ProbeNotRegistered("; ".join(problems))
    else:
        head = _git("rev-parse", "HEAD")
    return {"probe": probe_id, "row": row, "ledger_commit": head}


def _cell(row, field):
    for h, v in row.items():
        if field in h:
            return v
    return ""


def main(argv):
    import json
    if len(argv) > 1:
        print(json.dumps(assert_registered(argv[1]), indent=2))
        return 0
    regs = registrations()
    problems, head = check_ledger_committed()
    print(f"ledger commit {head[:8]}  ({'clean+pushed' if not problems else '; '.join(problems)})")
    for pid, row in regs.items():
        try:
            assert_registered(pid, strict_commit=False)
            print(f"  {pid:6s} REGISTERED")
        except ProbeNotRegistered as e:
            print(f"  {pid:6s} REFUSED  -- {str(e).split(chr(10))[0][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
