"""LEDGER G1 -- no probe runs before its bar is committed and pushed. HARDENED 2026-08-29.

Why this is code. A pre-registered rule a session can re-read in its own favour is not a
pre-registration (m7/CODEMAP.md, "Decision executors"). M7's step-selection rule was found
unapplied *by accident*, after it had already governed four arms and a promoted adoption; the fix
was `rule_audit.py`, and the lesson generalises: anything that depends on the ledger saying
something must be a program that reads it.

WHAT THE 2026-08-29 GATE CHANGED. v1 parsed the ledger's markdown tables and gated ENTRY POINTS.
Both were wrong:
  * prose fields are not a registration -- "approximately 0.45" and "budget split as registered"
    passed a presence check while leaving the decision open;
  * gating entry points is bypassed by calling a helper or an evaluator directly.
So the authority is now `m8/registry.json`, a machine-readable file whose rows must be complete,
and the gate is a STAMP: `stamp(probe_id)` returns a provenance block that every result artifact
must carry, and `write_result()` refuses to write one without it. A metric that reaches disk
without a registry sha did not run under a registration.

What it enforces, for a probe id:
  1. a complete row in `m8/registry.json` -- bar, endpoint, comparator, multiplicity,
     no_survivor -- with no empty or placeholder cell;
  2. no `TBD` anywhere in the bar. That placeholder exists precisely so that running before the
     noise floor is measured is impossible (LEDGER 4.7 / G4);
  3. `m8/LEDGER.md` and `m8/registry.json` both clean in git -- the registration must be the
     committed version, not the version on the disk of the session about to read the result;
  4. HEAD present on a remote branch. A bar that exists only locally can be rewritten by the
     same session that saw the number.

It deliberately does NOT try to check that the probe's code measures what the row says. That is
not mechanically checkable, and LEDGER 4.4 requires such rules to be listed as unverifiable rather
than reported as passes.
"""
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "m8" / "LEDGER.md"
REGISTRY = REPO / "m8" / "registry.json"
GUARDED_FILES = ("m8/LEDGER.md", "m8/registry.json")

REQUIRED = ("bar", "endpoint", "comparator", "multiplicity", "no_survivor")
_PLACEHOLDER = {"", "-", "--", "tbd", "todo", "n/a"}


class ProbeNotRegistered(RuntimeError):
    """Raised instead of running an unregistered or under-registered probe."""


def _sha256(p: Path):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def registry():
    return json.loads(REGISTRY.read_text())


def _git(*args):
    return subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True,
                          text=True, check=False).stdout.strip()


def check_committed():
    """The registration must be the committed one, and the commit must be pushed."""
    problems = []
    dirty = _git("status", "--porcelain", "--", *GUARDED_FILES)
    if dirty:
        problems.append(f"uncommitted changes in {GUARDED_FILES}: commit and push the "
                        f"registration before reading any number it binds (LEDGER G1)")
    head = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", head):
        problems.append(f"HEAD {head[:8]} is not on any remote branch -- push before running "
                        f"(a bar that exists only locally can be rewritten by the session that "
                        f"saw the number)")
    return problems, head


def assert_registered(probe_id, *, strict_commit=True):
    """Refuse unless `probe_id` is completely registered at the current commit. Returns a
    provenance stamp: the registry sha, the ledger sha, the commit, and the row itself."""
    reg = registry()
    probes = reg["probes"]
    row = probes.get(probe_id)
    if row is None:
        match = [k for k in probes if k.lower() == str(probe_id).lower()]
        row = probes[match[0]] if match else None
        probe_id = match[0] if match else probe_id
    if row is None:
        raise ProbeNotRegistered(
            f"probe {probe_id!r} has no row in m8/registry.json. Registered: {sorted(probes)}. "
            f"Registering it is a LEDGER 15 amendment, made BEFORE the run.")

    missing = [f for f in REQUIRED
               if str(row.get(f, "")).strip().lower() in _PLACEHOLDER]
    if missing:
        raise ProbeNotRegistered(
            f"probe {probe_id!r} is registered but INCOMPLETE: missing or placeholder {missing}. "
            f"LEDGER G1 requires bar, endpoint, comparator, multiplicity and no_survivor. A stub "
            f"is a placeholder for a registration, not a registration.")

    bar = str(row["bar"])
    if "tbd" in bar.lower():
        raise ProbeNotRegistered(
            f"probe {probe_id!r} has bar {bar!r}. A TBD bar is a REFUSAL, not a placeholder to "
            f"run through: measure the noise floor first (LEDGER 4.7 / G4), then freeze the bar "
            f"at max(planning_minimum, 2 x floor) by a LEDGER 15 amendment, then run.")
    # A bar can be fully worded and still be unfinished: B3's template is fixed but its numeric
    # floor term is not. `bar_pending` names what is still missing, and having it is a refusal --
    # otherwise a bar that merely READS complete would pass a keyword check.
    if row.get("bar_pending"):
        raise ProbeNotRegistered(
            f"probe {probe_id!r} declares bar_pending={row['bar_pending']!r}: the wording is "
            f"fixed but a term of the bar is not yet a number. Complete it by a LEDGER 15 "
            f"amendment before running.")
    if not row.get("noise_floor_exempt", False):
        floor = REPO / "results" / "m8_noise_floor.json"
        if not floor.exists():
            raise ProbeNotRegistered(
                f"probe {probe_id!r} is not noise-floor exempt and results/m8_noise_floor.json "
                f"does not exist. LEDGER G4: the floor is measured before any bar that reads it "
                f"is frozen. Probes whose bars do not need calibration must say so explicitly "
                f"with noise_floor_exempt + exempt_reason.")

    if strict_commit:
        problems, head = check_committed()
        if problems:
            raise ProbeNotRegistered("; ".join(problems))
    else:
        head = _git("rev-parse", "HEAD")

    return {"probe": probe_id, "row": row, "wave": row.get("wave"),
            "registry_sha256": _sha256(REGISTRY), "ledger_sha256": _sha256(LEDGER),
            "ledger_commit": head,
            "_note": "LEDGER G1 stamp. A result artifact without this block did not run under a "
                     "registration and may not be cited."}


def stamp(probe_id, *, strict_commit=True):
    """Alias with the name the call sites use."""
    return assert_registered(probe_id, strict_commit=strict_commit)


def write_result(path, payload, probe_id, *, strict_commit=True):
    """The ONLY sanctioned way for a probe to write a result. Refuses without a valid stamp, and
    embeds it, so gating does not depend on an entry point remembering to ask."""
    prov = assert_registered(probe_id, strict_commit=strict_commit)
    path = Path(path)
    body = dict(payload)
    body["_registration"] = prov
    path.write_text(json.dumps(body, indent=2, default=str))
    return prov


def classify_change(key, reg=None):
    """LEDGER 5.4, mechanically. -> 'qualifying_table' | 'qualifying_non_table' |
    'not_qualifying' | 'unknown'. An unknown key FAILS the qualifying condition; classification
    happens at manifest time, before the access, so a key cannot be argued into a category after a
    number exists."""
    s = (reg or registry())["ship_rule"]
    if key in s["qualifying_table_keys"]:
        return "qualifying_table"
    if key in s["qualifying_non_table_keys"]:
        return "qualifying_non_table"
    if key in s["not_qualifying_keys"]:
        return "not_qualifying"
    if any(fnmatch.fnmatch(key, p) for p in s.get("not_qualifying_patterns", [])):
        return "not_qualifying"
    return "unknown"


def main(argv):
    if len(argv) > 1:
        print(json.dumps(assert_registered(argv[1]), indent=2))
        return 0
    reg = registry()
    problems, head = check_committed()
    print(f"registry {_sha256(REGISTRY)[:12]}  ledger commit {head[:8]}  "
          f"({'clean+pushed' if not problems else '; '.join(problems)})")
    runnable = 0
    for pid in reg["probes"]:
        try:
            assert_registered(pid, strict_commit=False)
            wave = reg["probes"][pid].get("wave")
            print(f"  {pid:12s} wave {wave}  RUNNABLE")
            runnable += 1
        except ProbeNotRegistered as e:
            reason = str(e).split("\n")[0]
            reason = reason[reason.find("has bar") if "has bar" in reason else 0:][:88]
            print(f"  {pid:12s} wave {reg['probes'][pid].get('wave')}  REFUSED -- {reason}")
    print(f"\n{runnable} runnable / {len(reg['probes'])} registered")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
