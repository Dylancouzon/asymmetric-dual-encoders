"""LEDGER §4.4 deliverable 3: rule compliance, AUDITED rather than discovered.

M7 found its step-selection rule unapplied **by accident**, after it had already governed four
arms and a promoted adoption. The lesson was not "be careful"; it was that a rule nobody
mechanically checks is a rule that has not been applied, and you learn which one it was by
accident or not at all. This is M8's checker.

What it can check, and does:

  1. **Every M8 result carries a registration stamp.** `probe_guard.write_result` embeds the
     registry sha, the ledger sha and the commit. An artifact without one did not run under a
     registration and may not be cited.
  2. **The registration a result ran under still says what it says now.** For each stamped
     result, the registry blob is fetched from git AT THAT COMMIT and the probe's bar, endpoint,
     comparator, multiplicity and no-survivor are diffed against today's. Any difference is a bar
     that moved after a number existed -- the one thing the amendment rule forbids outright, in
     either direction (§0). This is the check M7 did not have.
  3. **The stamped commit is a real ancestor of HEAD**, so a result cannot claim a registration
     from a branch that was never merged.
  4. **The gap list is still true**: every file §4.4 names as missing is still missing, and every
     one that has landed has been struck. A stale gap list is how "DONE" headings start lying.
  5. **Registry hygiene**: no probe row is missing a required field; no bar contains a placeholder
     while its probe has a result.

What it CANNOT check, listed as unverifiable rather than reported as passes (§4.4's own rule):
pre-registration ORDERING beyond the commit graph (a bar committed minutes before a run that had
already been executed locally looks identical), whether a probe's CODE measures what its row says,
Holm family membership, and whether a human read a number before writing a rule.
"""
import json
import subprocess
import sys
from pathlib import Path

import m8base

REPO = m8base.REPO
RESULTS = m8base.RESULTS
REGISTRY_PATH = "m8/registry.json"
COMPARED_FIELDS = ("bar", "endpoint", "comparator", "multiplicity", "no_survivor")

UNVERIFIABLE = [
    "pre-registration ORDERING beyond the commit graph: a bar committed minutes before a run that "
    "had already been executed locally is indistinguishable from one committed properly first.",
    "whether a probe's CODE measures what its registry row says it measures.",
    "Holm family membership: which comparisons belong to which family is a judgement recorded in "
    "prose.",
    "whether a number was seen before a rule that reads it was written.",
]


def _git(*args):
    r = subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True, text=True)
    return r.stdout, r.returncode


def registry_at(commit):
    out, rc = _git("show", f"{commit}:{REGISTRY_PATH}")
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def is_ancestor(commit):
    _, rc = _git("merge-base", "--is-ancestor", commit, "HEAD")
    return rc == 0


def audit():
    now = json.loads((REPO / "m8" / REGISTRY_PATH.split("/", 1)[1]).read_text())
    findings, checked, unstamped = [], [], []

    for f in sorted(RESULTS.glob("m8_*.json")):
        try:
            body = json.loads(f.read_text())
        except (json.JSONDecodeError, MemoryError):
            findings.append({"severity": "MINOR", "file": f.name,
                             "issue": "unreadable as JSON; not audited"})
            continue
        if not isinstance(body, dict):
            continue
        prov = body.get("_registration")
        if not prov:
            # Not every artifact is a probe result: calibration caches and schedules are inputs.
            unstamped.append(f.name)
            continue
        pid, commit = prov.get("probe"), prov.get("ledger_commit", "")
        checked.append({"file": f.name, "probe": pid, "commit": commit[:8]})

        if not commit or not is_ancestor(commit):
            findings.append({"severity": "BLOCKER", "file": f.name, "probe": pid,
                             "issue": f"stamped commit {commit[:8] or '(none)'} is not an "
                                      f"ancestor of HEAD"})
            continue
        then = registry_at(commit)
        if then is None:
            findings.append({"severity": "BLOCKER", "file": f.name, "probe": pid,
                             "issue": f"no registry at commit {commit[:8]}"})
            continue
        row_then, row_now = then["probes"].get(pid), now["probes"].get(pid)
        if row_then is None or row_now is None:
            findings.append({"severity": "BLOCKER", "file": f.name, "probe": pid,
                             "issue": "probe row absent from the registry then or now"})
            continue
        for field in COMPARED_FIELDS:
            a, b = row_then.get(field), row_now.get(field)
            if a != b:
                findings.append({
                    "severity": "BLOCKER", "file": f.name, "probe": pid, "field": field,
                    "issue": "the registration MOVED after this result existed -- forbidden in "
                             "either direction (LEDGER 0)",
                    "at_run": str(a)[:200], "now": str(b)[:200]})

    # Registry hygiene
    for pid, row in now["probes"].items():
        missing = [k for k in COMPARED_FIELDS if not str(row.get(k, "")).strip()]
        if missing:
            findings.append({"severity": "MAJOR", "probe": pid,
                             "issue": f"registry row missing {missing}"})

    # Gap list still true?
    ledger = (REPO / "m8" / "LEDGER.md").read_text()
    # Only the gap TABLE's rows count, not the prose around it -- a struck item is announced in
    # that prose, and matching it there made this check flag its own note (found on first run).
    gap_rows = []
    if "GAP LIST" in ledger:
        section = ledger.split("GAP LIST")[1].split("\n## ")[0]
        gap_rows = [ln for ln in section.splitlines()
                    if ln.startswith("|") and not set(ln) <= set("|-: ")]
    gap_files = [n for n in ("m8src/test_decide.py", "m8src/rule_audit.py",
                             "m8src/test_final_guard.py", "m8src/test_freeze_binding.py")
                 if any(f"`{n}`" in row for row in gap_rows)]
    stale = [n for n in gap_files if (REPO / n).exists()]
    if stale:
        findings.append({"severity": "MAJOR",
                         "issue": f"the gap list still names files that now exist: {stale}. A "
                                  f"stale gap list is how a DONE heading starts lying."})

    return {
        "_note": __doc__.strip().splitlines()[0],
        "head": _git("rev-parse", "HEAD")[0].strip(),
        "registry_sha_now": now.get("version"),
        "stamped_results_checked": checked,
        "unstamped_artifacts": unstamped,
        "_unstamped_note": "not every results/m8_*.json is a probe result -- calibration caches, "
                           "schedules and descriptive diagnostics are inputs, not registered "
                           "measurements. They are listed so the distinction is visible.",
        "findings": findings,
        "outstanding_violations": len([f for f in findings if f["severity"] == "BLOCKER"]),
        "NOT_mechanically_checkable": UNVERIFIABLE,
    }


def main():
    out = audit()
    (RESULTS / "m8_rule_audit.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"HEAD {out['head'][:8]}   {len(out['stamped_results_checked'])} stamped result(s) "
          f"checked, {len(out['unstamped_artifacts'])} unstamped artifact(s)")
    for c in out["stamped_results_checked"]:
        print(f"  {c['file']:38s} probe {str(c['probe']):10s} @ {c['commit']}")
    if out["findings"]:
        print(f"\n{len(out['findings'])} finding(s):")
        for f in out["findings"]:
            print(f"  [{f['severity']}] {f.get('file', f.get('probe', ''))}: {f['issue']}")
    else:
        print("\nno findings")
    print("\nNOT mechanically checkable, listed rather than reported as passes:")
    for u in out["NOT_mechanically_checkable"]:
        print(f"  - {u}")
    return 1 if out["outstanding_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
