"""Every pre-registered rule against every arm family it binds. One matrix, checked from disk.

Written because of how the step-selection rule was found unapplied: by accident, while re-reading
the ledger for another reason, after it had already governed four arms and a promoted adoption.
The pre-freeze review's M5 is blunt about it -- "rule compliance is discovered, not audited" --
and a project whose whole claim rests on pre-registration cannot leave compliance to luck.

This is deliberately NOT a general rule engine. It encodes the handful of rules that bind arm
families mechanically, checks each against the committed artifacts, and reports UNKNOWN rather
than PASS for anything it cannot verify -- an audit that silently scores the unverifiable as
compliant is worse than none, because it converts an open question into a reassuring green row.

Usage: rule_audit.py
"""
import json
import re

from _paths import REPO, WORK

RUNS = WORK / "runs"


def _defaults():
    """Cfg field defaults, so a field added mid-family can be told apart from a real change."""
    from dataclasses import fields
    from train import Cfg
    return {f.name: (f.default if f.default is not None else None) for f in fields(Cfg)}


DEFAULTS = _defaults()


def cfg(run_id):
    """None for anything in work/runs that is not a run record -- the directory also holds dev
    caches and `*.fusion.json` sidecars, and assuming every JSON has a `cfg` crashed the audit."""
    p = RUNS / f"{run_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("cfg")
    except Exception:
        return None


def hist(run_id):
    p = RUNS / f"{run_id}.json"
    if not p.exists():
        return []
    try:
        h = json.loads(p.read_text()).get("history") or []
    except Exception:
        return []
    return [e for e in h if e.get("phase") in ("A", "B", "C")]


def arms(pattern):
    return sorted(p.stem for p in RUNS.glob("*.json")
                  if re.match(pattern, p.stem) and not p.stem.endswith(".meta"))


def _peak_rerun_of(rid, best_step):
    """Did some OTHER committed arm re-run this arm's recipe to its peak step?

    Without this the audit flags every deliberately-long exploratory arm, because the rule is
    satisfied by a SIBLING (`p35a-2m-1e3-x4000` peaks at 2500 and `p35w-2m-s2500` is the re-run),
    not by the arm itself. An audit that cannot tell those apart cries wolf and gets ignored,
    which is how the real violation stayed hidden.
    """
    c = cfg(rid)
    if c is None:
        return None
    for other in arms(r".*"):
        if other == rid:
            continue
        o = cfg(other)
        if o is None or o.get("steps_a") != best_step:
            continue
        if all(o.get(k) == v for k, v in c.items()
               if k not in ("run_id", "steps_a", "eval_every")):
            return other
    return None


# Documented reasons an arm is NOT bound by the step rule as literally written. Each names the
# ledger text that exempts it. Exemptions are listed, never silently applied -- the point of this
# file is that a gap is visible, and an exemption is a claim that must itself be checkable.
STEP_EXEMPT = {
    r"^p5s-": "LEDGER 'Recipe simplification', AMENDED before any simplification arm had a "
              "full-suite number: the A-phase step count is FIXED at the baseline's 2500, because "
              "an equivalence test that also selects a step varies a fifth thing -- with an "
              "instrument whose peak did not reproduce.",
    r"^p35w-": "an arm whose name marks it as the peak RE-RUN of a longer sibling is terminal by "
               "construction. 'Run long, find the peak, re-run once' does not recurse; applying "
               "the rule to the re-run would regress without end.",
}


def _step_exempt(rid):
    for pat, why in STEP_EXEMPT.items():
        if re.match(pat, rid):
            return why
    return None


def check_step_rule(family, pattern):
    """"An arm's step count is its best proxy eval, implemented by re-running to that step."

    An arm complies if its curve peaks at its final step, OR a sibling arm re-ran the same recipe
    to that peak. Anything else is a genuine gap and is named. This is the rule that was missed."""
    rows = {}
    for rid in arms(pattern):
        h = hist(rid)
        if len(h) < 2:
            continue
        best = max(h, key=lambda e: e["macro"])
        at_end = best["step"] == h[-1]["step"]
        rerun = None if at_end else _peak_rerun_of(rid, best["step"])
        exempt = None if (at_end or rerun) else _step_exempt(rid)
        rows[rid] = {"best_step": best["step"], "final_step": h[-1]["step"],
                     "best_macro": round(best["macro"], 5),
                     "peak_rerun_as": rerun, "exempt_reason": exempt,
                     "complies": bool(at_end or rerun or exempt)}
    return rows


def check_one_knob(family, base_id, pattern, allowed):
    """An ablation arm must differ from its base on ONLY the knobs its design names."""
    b = cfg(base_id)
    rows = {}
    if b is None:
        return {"_unknown": f"base {base_id} has no committed cfg"}
    ignore = {"run_id", "init", "eval_every", "steps_a", "steps_b"}
    for rid in arms(pattern):
        c = cfg(rid)
        if c is None:
            continue
        diff = sorted(k for k in set(b) | set(c)
                      if k not in ignore and b.get(k) != c.get(k))
        # A key PRESENT in one cfg and ABSENT in the other, whose value equals the field's
        # default, is a bookkeeping difference from adding a defaulted Cfg field mid-family --
        # the behaviour is identical, the record is not. Named rather than hidden, because
        # "the arms' recorded configs are not identical" is true and a reader should see why.
        book = [k for k in diff if (k not in b or k not in c)
                and (b.get(k, DEFAULTS.get(k)) == DEFAULTS.get(k)
                     or c.get(k, DEFAULTS.get(k)) == DEFAULTS.get(k))]
        real = [k for k in diff if k not in book]
        rows[rid] = {"differs_on": diff, "bookkeeping_only": book, "behavioural": real,
                     "complies": set(real) <= set(allowed) if allowed is not None else None}
    return rows


def main():
    out = {
        "_what": "pre-registered rules x the arm families they bind, checked against committed "
                 "artifacts. Written after the step-selection rule was found unapplied by "
                 "accident rather than by audit (pre-freeze review M5).",
        "_reading": "UNKNOWN is not PASS. A rule this file cannot verify mechanically is listed "
                    "under `not_mechanically_checkable` with the reason, so the gap is visible "
                    "rather than absent.",
        "rules": {},
        "not_mechanically_checkable": {},
    }

    # --- step selection ------------------------------------------------------------------
    step = {}
    for fam, pat in (("p4n negatives", r"^p4n-.*-a$"),
                     ("p5s simplification", r"^p5s-.*-a$"),
                     ("p4 mandatory ablations", r"^p4-.*-a$"),
                     ("p35 lever #2", r"^p35[abw]-.*$")):
        step[fam] = check_step_rule(fam, pat)
    out["rules"]["step_selection"] = {
        "_rule": "LEDGER 'Step selection': an arm's step count is its best proxy eval, "
                 "implemented by re-running to that step. AMENDED 2026-08-28 for decisions whose "
                 "numbers did not yet exist: match the baseline's step count instead, because the "
                 "proxy peak did not reproduce and inverted the full-suite ordering.",
        "_status": "the p4n family was found NON-COMPLIANT on 2026-08-28 and corrected; the "
                   "corrected arms then failed the negatives bar and the avenue closed. Arms "
                   "listed complies=false that predate the amendment are the historical record, "
                   "not new violations.",
        "families": step}

    # --- one-knob ablations --------------------------------------------------------------
    out["rules"]["one_knob"] = {
        "_rule": "an ablation arm varies only what its design names; the negatives design says "
                 "'vary ONLY the negatives, from the candidate's own B checkpoint at its own A "
                 "recipe'.",
        "p4n_vs_candidate": check_one_knob("p4n", "p35w-2m-s2500", r"^p4n-.*-a$",
                                           allowed=["hard_neg_k", "hard_neg_source",
                                                    "init_preproc", "pool_mode"]),
    }

    # --- things this file must NOT score green -------------------------------------------
    out["not_mechanically_checkable"] = {
        "pre_registration_ordering": "that each bar was committed BEFORE its numbers. Verifiable "
                                     "only from `git log -p m7/LEDGER.md` commit order against "
                                     "result-artifact commit order; the review verified the "
                                     "step-rule case by hand (5cabf45 precedes 24ba6c6).",
        "holm_family_membership": "which arms constitute a family is a judgement fixed in prose. "
                                  "negatives_decide.py and lever4_readjudicate.py encode theirs; "
                                  "nothing checks that the encoded family matches the written one.",
        "six_set_access": "convention-based, not enforced -- any script can read committed qrels. "
                          "m7/SIX_ACCESS.log is an audit trail, and the three known deviations are "
                          "enumerated in the ledger.",
        "dev_component_pinning": "enforced at runtime by dev_eval.dev_components() and "
                                 "heldout.verify_pinned(), not here.",
    }

    p = REPO / "results" / "m7_rule_audit.json"
    p.write_text(json.dumps(out, indent=1))
    for name, r in out["rules"].items():
        print(f"\n== {name}")
        fams = r.get("families", {k: v for k, v in r.items() if not k.startswith("_")})
        for fam, rows in fams.items():
            if fam.startswith("_"):
                continue
            bad = [k for k, v in rows.items() if v.get("complies") is False]
            ex = [k for k, v in rows.items() if v.get("exempt_reason")]
            bk = [k for k, v in rows.items() if v.get("bookkeeping_only")]
            print(f"  {fam}: {len(rows)} arms, {len(bad)} non-compliant"
                  + (f" -> {bad}" if bad else "")
                  + (f"; {len(ex)} exempt (documented)" if ex else "")
                  + (f"; {len(bk)} with bookkeeping-only cfg drift {bk}" if bk else ""))
    print(f"\n{len(out['not_mechanically_checkable'])} rules are NOT mechanically checkable and "
          f"are listed as such, not as passes.")
    print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
