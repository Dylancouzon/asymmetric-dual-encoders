"""Realized query EXPOSURE per arm -- what the sampler actually presented, not what was available.

astra's recommendation #2, 2026-09-08. The point it corrects: family A's arms are all
form-balanced, so each form present in an arm receives an EQUAL share of query presentations no
matter how much text backs it. `by_form` in the manifest is AVAILABLE TEXTS and says nothing about
exposure. So an arm with more forms gives each existing form proportionally LESS training, and
A4-A3 is therefore a test of RE-ALLOCATING exposure to generated forms -- not a test of whether
generated data helps.

Reads only arm records already on disk. No teacher encoding, no GPU, no training.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# the seven forms produced by the Qwen generator; the rest come from real text
GENERATED = {"argument", "comparison", "conversational", "finance", "health", "howto", "yesno"}


def arm_exposure(rec):
    """-> one row: forms present, presentations per form, and the generated share of exposure."""
    man = rec.get("assemble_manifest") or {}
    q = man.get("query") or {}
    by_form = q.get("by_form") or {}
    forms = sorted(by_form)
    pattern = man.get("pattern") or rec.get("pattern") or "75/25"
    try:
        qs, _ds = (int(x) for x in str(pattern).split("/"))
    except Exception:
        qs = 75
    # the dose lives in `recipe.dose_examples`; the manifest carries shape, not dose
    recipe = rec.get("recipe") or {}
    examples = recipe.get("dose_examples") or man.get("dose_examples")
    pattern = recipe.get("pattern") or pattern
    q_pres = int(examples * qs / 100) if examples else None
    per_form = int(q_pres / len(forms)) if q_pres and forms else None
    gen = [f for f in forms if f in GENERATED]
    return {
        "arm": rec.get("arm"),
        "pattern": pattern,
        "examples": examples,
        "query_presentations": q_pres,
        "n_forms": len(forms),
        "forms": forms,
        "available_texts_by_form": by_form,
        "presentations_per_form": per_form,
        "n_generated_forms": len(gen),
        "generated_share_of_query_presentations": (len(gen) / len(forms)) if forms else None,
        "_what": ("presentations_per_form is q_pres/n_forms because the sampler is form-balanced: "
                  "every form present gets an equal share of presentations regardless of how much "
                  "text backs it. available_texts_by_form is NOT exposure."),
    }


def build(arms=None):
    arms = arms or [p.stem.replace("m10_arm_", "") for p in sorted(RESULTS.glob("m10_arm_*.json"))]
    rows = []
    for a in arms:
        p = RESULTS / f"m10_arm_{a}.json"
        if not p.exists():
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(rec, dict) or "assemble_manifest" not in rec:
            continue
        rows.append(arm_exposure(rec))
    return rows


def main():
    rows = build()
    hdr = f"{'arm':<14}{'forms':>6}{'q_pres':>12}{'per form':>10}{'gen forms':>10}{'gen share':>11}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        gs = r["generated_share_of_query_presentations"]
        print(f"{str(r['arm']):<14}{r['n_forms']:>6}{(r['query_presentations'] or 0):>12,}"
              f"{(r['presentations_per_form'] or 0):>10,}{r['n_generated_forms']:>10}"
              f"{(f'{gs:.1%}' if gs is not None else '-'):>11}")
    out = RESULTS / "m10_exposure_table.json"
    out.write_text(json.dumps({"_what": __doc__.strip().split("\n")[0], "arms": rows}, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")
    # the comparison astra's argument turns on
    d = {r["arm"]: r for r in rows}
    if "A3" in d and "ANCHOR" in d:
        a3, a4 = d["A3"], d["ANCHOR"]
        print(f"\n  an existing form gets {a3['presentations_per_form']:,} presentations in A3 "
              f"and {a4['presentations_per_form']:,} in A4/ANCHOR "
              f"({a4['presentations_per_form']/a3['presentations_per_form']:.2f}x)")
        print("  -> A4-A3 re-allocates exposure to generated forms; it is not a synthetic-vs-real test")


if __name__ == "__main__":
    main()
