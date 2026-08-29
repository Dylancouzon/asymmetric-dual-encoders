"""Retention against the teacher, three ways, because one number hides the thing that matters.

The project has been quoting retention on the six-component dev macro. Four of those six
components are Wikipedia or train-adjacent (m7/LEDGER.md, "THE DEV MACRO IS A BIASED ESTIMATOR"),
and all six of the CONFIRMATORY datasets are out-of-domain, so the macro is the least predictive
of the three figures below and the one most likely to be quoted.

  all-six        the dev macro. Flattering, and the one to stop leading with.
  text-backed    the four components that have document text, i.e. where BM25 and potion have a
                 row and the gate's G1/G3 comparisons run.
  out-of-domain  cqadup-programmers + cqadup-physics ONLY -- the two components outside the TRAIN
                 mix and its Wikipedia family, and the nearest analogue this suite has to the six.

BM25 is printed on the same rows as the floor, because "retains 93% of its teacher" and "beats
BM25 by 0.045 where it counts" are the same system described honestly and dishonestly.

Usage: retention.py <compare_tag> <artifact_key>
       retention.py steprule p4n-mixed32-s1000-a
"""
import json
import sys

import numpy as np

from _paths import REPO, WORK

TEACHER_ROW = "stella-400M-v5-symmetric"
GROUPS = {
    "all_six": None,                       # every component present
    "text_backed": ("nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics"),
    "out_of_domain": ("cqadup-programmers", "cqadup-physics"),
}


def macro(per, comps):
    use = [c for c in (comps or per) if c in per]
    return float(np.mean([per[c] for c in use])), use


def main(tag, key):
    d = json.loads((REPO / f"results/m7_compare_full_{tag}.json").read_text())
    ours = d["per_component_unrounded"][f"{key}|fp16"]
    refs = json.loads((WORK / "devres" / "refs-stella-400M-v5.json").read_text())
    ref_means = {s: {c: float(np.mean(list(v.values()))) for c, v in per.items()}
                 for s, per in refs.items()}

    rows = {}
    for name, comps in GROUPS.items():
        ours_m, used = macro(ours, comps)
        row = {"components": used, "system": ours_m}
        for s, per in ref_means.items():
            m, u = macro(per, used)
            # only comparable when the reference covers every component in the group
            row[s] = m if set(u) == set(used) else None
        t = row.get(TEACHER_ROW)
        row["retention_vs_teacher"] = ours_m / t if t else None
        for s in ("bm25", "potion-retrieval-32M"):
            row[f"delta_vs_{s}"] = (ours_m - row[s]) if row.get(s) is not None else None
        rows[name] = row

    out = {"_what": "retention against the frozen teacher, and the floor, on three nested "
                    "component groups. The all-six figure is the flattering one; the "
                    "out-of-domain pair is the nearest analogue to the confirmatory six.",
           "_status": "exploratory dev evidence; not a prediction of the six",
           "artifact": key, "compare_artifact": f"m7_compare_full_{tag}.json",
           "teacher_row": TEACHER_ROW, "groups": rows}
    (REPO / "results" / f"m7_retention_{key}.json").write_text(json.dumps(out, indent=1))

    print(f"{key}")
    for name, r in rows.items():
        ret = f"{r['retention_vs_teacher']:.3f}" if r["retention_vs_teacher"] else "  -  "
        bm = f"{r['delta_vs_bm25']:+.4f}" if r.get("delta_vs_bm25") is not None else "   -   "
        print(f"  {name:14s} n={len(r['components'])}  ours {r['system']:.4f}  "
              f"teacher {r[TEACHER_ROW]:.4f}  retention {ret}  vs bm25 {bm}")
    print(f"wrote results/m7_retention_{key}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
