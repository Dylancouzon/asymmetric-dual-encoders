"""Assemble the learnability probe into one table, with CIs, and state what it changes.

Reads every results/m7_learnability_<encoder>.json plus results/m7_teacher_probe.json, and reports
per candidate: the teacher's own CEILING on the two CQADupStack dev components, the closed-form
TABLE's score against that same teacher's documents, and the ratio. Then paired-bootstraps each
candidate's table against the INCUMBENT's table -- valid pairing, because every table is scored on
the same components, queries and qrels; only the teacher behind them differs.

Why the ratio is not the deciding column: what ships is the table, so the table's absolute score is
the criterion and the ratio is diagnostic. A candidate can have the best ceiling, the worst ratio,
and a table below the incumbent's -- which is what the first run of this probe found.

    ../.venv/bin/python scripts/learnability_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

import boot
from _paths import REPO

INCUMBENT = "bge-base-en-v1.5"
COMPONENTS = ("cqadup-programmers", "cqadup-physics")


def main():
    probe = json.load(open(REPO / "results" / "m7_teacher_probe.json"))["candidates"]
    rows, per_query = {}, {}
    for p in sorted((REPO / "results").glob("m7_learnability_*.json")):
        d = json.loads(p.read_text())
        enc = d["encoder"]
        best = d["best_lambda"]
        b = d["lambdas"][best]
        ceiling = probe.get(enc, {}).get("macro_cqadupstack")
        rows[enc] = {"dim": d["dim"], "best_lambda": float(best),
                     "table_macro_2": b["dev_macro_2"],
                     "cosine_agreement": b["dev_cosine_mean"],
                     "teacher_ceiling_2": ceiling,
                     "ratio_table_over_ceiling": (round(b["dev_macro_2"] / ceiling, 4)
                                                  if ceiling else None),
                     "lambda_curve": {k: v["dev_macro_2"] for k, v in d["lambdas"].items()}}
        per_query[enc] = {c: {k: float(v) for k, v in b["per_query"][c].items()}
                          for c in COMPONENTS}

    ranked = sorted(rows, key=lambda k: -rows[k]["table_macro_2"])
    vs_inc = {}
    if INCUMBENT in per_query:
        for enc in ranked:
            if enc == INCUMBENT:
                continue
            r = boot.paired(per_query[enc], per_query[INCUMBENT], alternative="two-sided")
            vs_inc[enc] = r
            print(f"  {enc:22s} table {rows[enc]['table_macro_2']:.4f}  "
                  f"ceiling {rows[enc]['teacher_ceiling_2']}  "
                  f"ratio {rows[enc]['ratio_table_over_ceiling']}  |  vs incumbent "
                  f"d={r['delta']:+.4f} CI={r['ci95']} boot-tail={r['boot_tail_str']} "
                  f"{'RESOLVED' if r['resolved'] else 'UNRESOLVED'}")
    out = {"_note": "Teacher candidates ranked by the score of the CLOSED-FORM table fitted against "
                    "them, not by their own symmetric ceiling. Fitted on TRAIN query vectors, "
                    "measured on the two CQADupStack dev components against that teacher's own "
                    "documents. Paired bootstrap against the incumbent's table: same components, "
                    "queries and qrels, so the pairing is valid. Closed-form and flat -- training "
                    "moves a table (see m7_ridge_vs_trained.json), so this ranks candidates rather "
                    "than predicting final scores, and it is a DEV selection: the six are unread.",
           "components": list(COMPONENTS), "incumbent": INCUMBENT,
           "ranked_by_table_macro": ranked, "candidates": rows,
           "vs_incumbent_boot": vs_inc}
    (REPO / "results" / "m7_learnability_report.json").write_text(json.dumps(out, indent=1))
    print("\nranking by what ships (table macro): " + " > ".join(
        f"{e} {rows[e]['table_macro_2']:.4f}" for e in ranked))
    print("wrote results/m7_learnability_report.json")


if __name__ == "__main__":
    main()
