"""E15 (exploratory): make the paper's teacher-choice and build-cost decisions explicit.

Uses only five committed result receipts and the historical Zero ledger, never raw evaluation
data. Reports selection consequences, task sensitivity, size correlations without ratio coupling,
and Nano's training-loop cost at its recorded historical rental rate. Does not train a model.

    .venv/bin/python m15src/e15_decision_audit.py
"""
import argparse
import json
import re

import numpy as np
from scipy.stats import spearmanr

from common import REPO, receipt, sha_file, utc_now, write_result

OUT = REPO / "results" / "m15_e15_decision_audit.json"
SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN = ("scifact", "nfcorpus", "scidocs", "trec-covid")
INPUTS = ("results/m15_e8_towers.json", "results/m15_e8x_towers.json",
          "results/m15_e13_robustness.json", "results/m13_build_record.json",
          "results/m13_build_allocation.json", "m7/LEDGER.md")


def selection(rows):
    screen = max(rows, key=lambda n: rows[n]["dev_table"])
    teacher = max(rows, key=lambda n: rows[n]["six_ceiling_macro_all6"])
    best = max(rows, key=lambda n: rows[n]["six_table_macro_all6"])
    macro = {n: float(np.mean([c["six_table"][d] for d in CLEAN])) for n, c in rows.items()}
    best_clean = max(macro, key=macro.get)
    per = {}
    for ds in SIX:
        ranking = sorted(rows, key=lambda n: rows[n]["six_table"][ds], reverse=True)
        per[ds] = {"screen_choice_rank": ranking.index(screen) + 1,
                   "screen_choice_score": rows[screen]["six_table"][ds],
                   "best_table": ranking[0], "best_table_score": rows[ranking[0]]["six_table"][ds],
                   "screen_choice_regret": rows[ranking[0]]["six_table"][ds] - rows[screen]["six_table"][ds]}
    return {"n_checkpoints": len(rows), "dev_screen_choice": screen,
            "best_public_teacher": teacher, "best_public_table": best,
            "screen_table_macro": rows[screen]["six_table_macro_all6"],
            "teacher_choice_table_macro": rows[teacher]["six_table_macro_all6"],
            "screen_minus_teacher_choice": rows[screen]["six_table_macro_all6"] - rows[teacher]["six_table_macro_all6"],
            "screen_regret_all6": rows[best]["six_table_macro_all6"] - rows[screen]["six_table_macro_all6"],
            "best_clean4_table": best_clean, "screen_clean4_table_macro": macro[screen],
            "best_clean4_table_macro": macro[best_clean], "screen_regret_clean4": macro[best_clean] - macro[screen],
            "screen_vs_clean4_rho": float(spearmanr([c["dev_table"] for c in rows.values()], list(macro.values()))[0]),
            "dimension_vs_absolute_table_rho": float(spearmanr([c["dim"] for c in rows.values()],
                                                                [c["six_table_macro_all6"] for c in rows.values()])[0]),
            "dimension_vs_teacher_rho": float(spearmanr([c["dim"] for c in rows.values()],
                                                         [c["six_ceiling_macro_all6"] for c in rows.values()])[0]),
            "dimension_vs_retention_rho": float(spearmanr([c["dim"] for c in rows.values()],
                                                           [c["six_table_macro_all6"] / c["six_ceiling_macro_all6"] for c in rows.values()])[0]),
            "per_dataset": per}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUT.relative_to(REPO)),
                        help="unused result path inside the repository; existing files are refused")
    args = parser.parse_args()
    out = (REPO / args.output).resolve()
    if not out.is_relative_to(REPO):
        parser.error("--output must be inside the repository")
    out.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    blobs = {p: json.loads((REPO / p).read_text()) for p in INPUTS if p.endswith(".json")}
    ten = {n: c for n, c in blobs[INPUTS[0]]["configs"].items() if n != "arctic-embed-l-mean"}
    pooled = {**ten, **blobs[INPUTS[1]]["new_configs"]}
    build, allocation = blobs[INPUTS[3]], blobs[INPUTS[4]]
    hours = build["training"]["seconds"] / 3600
    price = allocation["total_price_usd_per_hour"]
    ledger = (REPO / INPUTS[-1]).read_text()
    target = re.search(r"query targets \(~(\d+)[–-](\d+) h\)", ledger)
    train = re.search(r"Retraining itself is ~(\d+) minutes", ledger)
    if target is None or train is None:
        raise RuntimeError("historical Zero cost estimates moved; inspect the source before quoting")
    inputs = [{"path": p, "sha256": sha_file(REPO / p)} for p in INPUTS]
    write_result(out, {
        "status": "COMPLETE", "measurement": "E15 (exploratory, derived from existing receipts)",
        "selection_registered_ten": selection(ten), "selection_pooled_26": selection(pooled),
        "screen_timing": {"minutes_per_configuration": blobs[INPUTS[2]]["screen_minutes_per_tower_a100"],
                          "kind": "coarse historical intervals between output files, not an isolated cold-start benchmark",
                          "exclusions": "model downloads, data preparation, full document-index construction; Stella reuses cached fit-query vectors"},
        "nano_build": {"gpu": build["environment"]["gpu"], "training_examples": build["dose_run_examples"],
                       "training_steps": build["training"]["steps_run"], "training_seconds": build["training"]["seconds"],
                       "training_hours": hours, "historical_total_rental_usd_per_hour": price,
                       "training_time_priced_usd": hours * price,
                       "cost_kind": "training duration multiplied by the recorded rental rate; not a separately itemized invoice or complete project cost",
                       "exclusions": "data collection and generation, teacher-target preparation, recipe search, failed runs, evaluation, model export, engineering time"},
        "zero_build_historical_estimate": {"teacher_target_hours": [int(target.group(1)), int(target.group(2))],
                                           "retraining_minutes": int(train.group(1)),
                                           "kind": "approximate planning estimates in m7/LEDGER.md, not measured end-to-end timings or a dollar quote"},
        "limits": ["The strongest public teacher is a hindsight selection rule, not a prospectively tested model-card recommendation.",
                   "The screen chooses closed-form tables, not the trained Zero recipe or transformer students.",
                   "All-six and clean-four targets differ; checkpoint selection is task dependent.",
                   "Dimension and retention share a teacher-quality confound; no causal size claim.",
                   "No new training or evaluation access; only published aggregate receipts were read."],
        "receipt": receipt(__file__, inputs, started, ("scipy",))})


if __name__ == "__main__":
    main()
