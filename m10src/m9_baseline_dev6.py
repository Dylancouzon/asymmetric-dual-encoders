"""M10.0-c -- the baseline row.

Per-component DEV-6 read of the FROZEN M9 candidate (`m9/FREEZE.json`), including
`heldout-longq`, on the box. DEV-6 only -- no six-set, no reserved, no LoTTE. Every later
retention-vs-M9 comparison in M10 divides by this row.

Usage: .venv/bin/python m10src/m9_baseline_dev6.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m9src"))

import torch  # noqa: E402

import m9base  # noqa: E402  (installs paths_guard, sets sys.path, pins M7_ENCODER)
import eval9   # noqa: E402
import guard9  # noqa: E402
import nano    # noqa: E402

FREEZE = REPO / "m9" / "FREEZE.json"
M9_RESULTS = REPO / "m9" / "RESULTS.md"
OUT = REPO / "results" / "m10_m9_baseline_dev6.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    freeze = json.loads(FREEZE.read_text())
    ckpt_path = REPO / freeze["checkpoint"]
    print(f"verifying checkpoint sha256: {ckpt_path}", flush=True)
    got = sha256_of(ckpt_path)
    want = freeze["checkpoint_sha256"]
    if got != want:
        raise SystemExit(f"REFUSED: sha256 mismatch. got {got}, want {want} "
                         f"(m9/FREEZE.json). Not scoring an unverified checkpoint.")
    print(f"sha256 OK: {got}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_key = freeze["teacher"]
    student_key = freeze["student"]
    query_prefix = freeze.get("student_query_prefix", "")
    assert teacher_key == eval9.INCUMBENT, f"unexpected teacher {teacher_key!r} in FREEZE.json"

    print(f"loading student {student_key!r} onto {device}", flush=True)
    model = nano.Nano(student_key).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    print(f"checkpoint step {blob.get('step')}, config_hash {blob.get('config_hash', '')[:12]}",
          flush=True)

    comps = eval9.components("DEV6")
    print(f"scoring DEV-6 components: {comps}", flush=True)
    t1 = time.time()
    per = eval9.eval_student(model, teacher_key, comps=comps, query_prefix=query_prefix)
    elapsed_score = time.time() - t1
    print(f"scored in {elapsed_score:.1f}s", flush=True)

    m = eval9.macros(per, teacher_key)["DEV6"]
    ceil = guard9.registry()["ceilings"][teacher_key]["DEV6"]
    means = m["means"]
    macro = m["macro"]
    retention = round(macro / ceil, 4)

    # Teacher DEV-6 numbers, if cached (per-component ceiling for retention = candidate/teacher).
    teacher_cache = REPO / "results" / f"m9_dev_symmetric_{teacher_key}.json"
    teacher_dev6 = None
    teacher_source = None
    if teacher_cache.exists():
        td = json.loads(teacher_cache.read_text())
        tmeans = td.get("macros", {}).get("DEV6", {}).get("means")
        if tmeans and set(tmeans) >= set(means):
            teacher_dev6 = {k: tmeans[k] for k in means}
            teacher_source = str(teacher_cache.relative_to(REPO))
    else:
        teacher_source = "not cached; not cheaply available without re-running teacher9 symmetric pass"

    per_component_table = {}
    for k in means:
        row = {"candidate": means[k]}
        if teacher_dev6 is not None:
            row["teacher"] = teacher_dev6[k]
            row["retention"] = round(means[k] / teacher_dev6[k], 4) if teacher_dev6[k] else None
        per_component_table[k] = row

    # Reproduction check vs m9/RESULTS.md's reported DEV-6 anchor numbers, if present.
    # m9/RESULTS.md reports m9s1/m9s1c/m9s1b DEV-6 macros, not the FROZEN longrun candidate
    # (a different arm/checkpoint) -- so this is a documentation cross-check, not an identity.
    reported_candidates = {
        "m9s1_anchor_final": 0.4806,
        "m9s1_anchor_final_precise": 0.48041,
        "m9s1c_random_head": 0.45620,
        "m9s1b_seed1": 0.48271,
        "freeze_dev_screen3_SCREEN3_not_DEV6": freeze.get("dev_screen3"),
    }
    max_abs_diff_vs_m9s1_anchor = round(abs(macro - reported_candidates["m9s1_anchor_final_precise"]), 6)

    out = {
        "_note": ("M10.0-c baseline row. DEV-6 per-component nDCG@10 of the FROZEN M9 candidate "
                 "(m9/FREEZE.json, checkpoint step " + str(freeze.get("step")) + "). This is a "
                 "diagnostic baseline: it is NOT the same arm as m9s1/m9s1c/m9s1b in m9/RESULTS.md "
                 "(those are Stage-A anchor-curve arms at 30,349 steps; the frozen candidate is "
                 "the post-cooldown longrun checkpoint at step " + str(freeze.get("step")) +
                 ", 457,265 steps). No six-set, reserved or LoTTE data was read."),
        "checkpoint": str(freeze["checkpoint"]),
        "checkpoint_sha256": got,
        "checkpoint_step": freeze.get("step"),
        "teacher": teacher_key,
        "student": student_key,
        "student_query_prefix": query_prefix,
        "eval9_config": {
            "surface": "DEV6",
            "components": comps,
            "chunk": eval9.CHUNK,
        },
        "per_component": per_component_table,
        "dev6_macro": macro,
        "dev6_ceiling_DEV6": ceil,
        "dev6_retention": retention,
        "teacher_dev6_source": teacher_source,
        "reproduction_check_vs_m9_RESULTS_md": {
            "note": ("m9/RESULTS.md's DEV-6 rows (m9s1/m9s1c/m9s1b) are a DIFFERENT arm from the "
                    "frozen longrun candidate scored here (Stage-A anchor at 30,349 steps vs the "
                    "frozen candidate at 457,265 steps) -- reported for context, not as an "
                    "identity check. See FREEZE.json's provenance note."),
            "candidates_reported_in_m9_RESULTS_md": reported_candidates,
            "this_dev6_macro": macro,
            "max_abs_diff_vs_m9s1_anchor_dev6": max_abs_diff_vs_m9s1_anchor,
        },
        "wall_time_s": {
            "checkpoint_verify_and_load": round(t1 - t0, 1),
            "scoring": round(elapsed_score, 1),
            "total": round(time.time() - t0, 1),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}", flush=True)
    print(json.dumps({"dev6_macro": macro, "dev6_retention": retention, "means": means}, indent=1))


if __name__ == "__main__":
    main()
