"""E24: prospective teacher-selection test (m15/MEASUREMENTS.md, E24). Runpod.

    e24_prospective.py preflight         tokenizer check for the eight roster checkpoints
    e24_prospective.py teacher <name>    teacher's own six-set score (its query path) for one checkpoint
    e24_prospective.py predict           commit predictions from the E21 fit BEFORE any student is scored
    e24_prospective.py dev <name>        E8 table recipe: dev grid (reuses e8_towers with E24 paths)
    e24_prospective.py freeze            freeze table lambdas
    e24_prospective.py six <name>        E8 table recipe: six-set scoring
    e24_prospective.py head-dev <name>   E19 head recipe: dev grid
    e24_prospective.py head-freeze       freeze head lambdas
    e24_prospective.py head-six <name>   E19 head recipe: six-set scoring
    e24_prospective.py assemble          results/m15_e24_prospective.json
    e24_prospective.py all               every step in order; stops if predictions are not committed

Order matters and is enforced: teacher scores first (they are model inputs), then the prediction
file with its sha256, then students. The roster is fixed in the method; a checkpoint that fails the
tokenizer check is excluded, not replaced.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("m15src", "m8src", "m7src"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))
import e8_towers as E8          # noqa: E402
import e8x_towers as E8X        # noqa: E402  (rewrites E8.CONFIGS/OUT_DIR; we override below)
import e19_head_screen as E19   # noqa: E402
from common import load_public, receipt, sha_file, utc_now, write_result  # noqa: E402

ROSTER = ("e5-large-v1", "bge-large-en-v1", "e5-base-unsupervised", "nomic-embed-text-v1",
          "msmarco-bert-base-dot-v5", "msmarco-distilbert-base-v4", "all-minilm-l12-v1",
          "paraphrase-minilm-l6-v2")
OUT_DIR = REPO / "work" / "m15" / "e24"
TABLE_DIR = OUT_DIR / "table"
HEAD_DIR = OUT_DIR / "head"
PREDICTIONS = OUT_DIR / "predictions.json"
RESULT = REPO / "results" / "m15_e24_prospective.json"
E21 = REPO / "results" / "m15_e21_width_model.json"
E19_RESULT = REPO / "results" / "m15_e19_head_screen.json"
STELLA_VOCAB = E8X.STELLA_VOCAB

# Route the reused E8/E19 drivers to E24 output directories.
E8.CONFIGS = ROSTER
E8.OUT_DIR = TABLE_DIR
E8.FROZEN = TABLE_DIR / "frozen_lambdas.json"
E8.PREFLIGHT = OUT_DIR / "preflight.json"
E19.OUT_DIR = HEAD_DIR
E19.FROZEN = HEAD_DIR / "frozen_lambdas.json"
E19.eligible = lambda: [n for n in ROSTER if n in json.loads(E8.PREFLIGHT.read_text())["eligible"]]
E8.eligible = E19.eligible


def preflight():
    import hashlib
    import torch
    from transformers import AutoTokenizer
    if E8.PREFLIGHT.exists():
        return
    E8.fit_list()
    os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
    import m8base  # noqa: F401
    import encoders
    rep = {"started_utc": utc_now(), "gpu": torch.cuda.get_device_name(0), "tokenizers": {}}
    for name in ROSTER:
        spec = encoders.get(name)
        kw = {"trust_remote_code": True} if spec.trust_remote_code else {}
        tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, **kw)
        vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
        h = hashlib.sha256("\n".join(t for t, _ in vocab).encode()).hexdigest()
        rep["tokenizers"][name] = {"vocab_size": len(tok), "cls_id": tok.cls_token_id,
                                   "ordered_vocab_sha256": h,
                                   "compatible": len(tok) == 30522 and tok.cls_token_id == 101 and h == STELLA_VOCAB}
        print(name, rep["tokenizers"][name], flush=True)
    rep["eligible"] = [n for n in ROSTER if rep["tokenizers"][n]["compatible"]]
    rep["excluded"] = [n for n in ROSTER if n not in rep["eligible"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    E8.atomic_json(E8.PREFLIGHT, rep)


def teacher(name):
    """The teacher's own six-set score, scored once; an input to the prediction, not a student."""
    out = OUT_DIR / f"teacher-{name}.json"
    if out.exists():
        print(f"{name}: teacher already done", flush=True)
        return
    t = E8.Tower(name)
    scores, pins = {}, {}
    for ds in E8.SIX:
        data = load_public(ds)
        dv = t.docs(f"e8-six-{ds}-docs", data["doc_texts"])
        scores[ds] = E8._mean(t.score(t.ceiling_queries(f"e8-six-{ds}-q", data["q_texts"]),
                                      data["q_ids"], dv, data["doc_ids"], data["qrels"]))
        pins[ds] = data["pins"]
        print(f"  {name} {ds}: teacher {scores[ds]:.4f}", flush=True)
        del dv
        t.m8base.empty_cache()
    E8.atomic_json(out, {"name": name, "status": "COMPLETE", "dim": t.spec.dim, "repo": t.spec.repo,
                         "revision": t.spec.revision, "six": scores,
                         "six_macro_all6": float(np.mean([scores[d] for d in E8.SIX])), "pins": pins})


def predict():
    """Predictions from the E21 teacher+width fit on the 26, written before any student is fitted."""
    if PREDICTIONS.exists():
        print("predictions already committed", flush=True)
        return
    for n in E19.eligible():
        if (TABLE_DIR / f"dev-{n}.json").exists() or (HEAD_DIR / f"dev-{n}.json").exists():
            raise SystemExit(f"E24 STOP: a student was fitted for {n} before predictions were committed")
    e19 = json.loads(E19_RESULT.read_text())["configs"]
    rows = {n: r for n, r in e19.items() if n != "arctic-embed-l-mean"}
    names = list(rows)
    teach = np.array([rows[n]["teacher_six_macro_all6"] for n in names])
    width = np.log2([rows[n]["dim"] for n in names])
    X1 = np.column_stack([np.ones(len(names)), teach, width])
    fits = {}
    for recipe, key in (("table", "recipe1_six_table_macro_all6"), ("head", "six_head_macro_all6")):
        y = np.array([rows[n][key] for n in names])
        fits[recipe] = np.linalg.lstsq(X1, y, rcond=None)[0].tolist()
    preds = {}
    for n in E19.eligible():
        t = json.loads((OUT_DIR / f"teacher-{n}.json").read_text())
        x = np.array([1.0, t["six_macro_all6"], np.log2(t["dim"])])
        preds[n] = {"teacher_six_macro_all6": t["six_macro_all6"], "dim": t["dim"],
                    "predicted": {r: float(x @ np.array(b)) for r, b in fits.items()}}
    blob = {"written_utc": utc_now(), "fit_on": names, "coefficients_intercept_teacher_log2width": fits,
            "predictions": preds, "e19_result_sha256": sha_file(E19_RESULT)}
    PREDICTIONS.write_text(json.dumps(blob, indent=2))
    print(f"PREDICTIONS sha256 {sha_file(PREDICTIONS)}", flush=True)


def _require_predictions():
    if not PREDICTIONS.exists():
        raise SystemExit("E24 STOP: predictions must be committed before any student is fitted")


def assemble():
    from scipy.stats import spearmanr
    pred = json.loads(PREDICTIONS.read_text())
    names = [n for n in E19.eligible() if (TABLE_DIR / f"six-{n}.json").exists()
             and (HEAD_DIR / f"six-{n}.json").exists()]
    rows = {}
    for n in names:
        t = json.loads((OUT_DIR / f"teacher-{n}.json").read_text())
        tab = json.loads((TABLE_DIR / f"six-{n}.json").read_text())
        hd = json.loads((HEAD_DIR / f"six-{n}.json").read_text())
        tdev = json.loads((TABLE_DIR / f"dev-{n}.json").read_text())
        hdev = json.loads((HEAD_DIR / f"dev-{n}.json").read_text())
        rows[n] = {"dim": t["dim"], "teacher_six_macro_all6": t["six_macro_all6"],
                   "table": float(np.mean([tab["table"][d] for d in E8.SIX])) if tab["status"] == "COMPLETE" else None,
                   "head": float(np.mean([hd["table"][d] for d in E8.SIX])) if hd["status"] == "COMPLETE" else None,
                   "table_dev": tdev.get("best_dev_macro_2"), "head_dev": hdev.get("best_dev_macro_2"),
                   "predicted_table": pred["predictions"][n]["predicted"]["table"],
                   "predicted_head": pred["predictions"][n]["predicted"]["head"]}
    stats = {}
    for recipe in ("table", "head"):
        ok = [n for n in names if rows[n][recipe] is not None]
        actual = np.array([rows[n][recipe] for n in ok])
        p = np.array([rows[n][f"predicted_{recipe}"] for n in ok])
        t = np.array([rows[n]["teacher_six_macro_all6"] for n in ok])
        dev = np.array([rows[n][f"{recipe}_dev"] for n in ok])
        best = float(actual.max())
        stats[recipe] = {"n": len(ok),
                         "spearman_pred_vs_actual": float(spearmanr(p, actual)[0]),
                         "spearman_pred_vs_actual_bootstrap95": E8X.boot_spearman(p, actual),
                         "spearman_teacher_vs_actual": float(spearmanr(t, actual)[0]),
                         "spearman_dev_vs_actual": float(spearmanr(dev, actual)[0]),
                         "mae": float(np.mean(np.abs(p - actual))),
                         "regret": {"strongest_teacher": best - float(actual[int(np.argmax(t))]),
                                    "model_pick": best - float(actual[int(np.argmax(p))]),
                                    "dev_screen": best - float(actual[int(np.argmax(dev))])}}
    write_result(RESULT, {
        "status": "COMPLETE", "measurement": "E24 (exploratory, pre-specified; predictions committed before scoring)",
        "scope": "eight pre-declared checkpoints never scored before; the E21 fit on 26 predicts them; "
                 "tokenizer-incompatible checkpoints excluded, not replaced",
        "predictions_sha256": sha_file(PREDICTIONS), "predictions_written_utc": pred["written_utc"],
        "preflight": json.loads(E8.PREFLIGHT.read_text()), "rows": rows, "stats": stats,
        "receipt": receipt(__file__, [{"predictions": str(PREDICTIONS.relative_to(REPO)), "sha256": sha_file(PREDICTIONS)},
                                      {"e21": str(E21.relative_to(REPO))}], utc_now(), ("torch", "transformers", "scipy"))})


def all_steps():
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'm15src'}:{REPO / 'm8src'}:{REPO / 'm7src'}"}
    run = lambda *a, **k: subprocess.run([py, __file__, *a], check=True, cwd=REPO, env={**env, **k})
    run("preflight")
    for n in E19.eligible():
        run("teacher", n, M7_ENCODER=n)
    run("predict")
    for n in E19.eligible():
        run("dev", n, M7_ENCODER=n)
    if not E8.FROZEN.exists():
        run("freeze")
    for n in E19.eligible():
        run("six", n, M7_ENCODER=n)
    for n in E19.eligible():
        run("head-dev", n, M7_ENCODER=n)
    if not E19.FROZEN.exists():
        run("head-freeze")
    for n in E19.eligible():
        run("head-six", n, M7_ENCODER=n)
    run("assemble")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["all"]
    if cmd in ("dev", "six", "head-dev", "head-six"):
        _require_predictions()
    {"preflight": preflight, "predict": predict, "assemble": assemble, "all": all_steps,
     "teacher": lambda: teacher(rest[0]),
     "dev": lambda: E8.dev(rest[0]), "freeze": E8.freeze, "six": lambda: E8.six(rest[0]),
     "head-dev": lambda: E19.dev(rest[0]), "head-freeze": E19.freeze, "head-six": lambda: E19.six(rest[0])}[cmd]()
