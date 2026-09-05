"""The registered 90-step smoke of EVERY arm shape (§Screen), before any registered arm runs.

`m10/screen_registry.json` locks 16 trained arms. They share one trainer, so the arms that differ
only in DATA or DOSE share a code shape, but the ones that differ in student, feature width, head
form, mix pattern, batch size, objective or head init do not — and a shape error in any of them
surfaces at step 1 of a run that was going to cost GPU-hours. §Screen therefore requires all of
them smoked at 90 steps first. Only the anchor's shape had been.

Runs on **CPU by default** so it never contends with a training arm for the card; `--device cuda`
when the card is free. 90 steps at batch 32 is seconds per shape either way.

**On CPU, cap the sequence length for the big-batch shapes.** `E-bs128` at 512 tokens took the box
from 10 GB free to 2 GB and cost the running calibration arm 6% of its rate (945 -> 885 ex/s):
`output_hidden_states=True` keeps every layer's states for 128x512 positions, and CPU torch has no
device limit to push back. `--max-len` bounds it; the GPU run does not need it, and the shape being
smoked is the head and the loop, not the sequence length. The record says which was used.

**Results are written after EVERY shape**, not at the end, so a kill or an OOM leaves the shapes
that did pass. The first run of this file was killed during `E-bs128` to protect the calibration
and lost all eight passing records to an end-of-run write.

What it checks per shape, and why each has already caught something:
  1. the model CONSTRUCTS               -- `G-384` raised `KeyError: 1`; `LAYERS` had no 1-layer key
  2. params <= the 35M cap (hard, Dylan 2026-09-01)
  3. 90 steps run with a finite loss, through the real `trainer10.train_arm`
  4. the arm's REGISTERED warm start actually runs -- note its *numbers* are degenerate here:
     256 fit texts against 1,153 ridge parameters interpolates, so G-MLP's train objective reads
     0.0. This checks the PATH, not the fit; `test_warmstart` runs the overdetermined case.
     G-MLP's three-solve recipe
     (`nano10.warm_start_mlp`) and C-M9init's zero-padded head (`nano10.warm_start_from_m9`) were
     missing when this file was written and are now exercised here rather than assumed

It writes `results/m10_arm_smoke.json` and exits non-zero if any shape fails, so it can gate.
"""
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "results" / "m10_arm_smoke.json"
STEPS = 90

import numpy as np
import torch

# Every registered arm, mapped to the knobs that make its shape. `like` names another arm whose
# shape it shares (data- or dose-only differences), so the report says which arms a smoke covers.
SHAPES = {
    "ANCHOR":       dict(student="bge-small",  n_layers=3, head="linear", pattern="75/25",
                         batch=32,  loss="squared_l2"),
    "F-MiniLM-L6":  dict(student="MiniLM-L6",  n_layers=3, head="linear", pattern="75/25",
                         batch=32,  loss="squared_l2"),
    "F-MiniLM-L12": dict(student="MiniLM-L12", n_layers=3, head="linear", pattern="75/25",
                         batch=32,  loss="squared_l2"),
    "G-384":        dict(student="bge-small",  n_layers=1, head="linear", pattern="75/25",
                         batch=32,  loss="squared_l2"),
    "G-1536":       dict(student="bge-small",  n_layers=4, head="linear", pattern="75/25",
                         batch=32,  loss="squared_l2"),
    "G-MLP":        dict(student="bge-small",  n_layers=3, head="mlp",    pattern="75/25",
                         batch=32,  loss="squared_l2", warm_start="mlp"),
    "B-100/0":      dict(student="bge-small",  n_layers=3, head="linear", pattern="100/0",
                         batch=32,  loss="squared_l2"),
    "B-50/50":      dict(student="bge-small",  n_layers=3, head="linear", pattern="50/50",
                         batch=32,  loss="squared_l2"),
    "E-bs128":      dict(student="bge-small",  n_layers=3, head="linear", pattern="75/25",
                         batch=128, loss="squared_l2"),
    "D-NORM":       dict(student="bge-small",  n_layers=3, head="linear", pattern="75/25",
                         batch=32,  loss="leaf_norm_e2"),
    "D-COV":        dict(student="bge-small",  n_layers=3, head="linear", pattern="75/25",
                         batch=32,  loss="document_covariance_weighted"),
    "C-M9init":     dict(student="bge-small",  n_layers=3, head="linear", pattern="75/25",
                         batch=32,  loss="squared_l2", warm_start="m9"),
}
# arms whose shape is covered by one of the above
COVERS = {"ANCHOR": ["A1", "A2", "A3", "A4", "F-bge-small"]}
N_TEXTS = 4096
# C-M9init's init. **This is step 450,000 = 3.69B tokens, NOT the 3.74B where M9's plateau rule
# fired**: `ckpt_every` was 15,000, so the final ~6,543 steps were never checkpointed and the
# newest surviving weights of the M9 long run are these. Immaterial for an init, but the arm must
# not be described as starting from "the M9 candidate" without the qualification.
M9_CANDIDATE = REPO / "work" / "m9long" / "ckpt" / "step450000.pt"


def corpus(verbose=True):
    """A small slice of the REAL corpora -- a smoke on synthetic tensors proves nothing about the
    data path, which is where two of M10's four launch failures were."""
    import data10 as D
    t0 = time.time()
    texts, rows, _meta = D.m9_query_pool()
    texts, rows = texts[:N_TEXTS], rows[:N_TEXTS]
    T = D.m9_query_targets(rows)
    dtexts, dvecs, _dm = D.m9_doc_pool(N_TEXTS, seed=0)
    if verbose:
        print(f"  corpus: {len(texts):,} queries, {len(dtexts):,} documents "
              f"({time.time() - t0:.0f}s)", flush=True)
    return texts, T, dtexts, dvecs


def _write(recs, device, max_len):
    failed = [r["arm"] for r in recs if not r.get("passed")]
    no_ws = [r["arm"] for r in recs if not r.get("warm_start_implemented")]
    OUT.write_text(json.dumps(
        {"_what": f"§Screen's registered {STEPS}-step smoke of every arm shape",
         "steps": STEPS, "device": device, "n_texts": N_TEXTS, "max_len": max_len,
         "shapes_registered": list(SHAPES), "shapes_run": [r["arm"] for r in recs],
         "arms": recs, "failed": failed, "warm_start_not_implemented": no_ws,
         "all_shapes_pass": not failed and len(recs) == len(SHAPES),
         "_partial": len(recs) != len(SHAPES)}, indent=1))
    return failed, no_ws


def smoke_one(name, spec, corp, device="cpu", max_len=512, verbose=True):
    import data10 as D
    import nano10 as N
    import trainer10 as Tr
    texts, T, dtexts, dvecs = corp
    rec = {"arm": name, "spec": {k: v for k, v in spec.items()}, "covers": COVERS.get(name, []),
           "device": device}
    t0 = time.time()
    try:
        torch.manual_seed(0)
        m = N.Nano10(spec["student"], n_layers=spec["n_layers"], head=spec["head"]).to(device)
    except Exception as e:
        rec.update(constructed=False, error=f"{type(e).__name__}: {e}", passed=False)
        return rec
    rec.update(constructed=True, d_in=m.d_in, params=m.n_params(), under_cap=m.under_cap())

    # the arm's REGISTERED warm start, actually run
    ws = spec.get("warm_start", "linear")
    rec["warm_start_registered"] = ws
    try:
        if ws == "mlp":
            rec["warm_start_record"] = N.warm_start_mlp(m, texts[:256], T[:256], lam=1e-4)
        elif ws == "m9":
            rec["warm_start_record"] = N.warm_start_from_m9(m, M9_CANDIDATE)
        else:
            X = N.pooled_features(m, texts[:256])
            rec["warm_start_record"] = N.warm_start_linear(m, X, T[:256], lam=1e-4)
        rec["warm_start_implemented"] = True
    except Exception as e:
        rec["warm_start_implemented"] = False
        rec["warm_start_error"] = f"{type(e).__name__}: {e}"

    b = spec["batch"]
    try:
        qi = D.pretokenize(m.tok, texts, max_len=max_len)
        di = D.pretokenize(m.tok, dtexts, max_len=max_len)
        q = D.Stream(qi, T, pad_id=m.tok.pad_token_id, batch_size=b, seed=0)
        d = D.Stream(di, dvecs, pad_id=m.tok.pad_token_id, batch_size=b, seed=0)
        sigma = None
        if spec["loss"] == "document_covariance_weighted":
            sigma = torch.from_numpy(np.asarray(N.cov_matrix(dvecs), dtype=np.float32)).to(device)
        r = Tr.train_arm(m, D.batch_fn(q, d), total_steps=STEPS, pattern=spec["pattern"],
                         peak=1e-4, loss_name=spec["loss"], sigma=sigma, seed=0,
                         device=device, batch_size=b, log_every=0)
        rec.update(steps_run=r.get("steps_run"), stopped=r.get("stopped"),
                   examples=r.get("examples"), examples_per_s=r.get("examples_per_s"),
                   mix=r.get("mix"))
        ok = (r.get("steps_run") == STEPS and not r.get("stopped"))
        rec["passed"] = bool(ok and m.under_cap())
    except Exception as e:
        import traceback
        rec.update(passed=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1200:])
    rec["seconds"] = round(time.time() - t0, 1)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        flag = "PASS" if rec.get("passed") else "FAIL"
        print(f"  {flag}  {name:14s} d_in={rec.get('d_in','?'):>5} "
              f"params={rec.get('params',0)/1e6:5.1f}M steps={rec.get('steps_run','-')} "
              f"{rec.get('examples_per_s') or 0:.0f} ex/s "
              f"{rec.get('error','')}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-len", type=int, default=512,
                    help="cap tokenized length; 128 keeps E-bs128 off the CPU memory cliff")
    a = ap.parse_args()
    names = a.only or list(SHAPES)
    print(f"{STEPS}-step arm-shape smoke on {a.device}, max_len {a.max_len}: "
          f"{len(names)} shapes", flush=True)
    corp = corpus()
    recs = []
    for n in names:
        recs.append(smoke_one(n, SHAPES[n], corp, device=a.device, max_len=a.max_len))
        failed, no_ws = _write(recs, a.device, a.max_len)     # after EVERY shape
    print(f"\n{len(recs) - len(failed)}/{len(recs)} shapes pass; wrote {OUT}")
    if no_ws:
        print(f"registered warm start NOT implemented: {', '.join(no_ws)}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
