"""M17 bundle export: fold once, average the registered snapshots, gate the bundle.

Three things, in this order:

1. **Fold.** Effective float32 rows are `softplus(w)[:, None] * rows`, computed ONCE from an
   unfolded training checkpoint. A checkpoint already marked `weights_folded` is refused.
2. **Average.** `checkpoint_averaging`: an equal arithmetic mean of the effective float32 rows
   of the registered step-bound snapshots — no separate averaging of rows and scalars, no
   per-row or global rescaling, no mixing of tokenizers, runs or row scales, and the result is
   quantized once. Effective-table RMS is recorded at every snapshot so scale drift is visible.
3. **Gate.** A SEPARATE set of M17 gates modeled on `m11/release/push.py` and
   `m11/release/verify_bundle.py`. They never touch M11's gates and never relax them: v1's gates
   stay bound to `m7/FREEZE.json:table_sha256`, which no M17 artifact can or should satisfy.
   What M17 inherits is the METHOD — compare the shipped numpy path against the torch path on
   adversarial fixtures, hash what is staged, and assert the document-encoder metadata.

The frozen `encoder_spec` is COPIED from `m7/FREEZE.json` into the bundle and asserted field by
field (revision, dim, pooling, post-pooling Dense, prefixes). The bundle records training and
evaluation bank hashes in its provenance; it does NOT require a production corpus hash, because
the same document space serves different users' corpora.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import freeze, registry, sha_file, sha_json, write_json

BUNDLE_FILES = ("model.npz", "config.json", "tokenizer.json", "provenance.json")
SPEC_FIELDS = ("repo", "revision", "dim", "pooling", "post_dense", "query_prefix", "doc_prefix",
               "max_length", "tokenizer_id", "cls_id")


def effective_rows(npz_path):
    """Effective float32 rows of ONE unfolded checkpoint, plus its scale diagnostic.

    `m7src.table.save_table` stores `token_weights` as `softplus(w_raw)` — already the
    positive scalar the forward multiplies by — so the fold is one multiply, not a second
    softplus.
    """
    p = Path(npz_path)
    meta = json.loads((p.parent / (p.stem + ".meta.json")).read_text())
    if meta.get("weights_folded"):
        raise SystemExit(f"M17 EXPORT REFUSED: {npz_path} is already folded; folding twice "
                         "multiplies every row by its scalar a second time.")
    z = np.load(p)
    rows = z["rows_fp16"].astype(np.float32)
    w = z["token_weights"]
    if w.size:
        rows = np.asarray(w, dtype=np.float32)[:, None] * rows
    rms = float(np.sqrt((rows ** 2).mean()))
    return rows, {"path": str(npz_path), "rms": rms, "meta": meta}


def average_snapshots(paths, reg=None):
    """Equal mean of effective float32 rows. Refuses to mix tokenizers, runs or scales."""
    reg = reg or registry()
    want = [int(s) for s in reg["checkpoint_averaging"]["checkpoint_steps"]]
    rows, diag = [], []
    ident = None
    for p in paths:
        r, d = effective_rows(p)
        m = d["meta"]
        key = (m.get("m17_run_id"), m.get("tokenizer_sha256"), m.get("vocabulary_sha256"),
               r.shape)
        if ident is None:
            ident = key
        elif key != ident:
            raise SystemExit(f"M17 AVERAGING REFUSED: {p} has identity {key} but the first "
                             f"snapshot has {ident}; no cross-run, cross-tokenizer or "
                             "cross-shape averaging.")
        rows.append(r)
        diag.append({"step": m.get("m17_step"), **{k: v for k, v in d.items() if k != "meta"}})
    got = sorted(int(d["step"]) for d in diag if d["step"] is not None)
    if got != sorted(want):
        raise SystemExit(f"M17 AVERAGING REFUSED: snapshots at steps {got}, registry requires "
                         f"{sorted(want)}. A missing snapshot is an incomplete comparison, not "
                         "permission to average a different window.")
    mean = np.mean(np.stack(rows, 0), axis=0).astype(np.float32)
    diag.append({"step": "mean_last_three", "rms": float(np.sqrt((mean ** 2).mean()))})
    return mean, {"snapshots": diag, "operation": reg["checkpoint_averaging"]["operation"]}


def assert_encoder_spec(spec, reg=None):
    """The candidate bundle's document-encoder metadata must BE the frozen spec."""
    reg = reg or registry()
    fz_spec = freeze()["encoder_spec"]
    missing = [f for f in SPEC_FIELDS if spec.get(f) != fz_spec.get(f)]
    if missing:
        raise SystemExit(f"M17 BUILD REFUSED: encoder_spec fields {missing} differ from "
                         "m7/FREEZE.json. The document index is frozen; a table trained against "
                         "a different document space is not a drop-in v1.1.")
    if spec["revision"] != reg["teacher_revision"] or spec["dim"] != reg["dim"]:
        raise SystemExit("M17 BUILD REFUSED: encoder_spec disagrees with the registry's teacher "
                         "revision or dimension.")
    return fz_spec


def build_bundle(out_dir, eff_rows, tokenizer, provenance, reg=None, form="endpoint",
                 fallback_id=None):
    """Write the M17 candidate bundle. `eff_rows` are already folded effective float32 rows.

    `fallback_id` defaults to the frozen `encoder_spec.cls_id` (101 in this WordPiece
    vocabulary) and any other value is recorded in the config as a fixture table, which is the
    only situation it can legitimately arise in — the rehearsal's toy vocabulary is smaller
    than 101 rows.
    """
    from table import quantize_int8
    reg = reg or registry()
    fz = freeze()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = assert_encoder_spec(dict(fz["encoder_spec"]), reg)

    n_tok = tokenizer.get_vocab_size(with_added_tokens=True)
    if n_tok != eff_rows.shape[0]:
        raise SystemExit(f"M17 BUILD REFUSED: tokenizer has {n_tok} tokens but the table has "
                         f"{eff_rows.shape[0]} rows; a token id would index off the end.")
    q, scale = quantize_int8(eff_rows)
    np.savez(out / "model.npz", rows_fp16=eff_rows.astype(np.float16), rows_int8=q,
             int8_scale=scale, token_weights=np.zeros(0, dtype=np.float32),
             updates=np.zeros(0, dtype=np.int64))
    tokenizer.save(str(out / "tokenizer.json"))
    spec_cls = int(fz["encoder_spec"]["cls_id"])
    fb = spec_cls if fallback_id is None else int(fallback_id)
    if fb >= eff_rows.shape[0]:
        raise SystemExit(f"M17 BUILD REFUSED: fallback token id {fb} is outside a table of "
                         f"{eff_rows.shape[0]} rows.")
    config = {
        "_note": "M17 zero v1.1 CANDIDATE bundle. Not a release; no release authority is "
                 "implied by its existence (m17/registry.json screening_preferences...).",
        "preproc": fz["preproc"], "preproc_fingerprint": fz["preproc_fingerprint"],
        "fallback_token_id": fb,
        **({"fixture_table": True, "frozen_cls_id": spec_cls} if fb != spec_cls else {}),
        "vocab": int(eff_rows.shape[0]), "dim": int(eff_rows.shape[1]),
        "weights_folded": True, "form": form,
        "document_encoder": spec,
        "document_encoder_spec_source": reg["document_encoder_spec_source"],
    }
    write_json(out / "config.json", config)
    prov = {"_schema": "m17-bundle-provenance-v1", "form": form,
            "registry_status": reg.get("status"), **provenance}
    prov["model_npz_sha256"] = sha_file(out / "model.npz")
    prov["tokenizer_sha256"] = sha_file(out / "tokenizer.json")
    prov["config_sha256"] = sha_json(config)
    write_json(out / "provenance.json", prov)
    return out


# ---- M17 gates (separate from M11's; never weaken those) -----------------------------------

FIXTURES = [
    "what is a lookup table", "protein folding market impact", "zzzqx", "", "   ",
    "the the the the the the", "s3 bucket policy", "k8s ingress", "S3://bucket/key",
    "s3-compatible object storage", "xs3", "C++ and .NET", "[CLS] [SEP] [UNK] [MASK]",
    "electroencephalographically " + "a" * 120, " ".join(["retrieval augmented generation"] * 400),
]


def gate_files(bundle):
    got = sorted(p.name for p in Path(bundle).iterdir() if p.is_file())
    want = sorted(BUNDLE_FILES)
    extra, missing = sorted(set(got) - set(want)), sorted(set(want) - set(got))
    if missing or extra:
        raise SystemExit(f"M17 GATE files: missing {missing}, unexpected {extra}")
    return {"files": got}


def gate_artifact(bundle):
    """The staged bytes must be the bytes provenance claims (M11 gate 1's failure mode)."""
    b = Path(bundle)
    prov = json.loads((b / "provenance.json").read_text())
    got = sha_file(b / "model.npz")
    if got != prov.get("model_npz_sha256"):
        raise SystemExit(f"M17 GATE artifact: staged model.npz hashes {got[:12]} but provenance "
                         f"records {str(prov.get('model_npz_sha256'))[:12]}")
    return {"model_npz_sha256": got}


def gate_encoder_spec(bundle):
    cfg = json.loads((Path(bundle) / "config.json").read_text())
    fz = freeze()
    if cfg["preproc"] != fz["preproc"] or cfg["preproc_fingerprint"] != fz["preproc_fingerprint"]:
        raise SystemExit("M17 GATE preproc: bundle preprocessing differs from m7/FREEZE.json; "
                         "the query rule is frozen and the same for v1 and any v1.1 candidate.")
    assert_encoder_spec(cfg["document_encoder"])
    return {"preproc_fingerprint": cfg["preproc_fingerprint"]}


def gate_tokenizer(bundle):
    """Padding off, truncation at the frozen max_length, vocabulary size equal to the rows."""
    from tokenizers import Tokenizer
    b = Path(bundle)
    cfg = json.loads((b / "config.json").read_text())
    raw = json.loads((b / "tokenizer.json").read_text())
    if raw.get("padding") is not None:
        raise SystemExit("M17 GATE tokenizer: padding is enabled; ~500 [PAD] rows would enter "
                         "every bag (the v1 sanitisation exists for this).")
    tok = Tokenizer.from_file(str(b / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    if n != cfg["vocab"]:
        raise SystemExit(f"M17 GATE tokenizer: {n} tokens vs {cfg['vocab']} rows")
    return {"vocab": n, "truncation_in_file": raw.get("truncation")}


def gate_conformance(bundle, tol=1e-5):
    """The shipped numpy loader must reproduce the torch query path on both variants."""
    from table import Preproc, QueryTable
    import loader_np
    b = Path(bundle)
    cfg = json.loads((b / "config.json").read_text())
    pre = Preproc(**cfg["preproc"])
    z = np.load(b / "model.npz")
    out = {}
    for variant in ("fp16", "int8"):
        rows = (z["rows_fp16"].astype(np.float32) if variant == "fp16"
                else z["rows_int8"].astype(np.float32) * z["int8_scale"][:, None])
        ref = QueryTable(rows, learned_weights=False,
                         fallback_id=cfg["fallback_token_id"]).to("cpu").eval()
        a = _torch_encode(ref, b / "tokenizer.json", pre, FIXTURES)
        for mode in ("eager_fp32", "resident_int8"):
            if mode == "resident_int8" and variant != "int8":
                continue
            enc = loader_np.M17QueryEncoder(b, variant=variant, mode=mode)
            c = enc.encode(FIXTURES)
            dev = float(np.abs(a - c).max())
            solo = max(float(np.abs(enc.encode(t)[0] - a[i]).max())
                       for i, t in enumerate(FIXTURES))
            out[f"{variant}/{mode}"] = {"max_abs": dev, "b1_max_abs": solo}
            if dev > tol or solo > tol:
                raise SystemExit(f"M17 GATE conformance: {variant}/{mode} does not reproduce the "
                                 f"torch query path (max-abs {dev:.3e}, b=1 {solo:.3e})")
    return out


def _torch_encode(model, tokenizer_json, pre, texts):
    """Torch reference using the BUNDLE's tokenizer, not the teacher's `AutoTokenizer`.

    The bundle ships an extended vocabulary; loading the teacher's tokenizer here would compare
    the new table against the old tokenization and pass for the wrong reason.
    """
    import torch
    from tokenizers import Tokenizer
    from table import EPS, _bag_index, occurrence_weights, ragged
    import torch.nn.functional as F
    tok = Tokenizer.from_file(str(tokenizer_json))
    tok.no_padding()
    tok.enable_truncation(max_length=pre.max_length)
    ids = [e.ids for e in tok.encode_batch(list(texts))]
    out = np.empty((len(texts), model.rows.shape[1]), dtype=np.float32)
    with torch.no_grad():
        flat, off, lens = ragged(ids, "cpu")
        psw = occurrence_weights(ids, pre.pool_mode, device="cpu")
        s = F.embedding_bag(flat, model.rows, off, mode="sum", per_sample_weights=psw)
        denom = torch.zeros_like(s[:, 0]).index_add_(0, _bag_index(off, lens), psw)
        mean = s / denom.clamp_min(EPS).unsqueeze(1)
        norm = mean.norm(dim=1, keepdim=True)
        fb = model.fallback_vector()
        out[:] = torch.where(norm > EPS, mean / norm.clamp_min(EPS),
                             fb.expand_as(mean)).float().numpy()
    return out


GATES = (gate_files, gate_artifact, gate_encoder_spec, gate_tokenizer, gate_conformance)


def run_gates(bundle, log=print):
    report = {}
    for g in GATES:
        report[g.__name__] = g(bundle)
        log(f"  PASS {g.__name__}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshots", nargs="*", default=[], help="unfolded snapshot npz paths")
    ap.add_argument("--endpoint", default=None, help="unfolded endpoint npz path")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--form", choices=("endpoint", "mean_last_three"), default="endpoint")
    ap.add_argument("--provenance", default=None, help="JSON file merged into provenance.json")
    ap.add_argument("--gates", action="store_true")
    args = ap.parse_args(argv)
    from tokenizers import Tokenizer
    reg = registry()
    if args.form == "mean_last_three":
        rows, diag = average_snapshots(args.snapshots, reg)
    else:
        rows, d = effective_rows(args.endpoint)
        diag = {"snapshots": [{"step": d["meta"].get("m17_step"), "rms": d["rms"]}]}
    prov = json.loads(Path(args.provenance).read_text()) if args.provenance else {}
    out = build_bundle(args.out, rows, Tokenizer.from_file(args.tokenizer),
                       {**prov, "averaging": diag}, reg, form=args.form)
    if args.gates:
        run_gates(out)
    print(f"bundle: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
