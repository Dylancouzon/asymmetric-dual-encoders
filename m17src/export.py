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

from common import admit_read, freeze, registry, sha_file, sha_json, write_json

BUNDLE_FILES = ("model.npz", "config.json", "tokenizer.json", "provenance.json")
ATTRIBUTION_SRC = "research/m17-k8s-attribution.md"
ATTRIBUTION_NAME = "ATTRIBUTION.md"
ATTRIBUTION_TRIGGER = "k8s-docs-en"
SPEC_FIELDS = ("repo", "revision", "dim", "pooling", "post_dense", "query_prefix", "doc_prefix",
               "max_length", "tokenizer_id", "cls_id", "config_kwargs")
# The torch-vs-numpy conformance bound. It is NOT the registry's loader-vs-loader
# `int8_resident_loading.parity_max_abs` (1e-6): the two paths differ in accumulation order,
# so this bound is looser and is recorded in the bundle's provenance.
CONFORMANCE_TOL = 1e-5


def effective_rows(npz_path):
    """Effective float32 rows of ONE unfolded checkpoint, plus its scale diagnostic.

    `m7src.table.save_table` stores `token_weights` as `softplus(w_raw)` — already the
    positive scalar the forward multiplies by — so the fold is one multiply, not a second
    softplus. `rows_fp32` (written by `train._save_table`) is preferred over the legacy
    `rows_fp16` array: FP16 rounding before folding can collapse two different snapshots.
    """
    p = admit_read(npz_path)
    meta = json.loads(admit_read(p.parent / (p.stem + ".meta.json")).read_text())
    if meta.get("weights_folded"):
        raise SystemExit(f"M17 EXPORT REFUSED: {npz_path} is already folded; folding twice "
                         "multiplies every row by its scalar a second time.")
    z = np.load(p)
    stored = "rows_fp32" if "rows_fp32" in z.files else "rows_fp16"
    rows = z[stored].astype(np.float32)
    w = z["token_weights"]
    if w.size:
        rows = np.asarray(w, dtype=np.float32)[:, None] * rows
    rms = float(np.sqrt((rows ** 2).mean()))
    return rows, {"path": str(npz_path), "rms": rms, "rows_stored_as": stored, "meta": meta}


def average_snapshots(paths, reg=None):
    """Equal mean of effective float32 rows. Refuses to mix tokenizers, runs or scales."""
    reg = reg or registry()
    want = [int(s) for s in reg["checkpoint_averaging"]["checkpoint_steps"]]
    paths = list(paths)
    if len(paths) != len(want):
        raise SystemExit(f"M17 AVERAGING REFUSED: {len(paths)} snapshots supplied, the registry "
                         f"registers exactly {len(want)} ({sorted(want)}). A fourth table is "
                         "not the registered mean.")
    rows, diag = [], []
    ident = None
    for p in paths:
        r, d = effective_rows(p)
        m = d["meta"]
        key = tuple(m.get(f) for f in ("m17_run_id", "tokenizer_sha256", "vocabulary_sha256",
                                       "candidate_cache_sha256")) + (r.shape,)
        if any(v in (None, "") for v in key[:4]):
            raise SystemExit(f"M17 AVERAGING REFUSED: {p} does not record its run, tokenizer, "
                             "vocabulary and cache identities; a missing field is not a match.")
        if m.get("m17_step") is None:
            raise SystemExit(f"M17 AVERAGING REFUSED: {p} records no `m17_step`; an unstepped "
                             "snapshot cannot be checked against the registered window.")
        if ident is None:
            ident = key
        elif key != ident:
            raise SystemExit(f"M17 AVERAGING REFUSED: {p} has identity {key} but the first "
                             f"snapshot has {ident}; no cross-run, cross-tokenizer or "
                             "cross-shape averaging.")
        rows.append(r)
        diag.append({"step": int(m["m17_step"]), **{k: v for k, v in d.items() if k != "meta"}})
    got = sorted(int(d["step"]) for d in diag)
    if len(set(got)) != len(got) or got != sorted(want):
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


def check_table_limits(eff_rows, reg=None):
    """Actual shape against the frozen space and the registered caps.

    A table can be internally consistent and still be the wrong artifact: 16 dimensions
    stamped with the 1024-dimensional document spec, more added rows than the cap allows, or
    a parameter count over the 35M student cap. Rows AND their scalars count.
    """
    reg = reg or registry()
    rows, dim = int(eff_rows.shape[0]), int(eff_rows.shape[1])
    base, add_max = int(reg["base_vocab"]), int(reg["added_rows_max"])
    if dim != int(reg["dim"]):
        raise SystemExit(f"M17 BUILD REFUSED: the table has {dim} dimensions, the frozen "
                         f"document space is {reg['dim']}-dimensional.")
    added = rows - base
    if added < 0 or added > add_max:
        raise SystemExit(f"M17 BUILD REFUSED: {rows} rows is base_vocab {base} + {added} added; "
                         f"the registered range is 0..{add_max} added rows.")
    params = rows * (dim + 1)                       # rows plus one learned scalar per row
    cap = int(reg["student_parameter_cap"])
    if params > cap:
        raise SystemExit(f"M17 BUILD REFUSED: {params} parameters (rows plus scalars) exceeds "
                         f"the {cap} student cap.")
    return {"rows": rows, "dim": dim, "added_rows": added, "parameters": params,
            "parameter_cap": cap}


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
    fixture = fallback_id is not None and int(fallback_id) != int(fz["encoder_spec"]["cls_id"])
    if not fixture:
        check_table_limits(eff_rows, reg)

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
    # CC BY 4.0 attribution ships WITH the derived weights whenever the vocabulary was mined
    # from the Kubernetes documentation (research/m7-data-licensing.md, Kubernetes row).
    sources = provenance.get("vocabulary_sources") or provenance.get("sources") or []
    attribution = None
    if ATTRIBUTION_TRIGGER in list(sources):
        src = admit_read(Path(__file__).resolve().parents[1] / ATTRIBUTION_SRC)
        (out / ATTRIBUTION_NAME).write_text(src.read_text())
        attribution = ATTRIBUTION_NAME

    config = {
        "_note": "M17 zero v1.1 CANDIDATE bundle. Not a release; no release authority is "
                 "implied by its existence (m17/registry.json screening_preferences...).",
        **({"attribution": attribution} if attribution else {}),
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
    if attribution:
        prov["attribution_sha256"] = sha_file(out / ATTRIBUTION_NAME)
        prov["attribution_source"] = ATTRIBUTION_SRC
    prov["table_limits"] = ({"fixture_table": True} if fixture
                            else check_table_limits(eff_rows, reg))
    prov["conformance_tolerance_max_abs"] = CONFORMANCE_TOL
    prov["conformance_tolerance_note"] = (
        "torch-vs-numpy query-path bound for gate_conformance; the loader-vs-loader bound is "
        "the registry's int8_resident_loading.parity_max_abs and is measured separately.")
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
    """EVERY directory entry, not only the ones the allowlist expects."""
    b = Path(bundle)
    cfg = (json.loads(admit_read(b / "config.json").read_text())
           if (b / "config.json").exists() else {})
    want = set(BUNDLE_FILES) | ({ATTRIBUTION_NAME} if cfg.get("attribution") else set())
    entries = sorted(p.name for p in b.iterdir())
    nonfiles = sorted(p.name for p in b.iterdir() if not p.is_file())
    if nonfiles:
        raise SystemExit(f"M17 GATE files: {nonfiles} are not plain files; a bundle is a flat "
                         "set of known artifacts.")
    extra, missing = sorted(set(entries) - want), sorted(want - set(entries))
    if missing or extra:
        raise SystemExit(f"M17 GATE files: missing {missing}, unexpected {extra}")
    return {"files": entries}


def gate_artifact(bundle):
    """EVERY hash provenance records must be the hash of the staged bytes."""
    b = Path(bundle)
    prov = json.loads(admit_read(b / "provenance.json").read_text())
    cfg = json.loads(admit_read(b / "config.json").read_text())
    got = {"model_npz_sha256": sha_file(b / "model.npz"),
           "tokenizer_sha256": sha_file(b / "tokenizer.json"),
           "config_sha256": sha_json(cfg)}
    if cfg.get("attribution"):
        got["attribution_sha256"] = sha_file(b / cfg["attribution"])
    for k, v in got.items():
        if v != prov.get(k):
            raise SystemExit(f"M17 GATE artifact: staged {k} is {v[:12]} but provenance records "
                             f"{str(prov.get(k))[:12]}")
    return got


def gate_encoder_spec(bundle):
    cfg = json.loads(admit_read(Path(bundle) / "config.json").read_text())
    fz = freeze()
    if cfg["preproc"] != fz["preproc"] or cfg["preproc_fingerprint"] != fz["preproc_fingerprint"]:
        raise SystemExit("M17 GATE preproc: bundle preprocessing differs from m7/FREEZE.json; "
                         "the query rule is frozen and the same for v1 and any v1.1 candidate.")
    assert_encoder_spec(cfg["document_encoder"])
    z = np.load(admit_read(Path(bundle) / "model.npz"))
    rows = z["rows_int8"]
    out = {"preproc_fingerprint": cfg["preproc_fingerprint"], "shape": list(rows.shape)}
    if int(rows.shape[0]) != int(cfg["vocab"]) or int(rows.shape[1]) != int(cfg["dim"]):
        raise SystemExit(f"M17 GATE encoder_spec: the staged table is {rows.shape} but the "
                         f"config declares {(cfg['vocab'], cfg['dim'])}.")
    if cfg.get("fixture_table"):
        out["fixture_table"] = True                # a rehearsal table, recorded as such
        return out
    out["limits"] = check_table_limits(np.zeros((rows.shape[0], rows.shape[1]), dtype=np.float32))
    return out


def gate_tokenizer(bundle):
    """Padding off, truncation at the frozen max_length, vocabulary size equal to the rows."""
    from tokenizers import Tokenizer
    b = Path(bundle)
    cfg = json.loads(admit_read(b / "config.json").read_text())
    raw = json.loads(admit_read(b / "tokenizer.json").read_text())
    if raw.get("padding") is not None:
        raise SystemExit("M17 GATE tokenizer: padding is enabled; ~500 [PAD] rows would enter "
                         "every bag (the v1 sanitisation exists for this).")
    tok = Tokenizer.from_file(str(admit_read(b / "tokenizer.json")))
    n = tok.get_vocab_size(with_added_tokens=True)
    if n != cfg["vocab"]:
        raise SystemExit(f"M17 GATE tokenizer: {n} tokens vs {cfg['vocab']} rows")
    return {"vocab": n, "truncation_in_file": raw.get("truncation")}


def gate_conformance(bundle, tol=None):
    """The shipped numpy loader must reproduce the torch query path on both variants.

    Non-finite values are rejected BEFORE any tolerance comparison: `nan > tol` is False, so a
    NaN table would otherwise pass the gate it exists to fail.
    """
    from table import Preproc, QueryTable
    import loader_np
    tol = CONFORMANCE_TOL if tol is None else float(tol)
    b = Path(bundle)
    cfg = json.loads(admit_read(b / "config.json").read_text())
    pre = Preproc(**cfg["preproc"])
    z = np.load(admit_read(b / "model.npz"))
    scale = z["int8_scale"]
    if scale.shape != (z["rows_int8"].shape[0],):
        raise SystemExit(f"M17 GATE conformance: int8_scale has shape {scale.shape} for "
                         f"{z['rows_int8'].shape[0]} rows; one positive scale per row is "
                         "required.")
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise SystemExit("M17 GATE conformance: int8_scale is not finite and strictly positive.")
    for name in ("rows_fp16", "rows_int8"):
        if not np.isfinite(z[name].astype(np.float32)).all():
            raise SystemExit(f"M17 GATE conformance: {name} contains non-finite values; a NaN "
                             "table cannot be compared against a tolerance.")
    out = {"tolerance": tol, "tolerance_source": "export.CONFORMANCE_TOL (torch vs numpy)"}
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
            if not (np.isfinite(a).all() and np.isfinite(c).all()):
                raise SystemExit(f"M17 GATE conformance: {variant}/{mode} produced non-finite "
                                 "query vectors; there is nothing to compare.")
            dev = float(np.abs(a - c).max())
            solo = max(float(np.abs(enc.encode(t)[0] - a[i]).max())
                       for i, t in enumerate(FIXTURES))
            out[f"{variant}/{mode}"] = {"max_abs": dev, "b1_max_abs": solo}
            if not (np.isfinite(dev) and np.isfinite(solo)):
                raise SystemExit(f"M17 GATE conformance: {variant}/{mode} error statistic is not "
                                 "finite; `nan > tol` is False and would pass this gate.")
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
    tok = Tokenizer.from_file(str(admit_read(tokenizer_json)))
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
    prov = json.loads(admit_read(args.provenance).read_text()) if args.provenance else {}
    out = build_bundle(args.out, rows, Tokenizer.from_file(str(admit_read(args.tokenizer))),
                       {**prov, "averaging": diag}, reg, form=args.form)
    if args.gates:
        run_gates(out)
    print(f"bundle: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
