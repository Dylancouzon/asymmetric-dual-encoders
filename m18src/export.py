"""Atomic internal M18 int8 bundle export and serving-parity gates.

Forked from M17's exporter but removes checkpoint averaging, the erroneous 35M table cap and all
M17 source/lock hooks. M18 publishes one selected registered checkpoint (including step 0/V0-Q)
as a flat bundle only after the staged directory passes byte, tokenizer, dimension and NumPy
conformance checks.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from common import (atomic_save_npz, admit_read, admit_write, freeze, registry, sha_array,
                    sha_bytes, sha_file, sha_json, write_json)

BUNDLE_FILES = ("model.npz", "config.json", "tokenizer.json", "provenance.json")
FIXTURES = ("qdrant hnsw ef_construct", "CUDA_ERROR_OUT_OF_MEMORY", "grpc 500 port 6333",
            "S3 vs S4", "550e8400-e29b-41d4-a716-446655440000", "", "the the the")


def check_table_limits(rows, reg=None, fixture=False):
    reg = reg or registry()
    n, dim = map(int, rows.shape)
    if fixture:
        return {"rows": n, "dim": dim, "fixture": True}
    base = int(reg["models"]["zero_v1"]["vocab"])
    max_added = int(reg["vocabulary"]["max_added_rows"])
    if dim != int(reg["models"]["teacher"]["dim"]):
        raise SystemExit(f"M18 EXPORT REFUSED: table dim {dim} is not frozen dim "
                         f"{reg['models']['teacher']['dim']}")
    if not base <= n <= base + max_added:
        raise SystemExit(f"M18 EXPORT REFUSED: {n} rows outside {base}..{base+max_added}")
    return {"rows": n, "dim": dim, "added_rows": n - base,
            "resident_int8_bytes": n * dim + n * 4}


def _quantize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    scale = np.abs(rows).max(1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    codes = np.rint(rows / scale[:, None]).clip(-127, 127).astype(np.int8)
    return codes, scale


def _preprocessing_config(variant, implementation_sha256):
    if variant == "T0":
        return {"variant": "T0", "text": "raw", "token_mask": None,
                "implementation_sha256": implementation_sha256}
    if variant == "T1":
        return {"variant": "T1", "text": "typed_ephemeral", "token_mask": None,
                "implementation_sha256": implementation_sha256}
    if variant == "T2":
        return {"variant": "T2", "text": "raw", "token_mask": "^##[0-9]+$",
                "implementation_sha256": implementation_sha256}
    raise ValueError(variant)


def build_bundle(out_dir, rows, tokenizer, provenance, variant="T0",
                 preprocessing_sha256="", fixture=False, run_gates=True):
    """Stage every file, gate the staged bundle, then rename the directory atomically."""
    reg = registry()
    out = Path(admit_write(out_dir))
    if out.exists():
        raise SystemExit(f"M18 EXPORT REFUSED: immutable bundle destination exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.with_name(out.name + f".building-{os.getpid()}")
    if stage.exists():
        raise SystemExit(f"M18 EXPORT REFUSED: stale staging directory exists: {stage}")
    stage.mkdir()
    rows = np.asarray(rows, dtype=np.float32)
    limits = check_table_limits(rows, reg, fixture=fixture)
    if tokenizer.get_vocab_size(with_added_tokens=True) != rows.shape[0]:
        raise SystemExit("M18 EXPORT REFUSED: tokenizer/table row count mismatch")
    tokenizer_sha = sha_bytes(tokenizer.to_str().encode())
    identity = dict(provenance).get("training_snapshot") or dict(provenance).get("table_identity")
    if not fixture:
        required = {"variant": variant, "tokenizer_sha256": tokenizer_sha,
                    "preprocessing_sha256": preprocessing_sha256,
                    "rows_sha256": sha_array(rows)}
        if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in required.items()):
            raise SystemExit("M18 EXPORT REFUSED: table/tokenizer/variant/preprocessing identity mismatch")
    codes, scales = _quantize(rows)
    prep = _preprocessing_config(variant, preprocessing_sha256)
    try:
        atomic_save_npz(stage / "model.npz", rows_fp16=rows.astype(np.float16),
                        rows_int8=codes, int8_scale=scales)
        tokenizer.no_padding()
        tokenizer.save(str(stage / "tokenizer.json"))
        fz = freeze()
        config = {"_schema": "m18-internal-zero-bundle-v1", "internal_only": True,
                  "variant": variant, "vocab": int(rows.shape[0]), "dim": int(rows.shape[1]),
                  "fallback_token_id": int(fz["encoder_spec"]["cls_id"] if not fixture else 0),
                  "preproc": fz["preproc"] if not fixture else {
                      "prefix": "", "add_special_tokens": True, "max_length": 512,
                      "pool_mode": "sqrt"},
                  "query_preprocessing": prep,
                  "query_preprocessing_fingerprint": sha_json(prep),
                  "document_encoder": fz["encoder_spec"], "weights_folded": True,
                  "recommended_variant": "int8"}
        write_json(stage / "config.json", config)
        prov = {"_schema": "m18-internal-zero-provenance-v1", **dict(provenance),
                "table_float32_sha256": sha_array(rows), "limits": limits,
                "model_npz_sha256": sha_file(stage / "model.npz"),
                "tokenizer_sha256": sha_file(stage / "tokenizer.json"),
                "config_sha256": sha_file(stage / "config.json")}
        write_json(stage / "provenance.json", prov)
        gates = gate_bundle(stage, float_rows=rows) if run_gates else {}
        write_json(stage / "gates.json", gates)
        # gates.json is deliberately part of the immutable bundle and checked as an allowed file.
        os.replace(stage, out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return out


def gate_bundle(bundle, float_rows=None):
    import loader_np
    from tokenizers import Tokenizer

    b = Path(bundle)
    allowed = set(BUNDLE_FILES) | {"gates.json"}
    names = {p.name for p in b.iterdir() if p.is_file()}
    if not set(BUNDLE_FILES).issubset(names) or names - allowed:
        raise SystemExit(f"M18 GATE REFUSED: bundle files are {sorted(names)}")
    cfg = json.loads(admit_read(b / "config.json").read_text())
    prov = json.loads(admit_read(b / "provenance.json").read_text())
    got = {"model_npz_sha256": sha_file(b / "model.npz"),
           "tokenizer_sha256": sha_file(b / "tokenizer.json"),
           "config_sha256": sha_file(b / "config.json")}
    if any(got[k] != prov.get(k) for k in got):
        raise SystemExit("M18 GATE REFUSED: generated-byte hashes do not match provenance")
    z = np.load(admit_read(b / "model.npz"))
    if z["rows_int8"].shape != (cfg["vocab"], cfg["dim"]):
        raise SystemExit("M18 GATE REFUSED: config/table shape mismatch")
    tok = Tokenizer.from_file(str(admit_read(b / "tokenizer.json")))
    if tok.get_vocab_size(with_added_tokens=True) != cfg["vocab"]:
        raise SystemExit("M18 GATE REFUSED: tokenizer/table row mismatch")
    identity = prov.get("training_snapshot") or prov.get("table_identity")
    if not prov.get("fixture"):
        expected = {"variant": cfg["variant"], "tokenizer_sha256": sha_file(b / "tokenizer.json"),
                    "preprocessing_sha256": cfg["query_preprocessing"]["implementation_sha256"],
                    "rows_sha256": prov.get("table_float32_sha256")}
        if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected.items()):
            raise SystemExit("M18 GATE REFUSED: provenance identity does not bind bundle bytes")
    parity = loader_np.parity(b, texts=FIXTURES, float_rows=float_rows,
                              tol=float(registry()["serving"]["loader_parity_max_abs"]))
    if not parity["pass"]:
        raise SystemExit("M18 GATE REFUSED: eager/resident NumPy loader mismatch")
    qerr = parity.get("vs_exported_float_rows_max_abs", 0.0)
    if qerr > float(registry()["serving"]["int8_error_max_abs"]):
        raise SystemExit(f"M18 GATE REFUSED: int8 query error {qerr:.3e}")
    training_parity = _training_forward_parity(b, float_rows, FIXTURES) if float_rows is not None else {}
    if training_parity and training_parity["max_abs"] > float(registry()["serving"]["loader_parity_max_abs"]):
        raise SystemExit("M18 GATE REFUSED: training-forward and serving float outputs differ")
    return {"generated_hashes": got, "loader": parity, "training_forward": training_parity,
            "table": check_table_limits(z["rows_int8"], fixture=bool(prov.get("fixture")))}


def _training_forward_parity(bundle, rows, texts):
    """Compare the actual torch training forward rule with serving's float reference."""
    import loader_np
    import preprocess
    import train
    from tokenizers import Tokenizer
    cfg = json.loads(admit_read(Path(bundle) / "config.json").read_text())
    tok = Tokenizer.from_file(str(admit_read(Path(bundle) / "tokenizer.json")))
    tok.enable_truncation(max_length=int(cfg["preproc"]["max_length"])); tok.no_padding()
    variant = cfg["variant"]
    ids = [preprocess.active_ids(t, tok, variant) for t in texts]
    model = train.build_model(np.asarray(rows, np.float32), np.empty((0, rows.shape[1]), np.float32),
                              fallback_id=int(cfg["fallback_token_id"]), device="cpu")
    with __import__("torch").no_grad():
        actual = model.forward_ids(ids).cpu().numpy()
    expected = loader_np._reference_encode(np.asarray(rows, np.float32),
                                            Path(bundle) / "tokenizer.json", cfg, texts, variant)
    return {"max_abs": float(np.abs(actual - expected).max()), "queries": len(texts)}


def main(argv=None):
    from tokenizers import Tokenizer
    import train
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot", help="snapshot directory containing table.npz/meta.json")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=("T0", "T1", "T2"), required=True)
    ap.add_argument("--preprocessing-sha256", required=True)
    args = ap.parse_args(argv)
    rows, meta = train.load_snapshot(args.snapshot)
    build_bundle(args.out, rows, Tokenizer.from_file(str(admit_read(args.tokenizer))),
                 {"training_snapshot": meta}, variant=args.variant,
                 preprocessing_sha256=args.preprocessing_sha256)


if __name__ == "__main__":
    main()
