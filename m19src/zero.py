"""Deterministic M19 row algebra, compact export and resident-int8 serving loader.

Forked semantically from M18's ``export.py`` and ``loader_np.py`` at commit bfa7257. M19 removes
training/checkpoints and constructs only V0-compose and T0-teacher rows. Inherited int8 codes,
scales, token IDs and pooling values remain exact; the full table is never expanded to float32.
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from m19src.common import (M19, REGISTRY_PATH, RELEASE_BUNDLE, admit_read, admit_write,
                           atomic_create_bytes, load_json, sha_array, sha_bytes, sha_file,
                           sha_file_unchecked, sha_json)
from m19src.term_inventory import _extend_tokenizer

EPS = 1e-6


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("non-finite or degenerate query sum")
    return (vector / norm).astype(np.float32)


def _gather_sum(ids, gather):
    counts = Counter(map(int, ids))
    total = np.zeros_like(np.asarray(gather(next(iter(counts))), dtype=np.float32))
    for token_id in sorted(counts):
        total += np.sqrt(np.float32(counts[token_id])) * np.asarray(gather(token_id), np.float32)
    return total


def dequant_row(codes, scales, token_id):
    return codes[int(token_id)].astype(np.float32) * np.float32(scales[int(token_id)])


def quantize_rows(rows):
    rows = np.asarray(rows, dtype=np.float32)
    maxima = np.abs(rows).max(axis=1)
    scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype(np.float32)
    codes = np.rint(rows / scales[:, None]).clip(-127, 127).astype(np.int8)
    return codes, scales


def load_base(model_path=RELEASE_BUNDLE / "model.npz"):
    with np.load(admit_read(model_path)) as archive:
        codes = np.asarray(archive["rows_int8"], dtype=np.int8).copy()
        scales = np.asarray(archive["int8_scale"], dtype=np.float32).copy()
    if codes.ndim != 2 or scales.shape != (codes.shape[0],):
        raise SystemExit("M19 ROW REFUSED: malformed released codes/scales")
    return codes, scales


def construct_added_rows(base_codes, base_scales, base_tokenizer_payload, roster, teacher_vectors):
    """Return V0/T0 added float rows and per-term algebra receipts.

    ``teacher_vectors`` maps each roster term to its pinned-prefix unit Stella query vector.
    ``alpha`` is the norm of the released bare-query pre-normalization sum. The pooling denominator
    cancels under final normalization, so it is intentionally absent from the construction.
    """
    base = Tokenizer.from_str(base_tokenizer_payload.decode("utf-8"))
    base.no_padding(); base.no_truncation()
    terms = [row["term"] for row in roster["terms"]]
    extended, audit = _extend_tokenizer(base_tokenizer_payload, terms)
    expected_audit = roster["selected_added_token_audit"]
    if (audit["term_ids"] != expected_audit["term_ids"]
            or audit["inherited_vocab_sha256"] != expected_audit["inherited_vocab_sha256"]):
        raise SystemExit("M19 ROW REFUSED: tokenizer extension differs from roster lock")
    dim = int(base_codes.shape[1])
    v0_rows, t0_rows, receipts = [], [], []
    for row in roster["terms"]:
        term = row["term"]
        original = base.encode(term, add_special_tokens=True).ids
        exact = extended.encode(term, add_special_tokens=True).ids
        added_id = audit["term_ids"][term]
        if exact.count(added_id) != 1 or any(i >= len(base_codes) and i != added_id for i in exact):
            raise SystemExit(f"M19 ROW REFUSED: non-exact added-token behavior for {term}")
        old_sum = _gather_sum(original, lambda token_id: dequant_row(base_codes, base_scales, token_id))
        remaining = [token_id for token_id in exact if token_id != added_id]
        fixed_sum = _gather_sum(remaining, lambda token_id: dequant_row(base_codes, base_scales, token_id))
        composition = old_sum - fixed_sum
        teacher = _normalize(teacher_vectors[term])
        alpha = float(np.linalg.norm(old_sum))
        if not np.isfinite(alpha) or alpha <= EPS:
            raise SystemExit(f"M19 ROW REFUSED: invalid released sum norm for {term}")
        target = (np.float32(alpha) * teacher - fixed_sum).astype(np.float32)
        pre_vector = _normalize(fixed_sum + target)
        pre_cosine = float(np.dot(pre_vector, teacher))
        if pre_cosine < 0.999999:
            raise SystemExit(f"M19 ROW REFUSED: {term} prequant cosine {pre_cosine}")
        v0_rows.append(composition.astype(np.float32))
        t0_rows.append(target)
        receipts.append({
            "term": term,
            "added_id": added_id,
            "original_ids": list(map(int, original)),
            "exact_ids": list(map(int, exact)),
            "remaining_ids": list(map(int, remaining)),
            "alpha_old_sum_norm": alpha,
            "old_sum_sha256": sha_array(old_sum),
            "fixed_sum_sha256": sha_array(fixed_sum),
            "composition_row_sha256": sha_array(composition.astype(np.float32)),
            "teacher_row_sha256": sha_array(target),
            "prequant_bare_cosine": pre_cosine,
        })
    return {
        "V0-compose": np.stack(v0_rows),
        "T0-teacher": np.stack(t0_rows),
        "tokenizer": extended,
        "tokenizer_audit": audit,
        "receipts": receipts,
        "dim": dim,
    }


def compact_table(base_codes, base_scales, added_float_rows):
    added_codes, added_scales = quantize_rows(added_float_rows)
    return (np.concatenate([base_codes, added_codes], axis=0),
            np.concatenate([base_scales, added_scales], axis=0))


class M19QueryEncoder:
    """Compact NumPy query encoder; only referenced rows are dequantized per query."""

    def __init__(self, codes, scales, tokenizer, config):
        self.codes = np.asarray(codes, dtype=np.int8)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.tokenizer = tokenizer
        self.config = config
        pre = config["preproc"]
        if pre != {"add_special_tokens": True, "max_length": 512,
                   "pool_mode": "sqrt", "prefix": ""}:
            raise ValueError(f"unexpected inherited pooling parameters: {pre}")
        if self.scales.shape != (self.codes.shape[0],):
            raise ValueError("codes/scales shape mismatch")
        if tokenizer.get_vocab_size(with_added_tokens=True) != self.codes.shape[0]:
            raise ValueError("tokenizer/table vocabulary mismatch")
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=int(pre["max_length"]))
        self.fallback_id = int(config["fallback_token_id"])

    @classmethod
    def from_bundle(cls, bundle, *, verification):
        bundle = Path(bundle)
        verify_bundle(bundle, verification=verification)
        config = load_json(bundle / "config.json")
        with np.load(admit_read(bundle / "model.npz")) as archive:
            codes = np.asarray(archive["rows_int8"], dtype=np.int8).copy()
            scales = np.asarray(archive["int8_scale"], dtype=np.float32).copy()
        tokenizer = Tokenizer.from_file(str(admit_read(bundle / "tokenizer.json")))
        return cls(codes, scales, tokenizer, config)

    @property
    def resident_table_bytes(self):
        return int(self.codes.nbytes + self.scales.nbytes)

    def _row(self, token_id):
        return dequant_row(self.codes, self.scales, token_id)

    def encode_ids(self, ids):
        if not ids:
            return _normalize(self._row(self.fallback_id))
        return _normalize(_gather_sum(ids, self._row))

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self.encode_ids(item.ids) for item in self.tokenizer.encode_batch(texts)])


def algebra_gates(base_codes, base_scales, built, teacher_vectors, registry=None):
    registry = registry or load_json(REGISTRY_PATH)
    gates = registry["numerical_gates"]
    reports = {}
    for variant in ("V0-compose", "T0-teacher"):
        codes, scales = compact_table(base_codes, base_scales, built[variant])
        cfg = load_json(RELEASE_BUNDLE / "config.json")
        encoder = M19QueryEncoder(codes, scales, built["tokenizer"], cfg)
        variant_reports = {}
        for receipt in built["receipts"]:
            term = receipt["term"]
            actual = encoder.encode(term)[0]
            if variant == "T0-teacher":
                reference = _normalize(teacher_vectors[term])
            else:
                original_ids = receipt["original_ids"]
                reference = _normalize(_gather_sum(
                    original_ids, lambda token_id: dequant_row(base_codes, base_scales, token_id)
                ))
            cosine = float(np.dot(actual, reference))
            max_abs = float(np.abs(actual - reference).max())
            variant_reports[term] = {"cosine": cosine, "max_abs": max_abs}
            if cosine < float(gates["int8_bare_cosine_minimum"]):
                raise SystemExit(f"M19 ROW REFUSED: {variant} {term} int8 cosine {cosine}")
            if max_abs > float(gates["int8_query_vector_max_abs"]):
                raise SystemExit(f"M19 ROW REFUSED: {variant} {term} int8 max abs {max_abs}")
        reports[variant] = {
            "terms": variant_reports,
            "resident_table_bytes": encoder.resident_table_bytes,
            "expected_resident_table_bytes": int(base_codes.nbytes + base_scales.nbytes
                                                   + len(built["receipts"]) * (built["dim"] + 4)),
            "no_full_float32_table": not hasattr(encoder, "rows"),
        }
        if reports[variant]["resident_table_bytes"] != reports[variant]["expected_resident_table_bytes"]:
            raise SystemExit(f"M19 ROW REFUSED: {variant} resident byte increment differs")
    return reports


def bundle_payload(variant, codes, scales, tokenizer, base_config, provenance):
    """Return immutable bundle components; the transaction publisher owns filesystem writes."""
    required = {
        "inheritance_identity", "roster_identity", "base_codes_sha256",
        "base_scales_sha256", "base_tokenizer_sha256", "pooling_identity_sha256",
        "selected_added_token_audit",
    }
    if required - set(provenance):
        raise ValueError(f"bundle provenance missing {sorted(required - set(provenance))}")
    config = {
        "_schema": "m19-internal-zero-bundle-v1",
        "internal_only": True,
        "variant": variant,
        "vocab": int(codes.shape[0]),
        "dim": int(codes.shape[1]),
        "fallback_token_id": int(base_config["fallback_token_id"]),
        "preproc": dict(base_config["preproc"]),
        "weights_folded": base_config["weights_folded"],
        "learned_weights": base_config["learned_weights"],
        "document_encoder": dict(base_config["document_encoder"]),
        "recommended_variant": "resident_int8",
    }
    model = {"rows_int8": np.asarray(codes, np.int8),
             "int8_scale": np.asarray(scales, np.float32)}
    tokenizer_bytes = tokenizer.to_str().encode()
    return {
        "config": config,
        "model": model,
        "tokenizer_bytes": tokenizer_bytes,
        "provenance": {
            "_schema": "m19-internal-zero-provenance-v1",
            **provenance,
            "variant": variant,
            "codes_sha256": sha_array(model["rows_int8"]),
            "scales_sha256": sha_array(model["int8_scale"]),
            "tokenizer_sha256": sha_bytes(tokenizer_bytes),
            "config_identity_sha256": sha_json(config),
        },
    }


def _npy_bytes(array):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz(arrays):
    """NPZ bytes with fixed metadata so interruption/resume can compare exact content."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return stream.getvalue()


def bundle_files(payload):
    model_bytes = _deterministic_npz(payload["model"])
    config_bytes = (json.dumps(payload["config"], indent=2, sort_keys=True) + "\n").encode()
    tokenizer_bytes = payload["tokenizer_bytes"]
    provenance = {
        **payload["provenance"],
        "file_sha256": {
            "model.npz": sha_bytes(model_bytes),
            "config.json": sha_bytes(config_bytes),
            "tokenizer.json": sha_bytes(tokenizer_bytes),
        },
    }
    provenance_bytes = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    files = {
        "model.npz": model_bytes,
        "config.json": config_bytes,
        "tokenizer.json": tokenizer_bytes,
        "provenance.json": provenance_bytes,
    }
    complete = {
        "_schema": "m19-bundle-complete-v1",
        "state": "complete",
        "variant": payload["config"]["variant"],
        "files": {name: sha_bytes(content) for name, content in sorted(files.items())},
    }
    files["complete.json"] = (json.dumps(
        {**complete, "identity_sha256": sha_json(complete)}, indent=2, sort_keys=True
    ) + "\n").encode()
    return files


def _publish_file(path, content):
    path = Path(path)
    expected = sha_bytes(content)
    if path.exists():
        if sha_file_unchecked(path) != expected:
            raise SystemExit(f"M19 BUNDLE REFUSED: existing file differs: {path}")
        return
    try:
        atomic_create_bytes(path, content)
    except FileExistsError:
        if sha_file_unchecked(path) != expected:
            raise SystemExit(f"M19 BUNDLE REFUSED: competing file differs: {path}")


def publish_bundle(out_dir, payload, *, verification):
    """Resumably publish immutable files; ``complete.json`` is always last."""
    out = Path(admit_write(out_dir))
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not out.is_dir():
            raise SystemExit(f"M19 BUNDLE REFUSED: output is not a directory: {out}")
    files = bundle_files(payload)
    for name in ("model.npz", "config.json", "tokenizer.json", "provenance.json"):
        _publish_file(out / name, files[name])
    _publish_file(out / "complete.json", files["complete.json"])
    return verify_bundle(out, verification=verification)


def _verify_context(provenance, verification):
    required = {
        "base_codes", "base_scales", "base_tokenizer_payload", "roster",
        "inheritance_identity", "pooling_identity_sha256",
    }
    if required - set(verification):
        raise ValueError(f"bundle verification missing {sorted(required - set(verification))}")
    base_codes = np.asarray(verification["base_codes"], dtype=np.int8)
    base_scales = np.asarray(verification["base_scales"], dtype=np.float32)
    roster = verification["roster"]
    roster_identity = roster.get("identity_sha256") or sha_json(roster)
    expected = {
        "inheritance_identity": verification["inheritance_identity"],
        "roster_identity": roster_identity,
        "base_codes_sha256": sha_array(base_codes),
        "base_scales_sha256": sha_array(base_scales),
        "base_tokenizer_sha256": sha_bytes(verification["base_tokenizer_payload"]),
        "pooling_identity_sha256": verification["pooling_identity_sha256"],
        "selected_added_token_audit": roster["selected_added_token_audit"],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise SystemExit("M19 BUNDLE REFUSED: provenance differs from verified inheritance/roster")
    return base_codes, base_scales, roster


def verify_bundle(bundle, *, verification):
    bundle = Path(bundle)
    complete_path = admit_read(bundle / "complete.json")
    if not complete_path.is_file():
        raise SystemExit("M19 BUNDLE REFUSED: bundle is incomplete")
    complete = json.loads(complete_path.read_text())
    body = {key: value for key, value in complete.items() if key != "identity_sha256"}
    if complete.get("identity_sha256") != sha_json(body) or complete.get("state") != "complete":
        raise SystemExit("M19 BUNDLE REFUSED: invalid completion identity")
    for name, expected in complete["files"].items():
        if sha_file(bundle / name) != expected:
            raise SystemExit(f"M19 BUNDLE REFUSED: {name} differs from completion manifest")
    config = load_json(bundle / "config.json")
    provenance = load_json(bundle / "provenance.json")
    base_codes, base_scales, roster = _verify_context(provenance, verification)
    pooling = {
        "fallback_token_id": config["fallback_token_id"],
        "learned_weights": config["learned_weights"],
        "preproc": config["preproc"],
        "weights_folded": config["weights_folded"],
    }
    if sha_json(pooling) != verification["pooling_identity_sha256"]:
        raise SystemExit("M19 BUNDLE REFUSED: inherited pooling identity changed")
    with np.load(admit_read(bundle / "model.npz")) as archive:
        codes = np.asarray(archive["rows_int8"], dtype=np.int8)
        scales = np.asarray(archive["int8_scale"], dtype=np.float32)
        if set(archive.files) != {"rows_int8", "int8_scale"}:
            raise SystemExit("M19 BUNDLE REFUSED: unexpected/eager table array")
        if codes.shape != (config["vocab"], config["dim"]) or scales.shape != (config["vocab"],):
            raise SystemExit("M19 BUNDLE REFUSED: table/config shape mismatch")
        if not np.array_equal(codes[:len(base_codes)], base_codes):
            raise SystemExit("M19 BUNDLE REFUSED: inherited row codes changed")
        if not np.array_equal(scales[:len(base_scales)], base_scales):
            raise SystemExit("M19 BUNDLE REFUSED: inherited row scales changed")
    tokenizer = Tokenizer.from_file(str(admit_read(bundle / "tokenizer.json")))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config["vocab"]:
        raise SystemExit("M19 BUNDLE REFUSED: tokenizer/table size mismatch")
    base = Tokenizer.from_str(verification["base_tokenizer_payload"].decode("utf-8"))
    base_vocab = base.get_vocab(with_added_tokens=True)
    extended_vocab = tokenizer.get_vocab(with_added_tokens=True)
    if any(extended_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise SystemExit("M19 BUNDLE REFUSED: inherited tokenizer IDs changed")
    audit = roster["selected_added_token_audit"]
    if ({term: extended_vocab.get(term) for term in audit["term_ids"]} != audit["term_ids"] or
            len(extended_vocab) != audit["final_vocab"] or
            len(base_vocab) != audit["base_vocab"]):
        raise SystemExit("M19 BUNDLE REFUSED: selected AddedToken audit changed")
    return {
        "identity_sha256": complete["identity_sha256"],
        "variant": complete["variant"],
        "files": complete["files"],
        "resident_table_bytes": int(codes.nbytes + scales.nbytes),
    }
