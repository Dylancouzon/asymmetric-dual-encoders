"""Build the frozen M19 V0-compose and T0-teacher compact bundles locally."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from m19src import inherit, zero
from m19src.common import (M19, RESULTS, WORK, admit_read, atomic_create_bytes, load_json,
                           sha_array, sha_bytes, sha_file_unchecked, sha_json)

TEACHER_ARRAY = WORK / "candidate" / "teacher-vectors.npy"
TEACHER_RECEIPT = RESULTS / "m19_teacher_receipt.json"
ROW_RECEIPT = RESULTS / "m19_row_receipt_t0.json"
ALGEBRA_RECEIPT = RESULTS / "m19_algebra_gates.json"
BUILD_RECEIPT = RESULTS / "m19_candidate_build.json"
BUNDLE_ROOT = WORK / "bundles"
TEACHER_SNAPSHOT_LOCK = M19 / "teacher-snapshot-lock-v1.json"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _publish(path, payload):
    try:
        atomic_create_bytes(path, payload)
        return "created"
    except FileExistsError:
        if sha_file_unchecked(path) != sha_bytes(payload):
            raise SystemExit(f"M19 CANDIDATE REFUSED: existing output differs: {path}")
        return "verified"


def _npy_bytes(array):
    import io
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def verify_teacher_snapshot(teacher_path):
    teacher_path = Path(teacher_path).resolve()
    lock = load_json(TEACHER_SNAPSHOT_LOCK)
    if (lock.get("_schema") != "m19-teacher-snapshot-lock-v1" or
            Path(lock.get("snapshot_path", "")).resolve() != teacher_path or
            lock.get("revision") != teacher_path.name):
        raise SystemExit("M19 CANDIDATE REFUSED: teacher snapshot lock differs")
    files = {str(path.relative_to(teacher_path)): path for path in teacher_path.rglob("*")
             if path.is_file()}
    if set(files) != set(lock.get("files", {})):
        raise SystemExit("M19 CANDIDATE REFUSED: teacher snapshot file set differs")
    for name, expected in lock["files"].items():
        if sha_file_unchecked(files[name]) != expected:
            raise SystemExit(f"M19 CANDIDATE REFUSED: teacher snapshot file differs: {name}")
    return lock


def _encode_teachers(teacher_path, terms, prefix, config_kwargs, device):
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    teacher_path = Path(teacher_path).resolve()
    verify_teacher_snapshot(teacher_path)
    revision = teacher_path.name
    if not teacher_path.is_dir():
        raise SystemExit(f"M19 CANDIDATE REFUSED: teacher snapshot is missing: {teacher_path}")
    torch.manual_seed(19019)
    model = SentenceTransformer(
        str(teacher_path), trust_remote_code=True, local_files_only=True, device=device,
        config_kwargs=dict(config_kwargs),
    )
    model.eval()
    values = model.encode(
        [prefix + term for term in terms], batch_size=len(terms), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    verify_teacher_snapshot(teacher_path)
    norms = np.linalg.norm(values, axis=1)
    if (values.shape != (len(terms), 1024) or not np.isfinite(values).all() or
            float(np.max(np.abs(norms - 1))) > 1e-6):
        raise SystemExit("M19 CANDIDATE REFUSED: malformed teacher vectors")
    runtime = {
        "snapshot_path": str(teacher_path),
        "revision": revision,
        "config_kwargs": dict(config_kwargs),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "snapshot_files_sha256": {
            name: sha_file_unchecked(teacher_path / name)
            for name in ("config.json", "model.safetensors", "2_Dense_1024/model.safetensors")
        },
    }
    return values, runtime


def _boundary_and_parity(base_codes, base_scales, base_tokenizer_payload, built, base_config):
    base_tokenizer = Tokenizer.from_str(base_tokenizer_payload.decode())
    base_encoder = zero.M19QueryEncoder(base_codes, base_scales, base_tokenizer, base_config)
    t0_codes, t0_scales = zero.compact_table(
        base_codes, base_scales, built["T0-teacher"])
    t0_encoder = zero.M19QueryEncoder(
        t0_codes, t0_scales, Tokenizer.from_str(built["tokenizer"].to_str()), base_config)
    term_ids = built["tokenizer_audit"]["term_ids"]
    boundary = {}
    for term, token_id in term_ids.items():
        fixtures = {
            "bare": (term, 1), "case": (term.upper(), 1),
            "punctuation": (f"({term})", 1), "possessive": (term + "'s", 1),
            "plural": (term + "s", 0), "prefix_substring": ("pre" + term, 0),
            "suffix_substring": (term + "post", 0), "underscore": (f"x_{term}_y", 0),
        }
        observed = {
            name: built["tokenizer"].encode(text, add_special_tokens=False).ids.count(token_id)
            for name, (text, _) in fixtures.items()
        }
        expected = {name: count for name, (_, count) in fixtures.items()}
        if observed != expected:
            raise SystemExit(f"M19 CANDIDATE REFUSED: boundary behavior differs for {term}")
        boundary[term] = observed
    no_match = ["vector search", "collection aliases", "payload filters", "snapshot recovery"]
    tokenization_equal = all(
        base_tokenizer.encode(text).ids == built["tokenizer"].encode(text).ids for text in no_match)
    max_abs = float(np.max(np.abs(base_encoder.encode(no_match) - t0_encoder.encode(no_match))))
    if not tokenization_equal or max_abs > 1e-6:
        raise SystemExit("M19 CANDIDATE REFUSED: no-match tokenizer/encoder parity failed")
    return {
        "_schema": "m19-tokenizer-parity-v1",
        "boundary_match_counts": boundary,
        "no_match_fixture_sha256": sha_json(no_match),
        "no_match_tokenization_equal": tokenization_equal,
        "no_match_encoder_max_abs": max_abs,
    }


def build(teacher_path, device="cuda"):
    registry = load_json(M19 / "registry.json")
    inheritance = inherit.verify()
    roster = load_json(M19 / "term-roster-lock-v3.json")
    base_config = load_json(Path(inheritance["inherited_data"]["zero_v1_config"]["path"]))
    base_tokenizer_payload = admit_read(
        inheritance["inherited_data"]["zero_v1_tokenizer"]["path"]).read_bytes()
    base_codes, base_scales = zero.load_base(
        Path(inheritance["inherited_data"]["zero_v1_model"]["path"]))
    table = inheritance["identities"]["released_effective_table"]["table"]
    if (sha_array(base_codes) != table["codes_sha256"] or
            sha_array(base_scales) != table["scales_sha256"]):
        raise SystemExit("M19 CANDIDATE REFUSED: released compact rows differ from inheritance")

    terms = [row["term"] for row in roster["terms"]]
    document_encoder = registry["inheritance"]["document_encoder"]
    expected_revision = document_encoder.rsplit("@", 1)[1]
    if Path(teacher_path).resolve().name != expected_revision:
        raise SystemExit("M19 CANDIDATE REFUSED: teacher snapshot revision differs")
    prefix = registry["candidate"]["query_prefix"]
    if base_config["document_encoder"]["query_prefix"] != prefix:
        raise SystemExit("M19 CANDIDATE REFUSED: released and registered query prefixes differ")
    teacher_array, runtime = _encode_teachers(
        teacher_path, terms, prefix, base_config["document_encoder"]["config_kwargs"], device)
    teacher_vectors = {term: teacher_array[index] for index, term in enumerate(terms)}
    teacher_receipt = {
        "_schema": "m19-teacher-vectors-v1",
        "model": document_encoder,
        "query_prefix": prefix,
        "terms": terms,
        "dtype": str(teacher_array.dtype),
        "shape": list(teacher_array.shape),
        "vectors_sha256": sha_array(teacher_array),
        "runtime": runtime,
    }
    teacher_payload = _npy_bytes(teacher_array)
    _publish(TEACHER_ARRAY, teacher_payload)
    _publish(TEACHER_RECEIPT, _json_bytes(teacher_receipt))

    built = zero.construct_added_rows(
        base_codes, base_scales, base_tokenizer_payload, roster, teacher_vectors)
    algebra = zero.algebra_gates(
        base_codes, base_scales, built, teacher_vectors, registry=registry,
        base_config=base_config)
    parity = _boundary_and_parity(
        base_codes, base_scales, base_tokenizer_payload, built, base_config)
    algebra_receipt = {
        "_schema": "m19-algebra-gates-v1",
        "construction": built["receipts"],
        "gates": algebra,
        "tokenizer_parity": parity,
    }
    _publish(ALGEBRA_RECEIPT, _json_bytes(algebra_receipt))

    verification = {
        "base_codes": base_codes, "base_scales": base_scales,
        "base_tokenizer_payload": base_tokenizer_payload, "roster": roster,
        "inheritance_identity": inheritance["identity_sha256"],
        "pooling_identity_sha256": inheritance["identities"]["released_effective_table"][
            "pooling_sha256"],
    }
    provenance = {
        "inheritance_identity": inheritance["identity_sha256"],
        "roster_identity": roster["identity_sha256"],
        "base_codes_sha256": sha_array(base_codes),
        "base_scales_sha256": sha_array(base_scales),
        "base_tokenizer_sha256": sha_bytes(base_tokenizer_payload),
        "pooling_identity_sha256": verification["pooling_identity_sha256"],
        "selected_added_token_audit": roster["selected_added_token_audit"],
    }
    bundles = {}
    for variant, dirname in (("V0-compose", "v0-compose"), ("T0-teacher", "t0-teacher")):
        codes, scales = zero.compact_table(base_codes, base_scales, built[variant])
        payload = zero.bundle_payload(
            variant, codes, scales, built["tokenizer"], base_config, provenance)
        bundles[variant] = zero.publish_bundle(
            BUNDLE_ROOT / dirname, payload, verification=verification)

    t0_codes, t0_scales = zero.compact_table(base_codes, base_scales, built["T0-teacher"])
    row_receipt = {
        "_schema": "m19-row-receipt-v1", "variant": "T0-teacher",
        "codes_sha256": sha_array(t0_codes), "scales_sha256": sha_array(t0_scales),
        "tokenizer_sha256": sha_bytes(built["tokenizer"].to_str().encode()),
        "algebra_receipts_sha256": sha_json(built["receipts"]),
        "algebra_gates_sha256": sha_json(algebra),
    }
    _publish(ROW_RECEIPT, _json_bytes(row_receipt))
    result = {
        "_schema": "m19-candidate-build-v1", "state": "complete",
        "inheritance_identity": inheritance["identity_sha256"],
        "roster_identity": roster["identity_sha256"],
        "teacher_vectors_file_sha256": sha_bytes(teacher_payload),
        "teacher_vectors_identity_sha256": sha_array(teacher_array),
        "teacher_receipt_sha256": sha_bytes(_json_bytes(teacher_receipt)),
        "row_receipt_sha256": sha_bytes(_json_bytes(row_receipt)),
        "algebra_receipt_sha256": sha_bytes(_json_bytes(algebra_receipt)),
        "bundles": bundles,
    }
    _publish(BUILD_RECEIPT, _json_bytes(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = build(args.teacher_path, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
