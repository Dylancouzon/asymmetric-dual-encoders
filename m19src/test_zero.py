import numpy as np
from pathlib import Path
import pytest
import json
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from m19src import zero
from m19src.common import sha_json


def _fixture():
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "k": 3, "##8": 4, "##s": 5,
             "other": 6}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    payload = tokenizer.to_str().encode()
    rows = np.array([
        [0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.0, -0.1], [-0.1, 0.1, 0.2, 0.1],
        [0.5, -0.2, 0.1, 0.0], [0.0, 0.3, -0.1, 0.2], [0.1, 0.0, 0.4, -0.2],
        [0.3, 0.2, 0.1, 0.0],
    ], dtype=np.float32)
    codes, scales = zero.quantize_rows(rows)
    base_vocab = tokenizer.get_vocab(with_added_tokens=True)
    roster = {
        "terms": [{"term": "k8s"}],
        "selected_added_token_audit": {
            "term_ids": {"k8s": 7}, "inherited_vocab_sha256": sha_json(base_vocab),
            "base_vocab": 7, "final_vocab": 8,
        },
    }
    teacher = {"k8s": np.array([0.2, 0.7, -0.1, 0.4], dtype=np.float32)}
    return payload, codes, scales, roster, teacher


def _config():
    return {
        "preproc": {"add_special_tokens": True, "max_length": 512,
                    "pool_mode": "sqrt", "prefix": ""},
        "fallback_token_id": 1,
        "weights_folded": True,
        "learned_weights": False,
        "document_encoder": {"dim": 4},
    }


def _verification(payload, codes, scales, roster):
    pooling = {
        "fallback_token_id": _config()["fallback_token_id"],
        "learned_weights": _config()["learned_weights"],
        "preproc": _config()["preproc"],
        "weights_folded": _config()["weights_folded"],
    }
    roster = {**roster, "identity_sha256": "9" * 64}
    return {
        "base_codes": codes, "base_scales": scales, "base_tokenizer_payload": payload,
        "roster": roster, "inheritance_identity": "8" * 64,
        "pooling_identity_sha256": sha_json(pooling),
    }


def _provenance(verification):
    return {
        "inheritance_identity": verification["inheritance_identity"],
        "roster_identity": verification["roster"]["identity_sha256"],
        "base_codes_sha256": zero.sha_array(verification["base_codes"]),
        "base_scales_sha256": zero.sha_array(verification["base_scales"]),
        "base_tokenizer_sha256": zero.sha_bytes(verification["base_tokenizer_payload"]),
        "pooling_identity_sha256": verification["pooling_identity_sha256"],
        "selected_added_token_audit": verification["roster"]["selected_added_token_audit"],
    }


def test_teacher_row_hits_direction_and_compose_preserves_bare_vector():
    payload, codes, scales, roster, teacher = _fixture()
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    t0_codes, t0_scales = zero.compact_table(codes, scales, built["T0-teacher"])
    v0_codes, v0_scales = zero.compact_table(codes, scales, built["V0-compose"])
    t0 = zero.M19QueryEncoder(t0_codes, t0_scales, built["tokenizer"], _config())
    v0 = zero.M19QueryEncoder(v0_codes, v0_scales, built["tokenizer"], _config())
    base = Tokenizer.from_str(payload.decode())
    released = zero.M19QueryEncoder(codes, scales, base, _config())
    assert float(np.dot(t0.encode("k8s")[0], zero._normalize(teacher["k8s"]))) >= 0.999
    assert np.max(np.abs(v0.encode("k8s") - released.encode("k8s"))) <= 0.02
    assert built["receipts"][0]["prequant_bare_cosine"] >= 0.999999


def test_inherited_codes_scales_and_unmatched_query_are_exact():
    payload, codes, scales, roster, teacher = _fixture()
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    new_codes, new_scales = zero.compact_table(codes, scales, built["T0-teacher"])
    assert np.array_equal(new_codes[:len(codes)], codes)
    assert np.array_equal(new_scales[:len(scales)], scales)
    base = zero.M19QueryEncoder(codes, scales, Tokenizer.from_str(payload.decode()), _config())
    extended = zero.M19QueryEncoder(new_codes, new_scales, built["tokenizer"], _config())
    assert np.array_equal(base.encode("other"), extended.encode("other"))
    assert not hasattr(extended, "rows")
    assert extended.resident_table_bytes == base.resident_table_bytes + 8


def test_algebra_gates_measure_expected_compact_bytes(monkeypatch):
    payload, codes, scales, roster, teacher = _fixture()
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    monkeypatch.setattr(zero, "load_json", lambda path: _config())
    registry = {"numerical_gates": {
        "int8_bare_cosine_minimum": 0.99, "int8_query_vector_max_abs": 0.05,
    }}
    report = zero.algebra_gates(codes, scales, built, teacher, registry=registry)
    assert report["T0-teacher"]["no_full_float32_table"]
    assert report["T0-teacher"]["resident_table_bytes"] == codes.nbytes + scales.nbytes + 8


def test_bundle_payload_keeps_pooling_values_and_only_compact_arrays():
    payload, codes, scales, roster, teacher = _fixture()
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    new_codes, new_scales = zero.compact_table(codes, scales, built["T0-teacher"])
    verification = _verification(payload, codes, scales, roster)
    bundle = zero.bundle_payload("T0-teacher", new_codes, new_scales, built["tokenizer"],
                                 _config(), _provenance(verification))
    assert set(bundle["model"]) == {"rows_int8", "int8_scale"}
    assert bundle["config"]["preproc"] == _config()["preproc"]
    assert bundle["config"]["fallback_token_id"] == 1


def test_deterministic_bundle_bytes_and_resumable_publication(tmp_path, monkeypatch):
    payload, codes, scales, roster, teacher = _fixture()
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    new_codes, new_scales = zero.compact_table(codes, scales, built["T0-teacher"])
    verification = _verification(payload, codes, scales, roster)
    bundle = zero.bundle_payload("T0-teacher", new_codes, new_scales, built["tokenizer"],
                                 _config(), _provenance(verification))
    assert zero.bundle_files(bundle) == zero.bundle_files(bundle)
    monkeypatch.setattr(zero, "admit_write", lambda path: Path(path))
    monkeypatch.setattr(zero, "admit_read", lambda path: Path(path))
    monkeypatch.setattr(zero, "load_json", lambda path: json.loads(Path(path).read_text()))

    def create(path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as handle:
            handle.write(content)

    monkeypatch.setattr(zero, "atomic_create_bytes", create)
    monkeypatch.setattr(zero, "sha_file", lambda path: zero.sha_file_unchecked(path))
    out = tmp_path / "bundle"
    first = zero.publish_bundle(out, bundle, verification=verification)
    second = zero.publish_bundle(out, bundle, verification=verification)
    assert first == second
    assert first["resident_table_bytes"] == new_codes.nbytes + new_scales.nbytes


def test_incomplete_bundle_is_not_readable(tmp_path, monkeypatch):
    out = tmp_path / "bundle"
    out.mkdir()
    monkeypatch.setattr(zero, "admit_read", lambda path: Path(path))
    with pytest.raises(SystemExit, match="incomplete"):
        zero.verify_bundle(out, verification={})


def test_loader_refuses_incomplete_or_inheritance_wrong_bundle(tmp_path, monkeypatch):
    payload, codes, scales, roster, teacher = _fixture()
    verification = _verification(payload, codes, scales, roster)
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    new_codes, new_scales = zero.compact_table(codes, scales, built["T0-teacher"])
    bundle = zero.bundle_payload("T0-teacher", new_codes, new_scales, built["tokenizer"],
                                 _config(), _provenance(verification))
    monkeypatch.setattr(zero, "admit_write", lambda path: Path(path))
    monkeypatch.setattr(zero, "admit_read", lambda path: Path(path))
    monkeypatch.setattr(zero, "load_json", lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(zero, "atomic_create_bytes", lambda path, content: (
        Path(path).parent.mkdir(parents=True, exist_ok=True), Path(path).write_bytes(content)
    ))
    monkeypatch.setattr(zero, "sha_file", lambda path: zero.sha_file_unchecked(path))
    out = tmp_path / "bundle"
    zero.publish_bundle(out, bundle, verification=verification)
    loaded = zero.M19QueryEncoder.from_bundle(out, verification=verification)
    assert loaded.codes.shape == new_codes.shape
    wrong = {**verification, "inheritance_identity": "7" * 64}
    with pytest.raises(SystemExit, match="provenance differs"):
        zero.M19QueryEncoder.from_bundle(out, verification=wrong)
    (out / "complete.json").unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        zero.M19QueryEncoder.from_bundle(out, verification=verification)


def test_loader_refuses_rehashed_added_row_and_wrong_expected_identity(tmp_path, monkeypatch):
    payload, codes, scales, roster, teacher = _fixture()
    verification = _verification(payload, codes, scales, roster)
    built = zero.construct_added_rows(codes, scales, payload, roster, teacher)
    new_codes, new_scales = zero.compact_table(codes, scales, built["T0-teacher"])
    bundle = zero.bundle_payload("T0-teacher", new_codes, new_scales, built["tokenizer"],
                                 _config(), _provenance(verification))
    monkeypatch.setattr(zero, "admit_write", lambda path: Path(path))
    monkeypatch.setattr(zero, "admit_read", lambda path: Path(path))
    monkeypatch.setattr(zero, "load_json", lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(zero, "atomic_create_bytes", lambda path, content: (
        Path(path).parent.mkdir(parents=True, exist_ok=True), Path(path).write_bytes(content)
    ))
    monkeypatch.setattr(zero, "sha_file", lambda path: zero.sha_file_unchecked(path))
    out = tmp_path / "bundle"
    report = zero.publish_bundle(out, bundle, verification=verification)
    with pytest.raises(SystemExit, match="expected build"):
        zero.verify_bundle(out, verification=verification, expected_identity="7" * 64)

    altered = new_codes.copy()
    altered[-1] = -altered[-1]
    model_bytes = zero._deterministic_npz({"rows_int8": altered, "int8_scale": new_scales})
    (out / "model.npz").write_bytes(model_bytes)
    complete = json.loads((out / "complete.json").read_text())
    complete["files"]["model.npz"] = zero.sha_bytes(model_bytes)
    body = {key: value for key, value in complete.items() if key != "identity_sha256"}
    complete["identity_sha256"] = zero.sha_json(body)
    (out / "complete.json").write_text(json.dumps(complete, sort_keys=True))
    with pytest.raises(SystemExit, match="model arrays differ from provenance"):
        zero.verify_bundle(out, verification=verification,
                           expected_identity=complete["identity_sha256"])
