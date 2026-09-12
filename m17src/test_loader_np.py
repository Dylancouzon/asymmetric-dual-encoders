"""Resident-int8 vs eager-fp32: exact parity, and the query-rule fixtures that bind it.

Parity here is expected to be EXACT (the registry allows 1e-6), because both modes do float32
arithmetic in the same `np.unique` order over the same gathered rows. The fixtures exist so a
"cleanup" that drops special tokens, changes the empty fallback or loses truncation fails.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import loader_np


@pytest.fixture
def bundle(rehearsal):
    return rehearsal["root"] / "bundle-endpoint"


def test_parity_is_exact_and_within_the_registered_bound(bundle):
    rep = loader_np.parity(bundle)
    assert rep["pass"] and rep["loader_parity_max_abs"] == 0.0


def test_resident_holds_codes_and_scales_not_a_float_table(bundle):
    eager = loader_np.M17QueryEncoder(bundle, variant="int8", mode="eager_fp32")
    resident = loader_np.M17QueryEncoder(bundle, variant="int8", mode="resident_int8")
    assert eager.rows is not None and resident.rows is None
    assert resident.weight_bytes < eager.weight_bytes
    # weight bytes are an ARRAY measurement; RSS is a separate, larger process measurement
    assert loader_np.rss_kb() is None or loader_np.rss_kb() * 1024 > eager.weight_bytes


def test_fp16_variant_cannot_claim_the_resident_path(bundle):
    with pytest.raises(ValueError, match="resident_int8 needs the int8"):
        loader_np.M17QueryEncoder(bundle, variant="fp16", mode="resident_int8")


def test_unknown_mode_is_refused(bundle):
    with pytest.raises(ValueError, match="mode must be one of"):
        loader_np.M17QueryEncoder(bundle, mode="lazy_maybe")


@pytest.mark.parametrize("mode", loader_np.MODES)
def test_query_rule_fixtures(bundle, mode):
    enc = loader_np.M17QueryEncoder(bundle, variant="int8", mode=mode)
    v = enc.encode(loader_np.DEFAULT_FIXTURES)
    assert v.dtype == np.float32 and v.shape[0] == len(loader_np.DEFAULT_FIXTURES)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    # An empty STRING still tokenizes to [CLS] [SEP] — those are ordinary rows, exactly as in
    # the v1 encoder — so the fallback is reserved for an empty id list or a degenerate sum.
    assert np.allclose(enc.encode("")[0], enc.encode([""])[0])
    assert np.allclose(enc._encode_ids([]), enc._fallback, atol=1e-6)


def test_repeated_tokens_are_sqrt_saturated(bundle):
    """sqrt(3) weight on a thrice-repeated token, not 3x and not 1x."""
    enc = loader_np.M17QueryEncoder(bundle, variant="int8", mode="resident_int8")
    ids = enc.tokenizer.encode("storage storage storage").ids
    uniq, counts = np.unique(np.asarray(ids), return_counts=True)
    assert counts.max() == 3, "the fixture must actually repeat a token"
    w = np.sqrt(counts, dtype=np.float32)
    want = (enc._gather(uniq) * w[:, None]).sum(0) / float(w.sum())
    want = want / np.linalg.norm(want)
    assert np.allclose(enc.encode("storage storage storage")[0], want, atol=1e-6)
    linear = (enc._gather(uniq) * counts[:, None]).sum(0)
    assert not np.allclose(want, linear / np.linalg.norm(linear), atol=1e-6)


def test_truncation_is_enabled_at_the_frozen_length(bundle):
    enc = loader_np.M17QueryEncoder(bundle, variant="int8", mode="resident_int8")
    cfg = json.loads((bundle / "config.json").read_text())
    long = " ".join(["storage bucket policy"] * 500)
    assert len(enc.tokenizer.encode(long).ids) <= int(cfg["preproc"]["max_length"])


def test_special_tokens_stay_ordinary_rows(bundle):
    enc = loader_np.M17QueryEncoder(bundle, variant="int8", mode="resident_int8")
    ids = enc.tokenizer.encode("storage").ids
    assert ids[0] == enc.fallback_id, "CLS must be present in the bag"
    assert len(ids) >= 3, "CLS and SEP must not be stripped"


def test_a_vocab_row_mismatch_is_refused(bundle, tmp_path):
    import shutil
    d = tmp_path / "b"
    shutil.copytree(bundle, d)
    z = dict(np.load(d / "model.npz"))
    z["rows_int8"] = z["rows_int8"][:-1]
    z["int8_scale"] = z["int8_scale"][:-1]
    z["rows_fp16"] = z["rows_fp16"][:-1]
    np.savez(d / "model.npz", **z)
    with pytest.raises(ValueError, match="index off the end"):
        loader_np.M17QueryEncoder(d, variant="int8", mode="resident_int8")
