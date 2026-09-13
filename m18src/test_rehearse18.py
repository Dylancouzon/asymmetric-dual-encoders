from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m18src"))
import rehearse18
import export


def test_synthetic_end_to_end(tmp_path):
    result = rehearse18.build(tmp_path / "world", device="cpu", log=lambda *_: None)
    assert result["resume_vs_uninterrupted_max_abs"] == 0
    assert result["inherited_rows_unchanged"]
    assert result["new_rows_changed"]
    assert result["confirmation_second_claim_refused"]


def test_bundle_identity_uses_the_serialized_no_padding_tokenizer(tmp_path):
    tok = rehearse18.build_tokenizer()
    tok.enable_padding(length=16)
    rows = rehearse18.unit(np.random.default_rng(7).normal(
        size=(tok.get_vocab_size(with_added_tokens=True), rehearse18.DIM)))
    bundle = export.build_bundle(tmp_path / "bundle", rows, tok, {"fixture": True}, fixture=True)
    assert (bundle / "gates.json").exists()
