from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m18src"))
import rehearse18


def test_synthetic_end_to_end(tmp_path):
    result = rehearse18.build(tmp_path / "world", device="cpu", log=lambda *_: None)
    assert result["resume_vs_uninterrupted_max_abs"] == 0
    assert result["inherited_rows_unchanged"]
    assert result["new_rows_changed"]
    assert result["confirmation_second_claim_refused"]

