from m19src import rehearse


def test_one_command_rehearsal_resumes_and_is_stable(tmp_path):
    root = tmp_path / "work" / "m19" / "rehearsal"
    first = rehearse.run_rehearsal(root)
    second = rehearse.run_rehearsal(root)
    assert first == second
    assert first["state"] == "complete"
    assert first["outcome"] == "pass"
    assert first["interruption_boundary"] == "judgments-in-progress"
    assert first["audit_fraction"] >= 0.20
    assert first["audit_agreement"] >= 0.90
    assert first["metric_exposure_after_qrels_freeze"] is True
