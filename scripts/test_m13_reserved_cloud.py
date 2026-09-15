import hashlib
from pathlib import Path
import subprocess

import m13_reserved_cloud as M


def test_storage_accounting_includes_volume_and_container_disks():
    pod = {"volumeInGb": 500, "containerDiskInGb": 30}
    assert M.storage_hourly(pod) == 530 * .1 / 720


def test_expected_vector_space_covers_all_system_dimensions():
    counts = {dataset: {"n_docs": value} for dataset, value in zip(
        M.RESERVED_DATASETS, (5_416_568, 4_635_922, 22_998, 40_221))}
    assert M.expected_vector_bytes({"m7_untouched_final": counts}) == 44_023_565_568


def test_remote_pipeline_is_valid_shell_and_keeps_protected_scoring_last():
    assert M.BRANCH == "main"
    command = M.remote_command()
    subprocess.run(["bash", "-n", "-c", command], check=True)
    assert command.index("pre_encode.py --preflight-only") < command.index(
        "reserved_support --preflight-models")
    assert command.index("reserved_support --preflight-models") < command.index(
        "score13.py --reserved-only")
    assert "HF_DATASETS_OFFLINE=1 .venv/bin/python -u m13src/score13.py --reserved-only" in command


def test_checkpoint_prep_copies_once_and_refuses_changed_target(tmp_path):
    source_rel = Path("work/build/cycle3.pt")
    target_rel = Path("work/final/cycle3.pt")
    source = tmp_path / source_rel
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen checkpoint")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    command = M.checkpoint_prep_command(target_rel, source_rel, digest, remote=tmp_path)

    subprocess.run(["bash", "-n", "-c", command], check=True)
    subprocess.run(["bash", "-c", command], check=True)
    target = tmp_path / target_rel
    assert target.read_bytes() == source.read_bytes()
    assert not target.with_name(target.name + ".pending").exists()

    target.write_bytes(b"changed")
    failed = subprocess.run(["bash", "-c", command])
    assert failed.returncode != 0
    assert target.read_bytes() == b"changed"
