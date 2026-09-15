"""Synthetic watchdog tests; never use real experiment artifacts or processes."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("m13_monitor", Path(__file__).parents[1] / "scripts/m13_monitor.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = {"name": "train", "receipt": "train.json", "logs": ["train.log"], "stale_seconds": 100}
        self.config = {"root": str(self.root), "jobs": [self.job]}

    def write(self, name, value, timestamp=1000):
        path = self.root / name
        path.write_text(json.dumps(value))
        os.utime(path, (timestamp, timestamp))
        return path

    def poll(self, now=1050):
        with contextlib.redirect_stdout(io.StringIO()):
            return monitor.poll(self.config, now)["jobs"]

    def test_missing_is_pending_and_transitions_deduplicate(self):
        self.assertEqual(self.poll()[0]["state"], "pending")
        self.poll(1060)
        self.assertEqual(len((self.root / "work/m13-monitor/alerts.jsonl").read_text().splitlines()), 1)

    def test_running_stale_and_log_progress_recovery(self):
        self.write("train.json", {"status": "RUNNING"})
        self.assertEqual(self.poll()[0]["state"], "running")
        self.assertEqual(self.poll(1200)[0]["state"], "stale")
        self.write("train.log", "step 20", 1190)
        self.assertEqual(self.poll(1200)[0]["state"], "running")

    def test_failed_requires_exact_recovery_binding(self):
        receipt = self.write("train.json", {"status": "FAILED"})
        self.job["recovery_receipt"] = "recovery.json"
        self.write("recovery.json", {"status": "PASSED", "artifact_sha256": {"train.json": "wrong"}})
        self.assertEqual(self.poll()[0]["state"], "failed")
        self.write("recovery.json", {"status": "PASSED", "artifact_sha256": {
            "train.json": hashlib.sha256(receipt.read_bytes()).hexdigest()}})
        self.assertEqual(self.poll()[0]["state"], "recovered")

    def test_missing_dead_and_reused_controller_identity(self):
        self.write("train.json", {"status": "RUNNING"})
        self.job["pid_file"] = "pid"
        self.assertEqual(self.poll()[0]["state"], "dead")
        (self.root / "pid").write_text("1234")
        with patch.object(monitor, "process_identity", return_value="boot:1234:1"):
            self.assertEqual(self.poll()[0]["state"], "running")
        with patch.object(monitor, "process_identity", side_effect=ProcessLookupError):
            self.assertEqual(self.poll()[0]["state"], "dead")
        with patch.object(monitor, "process_identity", return_value="boot:1234:2"):
            result = self.poll()[0]
            self.assertEqual(result["state"], "dead")
            self.assertIn("identity changed", result["detail"])

    def test_waiting_dependency_and_handoff_grace(self):
        self.write("train.json", {"status": "RUNNING"}, 2000)
        self.write("follow.json", {"status": "RUNNING", "stage": "waiting"}, 1000)
        self.config["jobs"].append({"name": "follow", "receipt": "follow.json", "wait_for": "train", "stale_seconds": 100})
        self.assertEqual(self.poll(2050)[1]["state"], "waiting")
        self.write("train.json", {"status": "PASSED"}, 2050)
        self.assertEqual(self.poll(2100)[1]["state"], "running")
        self.assertEqual(self.poll(2200)[1]["state"], "stale")

    def test_log_failure_and_passed_terminal_precedence(self):
        self.write("train.json", {"status": "RUNNING"})
        (self.root / "train.log").write_text("Traceback (most recent call last):\nRuntimeError: failed\n")
        self.assertEqual(self.poll()[0]["state"], "error")
        self.write("train.json", {"status": "PASSED"})
        self.assertEqual(self.poll(5000)[0]["state"], "passed")

    def test_benign_error_note_is_not_failure(self):
        self.write("train.json", {"status": "RUNNING"})
        self.write("train.log", "NOTE E-bs128: batch 128 above ~128 tokens: driver error on this card; runs on the A100")
        self.assertEqual(self.poll()[0]["state"], "running")

    def test_permission_error_becomes_health_error(self):
        with patch.object(Path, "read_bytes", side_effect=PermissionError):
            self.assertEqual(monitor.check_job(self.job, self.root, {}, 1050)["state"], "error")

    def test_notification_command_requires_argv(self):
        self.config["notification_command"] = "echo unsafe"
        with self.assertRaisesRegex(ValueError, "argv list"):
            self.poll()

    def test_bad_receipt_and_notification_failure_remain_visible(self):
        (self.root / "train.json").write_text("{")
        self.config["notification_command"] = ["does-not-exist"]
        with patch.object(monitor.subprocess, "run", side_effect=FileNotFoundError("missing")) as run:
            result = self.poll()[0]
            self.assertEqual(result["state"], "error")
            self.assertIn("FileNotFoundError", result["notification_error"])
            self.assertEqual(run.call_args.kwargs["cwd"], self.root)
            self.assertIn("notification_error", self.poll()[0])
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
