#!/usr/bin/env python3
"""Offline policy tests for the M13 cloud-to-cloud migration supervisor."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location('m13_migrate_inputs', Path(__file__).with_name('m13_migrate_inputs.py'))
MIGRATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATE)


class MigrationPolicyTests(unittest.TestCase):
    def test_forced_sender_policy(self):
        self.assertTrue(MIGRATE.validate_forced_command(
            'rsync --server --sender -logDtpre.iLsfxCIvu . /'))
        self.assertTrue(MIGRATE.validate_forced_command(
            'rsync --server --sender -logDtpre.iLsfxCIvu . /home/dylan/asymetric-dual-encoders/'))
        for command in (
            'rsync --server --sender -logDtpre.iLsfxCIvu . /etc',
            'rsync --server --sender --delete . /',
            'scp /etc/shadow .',
            'rsync --server --receiver . /',
        ):
            self.assertFalse(MIGRATE.validate_forced_command(command))

    def test_endpoint_binding_replaces_only_endpoint_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'ssh_config'
            config.write_text('Host m13\n  HostName old.example\n  Port 22\n  User root\n')
            MIGRATE.update_ssh_endpoint(config, {'publicIp': '203.0.113.8', 'portMappings': {'22': 31234}})
            self.assertEqual(config.read_text(),
                             'Host m13\n  HostName 203.0.113.8\n  Port 31234\n  User root\n')

    def test_endpoint_binding_requires_both_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'ssh_config'
            config.write_text('Host m13\n  HostName old.example\n  Port 22\n')
            with self.assertRaises(RuntimeError):
                MIGRATE.update_ssh_endpoint(config, {'publicIp': '203.0.113.8'})

    def test_manifest_count_refusal_happens_before_payload_access(self):
        with self.assertRaisesRegex(RuntimeError, 'Unexpected transfer manifest'):
            MIGRATE.validate_manifest({'status': 'HASH_PINNED_NOT_LAUNCH_APPROVAL', 'files': 393, 'entries': []})

    def test_ready_receipt_is_atomic_and_contains_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            old = MIGRATE.READY
            try:
                MIGRATE.READY = Path(directory) / 'ready.json'
                receipt = {'status': 'PASSED', 'stage': 'awaiting-build-claim',
                           'verified_files': 394, 'source_pod_final_status': 'EXITED'}
                MIGRATE.write_ready(receipt)
                self.assertEqual(json.loads(MIGRATE.READY.read_text()), receipt)
                self.assertFalse((Path(directory) / 'ready.pending.json').exists())
            finally:
                MIGRATE.READY = old


if __name__ == '__main__':
    unittest.main()
