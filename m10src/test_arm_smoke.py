"""GPU smoke length follows physical card capacity, without allocating a GPU."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_smoke as smoke
import nano10


@pytest.mark.parametrize('arm,device,gib,requested,expected', [
    ('E-bs128', 'cuda', 10, 512, 128),
    ('E-bs128', 'cuda', 80, 512, 512),
    ('E-bs128', 'cuda:1', 80, 512, 512),
    ('E-bs128', 'cuda', 16, 512, 512),
    ('E-bs128', 'cuda', 10, 64, 64),
    ('E-bs32', 'cuda', 10, 512, 512),
    ('E-bs128', 'cpu', None, 512, 512),
])
def test_smoke_record_uses_device_appropriate_length(monkeypatch, arm, device,
                                                    gib, requested, expected):
    def properties(selected):
        assert selected == device
        assert gib is not None, 'CPU smoke must not query CUDA'
        return SimpleNamespace(total_memory=gib * 1024**3)

    def stop_before_model(*args, **kwargs):
        raise RuntimeError('synthetic construction stop')

    monkeypatch.setattr(smoke.torch.cuda, 'get_device_properties', properties)
    monkeypatch.setattr(nano10, 'Nano10', stop_before_model)
    record = smoke.smoke_one(arm, smoke.SHAPES[arm], ([], [], [], []),
                             device=device, max_len=requested, verbose=False)
    assert record['max_len'] == expected
    assert record['constructed'] is False
    assert record['error'] == 'RuntimeError: synthetic construction stop'
