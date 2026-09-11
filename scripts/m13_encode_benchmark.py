#!/usr/bin/env python3
"""Day-one stella encode timing on admitted SQuAD training passages only.

No evaluation, gradients, vector-cache writes or protected-corpus reads. The
10M estimate is a planning surrogate, not a measured reserved-corpus runtime.
Run under the cloud supervisor, which owns backup and Pod shutdown.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'work/train/stores/squad-ctx.json'
RESULT = REPO / 'results/m13_encode_benchmark.json'
SIZES = (1000, 10000)
BATCH_TOKENS = 32768
MAX_LENGTH = 512
TIMEOUT_SECONDS = 3600
SAFETY_FACTOR = 2.0


def allowance(measurements, setup_seconds):
    """Price the slower measured per-row rate, with explicit planning headroom."""
    if [m['passages'] for m in measurements] != list(SIZES):
        raise ValueError('Both registered measurement sizes are required')
    seconds = [m['seconds'] for m in measurements]
    if not all(math.isfinite(s) and s > 0 for s in seconds):
        raise ValueError('Encode durations must be finite and positive')
    if not math.isfinite(setup_seconds) or setup_seconds < 0:
        raise ValueError('Invalid setup duration')
    slowest = max(m['seconds'] / m['passages'] for m in measurements)
    projected = 10000000 * slowest
    return {'projected_passages': 10000000,
            'slowest_seconds_per_passage': slowest,
            'unadjusted_encode_hours': projected / 3600,
            'safety_factor': SAFETY_FACTOR,
            'setup_seconds': setup_seconds,
            'reserved_batch_allowance_hours':
                math.ceil((projected * SAFETY_FACTOR + setup_seconds) / 3600 * 10) / 10,
            'limitation': 'SQuAD training passages are a surrogate. Reserved corpus token '
                'lengths, downloads, cache serialization, hashing, search and evaluation were '
                'not measured. The factor of two is planning headroom, not a runtime guarantee; '
                'reconcile actual costs under the overall budget ceiling.'}


def select_texts(payload):
    ids, texts = payload['ids'], payload['texts']
    if len(ids) != len(texts) or len(texts) < max(SIZES):
        raise ValueError('Training store must contain at least 10000 aligned passages')
    # Deterministic spread across the store avoids selecting only its first topic.
    indices = [i * len(texts) // max(SIZES) for i in range(max(SIZES))]
    selected = [texts[i] for i in indices]
    if not all(isinstance(t, str) and t.strip() for t in selected):
        raise ValueError('Selected training passages must be nonempty strings')
    return selected


def save(record):
    temporary = RESULT.with_suffix('.pending.json')
    with temporary.open('w') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(RESULT)


def terminated(signum, _frame):
    raise RuntimeError(f'Encode benchmark interrupted by signal {signum}')


def main():
    # No arbitrary data/output path options: this executable cannot be repointed at
    # protected evaluation data or historical result files through its CLI.
    if len(sys.argv) != 1:
        raise SystemExit('Usage: .venv/bin/python scripts/m13_encode_benchmark.py')
    RESULT.parent.mkdir(exist_ok=True)
    with RESULT.open('x') as f:
        f.write('{"status": "STARTING"}\n')
    record = {'status': 'RUNNING', 'measurements': [],
              'source': str(SOURCE.relative_to(REPO)),
              'protected_evaluation_access': False,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'timeout_seconds': TIMEOUT_SECONDS}
    started = time.monotonic()
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminated)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        os.environ.update(M7_ENCODER='stella-400M-v5', M7_DEVICE='cuda',
                          HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        sys.path.insert(0, str(REPO / 'm7src'))
        import numpy as np
        import torch
        import teacher

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA GPU required')
        props = torch.cuda.get_device_properties(0)
        if 'A100' not in props.name or props.total_memory < 79 * 1024**3:
            raise RuntimeError('The allocation benchmark requires an A100 80GB')
        raw = SOURCE.read_bytes()
        texts = select_texts(json.loads(raw))
        record.update(source_sha256=hashlib.sha256(raw).hexdigest(),
                      selected_texts_sha256=teacher.sha_texts(texts),
                      selection='10000 evenly spaced store positions; every tenth then all',
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                                       text=True).strip(),
                      gpu=props.name, gpu_memory_bytes=props.total_memory,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      torch_num_threads=torch.get_num_threads(),
                      teacher={'model': teacher.SPEC.repo, 'revision': teacher.SPEC.revision,
                               'pooling': teacher.SPEC.pooling, 'post_dense': teacher.SPEC.post_dense,
                               'config_kwargs': teacher.SPEC.config_kwargs,
                               'prefix': teacher.SPEC.doc_prefix, 'dtype': 'fp32',
                               'max_length': MAX_LENGTH, 'batch_tokens': BATCH_TOKENS,
                               'path': 'm7src.teacher.encode (uncached, fp32 CPU output)',
                               'comparison_path': 'teacher.encode_cached uses the same encode '
                                   'kernel; cache IO excluded. score13 six-set document caches '
                                   'are fp32; the conditional reserved executor remains separate.'},
                      code_sha256={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
                                   for p in ('scripts/m13_encode_benchmark.py',
                                             'm7src/teacher.py', 'm7src/encoders.py')})
        del raw
        tok, _ = teacher.load_teacher(dtype=torch.float32, device='cuda')
        teacher.load_post_dense(teacher.SPEC, 'cuda')
        # Small explicit warmup; not included in either per-passage estimate.
        kwargs = dict(prefix=teacher.SPEC.doc_prefix, max_length=MAX_LENGTH,
                      batch_tokens=BATCH_TOKENS, dtype=torch.float32, device='cuda')
        teacher.encode(texts[:32], **kwargs)
        torch.cuda.synchronize()
        setup_seconds = time.monotonic() - started
        record['warmup_passages'] = 32
        for count in SIZES:
            subset = texts[::max(SIZES) // count]
            lengths = [len(tok(teacher.SPEC.doc_prefix + t, truncation=True,
                               max_length=MAX_LENGTH)['input_ids']) for t in subset]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            vectors = teacher.encode(subset, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - t0
            if vectors.shape != (count, teacher.SPEC.dim) or not np.isfinite(vectors).all():
                raise RuntimeError('Invalid encoded vectors')
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                raise RuntimeError('Encoded vectors are not normalized')
            measured = {'passages': count, 'seconds': elapsed,
                        'passages_per_second': count / elapsed,
                        'tokens_after_truncation': {'mean': float(np.mean(lengths)),
                                                   'p50': float(np.percentile(lengths, 50)),
                                                   'p95': float(np.percentile(lengths, 95)),
                                                   'max': max(lengths)},
                        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'allocator_retries': torch.cuda.memory_stats().get('num_alloc_retries'),
                        'output_sha256': hashlib.sha256(vectors.tobytes()).hexdigest()}
            record['measurements'].append(measured)
            save(record)
            print(json.dumps(measured), flush=True)
            del vectors
        record['allocation'] = allowance(record['measurements'], setup_seconds)
        record['status'] = 'PASSED'
    except BaseException as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.alarm(0)
        record['wall_seconds'] = time.monotonic() - started
        save(record)
    print(json.dumps(record['allocation']), flush=True)


if __name__ == '__main__':
    main()
