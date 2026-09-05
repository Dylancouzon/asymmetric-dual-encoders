"""The M10 training data pipeline: pretokenized, length-bucketed batches with teacher targets.

Wraps `m9src/data` rather than reimplementing it — that module is a `guard9` "train"-scope file
and is not edited, but its pools are exactly what M10 needs and its stella targets already exist,
so A1's corpus costs zero teacher compute. What is new here is the M10 batching:

- **Length buckets, not a global shuffle.** The measured rate difference is large: M9's two-chunk
  collate ran the M10 shape at 400 examples/s, bucketed single chunks at 890
  (`results/m10_rate_bench_real_box.json`). Padding to the batch maximum is the whole difference.
- **Query and document streams are separate**, because family B's 4-step window asks for a
  specific ratio of query to document steps and `trainer10` decides which stream a step draws
  from. Mixing them in one shuffled stream cannot express that.
- **The order is a function of (seed, step) alone**, so a resumed run draws exactly the batches an
  uninterrupted one would. `test_trainer10` proves resume determinism; it can only be true if the
  data order is reproducible from the step index, which is what `epoch_order` below guarantees.

Nothing here reads a protected surface: the pools are M8/M9's decontaminated TRAIN pools, and the
banned-row mask is applied by `m9src.data.doc_pool_rows` before anything is drawn.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np
import torch


def m9_query_pool():
    """-> (texts, target_rows, meta). The M9 screen pool and the rows of its cached stella
    targets. This is family A1's corpus and it needs no teacher pass."""
    import data as m9data
    texts, rows, meta = m9data.screen_query_pool()
    return texts, rows, meta


def m9_query_targets(rows):
    """-> (n, 1024) fp32 unit-norm teacher targets for those rows, from M9's frozen cache."""
    import data as m9data
    T = np.asarray(m9data.stella_query_targets()[rows], dtype=np.float32)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    bad = int((n < 1e-6).sum())
    if bad:
        raise SystemExit(f"{bad} teacher targets are ~zero; a target that is not a finite unit "
                         f"vector must never reach a trainer")
    return T / n


def m9_doc_pool(n, seed=0):
    """-> (texts, vectors, meta) for `n` documents drawn from the frozen stella document pool."""
    import data as m9data
    import pool as poolmod
    rows, meta = m9data.doc_pool_rows(n, seed)
    _index, vecs, _pmeta = poolmod.build()
    V = np.asarray(vecs[rows], dtype=np.float32)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return m9data.row_texts(rows), V, meta


def pretokenize(tok, texts, max_len=512, verbose=False, label=""):
    """-> list of int32 id arrays, no padding. Padding is a batching decision, not a corpus one."""
    import time
    t0, out = time.time(), []
    B = 2048
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], truncation=True, max_length=max_len,
                  add_special_tokens=True)["input_ids"]
        out += [np.asarray(e, dtype=np.int32) for e in enc]
        if verbose and (i // B) % 50 == 0 and i:
            print(f"  {label} {i:,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def length_buckets(id_lists, batch_size, seed=0):
    """-> list of index arrays, each one batch, grouped so a batch pads to near its own maximum.

    Sorted by length into contiguous runs, then the BATCHES are shuffled. Sorting alone would
    make every batch a fixed length band in a fixed order, which correlates batch content with
    training step; shuffling the batches breaks that while keeping the padding win.
    """
    # `PackedIds` (m10src/corpus_loader) carries its lengths, so a 5.3M-row corpus does not pay
    # 5.3M __getitem__ calls to find out what it already knows.
    lens = getattr(id_lists, "lengths", None)
    if lens is None:
        lens = np.array([len(x) for x in id_lists])
    order = np.argsort(lens, kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    batches = [b for b in batches if len(b) == batch_size]      # drop the ragged tail
    rng = np.random.default_rng(seed)
    rng.shuffle(batches)
    return batches


def collate(id_lists, idx, pad_id):
    """-> (input_ids, attention_mask) padded to the batch's own maximum, not the model's."""
    sel = [id_lists[int(i)] for i in idx]
    m = max(len(x) for x in sel)
    ids = np.full((len(sel), m), pad_id, dtype=np.int64)
    mask = np.zeros((len(sel), m), dtype=np.int64)
    for r, x in enumerate(sel):
        ids[r, :len(x)] = x
        mask[r, :len(x)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


class Stream:
    """One corpus (queries or documents) as an endless, step-addressable batch source."""

    def __init__(self, id_lists, targets, pad_id, batch_size=32, seed=0):
        self.ids, self.T, self.pad = id_lists, targets, pad_id
        self.batches = length_buckets(id_lists, batch_size, seed=seed)
        if not self.batches:
            raise ValueError("no full batches: corpus smaller than one batch")

    def __len__(self):
        return len(self.batches)

    def batch(self, k):
        """Batch `k`, wrapping. A pure function of k, which is what makes resume exact."""
        idx = self.batches[k % len(self.batches)]
        ids, mask = collate(self.ids, idx, self.pad)
        return ids, mask, torch.from_numpy(np.ascontiguousarray(self.T[idx]))


def kind_index(pattern, step):
    """-> how many steps before `step` drew from this step's stream. O(1) in the 4-step window.

    It replaces a call counter. A counter makes the stream position depend on how many times
    `batch_fn` has been called in THIS process, so an arm resumed at step 34 restarts both streams
    at batch 0 and trains on data the uninterrupted run had already seen -- the resume guarantee
    `test_trainer10` exists for, broken by the data path rather than by the loop. Derived from the
    step, the position is the same on both runs.
    """
    import nano10 as N
    w = N.WINDOWS[pattern]
    full, rem = divmod(int(step), len(w))
    k = w[int(step) % len(w)]
    return full * w.count(k) + w[:rem].count(k)


def batch_fn(q_stream, d_stream, pattern="75/25"):
    """-> the callable `trainer10.train_arm` wants. Each stream advances on ITS OWN counter, so
    changing family B's mix pattern re-weights the streams without re-ordering either of them.

    `pattern` must be the arm's: the counter is derived from it, and a mismatch would put the
    streams on positions the loop never asked for -- so the kind is checked, not assumed.
    """
    import nano10 as N

    def f(step, kind):
        want = N.mix_window(pattern, step)
        if kind != want:
            raise ValueError(f"step {step} is a {want!r} step under pattern {pattern!r}, asked "
                             f"for {kind!r}: batch_fn's pattern and the loop's disagree")
        s = q_stream if kind == "Q" else d_stream
        return s.batch(kind_index(pattern, step))
    return f


def manifest(**parts):
    blob = json.dumps(parts, sort_keys=True, default=str)
    return {"parts": parts, "sha256": hashlib.sha256(blob.encode()).hexdigest()}
