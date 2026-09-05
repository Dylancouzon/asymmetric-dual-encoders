"""Stella query targets for the M10 corpora, keyed by CONTENT HASH so a text is encoded once.

M7's `teacher.encode_cached` keys a whole corpus by the sha256 of its concatenated texts, which is
right for a fixed list and wrong here: M10's query corpus grows (harvest now, PAQ now, ~1.0M
generated queries later), the arms overlap (A3 = A2 + harvest), and a corpus-keyed cache would
re-encode all 5.3M texts the first time one string is appended. So the unit of caching is the
TEXT: `blake2b-128(text)` -> a row in an append-only fp16 memmap.

- **One prompt, one cache.** The directory is named by the encoder identity (repo, revision,
  pooling, post-Dense, prompt, max_length, store dtype). The prompt is `teacher.QUERY_PREFIX` read
  from the registry -- the same object `m9src/longrun.targets` passed for its query corpora -- so
  it cannot drift from M9's targets by being retyped.
- **Cold == warm.** `targets()` NEVER returns what the encoder just produced; it always reads the
  fp16 rows back and normalizes in fp32, so a first run and a re-run give bit-identical vectors
  (`m10/CODEMAP.md` pitfall 8).
- **One writer at a time.** An append is refresh -> duplicate lookup -> append -> meta commit, and
  two processes interleaving those can publish keys and vectors in different orders under the same
  `n`. `TargetCache.writer_lock()` is held across the whole sequence; readers take nothing and
  never truncate, so the live encode is safe to read from.
- **A 128-bit content hash is treated as text equality**, deliberately: at 10^7 texts the
  birthday probability of a blake2b-128 collision is ~10^-24, far below the probability of a bit
  flip in the store itself, so no secondary digest is kept (Codex 2026-09-05 finding 12, decided
  not to fix).
- **Resumable.** Vectors, then keys, then `meta.json`: `n` in the meta is the only authority, and
  anything past `n` rows left by a crash is truncated on the next open. A chunk is either fully
  recorded or not recorded at all.

CLI:
    .venv/bin/python m10src/targets10.py --sources harvest paq-a2 --limit 2000   # smoke + ETA
    .venv/bin/python m10src/targets10.py --sources harvest paq-a2                # the full pass
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np

import m9base                                  # pins M7_ENCODER=stella-400M-v5, installs the guard
import teacher

CACHE = REPO / "work" / "m10targets"
DIM = 1024
MAX_LENGTH = 512
KEY_BYTES = 16
CHUNK = 50_000                                  # texts per append; also the crash granularity
_KEY_DT = np.dtype([("a", "<u8"), ("b", "<u8")])


def key_of(text):
    """-> 16 content bytes. `surrogatepass` matches `teacher.sha_texts`, so a text that survives
    one hashing path survives the other."""
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=KEY_BYTES).digest()


def keys_of(texts):
    out = np.empty((len(texts), KEY_BYTES), dtype=np.uint8)
    for i, t in enumerate(texts):
        out[i] = np.frombuffer(key_of(t), dtype=np.uint8)
    return out


def _as_keydt(k):
    return np.ascontiguousarray(k).view(_KEY_DT).ravel()


def identity(prefix=None, max_length=MAX_LENGTH):
    import encoders
    s = encoders.active()
    return {"model": s.repo, "revision": s.revision, "pooling": s.pooling_key, "dim": s.dim,
            "post_dense": s.post_dense, "role": "query",
            "prefix": teacher.QUERY_PREFIX if prefix is None else prefix,
            "max_length": max_length, "store_dtype": "fp16", "hash": "blake2b-128"}


def cache_dir(ident=None):
    ident = ident or identity()
    blob = json.dumps(ident, sort_keys=True)
    return CACHE / f"query-{ident['model'].split('/')[-1]}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


class TargetCache:
    """An append-only fp16 store of teacher vectors, addressed by the hash of the text."""

    def __init__(self, d=None, ident=None, dim=DIM):
        self.ident = ident or identity()
        self.dim = dim
        self.dir = Path(d) if d else cache_dir(self.ident)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_p, self.keys_p, self.vecs_p = (self.dir / "meta.json", self.dir / "keys.u8",
                                                 self.dir / "vecs.f16")
        if self.meta_p.exists():
            m = json.loads(self.meta_p.read_text())
            if m["identity"] != self.ident:
                raise SystemExit(f"{self.dir}: cached under a different encoder identity; a cache "
                                 f"directory is named by its identity, so this is a bug, not a "
                                 f"stale export.\n  cached: {m['identity']}\n  wanted: {self.ident}")
            self.n = int(m["n"])
        else:
            self.n = 0
            self._write_meta()
        self._own_lock = False
        self._check_sizes(fix=False)
        self._sorted = None

    # -- storage -------------------------------------------------------------------------------
    def _write_meta(self):
        tmp = self.meta_p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"identity": self.ident, "n": self.n, "dim": self.dim,
                                   "key_bytes": KEY_BYTES}, indent=1))
        tmp.rename(self.meta_p)

    def _check_sizes(self, fix=False):
        """`n` is the authority. A crash between the vector write and the meta write leaves EXTRA
        bytes: unreferenced, and they would misalign the next append, so a writer truncates them.

        A file SHORTER than `n * row_bytes` is the opposite case and is not repairable: the old
        code zero-EXTENDED it, so half of a lost vector was normalized and trained as a target
        (Codex 2026-09-05 finding 8). It raises with the exact sizes instead.

        `fix=False` is the reader's mode -- it neither truncates nor creates, because a reader must
        not touch a store another process is legitimately appending to.
        """
        for name, p, row in (("keys", self.keys_p, KEY_BYTES),
                             ("vecs", self.vecs_p, 2 * self.dim)):
            want = self.n * row
            if not p.exists():
                if fix or want == 0:
                    p.touch()
                    continue
                raise SystemExit(f"{p} is missing but meta says n={self.n:,}")
            have = p.stat().st_size
            if have < want:
                raise SystemExit(
                    f"{p}: {have:,} bytes for n={self.n:,} rows x {row} bytes = {want:,} wanted. "
                    f"The store is SHORT by {want - have:,} bytes, which cannot be repaired by "
                    f"padding -- a partial vector is not a target. Restore the file, or lower "
                    f"`n` in {self.meta_p} to {have // row:,} and re-encode the rest.")
            if have > want and fix:
                self._require_lock("truncate excess bytes from")
                with open(p, "r+b") as fh:
                    fh.truncate(want)

    def _require_lock(self, what):
        """Every MUTATION goes through here. A lock nobody checks is a comment (Codex re-review
        2026-09-05): `append` and `refresh` were public and unguarded, so a second process could
        append with a stale `n` while another held the lock."""
        if not self._own_lock:
            raise SystemExit(f"refusing to {what} {self.dir} without the writer lock. Wrap the "
                             f"call in `with cache.writer_lock():`.")

    def refresh(self):
        """Re-read the meta another writer may have advanced, then repair excess bytes. Called
        under the writer lock, never on a read path."""
        self._require_lock("refresh")
        if self.meta_p.exists():
            self.n = int(json.loads(self.meta_p.read_text())["n"])
        self._check_sizes(fix=True)
        self._sorted = None
        return self.n

    @contextlib.contextmanager
    def writer_lock(self):
        """The single-writer lock: one file, created O_EXCL, held across refresh -> lookup ->
        append -> meta commit. Readers take nothing."""
        lp = self.dir / "lock"
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(
                f"{lp} exists: another process is appending to {self.dir}. Two writers can "
                f"publish keys and vectors in different orders under the same `n`. Wait for it, "
                f"or delete the lock if you are sure no writer is alive.")
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "at": time.time()}).encode())
            os.close(fd)
            self._own_lock = True
            yield lp
        finally:
            self._own_lock = False
            lp.unlink(missing_ok=True)

    def keys(self):
        if self.n == 0:
            return np.zeros((0, KEY_BYTES), dtype=np.uint8)
        return np.memmap(self.keys_p, dtype=np.uint8, mode="r", shape=(self.n, KEY_BYTES))

    def vecs(self):
        if self.n == 0:
            return np.zeros((0, self.dim), dtype=np.float16)
        return np.memmap(self.vecs_p, dtype=np.float16, mode="r", shape=(self.n, self.dim))

    def append(self, keys, vecs):
        """Vectors, then keys, then meta -- so a crash can only lose a chunk, never mis-pair one."""
        self._require_lock("append to")
        v = np.ascontiguousarray(vecs, dtype=np.float16)
        k = np.ascontiguousarray(keys, dtype=np.uint8)
        assert v.shape == (len(k), self.dim) and k.shape[1] == KEY_BYTES
        with open(self.vecs_p, "ab") as fh:
            fh.write(v.tobytes()); fh.flush()
        with open(self.keys_p, "ab") as fh:
            fh.write(k.tobytes()); fh.flush()
        self.n += len(k)
        self._write_meta()
        self._sorted = None

    # -- lookup --------------------------------------------------------------------------------
    def _index(self):
        if self._sorted is None:
            kk = _as_keydt(np.asarray(self.keys()))
            order = np.argsort(kk, kind="stable")
            self._sorted = (kk[order], order)
        return self._sorted

    def rows(self, keys):
        """-> int64 row per key, -1 where the text has never been encoded."""
        keys = np.asarray(keys, dtype=np.uint8)
        if len(keys) == 0:
            return np.zeros(0, dtype=np.int64)
        if self.n == 0:
            return np.full(len(keys), -1, dtype=np.int64)
        sk, order = self._index()
        q = _as_keydt(keys)
        pos = np.searchsorted(sk, q)
        pos_c = np.clip(pos, 0, len(sk) - 1)
        hit = sk[pos_c] == q
        out = np.where(hit, order[pos_c], -1).astype(np.int64)
        return out

    def rows_for(self, texts):
        return self.rows(keys_of(texts))


def encode_missing(texts, cache=None, chunk=CHUNK, batch_tokens=32768, verbose=True, label=""):
    """Encode whatever `texts` the cache does not hold. -> a report (never the vectors).

    Duplicates inside `texts` are collapsed by hash first, which matters: PAQ and the harvest are
    drawn independently and the 12 forms overlap in nothing but they are appended into one store.
    """
    cache = cache or TargetCache()
    with cache.writer_lock():
        return _encode_missing_locked(texts, cache, chunk, batch_tokens, verbose, label)


def _encode_missing_locked(texts, cache, chunk, batch_tokens, verbose, label):
    t0 = time.time()
    cache.refresh()
    keys = keys_of(texts)
    have = cache.rows(keys) >= 0
    todo_idx = np.flatnonzero(~have)
    # collapse duplicates among the missing, keeping first occurrence
    _u, first = np.unique(_as_keydt(keys[todo_idx]), return_index=True)
    todo_idx = todo_idx[np.sort(first)]
    n_dup = int((~have).sum() - len(todo_idx))
    rep = {"label": label, "n_texts": len(texts), "already_cached": int(have.sum()),
           "duplicates_collapsed": n_dup, "to_encode": len(todo_idx),
           "cache_dir": str(cache.dir), "chunk": chunk, "encoded": 0}
    if verbose:
        print(f"[{label}] {len(texts):,} texts: {int(have.sum()):,} cached, {n_dup:,} duplicate, "
              f"{len(todo_idx):,} to encode", flush=True)
    for i in range(0, len(todo_idx), chunk):
        sel = todo_idx[i:i + chunk]
        sub = [texts[int(j)] for j in sel]
        v = teacher.encode(sub, prefix=cache.ident["prefix"], max_length=cache.ident["max_length"],
                           batch_tokens=batch_tokens, verbose=False)
        a = np.asarray(v, dtype=np.float32)
        if not np.isfinite(a).all():
            raise SystemExit(f"[{label}] non-finite teacher vectors in chunk at {i}: a target that "
                             f"is not a finite unit vector must never reach a trainer")
        nrm = np.linalg.norm(a, axis=1)
        if not (0.99 < nrm.min() and nrm.max() < 1.01):
            raise SystemExit(f"[{label}] target norms {nrm.min():.4f}..{nrm.max():.4f}")
        cache.append(keys[sel], a.astype(np.float16))
        rep["encoded"] += len(sel)
        el = time.time() - t0
        if verbose:
            done = rep["encoded"]
            print(f"  [{label}] {done:,}/{len(todo_idx):,} @ {done / max(el, 1e-9):,.0f} texts/s "
                  f"({el:.0f}s elapsed, eta {(len(todo_idx) - done) / max(done / max(el, 1e-9), 1e-9) / 60:.0f}m)",
                  flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["texts_per_s"] = round(rep["encoded"] / max(rep["seconds"], 1e-9), 1)
    rep["cache_rows"] = cache.n
    return rep


def targets(texts, cache=None):
    """-> (n, 1024) fp32 unit-norm targets, ALWAYS read back from the fp16 store.

    Missing texts raise with the count and the command that fixes it, rather than training on a
    silently short matrix.
    """
    cache = cache or TargetCache()
    rows = cache.rows_for(texts)
    miss = int((rows < 0).sum())
    if miss:
        raise SystemExit(f"{miss:,} of {len(texts):,} texts have no teacher target in {cache.dir}. "
                         f"Run: .venv/bin/python m10src/targets10.py --sources <name...>")
    return normalize(np.asarray(cache.vecs()[rows], dtype=np.float32))


def normalize(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(n).all() or n.min() < 1e-6:
        raise SystemExit("a teacher target is ~zero or non-finite")
    return a / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="corpus_loader source names, e.g. harvest paq-a2")
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only the first N texts of each source (smoke); prints the ETA")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--report", default=str(REPO / "results" / "m10_targets10.json"))
    a = ap.parse_args()

    import corpus_loader as CL
    cache = TargetCache()
    print(f"cache {cache.dir} holds {cache.n:,} rows", flush=True)
    reps, total_texts = [], 0
    for name in a.sources:
        texts, _forms, man = CL.source_texts(name)
        total_texts += len(texts)
        use = texts[:a.limit] if a.limit else texts
        reps.append({**encode_missing(use, cache=cache, chunk=a.chunk,
                                      batch_tokens=a.batch_tokens, label=name),
                     "source_manifest": man})
    enc = sum(r["encoded"] for r in reps)
    sec = sum(r["seconds"] for r in reps)
    rate = enc / max(sec, 1e-9)
    out = {"_what": "stella query targets for the M10 corpora, content-hash keyed",
           "identity": cache.ident, "cache_dir": str(cache.dir), "cache_rows": cache.n,
           "limit": a.limit, "sources": reps, "texts_per_s": round(rate, 1),
           "corpus_texts_named": total_texts,
           "eta_hours_for_named_sources": round(total_texts / max(rate, 1e-9) / 3600, 2)}
    Path(a.report).write_text(json.dumps(out, indent=1))
    print(f"\n{enc:,} encoded in {sec:.0f}s = {rate:,.0f} texts/s; "
          f"the named sources are {total_texts:,} texts = "
          f"{out['eta_hours_for_named_sources']:.2f} h at this rate. Wrote {a.report}", flush=True)


if __name__ == "__main__":
    main()
