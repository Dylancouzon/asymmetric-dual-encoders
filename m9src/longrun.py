"""The M9.3 build: a resumable, stoppable, guarded trainer for a run measured in days.

Rewritten after Codex review #5 (`research/m9-codex-longrun-2026-08-30.md`), which returned DO NOT
LAUNCH on the first version and was right on every count that mattered. What that review changed,
because each is a way a seven-day run can be wasted or — worse — quietly wrong:

* **The loss is now the plain mean over the step's examples**, and says so in one place. The first
  version multiplied each source's mean by its token share *after* the shares had already set the
  batch composition, weighting a 95-token document ~6× a 16-token query while its docstring claimed
  otherwise. Token shares now decide *sampling* only; the objective is `Σ(‖v−t‖²) / N_examples`.
* **Integrity is checked against BYTES, not declarations.** The first version compared the hash in
  `corpora.json` with the hash in `meta.json` — two copies of the same claim — and never touched
  `flat.npy`, `offs.npy`, `pool_rows.npy` or the target caches. All are hashed now, and `offs.npy`
  matters most: corrupt it and one text's tokens are trained against another's teacher vector.
* **Resume refuses a different recipe.** The config's canonical hash and the corpus manifest hash
  are stored in the checkpoint and compared before any state is loaded, so a restart cannot produce
  a hybrid of old optimizer state and new hyperparameters.
* **Decay is resumable** — phase, origin and length live in the checkpoint, so an interrupted
  cooldown continues instead of restarting a fresh cosine from wherever it stopped.
* **The kill envelope exists**: non-finite loss or gradient, SCREEN-3 regression against the best
  checkpoint, a token-denominated plateau, and throughput collapse each stop the run and say why.
* **It runs under the guard**, in its own `build` scope, so it cannot train from dirty code.
* Cumulative dose is persisted, so `--hours` is a wall-clock budget for a session while the token
  and example counters are the run's true, resumable ledger.

    python m9src/longrun.py prepare     # tokenize + hash every corpus, once
    python m9src/longrun.py targets     # teacher vectors for the corpora that lack them
    python m9src/longrun.py verify      # integrity + config, without training
    python m9src/longrun.py train  --hours 24
    python m9src/longrun.py decay       # cooldown -> the servable artifact
    python m9src/longrun.py status
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import m9base
from m9base import RESULTS, WORK

import data as m9data     # noqa: E402
import eval9              # noqa: E402
import guard9             # noqa: E402
import nano               # noqa: E402

RUN = WORK / "m9long"
TOKENS = RUN / "corpora"
CKPT = RUN / "ckpt"
HISTORY = RUN / "history.jsonl"
HEARTBEAT = RUN / "heartbeat.json"
TERMINAL = RUN / "terminal.json"
MANIFEST = RUN / "manifest.json"
CONFIG = RUN / "config.json"
LOCKFILE = RUN / "trainer.lock"

QUERY_SOURCES = ("queries_pair", "nqopen", "triviaqa")
SPAN_SOURCES = ("pseudoq",)
DOC_SOURCES = ("documents",)


_BEAT = {"at": 0.0}


def beat(state, **kw):
    """Write the heartbeat on a 60-second timer of its OWN, not on the training log cadence.

    Tying it to `log_every` was a defect Codex found: at a 5x slowdown the log line arrives every
    ~763 s, which sits deliberately under a 900 s staleness threshold, so the exact failure the
    watchdog exists for kept it looking alive. It also emitted nothing at all during `verify`,
    target mapping, model construction or warm start, so a wedge there was invisible forever.
    """
    now = time.time()
    if state in ("train",) and now - _BEAT["at"] < 60:
        return
    _BEAT["at"] = now
    rec = {"wall": now, "state": state, "pid": os.getpid(), **kw}
    try:
        RUN.mkdir(parents=True, exist_ok=True)
        t = HEARTBEAT.with_suffix(".tmp")
        t.write_text(json.dumps(rec))
        os.replace(t, HEARTBEAT)
    except OSError:
        pass


def sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- corpora ----------

def _pack_streaming(tok, texts, prefix, max_len=512, batch=20_000, label=""):
    """-> (flat int32, offsets int64), tokenizing and packing in CHUNKS.

    Tokenizing everything first is a trap at this scale: a Python list of ~95 token ids costs
    ~3.5 kB once the list object and un-cached int objects are counted, so 6.15M documents is
    roughly 21 GB of transient heap before a byte is packed — which on a 25 GB box also running a
    training chain means the OOM killer takes both. Packed as int32 the same corpus is 2.3 GB.
    """
    chunks, lens_parts, total, t0 = [], [], 0, time.time()
    for i in range(0, len(texts), batch):
        ids = tok([prefix + t for t in texts[i:i + batch]], truncation=True,
                  max_length=max_len, add_special_tokens=True)["input_ids"]
        lens = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        flat = np.empty(int(lens.sum()), dtype=np.int32)
        pos = 0
        for x in ids:
            flat[pos:pos + len(x)] = x
            pos += len(x)
        chunks.append(flat)
        lens_parts.append(lens)
        total += len(ids)
        del ids
        if label and (i // batch) % 25 == 0 and i:
            el = time.time() - t0
            print(f"    {label} {total:,}/{len(texts):,} ({total/max(el,1e-9):,.0f}/s)", flush=True)
    lens = np.concatenate(lens_parts) if lens_parts else np.zeros(0, dtype=np.int64)
    offs = np.zeros(lens.size + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32), offs


def _save_corpus(name, flat, offs, meta):
    d = TOKENS / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "flat.npy", flat)
    np.save(d / "offs.npy", offs)
    meta = {**meta, "n": int(offs.size - 1), "n_tokens": int(offs[-1]),
            "mean_tokens": round(float(offs[-1]) / max(offs.size - 1, 1), 2)}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {name}: {meta['n']:,} texts, {meta['n_tokens']:,} tokens, "
          f"mean {meta['mean_tokens']}", flush=True)
    return meta


def load_corpus(name):
    d = TOKENS / name
    meta = json.loads((d / "meta.json").read_text())
    return np.load(d / "flat.npy", mmap_mode="r"), np.load(d / "offs.npy"), meta


def extra_texts():
    scr = RESULTS / "m9_extended_screen.json"
    blob = json.loads(scr.read_text())
    out = {}
    for name, row in blob["sources"].items():
        raw = json.loads((m9base.REPO / row["path"]).read_text())
        kept = json.loads((WORK / "decontam" / f"m9_kept_{name}.json").read_text())
        out[name] = ([str(raw[i]) for i in kept], row)
    return out


def _corpus_current(name, prefix, n, student, ident=None):
    """True when `name` is already tokenized with exactly this prefix, text count, STUDENT
    (tokenizers differ between students) and source identity, so a re-prepare under a new prompt
    policy re-tokenizes only the query corpora (~463K texts, minutes) without redoing the 6.15M
    documents. Declared prefix/count alone could bless a stale corpus from an older pool
    (Codex #8, blocker 3) -- the identity fields bind the actual source."""
    p = TOKENS / name / "meta.json"
    if not p.exists():
        return False
    m = json.loads(p.read_text())
    return (m.get("prefix") == prefix and m.get("n") == n and m.get("student") == student
            and all(m.get(k) == v for k, v in (ident or {}).items())
            and (TOKENS / name / "flat.npy").exists()
            and (TOKENS / name / "offs.npy").exists())


def prepare(student_key, doc_limit=None, prompt_policy="b"):
    """`student_key` and `prompt_policy` must be what the screen SELECTED: both the tokenizer and
    the prefix are baked into the tokenized corpus, so preparing under one recipe and training
    another would give the warm start and SGD different inputs (Codex review #7, blocker 1).
    `make_config` and `train` re-check."""
    RUN.mkdir(parents=True, exist_ok=True)
    tok = nano.Nano(student_key).tok
    tpl = guard9.registry()["templates"]
    qpre = tpl["query_policy_a_student"] if prompt_policy == "a" else tpl["query_policy_b_student"]
    t0 = time.time()

    qsrc = WORK / "m9_screen_queries.json"
    qsha = sha_file(qsrc)
    q = json.loads(qsrc.read_text())
    if _corpus_current("queries_pair", qpre, len(q), student_key, {"source_sha256": qsha}):
        print("  queries_pair: already tokenized under this recipe", flush=True)
    else:
        _save_corpus("queries_pair",
                     *_pack_streaming(tok, q, qpre, label="queries_pair"),
                     {"role": "query", "prefix": qpre, "prompt_policy": prompt_policy,
                      "student": student_key, "source": "work/m9_screen_queries.json",
                      "source_sha256": qsha})
    for name, (texts, row) in extra_texts().items():
        role = "query" if name in QUERY_SOURCES else "doc_span"
        pre = qpre if role == "query" else tpl["doc_student"]
        if _corpus_current(name, pre, len(texts), student_key,
                           {"kept_index_sha256": row["kept_index_sha256"]}):
            print(f"  {name}: already tokenized under this recipe", flush=True)
        else:
            _save_corpus(name, *_pack_streaming(tok, texts, pre, label=name),
                         {"role": role, "prefix": pre, "prompt_policy": prompt_policy,
                          "student": student_key, "what": row["what"],
                          "kept_index_sha256": row["kept_index_sha256"]})
        del texts

    r = guard9.registry()
    n_docs = doc_limit or r["data"]["n_eligible_doc_rows"]
    if _corpus_current("documents", tpl["doc_student"], n_docs, student_key,
                       {"doc_candidates_seed": r["data"]["doc_candidates_seed"]}) and \
            (TOKENS / "documents" / "pool_rows.npy").exists():
        print("  documents: already tokenized (doc prefix is policy-independent)", flush=True)
    else:
        rows, dmeta = m9data.doc_pool_rows(n_docs, r["data"]["doc_candidates_seed"])
        print(f"  documents: reading {rows.size:,} texts from the pool stores...", flush=True)
        texts = m9data.row_texts(rows)
        _save_corpus("documents",
                     *_pack_streaming(tok, texts, tpl["doc_student"], label="documents"),
                     {"role": "doc", "prefix": tpl["doc_student"], "student": student_key,
                      "doc_candidates_seed": r["data"]["doc_candidates_seed"], **dmeta})
        np.save(TOKENS / "documents" / "pool_rows.npy", rows)
        del texts
    print(f"prepare done in {time.time()-t0:.0f}s -- now run `targets`, `manifest`, `verify`",
          flush=True)


# --------------------------------------------------------------------------- targets ----------

def target_dir(name):
    return WORK / "enc9" / f"m9long-{name}"


def targets(batch_tokens=32768):
    """Teacher vectors for the corpora that do not already have them.

    `queries_pair` reuses M7's cached matrix and `documents` reads the frozen pool, so only the
    newly admitted sources need a pass. One stella pass over ~1.14M texts.
    """
    import teacher
    tpl = guard9.registry()["templates"]
    for name, (texts, _row) in extra_texts().items():
        d = target_dir(name)
        meta_p = d / "meta.json"
        if meta_p.exists() and json.loads(meta_p.read_text()).get("n") == len(texts):
            print(f"  {name}: cached", flush=True)
            continue
        role = "query" if name in QUERY_SOURCES else "doc"
        prefix = teacher.QUERY_PREFIX if role == "query" else ""
        print(f"  {name}: encoding {len(texts):,} texts ({role})...", flush=True)
        t0 = time.time()
        v = teacher.encode_cached(f"m9long-{name}", texts, prefix=prefix, max_length=512,
                                  batch_tokens=batch_tokens, verbose=True)
        a = np.asarray(v, dtype=np.float32)
        assert np.isfinite(a).all(), f"{name}: non-finite teacher vectors"
        nrm = np.linalg.norm(a, axis=1)
        assert 0.99 < nrm.min() and nrm.max() < 1.01, f"{name}: norms {nrm.min()}..{nrm.max()}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vecs.npy", a.astype(np.float16))
        meta_p.write_text(json.dumps(
            {"n": len(texts), "dim": int(a.shape[1]), "role": role, "prefix": prefix,
             "student_prefix": tpl["query_policy_b_student"] if role == "query"
             else tpl["doc_student"],
             "sha256": sha_file(d / "vecs.npy"), "seconds": round(time.time() - t0, 1)}, indent=1))
        print(f"  {name}: done in {time.time()-t0:.0f}s", flush=True)
        del a, texts


# --------------------------------------------------------------------------- manifest ---------

def build_manifest():
    """Hash every byte a training step can depend on. Declarations are not evidence."""
    man = {"corpora": {}, "targets": {}, "maps": {}}
    for name in QUERY_SOURCES + SPAN_SOURCES + DOC_SOURCES:
        d = TOKENS / name
        man["corpora"][name] = {
            "flat_sha256": sha_file(d / "flat.npy"), "offs_sha256": sha_file(d / "offs.npy"),
            **json.loads((d / "meta.json").read_text())}
    man["maps"]["pool_rows"] = sha_file(TOKENS / "documents" / "pool_rows.npy")
    man["maps"]["screen_rows"] = sha_file(WORK / "m9_screen_rows.npy")
    for name in QUERY_SOURCES[1:] + SPAN_SOURCES:
        d = target_dir(name)
        man["targets"][name] = {"vecs_sha256": sha_file(d / "vecs.npy"),
                                **json.loads((d / "meta.json").read_text())}
    man["targets"]["queries_pair"] = {"source": "M7 cached stella s2p matrix trainq-337981"}
    man["targets"]["documents"] = {"source": "frozen pool work/pool/stella-400M-v5"}
    man["manifest_sha256"] = canon(man)
    return man


def verify(strict=True):
    """Recompute every hash and compare with the published manifest. Cheap next to a wasted day."""
    assert MANIFEST.exists(), f"{MANIFEST} missing -- run `prepare`, `targets`, then `verify`"
    want = json.loads(MANIFEST.read_text())
    got = build_manifest()
    bad = []
    for section in ("corpora", "targets", "maps"):
        for k, v in want[section].items():
            g = got[section].get(k)
            if isinstance(v, dict):
                for f in ("flat_sha256", "offs_sha256", "vecs_sha256", "n"):
                    if f in v and g.get(f) != v[f]:
                        bad.append(f"{section}/{k}/{f}")
            elif g != v:
                bad.append(f"{section}/{k}")
    # Structural sanity beyond hash agreement: hashes prove the bytes are the bytes the manifest
    # saw, not that they parse. A corrupt offs.npy trains one text's tokens against another's
    # teacher vector silently (Codex #8, blocker 3).
    for name in QUERY_SOURCES + SPAN_SOURCES + DOC_SOURCES:
        flat, offs, meta = load_corpus(name)
        if not (offs[0] == 0 and offs[-1] == flat.size and offs.size == meta["n"] + 1
                and bool(np.all(np.diff(offs) > 0))):
            bad.append(f"structure/{name}")
    ok = not bad
    print(json.dumps({"manifest_sha256": want["manifest_sha256"], "recomputed_ok": ok,
                      "mismatches": bad}, indent=1))
    if strict and not ok:
        raise SystemExit(f"corpus/target integrity FAILED: {bad}")
    return got


# --------------------------------------------------------------------------- schedule ---------

def lr_at(step, cfg, phase):
    if step < cfg["warmup_steps"]:
        return cfg["lr_peak"] * (step + 1) / cfg["warmup_steps"]
    if phase["name"] != "decay":
        return cfg["lr_peak"]
    t = min(1.0, (step - phase["decay_from"]) / max(phase["decay_steps"] - 1, 1))
    return cfg["lr_final"] + 0.5 * (cfg["lr_peak"] - cfg["lr_final"]) * (1 + math.cos(math.pi * t))


# --------------------------------------------------------------------------- streams ----------

class Stream:
    """Deterministic, resumable, infinite. Position is (epoch, offset): a resume replays the same
    permutation from the same place, because a stream that reshuffled on restart would quietly
    turn a seven-day run into a different experiment."""

    def __init__(self, name, n, seed, epoch=0, offset=0):
        self.name, self.n, self.seed = name, n, seed
        self.epoch, self.offset, self._perm = epoch, offset, None

    def _p(self):
        if self._perm is None:
            self._perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        return self._perm

    def take(self, k):
        out = []
        while k > 0:
            p = self._p()
            m = min(k, self.n - self.offset)
            out.append(p[self.offset:self.offset + m])
            self.offset += m
            k -= m
            if self.offset >= self.n:
                self.epoch += 1
                self.offset = 0
                self._perm = None
        return np.concatenate(out) if len(out) > 1 else out[0]

    def state(self):
        return {"name": self.name, "epoch": self.epoch, "offset": self.offset, "seed": self.seed}


# --------------------------------------------------------------------------- targets ----------

class Targets:
    def __init__(self):
        self.q = np.asarray(m9data.stella_query_targets())
        self.qrows = np.load(WORK / "m9_screen_rows.npy")
        import pool as poolmod
        _i, self.pool, _m = poolmod.build()
        self.doc_rows = np.load(TOKENS / "documents" / "pool_rows.npy")
        self.extra = {n: np.load(target_dir(n) / "vecs.npy", mmap_mode="r")
                      for n in QUERY_SOURCES[1:] + SPAN_SOURCES}

    def check(self, corpora):
        """Validate every target cache BEFORE the optimizer is built -- a missing one used to
        surface only after the first backward pass had already run."""
        for name, (_f, offs, _m) in corpora.items():
            n = offs.size - 1
            if name == "queries_pair":
                assert self.qrows.size == n, f"{name}: {self.qrows.size} rows vs {n} texts"
            elif name == "documents":
                assert self.doc_rows.size == n, f"{name}: {self.doc_rows.size} rows vs {n} texts"
            else:
                a = self.extra[name]
                assert a.shape[0] == n, f"{name}: targets {a.shape[0]} vs {n} texts"
                assert a.shape[1] == 1024, f"{name}: dim {a.shape[1]}"
        return True

    def get(self, corpus, idx):
        if corpus == "queries_pair":
            return np.asarray(self.q[self.qrows[idx]], dtype=np.float32)
        if corpus == "documents":
            return np.asarray(self.pool[self.doc_rows[idx]], dtype=np.float32)
        return np.asarray(self.extra[corpus][idx], dtype=np.float32)


# --------------------------------------------------------------------------- training ---------

def collate(flat, offs, idx, pad_id, device):
    lens = (offs[idx + 1] - offs[idx]).astype(np.int64)
    n = int(lens.max())
    ii = np.full((len(idx), n), pad_id, dtype=np.int64)
    am = np.zeros((len(idx), n), dtype=np.int64)
    for k, (i, L) in enumerate(zip(idx, lens)):
        ii[k, :L] = flat[offs[i]:offs[i] + L]
        am[k, :L] = 1
    return (torch.from_numpy(ii).to(device, non_blocking=True),
            torch.from_numpy(am).to(device, non_blocking=True), int(lens.sum()))


def save_ckpt(path, blob):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        torch.save(blob, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def _proc_start(pid):
    """The PID's start time, so a recycled PID cannot impersonate the lock holder."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _acquire_lock():
    """O_CREAT|O_EXCL, not exists()-then-write. The watchdog can SIGTERM a trainer and start a
    replacement; if the old one is stuck in uninterruptible I/O, a check-then-write race puts two
    processes on the same `last.tmp`, and `os.replace` protects against one crashing writer, not
    two live ones."""
    RUN.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCKFILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "start": _proc_start(os.getpid()),
                                     "at": time.time()}).encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                held = json.loads(LOCKFILE.read_text())
            except (OSError, json.JSONDecodeError):
                held = None
            alive = held and _proc_start(held["pid"]) == held.get("start")
            if alive:
                raise SystemExit(
                    f"{LOCKFILE} is held by live pid {held['pid']} (start {held['start']}). "
                    f"Two trainers must never write {CKPT}.")
            print(f"clearing a stale lock from pid {held['pid'] if held else '?'}", flush=True)
            LOCKFILE.unlink(missing_ok=True)
    raise SystemExit("could not acquire the trainer lock")


def write_terminal(reason, step, cum, phase):
    """A registered stop, recorded atomically. Without this the watchdog sees only 'no PID' and
    restarts the run it just deliberately ended -- which would resurrect a first-eval failure, a
    regression stop, a plateau stop and a completed cooldown alike (Codex, blocker 1)."""
    t = TERMINAL.with_suffix(".tmp")
    t.write_text(json.dumps({"reason": reason, "step": step, "tokens": cum["tokens"],
                             "examples": cum["examples"], "phase": phase["name"],
                             "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall": time.time()},
                            indent=1))
    os.replace(t, TERMINAL)


def load_config():
    assert CONFIG.exists(), f"{CONFIG} missing -- M9.2 writes it when the recipe is locked"
    cfg = json.loads(CONFIG.read_text())
    cfg["_hash"] = canon({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


def train(cfg, hours=None, max_steps=None, start_decay=False, device="cuda", anneal=False):
    RUN.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    _acquire_lock()
    try:
        return _train(cfg, hours, max_steps, start_decay, device, anneal)
    finally:
        LOCKFILE.unlink(missing_ok=True)


def _train(cfg, hours, max_steps, start_decay, device, anneal=False):
    beat("verify")
    man = verify(strict=True)
    reconcile_history()
    names = [n for n in QUERY_SOURCES + SPAN_SOURCES + DOC_SOURCES if cfg["shares"].get(_grp(n))]
    corpora = {n: load_corpus(n) for n in names}
    doc_pre = guard9.registry()["templates"]["doc_student"]
    for n in names:
        want = cfg["student_query_prefix"] if n in QUERY_SOURCES else doc_pre
        got = corpora[n][2].get("prefix")
        got_student = corpora[n][2].get("student")
        if got != want or got_student != cfg["student"]:
            raise SystemExit(
                f"corpus {n!r} is tokenized with prefix {got!r} / student {got_student!r}; the "
                f"locked recipe requires prefix {want!r} / student {cfg['student']!r} (prompt "
                f"policy {cfg.get('prompt_policy')!r}). The warm start and SGD would train "
                f"different recipes. Re-run `longrun.py prepare --student {cfg['student']} "
                f"--prompt-policy {cfg.get('prompt_policy')}`, then `targets`, `manifest`, "
                f"`verify`.")
    beat("targets")
    tgt = Targets()
    tgt.check(corpora)
    beat("model")

    model = nano.Nano(cfg["student"]).to(device)
    pad_id = model.tok.pad_token_id
    dec = [p for _n, p in model.named_parameters() if p.dim() > 1]
    nod = [p for _n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": cfg["weight_decay"]},
                             {"params": nod, "weight_decay": 0.0}],
                            lr=cfg["lr_peak"], betas=tuple(cfg["betas"]), eps=cfg["eps"])

    step, cum = 0, {"tokens": 0, "examples": 0, "by_source": {n: 0 for n in names}}
    phase = {"name": "stable", "decay_from": None, "decay_steps": None}
    best = None
    streams = {n: Stream(n, corpora[n][2]["n"], cfg["seed"] + i) for i, n in enumerate(names)}

    last = CKPT / "last.pt"
    if last.exists():
        blob = torch.load(last, map_location=device, weights_only=False)
        if blob["config_hash"] != cfg["_hash"]:
            raise SystemExit(
                f"the checkpoint was written under config {blob['config_hash'][:12]} and the "
                f"current config hashes {cfg['_hash'][:12]}. Resuming would blend old optimizer "
                f"state with new hyperparameters. Restore the config or start a new run.")
        if blob["manifest_hash"] != man["manifest_sha256"]:
            raise SystemExit(
                f"the corpora/targets changed under this run "
                f"({blob['manifest_hash'][:12]} -> {man['manifest_sha256'][:12]}).")
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        step, cum, phase, best = blob["step"], blob["cum"], blob["phase"], blob.get("best")
        for n, s in blob["streams"].items():
            if n in streams:
                streams[n] = Stream(n, corpora[n][2]["n"], s["seed"], s["epoch"], s["offset"])
        torch.set_rng_state(blob["torch_rng"].cpu())
        if blob.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in blob["cuda_rng"]])
        print(f"resumed: step {step:,}, {cum['tokens']/1e9:.3f}B tokens, phase {phase['name']}",
              flush=True)
    else:
        torch.manual_seed(cfg["seed"])
        beat("warm_start")
        q = json.loads((WORK / "m9_screen_queries.json").read_text())
        ws = nano.warm_start_head(model, q, np.asarray(tgt.q[tgt.qrows]),
                                  cfg["student_query_prefix"])
        print(f"warm-start head: {json.dumps(ws)}", flush=True)

    if start_decay and phase["name"] != "decay":
        phase = {"name": "decay", "decay_from": step, "decay_steps": cfg["decay_steps"]}
        print(f"cooldown begins at step {step:,} for {cfg['decay_steps']:,} steps", flush=True)

    # per-step example counts, fixed by the TOKEN shares -- sampling only, never loss weighting
    per_step = {}
    for n in names:
        share = cfg["shares"][_grp(n)] * _within(n, corpora, cfg)
        per_step[n] = max(1, int(round(cfg["tokens_per_step"] * share /
                                       corpora[n][2]["mean_tokens"])))
    print("examples/step: " + json.dumps(per_step) +
          f"  (total {sum(per_step.values())})", flush=True)

    step0 = None
    if step == 0:
        beat("eval0")
        step0 = evaluate(model, cfg, 0, cum, 0.0)
        with open(HISTORY, "a") as fh:
            fh.write(json.dumps({**step0, "step0": True}) + "\n")
        model.train()
        torch.cuda.empty_cache()

    t0, sess_tok, sess_ex, loss_acc, nlog = time.time(), 0, 0, 0.0, 0
    # ROLLING throughput, not the cumulative session mean. Cumulative hides exactly the failure
    # this guards: a 5x slowdown after three good days needs ~five more days to drag the average
    # under half, and every restart resets the baseline so a degraded rate becomes the new normal.
    samples = [(time.time(), 0)]
    base_p = RUN / "throughput_baseline.json"
    baseline = json.loads(base_p.read_text())["tok_per_s"] if base_p.exists() else None
    deadline = t0 + hours * 3600 if hours else None
    stop_reason = None
    rate = None            # rolling tok/s; set after the first step of this session
    model.train()

    while stop_reason is None:
        if max_steps and step >= max_steps:
            stop_reason = "max_steps"
            break
        if phase["name"] == "decay" and step >= phase["decay_from"] + phase["decay_steps"]:
            stop_reason = "cooldown complete" + (
                f" (entered on: {phase['trigger']})" if phase.get("trigger") else "")
            break
        if deadline and time.time() > deadline:
            stop_reason = "session wall-clock budget"
            break
        if (CKPT / "STOP").exists():
            stop_reason = "STOP file"
            break
        if phase["name"] == "stable" and cum["tokens"] >= cfg["stable_token_cap"]:
            # A registered end of the stable phase RUNS the cooldown rather than stopping with an
            # unannealed checkpoint (M92_LOCK §6; Codex review #7, blocker 2).
            phase = {"name": "decay", "decay_from": step, "decay_steps": cfg["decay_steps"],
                     "trigger": f"stable-phase token cap {cfg['stable_token_cap']:,}"}
            print(f"cooldown begins at step {step:,} for {cfg['decay_steps']:,} steps "
                  f"({phase['trigger']})", flush=True)
            # durable immediately: a restart before the next scheduled checkpoint must resume
            # into the cooldown, not into another 1.7 h of stable LR (Codex #8, blocker 5)
            save_ckpt(CKPT / "last.pt", _blob(model, opt, step, streams, cfg, cum, phase, best, man))
        if (anneal and deadline and phase["name"] == "stable" and rate and step > 0
                and (deadline - time.time()) <
                1.25 * cfg["decay_steps"] * cfg["tokens_per_step"] / rate):
            # The horizon must not truncate the anneal: total was sized as cap + cooldown with
            # ZERO margin, so any slowdown means the cap is never reached and the wall clock
            # stops an unannealed run with nobody at the machine (Codex #8, blocker 5). Enter the
            # cooldown while it still fits, with a 25% margin at the current measured rate.
            phase = {"name": "decay", "decay_from": step, "decay_steps": cfg["decay_steps"],
                     "trigger": f"wall-clock horizon approaching ({deadline - time.time():,.0f}s "
                                f"left at {rate:,.0f} tok/s)"}
            print(f"cooldown begins at step {step:,} for {cfg['decay_steps']:,} steps "
                  f"({phase['trigger']})", flush=True)
            save_ckpt(CKPT / "last.pt", _blob(model, opt, step, streams, cfg, cum, phase, best, man))

        lr = lr_at(step, cfg, phase)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)

        n_ex = sum(per_step.values())
        step_loss, step_tok = 0.0, 0
        for name in names:
            flat, offs, _m = corpora[name]
            idx = streams[name].take(per_step[name])
            ii, am, ntok = collate(flat, offs, idx, pad_id, device)
            t = torch.from_numpy(tgt.get(name, idx)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(ii, am)
            v = F.normalize(v.float(), dim=-1, eps=1e-12)
            # THE objective, in one place: the plain mean over the step's examples. Token shares
            # set how many examples each source contributes and nothing else.
            part = ((v - t) ** 2).sum(-1).sum() / n_ex
            if not torch.isfinite(part):
                stop_reason = f"non-finite loss on {name} at step {step}"
                break
            part.backward()
            step_loss += float(part.detach())
            step_tok += ntok
            cum["by_source"][name] += len(idx)
        if stop_reason:
            break

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not torch.isfinite(gnorm):
            stop_reason = f"non-finite gradient norm at step {step}"
            break
        opt.step()
        step += 1
        cum["tokens"] += step_tok
        cum["examples"] += n_ex
        sess_tok += step_tok
        sess_ex += n_ex
        loss_acc += step_loss
        nlog += 1

        samples.append((time.time(), sess_tok))
        samples[:] = [x for x in samples if x[0] > time.time() - cfg["throughput_window_s"]] \
            or samples[-2:]
        rate = ((samples[-1][1] - samples[0][1]) / max(samples[-1][0] - samples[0][0], 1e-9))
        beat("train", step=step, tokens=cum["tokens"], examples=cum["examples"],
             tok_per_s=rate, phase=phase["name"], loss=step_loss, lr=lr,
             baseline=baseline, floor=(baseline * cfg["throughput_floor_frac"]) if baseline else None,
             stable_token_cap=cfg["stable_token_cap"], evals=len(read_history()))

        if step % cfg["log_every"] == 0:
            el = time.time() - t0
            print(f"  step {step:,} loss {loss_acc/nlog:.5f} lr {lr:.2e} "
                  f"{rate:,.0f} tok/s | cum {cum['tokens']/1e9:.3f}B tokens "
                  f"({cum['tokens']/cfg['stable_token_cap']:.1%} of cap)", flush=True)
            # The watchdog's heartbeat is `beat("train", ...)` above, written on its own 60 s
            # timer every step. A second writer here raced it and was the only one that omitted
            # `state`, which the watchdog's no-progress rule keys on (Codex review #7, blocker 4).
            loss_acc, nlog = 0.0, 0
            # Freeze a baseline ONCE, after warmup, and persist it so restarts inherit it rather
            # than re-baselining onto a degraded rate.
            if baseline is None and el > cfg["throughput_baseline_after_s"]:
                baseline = rate
                base_p.write_text(json.dumps({"tok_per_s": rate, "at": time.time(),
                                              "measured_over_s": cfg["throughput_window_s"]}))
                print(f"  throughput baseline frozen at {rate:,.0f} tok/s "
                      f"(floor {rate*cfg['throughput_floor_frac']:,.0f})", flush=True)
            if baseline and rate < cfg["throughput_floor_frac"] * baseline:
                stop_reason = (f"throughput collapse: {rate:,.0f} tok/s over the last "
                               f"{cfg['throughput_window_s']}s against a frozen baseline of "
                               f"{baseline:,.0f}")
                break

        if step % cfg["eval_every"] == 0:
            beat("eval", step=step, tokens=cum["tokens"])
            rec = evaluate(model, cfg, step, cum, time.time() - t0)
            model.train()
            torch.cuda.empty_cache()      # the screen measured 1,990 -> 786 ex/s without this
            blob = _blob(model, opt, step, streams, cfg, cum, phase, best, man)
            save_ckpt(CKPT / f"step{step}.pt", {**blob, "eval": rec})
            save_ckpt(CKPT / "last.pt", blob)
            with open(HISTORY, "a") as fh:
                fh.write(json.dumps(rec) + "\n")     # after the checkpoint, so replay can dedupe
            # First-eval sanity gate: at ~164M tokens the build should already be past the
            # anchor's 59.5M-token result. If it is not, the recipe or the data is wrong and six
            # more days will not fix it -- stop now rather than discover it on day seven.
            hist_rows = read_history()
            trained_rows = [r for r in hist_rows if not r.get("step0")]
            if len(trained_rows) == 1:
                # An ABSOLUTE floor is unsupported: this build's mixture is not the anchor's, and
                # by its first evaluation it has seen only ~8.2M query tokens. So the absolute
                # number is logged, and the hard gate is against THIS run's own step-0 baseline --
                # a first evaluation below where the warm-started head started means something is
                # broken, whatever the mixture (Codex, threshold disposition). Counting only
                # TRAINED rows: the step-0 row is in history too, so `len(history)==1` was already
                # false at the first trained evaluation and the gate could never fire; and after a
                # pre-first-eval restart the baseline is recovered from history rather than from a
                # local that a resume leaves None (Codex #8, blocker 2).
                base_rec = next((r for r in hist_rows if r.get("step0")), None)
                base = base_rec["screen3"] if base_rec else None
                print(f"  first eval {rec['screen3']:.5f}; absolute floor "
                      f"{cfg['first_eval_floor']} (advisory); step-0 baseline "
                      f"{base if base is None else round(base, 5)}", flush=True)
                if base is not None and rec["screen3"] < base - cfg["first_eval_regression"]:
                    stop_reason = (f"first evaluation {rec['screen3']:.5f} is below the step-0 "
                                   f"baseline {base:.5f} by more than "
                                   f"{cfg['first_eval_regression']} -- training is making it worse")
            kill = stop_reason or check_kill(rec, cfg)
            if kill and kill.startswith("plateau:"):
                # A plateau is a registered stop that RUNS the cooldown (M92_LOCK §6), so enter
                # decay instead of stopping -- and never re-fire inside a cooldown already running
                # (Codex review #7, blocker 2).
                if phase["name"] == "stable":
                    phase = {"name": "decay", "decay_from": step,
                             "decay_steps": cfg["decay_steps"], "trigger": kill}
                    print(f"cooldown begins at step {step:,} for {cfg['decay_steps']:,} steps "
                          f"({kill})", flush=True)
                    # durable immediately, or a restart resumes into stable LR (Codex #8, bl. 5)
                    save_ckpt(CKPT / "last.pt",
                              _blob(model, opt, step, streams, cfg, cum, phase, best, man))
                kill = None
            stop_reason = kill
            if best is None or rec["screen3"] > best["screen3"]:
                best = {"step": step, "screen3": rec["screen3"], "tokens": cum["tokens"]}
        elif step % cfg["ckpt_every"] == 0:
            save_ckpt(CKPT / "last.pt", _blob(model, opt, step, streams, cfg, cum, phase, best, man))

    save_ckpt(CKPT / "last.pt", _blob(model, opt, step, streams, cfg, cum, phase, best, man))
    write_terminal(stop_reason, step, cum, phase)
    beat("stopped", step=step, tokens=cum["tokens"], reason=stop_reason)
    print(f"STOPPED: {stop_reason}\n  step {step:,}, cumulative {cum['tokens']/1e9:.3f}B tokens, "
          f"{cum['examples']:,} examples, phase {phase['name']}, best {best}", flush=True)
    return {"step": step, "cum": cum, "phase": phase, "best": best, "stop_reason": stop_reason}


def _grp(name):
    return ("queries" if name in QUERY_SOURCES else
            "spans" if name in SPAN_SOURCES else "documents")


def _within(name, corpora, cfg):
    """A group's share is split across its members in proportion to their token totals, so the
    three query sources form one logical batch instead of three tiny ones."""
    grp = _grp(name)
    sibs = [n for n in corpora if _grp(n) == grp]
    tot = sum(corpora[s][2]["n_tokens"] for s in sibs)
    return corpora[name][2]["n_tokens"] / tot


def _blob(model, opt, step, streams, cfg, cum, phase, best, man):
    return {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
            "streams": {k: s.state() for k, s in streams.items()},
            "cfg": cfg, "config_hash": cfg["_hash"], "manifest_hash": man["manifest_sha256"],
            "cum": cum, "phase": phase, "best": best,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def check_kill(rec, cfg):
    """The numeric kill envelope the mandate requires, read against cumulative TOKENS."""
    rows = read_history()
    if len(rows) < 2:
        return None
    best = max(r["screen3"] for r in rows)
    tail = rows[-2:]
    if all(best - r["screen3"] > cfg["regression_thresh"] for r in tail):
        return (f"SCREEN-3 regression: two consecutive evaluations more than "
                f"{cfg['regression_thresh']} below the best {best:.5f}")
    # Adjacent evaluations are ~164M tokens apart, so requiring rows[-2:] to span 1B tokens was a
    # rule that could never fire (Codex, blocker 5). Look back to the latest evaluation at or
    # before `now - plateau_tokens` -- about seven evals -- and compare against that.
    now = rows[-1]
    older = [r for r in rows if r["tokens"] <= now["tokens"] - cfg["plateau_tokens"]]
    if older:
        ref = older[-1]
        gain = now["screen3"] - ref["screen3"]
        if gain < cfg["plateau_gain"]:
            return (f"plateau: {gain:+.5f} over {(now['tokens']-ref['tokens'])/1e9:.2f}B tokens "
                    f"(step {ref['step']:,} -> {now['step']:,}), below {cfg['plateau_gain']}")
    return None


def reconcile_history():
    """Rebuild missing history rows from the `eval` blocks embedded in `step*.pt`.

    The checkpoint is written before the history line is appended, so a crash in between used to
    lose that evaluation permanently -- taking the first-eval gate and the regression/plateau
    windows with it. The checkpoints are the durable record; history is the index.
    """
    have = {r["step"] for r in read_history()}
    added = 0
    for p in sorted(CKPT.glob("step*.pt")):
        step = int(p.stem[4:])
        if step in have:
            continue
        try:
            rec = torch.load(p, map_location="cpu", weights_only=False).get("eval")
        except Exception:
            continue
        if rec:
            with open(HISTORY, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            added += 1
    if added:
        print(f"reconciled {added} evaluation(s) from checkpoints into history", flush=True)
    return added


def read_history():
    if not HISTORY.exists():
        return []
    out, seen = [], set()
    for line in HISTORY.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a torn final line must not break status or the kill rule
        if r["step"] in seen:
            continue
        seen.add(r["step"])
        out.append(r)
    return sorted(out, key=lambda r: r["step"])


def evaluate(model, cfg, step, cum, elapsed):
    per = eval9.eval_student(model, cfg["teacher"], comps=eval9.components("SCREEN3"))
    m = eval9.macros(per, cfg["teacher"])["SCREEN3"]
    ceil = guard9.registry()["ceilings"][cfg["teacher"]]["SCREEN3"]
    rec = {"step": step, "tokens": cum["tokens"], "examples": cum["examples"],
           "by_source": dict(cum["by_source"]), "elapsed_s": round(elapsed, 1),
           "screen3": m["macro"], "means": m["means"],
           "retention": round(m["macro"] / ceil, 4), "wall": time.time()}
    print(f"  EVAL step {step:,} ({cum['tokens']/1e9:.3f}B tokens): SCREEN-3 {rec['screen3']:.5f} "
          f"retention {rec['retention']}", flush=True)
    return rec


def status():
    rows = read_history()
    out = {"checkpoints": sorted(p.name for p in CKPT.glob("*.pt")) if CKPT.exists() else [],
           "evals": len(rows)}
    if rows:
        out["curve"] = [(r["step"], round(r["tokens"] / 1e9, 3), r["screen3"], r["retention"])
                        for r in rows[-12:]]
        out["best"] = max(rows, key=lambda r: r["screen3"])["screen3"]
        if len(rows) >= 2:
            a, b = rows[-2], rows[-1]
            dtok = (b["tokens"] - a["tokens"]) or 1
            # per MILLION tokens, which is the unit the dose is registered in
            out["slope_per_Mtok"] = round((b["screen3"] - a["screen3"]) / dtok * 1e6, 6)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "targets", "verify", "manifest", "train",
                                    "decay", "status"])
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--doc-limit", type=int, default=None)
    ap.add_argument("--student", default="bge-small-en-v1.5")
    ap.add_argument("--prompt-policy", choices=["a", "b"], default="b",
                    help="the screen-selected query prompt policy; baked into the tokenized corpus")
    ap.add_argument("--anneal-before-deadline", action="store_true",
                    help="enter the cooldown automatically when the session deadline no longer "
                         "fits stable + cooldown; the watchdog passes this so the horizon cannot "
                         "truncate the anneal")
    a = ap.parse_args()

    if a.cmd == "prepare":
        prepare(a.student, a.doc_limit, a.prompt_policy)
    elif a.cmd == "targets":
        targets()
    elif a.cmd == "manifest":
        man = build_manifest()
        MANIFEST.write_text(json.dumps(man, indent=2))
        print(f"wrote {MANIFEST} ({man['manifest_sha256'][:16]})")
    elif a.cmd == "verify":
        verify()
    elif a.cmd == "status":
        status()
    else:
        cfg = load_config()
        guard9.begin_run(cfg["run_id"])
        train(cfg, hours=a.hours, max_steps=a.max_steps, start_decay=(a.cmd == "decay"),
              anneal=a.anneal_before_deadline)


if __name__ == "__main__":
    main()
