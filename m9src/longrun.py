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
MANIFEST = RUN / "manifest.json"
CONFIG = RUN / "config.json"
LOCKFILE = RUN / "trainer.lock"

QUERY_SOURCES = ("queries_pair", "nqopen", "triviaqa")
SPAN_SOURCES = ("pseudoq",)
DOC_SOURCES = ("documents",)


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


def prepare(student_key, doc_limit=None):
    RUN.mkdir(parents=True, exist_ok=True)
    tok = nano.Nano(student_key).tok
    tpl = guard9.registry()["templates"]
    t0 = time.time()

    q = json.loads((WORK / "m9_screen_queries.json").read_text())
    _save_corpus("queries_pair",
                 *_pack_streaming(tok, q, tpl["query_policy_b_student"], label="queries_pair"),
                 {"role": "query", "prefix": tpl["query_policy_b_student"],
                  "source": "work/m9_screen_queries.json"})
    for name, (texts, row) in extra_texts().items():
        role = "query" if name in QUERY_SOURCES else "doc_span"
        pre = tpl["query_policy_b_student"] if role == "query" else tpl["doc_student"]
        _save_corpus(name, *_pack_streaming(tok, texts, pre, label=name),
                     {"role": role, "prefix": pre, "what": row["what"],
                      "kept_index_sha256": row["kept_index_sha256"]})
        del texts

    r = guard9.registry()
    rows, dmeta = m9data.doc_pool_rows(doc_limit or r["data"]["n_eligible_doc_rows"],
                                       r["data"]["doc_candidates_seed"])
    print(f"  documents: reading {rows.size:,} texts from the pool stores...", flush=True)
    texts = m9data.row_texts(rows)
    _save_corpus("documents", *_pack_streaming(tok, texts, tpl["doc_student"], label="documents"),
                 {"role": "doc", "prefix": tpl["doc_student"], **dmeta})
    np.save(TOKENS / "documents" / "pool_rows.npy", rows)
    del texts
    print(f"prepare done in {time.time()-t0:.0f}s -- now run `targets`, then `verify`", flush=True)


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


def load_config():
    assert CONFIG.exists(), f"{CONFIG} missing -- M9.2 writes it when the recipe is locked"
    cfg = json.loads(CONFIG.read_text())
    cfg["_hash"] = canon({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


def train(cfg, hours=None, max_steps=None, start_decay=False, device="cuda"):
    RUN.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    if LOCKFILE.exists():
        raise SystemExit(f"{LOCKFILE} exists -- another trainer may be running. Remove it only if "
                         f"you are certain no process is writing to {CKPT}.")
    LOCKFILE.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    try:
        return _train(cfg, hours, max_steps, start_decay, device)
    finally:
        LOCKFILE.unlink(missing_ok=True)


def _train(cfg, hours, max_steps, start_decay, device):
    man = verify(strict=True)
    names = [n for n in QUERY_SOURCES + SPAN_SOURCES + DOC_SOURCES if cfg["shares"].get(_grp(n))]
    corpora = {n: load_corpus(n) for n in names}
    tgt = Targets()
    tgt.check(corpora)

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

    t0, sess_tok, sess_ex, loss_acc, nlog = time.time(), 0, 0, 0.0, 0
    tput = []
    deadline = t0 + hours * 3600 if hours else None
    stop_reason = None
    model.train()

    while stop_reason is None:
        if max_steps and step >= max_steps:
            stop_reason = "max_steps"
            break
        if phase["name"] == "decay" and step >= phase["decay_from"] + phase["decay_steps"]:
            stop_reason = "cooldown complete"
            break
        if deadline and time.time() > deadline:
            stop_reason = "session wall-clock budget"
            break
        if (CKPT / "STOP").exists():
            stop_reason = "STOP file"
            break
        if phase["name"] == "stable" and cum["tokens"] >= cfg["stable_token_cap"]:
            stop_reason = f"stable-phase token cap {cfg['stable_token_cap']:,}"
            break

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

        if step % cfg["log_every"] == 0:
            el = time.time() - t0
            rate = sess_tok / el
            tput.append(rate)
            print(f"  step {step:,} loss {loss_acc/nlog:.5f} lr {lr:.2e} "
                  f"{rate:,.0f} tok/s | cum {cum['tokens']/1e9:.3f}B tokens "
                  f"({cum['tokens']/cfg['stable_token_cap']:.1%} of cap)", flush=True)
            loss_acc, nlog = 0.0, 0
            if len(tput) > 12 and rate < cfg["throughput_floor_frac"] * float(np.median(tput[:8])):
                stop_reason = (f"throughput collapse: {rate:,.0f} tok/s against an early median "
                               f"of {np.median(tput[:8]):,.0f}")
                break

        if step % cfg["eval_every"] == 0:
            rec = evaluate(model, cfg, step, cum, time.time() - t0)
            model.train()
            torch.cuda.empty_cache()      # the screen measured 1,990 -> 786 ex/s without this
            blob = _blob(model, opt, step, streams, cfg, cum, phase, best, man)
            save_ckpt(CKPT / f"step{step}.pt", {**blob, "eval": rec})
            save_ckpt(CKPT / "last.pt", blob)
            with open(HISTORY, "a") as fh:
                fh.write(json.dumps(rec) + "\n")     # after the checkpoint, so replay can dedupe
            stop_reason = stop_reason or check_kill(rec, cfg)
            if best is None or rec["screen3"] > best["screen3"]:
                best = {"step": step, "screen3": rec["screen3"], "tokens": cum["tokens"]}
        elif step % cfg["ckpt_every"] == 0:
            save_ckpt(CKPT / "last.pt", _blob(model, opt, step, streams, cfg, cum, phase, best, man))

    save_ckpt(CKPT / "last.pt", _blob(model, opt, step, streams, cfg, cum, phase, best, man))
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
    span = tail[-1]["tokens"] - tail[0]["tokens"]
    if span >= cfg["plateau_tokens"]:
        gain = tail[-1]["screen3"] - tail[0]["screen3"]
        if gain < cfg["plateau_gain"]:
            return (f"plateau: {gain:+.5f} over {span/1e9:.2f}B tokens, below "
                    f"{cfg['plateau_gain']}")
    return None


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
    a = ap.parse_args()

    if a.cmd == "prepare":
        prepare(a.student, a.doc_limit)
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
        train(cfg, hours=a.hours, max_steps=a.max_steps, start_decay=(a.cmd == "decay"))


if __name__ == "__main__":
    main()
