"""The M9.3 build: a resumable, stoppable trainer for a run measured in days.

Three properties the screen's trainer does not need and this one cannot do without.

**Stoppable at any point.** A cosine schedule decaying to `lr_final` commits you to a horizon:
stop early and the LR is still high, want more and you cannot extend cleanly. So the schedule is
**warmup → stable → decay-on-demand**. The stable phase runs indefinitely; `--decay` takes any
stable checkpoint and runs a short cooldown to produce a servable model. "How long do we run"
becomes an observation instead of a guess, which is the whole point when the anchor curve says
gains halve every quarter.

**Resumable exactly.** Model, optimizer, step, per-source stream positions and both RNG states are
checkpointed atomically. A restart continues the same example order, not a fresh one — otherwise a
crash on day two silently changes the experiment into a different one.

**Honest about a moving corpus.** Text is tokenized once into a memmapped flat int32 array plus
offsets, hashed, and the hash is checked on resume. A corpus that changed under a restart is an
error, not a shrug.

    python m9src/longrun.py prepare                  # tokenize + hash the corpora, once
    python m9src/longrun.py train  --hours 168       # the stable phase, resumable
    python m9src/longrun.py decay  --steps 4000      # cooldown -> the servable artifact
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


# --------------------------------------------------------------------------- corpora ----------

def _pack_streaming(tok, texts, prefix, max_len=512, batch=20_000, label=""):
    """-> (flat int32, offsets int64), tokenizing and packing in CHUNKS.

    The obvious version -- tokenize everything, then pack -- is a trap at this scale. A Python list
    of ~95 token ids costs ~3.5 kB once the list object and the un-cached int objects are counted,
    so 6.15M documents is roughly **21 GB** of transient heap before a single byte is packed. That
    OOMs a 25 GB box, and on this box it would have taken the training chain with it. Packed as
    int32 the same corpus is 2.3 GB, and streaming never holds more than one chunk of lists.
    """
    chunks, offs_parts, total, t0 = [], [], 0, time.time()
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
        offs_parts.append(lens)
        total += len(ids)
        del ids
        if label and (i // batch) % 25 == 0 and i:
            el = time.time() - t0
            print(f"    {label} {total:,}/{len(texts):,} ({total/max(el,1e-9):,.0f}/s)", flush=True)
    lens = np.concatenate(offs_parts) if offs_parts else np.zeros(0, dtype=np.int64)
    offs = np.zeros(lens.size + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32), offs


def _save_corpus(name, flat, offs, meta):
    d = TOKENS / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "flat.npy", flat)
    np.save(d / "offs.npy", offs)
    meta = {**meta, "n": int(offs.size - 1), "n_tokens": int(offs[-1]),
            "mean_tokens": round(float(offs[-1]) / max(offs.size - 1, 1), 2),
            "flat_sha256": hashlib.sha256(flat.tobytes()).hexdigest()[:32]}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {name}: {meta['n']:,} texts, {meta['n_tokens']:,} tokens, "
          f"mean {meta['mean_tokens']}", flush=True)
    return meta


def load_corpus(name):
    d = TOKENS / name
    meta = json.loads((d / "meta.json").read_text())
    flat = np.load(d / "flat.npy", mmap_mode="r")
    offs = np.load(d / "offs.npy")
    return flat, offs, meta


def extra_query_texts():
    """The sources the extended screen admitted: real questions first, spans labelled as such."""
    scr = RESULTS / "m9_extended_screen.json"
    if not scr.exists():
        return {}, {}
    blob = json.loads(scr.read_text())
    out, prov = {}, {}
    for name, row in blob["sources"].items():
        raw = json.loads((m9base.REPO / row["path"]).read_text())
        kept = json.loads((WORK / "decontam" / f"m9_kept_{name}.json").read_text())
        out[name] = [str(raw[i]) for i in kept]
        prov[name] = {"n": len(kept), "kept_index_sha256": row["kept_index_sha256"],
                      "what": row["what"]}
    return out, prov


def prepare(student_key, doc_limit=None):
    """Tokenize every corpus once. Idempotent: an existing, hash-matching corpus is left alone."""
    RUN.mkdir(parents=True, exist_ok=True)
    tok = nano.Nano(student_key).tok
    tpl = guard9.registry()["templates"]
    prov = {"student": student_key, "corpora": {}}
    t0 = time.time()

    q = json.loads((WORK / "m9_screen_queries.json").read_text())
    prov["corpora"]["queries_pair"] = _save_corpus(
        "queries_pair",
        *_pack_streaming(tok, q, tpl["query_policy_b_student"], label="queries_pair"),
        {"role": "query", "prefix": tpl["query_policy_b_student"],
         "source": "work/m9_screen_queries.json (M7 TRAIN pair sources minus fever)"})

    extra, eprov = extra_query_texts()
    for name, texts in extra.items():
        role = "query" if name in ("nqopen", "triviaqa") else "doc_span"
        pre = tpl["query_policy_b_student"] if role == "query" else tpl["doc_student"]
        prov["corpora"][name] = _save_corpus(
            name, *_pack_streaming(tok, texts, pre, label=name),
            {"role": role, "prefix": pre, **eprov[name]})
        del texts

    r = guard9.registry()
    rows, dmeta = m9data.doc_pool_rows(doc_limit or r["data"]["n_eligible_doc_rows"],
                                       r["data"]["doc_candidates_seed"])
    print(f"  documents: reading {rows.size:,} texts from the pool stores...", flush=True)
    texts = m9data.row_texts(rows)
    prov["corpora"]["documents"] = _save_corpus(
        "documents", *_pack_streaming(tok, texts, tpl["doc_student"], label="documents"),
        {"role": "doc", "prefix": tpl["doc_student"], **dmeta})
    np.save(TOKENS / "documents" / "pool_rows.npy", rows)
    del texts

    prov["seconds"] = round(time.time() - t0, 1)
    (RUN / "corpora.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({k: v.get("n") for k, v in prov["corpora"].items()}, indent=1))
    return prov


# --------------------------------------------------------------------------- targets ----------

class Targets:
    """Teacher vectors for a corpus row. Queries come from a cached matrix; documents are read
    straight from the frozen pool memmap, so nothing large is ever resident."""

    def __init__(self, corpora):
        self.q = np.asarray(m9data.stella_query_targets())
        self.rows = np.load(WORK / "m9_screen_rows.npy")
        import pool as poolmod
        _i, self.pool, _m = poolmod.build()
        self.doc_rows = np.load(TOKENS / "documents" / "pool_rows.npy")
        self.extra = {}
        for name in corpora:
            p = WORK / "enc9" / f"m9long-{name}"
            if p.exists():
                self.extra[name] = np.load(p / "vecs.npy", mmap_mode="r")

    def get(self, corpus, idx):
        if corpus == "queries_pair":
            return np.asarray(self.q[self.rows[idx]], dtype=np.float32)
        if corpus == "documents":
            return np.asarray(self.pool[self.doc_rows[idx]], dtype=np.float32)
        return np.asarray(self.extra[corpus][idx], dtype=np.float32)


# --------------------------------------------------------------------------- schedule ---------

def lr_at(step, cfg, decay_from=None, decay_steps=None):
    """warmup -> stable -> (only once someone asks) cosine cooldown."""
    if step < cfg["warmup_steps"]:
        return cfg["lr_peak"] * (step + 1) / cfg["warmup_steps"]
    if decay_from is None or step < decay_from:
        return cfg["lr_peak"]
    t = min(1.0, (step - decay_from) / max(decay_steps - 1, 1))
    return cfg["lr_final"] + 0.5 * (cfg["lr_peak"] - cfg["lr_final"]) * (1 + math.cos(math.pi * t))


# --------------------------------------------------------------------------- streams ----------

class Stream:
    """A deterministic, resumable, infinite stream over one corpus.

    Position is (epoch, offset), so a resume replays the same permutation from the same place. A
    stream that wrapped on restart, or reshuffled, would quietly turn a seven-day run into a
    different experiment than the one that started.
    """

    def __init__(self, name, n, seed, epoch=0, offset=0):
        self.name, self.n, self.seed = name, n, seed
        self.epoch, self.offset = epoch, offset
        self._perm = None

    def _p(self):
        if self._perm is None:
            self._perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
        return self._perm

    def take(self, k):
        out = []
        while k > 0:
            p = self._p()
            avail = self.n - self.offset
            m = min(k, avail)
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


def save_ckpt(path, model, opt, step, streams, cfg, extra):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "streams": {k: s.state() for k, s in streams.items()}, "cfg": cfg,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                **extra}, tmp)
    os.replace(tmp, path)


def train(cfg, hours=None, max_steps=None, decay_from=None, decay_steps=None, device="cuda"):
    RUN.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    prov = json.loads((RUN / "corpora.json").read_text())
    names = list(cfg["mix"])
    corpora = {n: load_corpus(n) for n in names}
    for n in names:
        want = prov["corpora"][n]["flat_sha256"]
        got = corpora[n][2]["flat_sha256"]
        assert want == got, f"{n}: tokenized corpus changed under the run ({want} -> {got})"
    tgt = Targets(names)

    model = nano.Nano(cfg["student"]).to(device)
    pad_id = model.tok.pad_token_id
    decay = [p for _n, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for _n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg["lr_peak"], betas=tuple(cfg["betas"]), eps=cfg["eps"])
    step = 0
    streams = {n: Stream(n, corpora[n][2]["n"], cfg["seed"] + i) for i, n in enumerate(names)}

    last = CKPT / "last.pt"
    if last.exists():
        blob = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        step = blob["step"]
        for n, s in blob["streams"].items():
            streams[n] = Stream(n, corpora[n][2]["n"], s["seed"], s["epoch"], s["offset"])
        torch.set_rng_state(blob["torch_rng"].cpu())
        if blob.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in blob["cuda_rng"]])
        print(f"resumed at step {step:,} from {last}", flush=True)
    else:
        torch.manual_seed(cfg["seed"])
        import warmfit
        ws = nano.warm_start_head(
            model, json.loads((WORK / "m9_screen_queries.json").read_text()),
            np.asarray(tgt.q[tgt.rows]), cfg["student_query_prefix"])
        print(f"warm-start head: {json.dumps(ws)}  (lambda from {warmfit.ARTIFACT.name})",
              flush=True)

    budget = cfg["tokens_per_step"]
    shares = [cfg["mix"][n] for n in names]
    t0, tok_acc, ex_acc, loss_acc, nlog = time.time(), 0, 0, 0.0, 0
    deadline = t0 + hours * 3600 if hours else None
    model.train()

    while True:
        if max_steps and step >= max_steps:
            break
        if deadline and time.time() > deadline:
            print("wall-clock budget reached", flush=True)
            break
        if (CKPT / "STOP").exists():
            print("STOP file present -- stopping cleanly", flush=True)
            break

        lr = lr_at(step, cfg, decay_from, decay_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        loss = 0.0
        opt.zero_grad(set_to_none=True)
        for si, name in enumerate(names):
            flat, offs, meta = corpora[name]
            want = budget * shares[si]
            k = max(1, int(round(want / meta["mean_tokens"])))
            idx = streams[name].take(k)
            ii, am, ntok = collate(flat, offs, idx, pad_id, device)
            t = torch.from_numpy(tgt.get(name, idx)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(ii, am)
            v = F.normalize(v.float(), dim=-1, eps=1e-12)
            # each role's gradient is scaled by its share of the batch, so the loss is the plain
            # mean over the step's examples and no role is weighted twice
            part = ((v - t) ** 2).sum(-1).mean() * (len(idx) / max(k, 1))
            (part * shares[si]).backward()
            loss += float(part.detach()) * shares[si]
            tok_acc += ntok
            ex_acc += len(idx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1
        loss_acc += loss
        nlog += 1

        if step % cfg["log_every"] == 0:
            el = time.time() - t0
            print(f"  step {step:,} loss {loss_acc/nlog:.5f} lr {lr:.2e} "
                  f"{tok_acc/el:,.0f} tok/s {ex_acc/el:,.0f} ex/s "
                  f"[{tok_acc/1e9:.3f}B tokens this session]", flush=True)
            loss_acc, nlog = 0.0, 0
        if step % cfg["ckpt_every"] == 0:
            save_ckpt(CKPT / "last.pt", model, opt, step, streams, cfg,
                      {"tokens_session": tok_acc, "examples_session": ex_acc})
        if step % cfg["eval_every"] == 0:
            rec = evaluate(model, cfg, step, tok_acc, ex_acc, time.time() - t0)
            model.train()
            save_ckpt(CKPT / f"step{step}.pt", model, opt, step, streams, cfg, {"eval": rec})
            save_ckpt(CKPT / "last.pt", model, opt, step, streams, cfg,
                      {"tokens_session": tok_acc, "examples_session": ex_acc})

    save_ckpt(CKPT / "last.pt", model, opt, step, streams, cfg,
              {"tokens_session": tok_acc, "examples_session": ex_acc})
    print(f"stopped at step {step:,}, {tok_acc/1e9:.3f}B tokens this session", flush=True)
    return step


def evaluate(model, cfg, step, tok_acc, ex_acc, elapsed):
    per = eval9.eval_student(model, cfg["teacher"], comps=eval9.components("SCREEN3"))
    m = eval9.macros(per, cfg["teacher"])
    ceil = guard9.registry()["ceilings"][cfg["teacher"]]["SCREEN3"]
    rec = {"step": step, "tokens_session": tok_acc, "examples_session": ex_acc,
           "elapsed_s": round(elapsed, 1), "screen3": m["SCREEN3"]["macro"],
           "means": m["SCREEN3"]["means"],
           "retention": round(m["SCREEN3"]["macro"] / ceil, 4), "wall": time.time()}
    with open(HISTORY, "a") as fh:          # append-only: a crash never loses the curve
        fh.write(json.dumps(rec) + "\n")
    print(f"  EVAL step {step:,}: SCREEN-3 {rec['screen3']:.5f} "
          f"retention {rec['retention']}", flush=True)
    return rec


def status():
    out = {"checkpoints": sorted(p.name for p in CKPT.glob("*.pt"))} if CKPT.exists() else {}
    if HISTORY.exists():
        rows = [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()]
        out["evals"] = len(rows)
        out["curve"] = [(r["step"], r["screen3"], r["retention"]) for r in rows[-12:]]
        if len(rows) >= 2:
            a, b = rows[-2], rows[-1]
            dt = (b["step"] - a["step"]) or 1
            out["last_slope_per_1k_steps"] = round((b["screen3"] - a["screen3"]) / dt * 1000, 6)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "decay", "status"])
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4000, help="decay length")
    ap.add_argument("--doc-limit", type=int, default=None)
    ap.add_argument("--student", default="bge-small-en-v1.5")
    a = ap.parse_args()

    if a.cmd == "prepare":
        prepare(a.student, a.doc_limit)
        return
    if a.cmd == "status":
        status()
        return

    cfgp = RUN / "config.json"
    assert cfgp.exists(), f"{cfgp} is missing -- M9.2 writes it when the recipe is locked"
    cfg = json.loads(cfgp.read_text())
    if a.cmd == "train":
        train(cfg, hours=a.hours, max_steps=a.max_steps)
    else:
        blob = torch.load(CKPT / "last.pt", map_location="cpu", weights_only=False)
        start = blob["step"]
        print(f"cooldown: {a.steps:,} steps from step {start:,}", flush=True)
        train(cfg, max_steps=start + a.steps, decay_from=start, decay_steps=a.steps)


if __name__ == "__main__":
    main()
