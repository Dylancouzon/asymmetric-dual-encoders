"""M10 step 0b -- rate re-measure on REAL tokenized corpora.

PLANNING §11 measured the HARDWARE bound (random ids, no data path): 683 ex/s blended.
M9's realized pipeline ran at ~10% of its roof. This script measures the PIPELINE on the
real M9 corpora with the real memmapped teacher targets, so the box-vs-cloud build
decision in the M10.2 lock reads a realized number, not a bound.

Three paths, same model/optimizer/shapes:
  m9      -- M9's realized path: random batch, per-step numpy collate, length_chunks,
             memmap target fetch, blocking H2D.
  bucket  -- M10's plan: length-bucketed homogeneous batches (one chunk per step),
             otherwise identical.
  bucket+ -- bucket with a background prefetch thread and pinned memory.
  (+compile variants of the last two on the fixed buckets.)

Reports examples/s, padded tok/s, GPU busy fraction, peak GB and num_alloc_retries.
No protected surface is touched: only work/m9long/corpora and work/enc9 target caches.
"""
import os, sys, json, time, threading, queue
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "m9src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "m7src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORP = REPO / "work" / "m9long" / "corpora"
ENC9 = REPO / "work" / "enc9"
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


class Nano(nn.Module):
    """M10 recipe shape: bge-small, per-token Linear over pooled layers 12/8/4, mean pool."""
    def __init__(s, layers=(12, 8, 4)):
        super().__init__()
        s.bb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        s.layers = layers
        s.head = nn.Linear(384 * len(layers), 1024)

    def forward(s, ii, am):
        hs = s.bb(input_ids=ii, attention_mask=am, output_hidden_states=True).hidden_states
        v = s.head(torch.cat([hs[i] for i in s.layers], -1))
        m = am.unsqueeze(-1).to(v.dtype)
        v = (v * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(v.float(), dim=-1, eps=1e-12)


def load_corpus(name):
    d = CORP / name
    meta = json.loads((d / "meta.json").read_text())
    return np.load(d / "flat.npy", mmap_mode="r"), np.load(d / "offs.npy"), meta


class PadTo1024:
    """The `documents` corpus's teacher targets live in the M7 pool memmap, which is 768-d here
    (the pool's active encoder), while the head emits 1024. The bench needs the READ PATTERN and
    the token lengths, not the vector values, so rows are read from the real memmap and padded.
    That understates the target-fetch bytes by 25% -- a component the prefetch arms already show
    is not binding (GPU busy 0.99-1.00)."""

    def __init__(s, vecs, rows):
        s.v, s.rows = vecs, rows
        s.shape = (rows.size, 1024)

    def __getitem__(s, idx):
        a = np.asarray(s.v[s.rows[idx]], dtype=np.float32)
        return np.pad(a, ((0, 0), (0, 1024 - a.shape[1])))


def load_targets(name):
    """Plain memmap target caches; `documents` comes from the M7 pool via PadTo1024."""
    if name == "documents":
        sys.path.insert(0, str(REPO / "m7src"))
        import pool as poolmod
        _i, vecs, _m = poolmod.build()
        rows = np.load(CORP / "documents" / "pool_rows.npy")
        return PadTo1024(vecs, rows)
    p = ENC9 / f"m9long-{name}" / "vecs.npy"
    return np.load(p, mmap_mode="r")


def length_chunks(lens, cap_positions=16384):
    lens = np.asarray(lens)
    order = np.argsort(lens, kind="stable")
    chunks, cur, cur_max = [], [], 0
    for k in order:
        n = int(lens[k]); m = max(cur_max, n)
        if cur and (len(cur) + 1) * m > cap_positions:
            chunks.append(np.asarray(cur, dtype=np.int64)); cur, cur_max = [int(k)], n
        else:
            cur.append(int(k)); cur_max = m
    chunks.append(np.asarray(cur, dtype=np.int64))
    return chunks


def _ceil_bucket(n, ladder=(16, 32, 64, 128, 256, 512)):
    for b in ladder:
        if n <= b:
            return b
    return ladder[-1]


def collate(flat, offs, idx, pad_id, pin=False, fixed=False):
    lens = (offs[idx + 1] - offs[idx]).astype(np.int64)
    n = int(lens.max())
    if fixed:
        b = _ceil_bucket(n)
        assert b >= n, f"fixed bucket {b} would truncate a {n}-token row"
        n = b
    ii = np.full((len(idx), n), pad_id, dtype=np.int64)
    am = np.zeros((len(idx), n), dtype=np.int64)
    for k, (i, L) in enumerate(zip(idx, lens)):
        ii[k, :L] = flat[offs[i]:offs[i] + L]; am[k, :L] = 1
    t_ii, t_am = torch.from_numpy(ii), torch.from_numpy(am)
    if pin:
        t_ii, t_am = t_ii.pin_memory(), t_am.pin_memory()
    return t_ii, t_am, int(lens.sum()), int(len(idx) * n)


def step(model, opt, batches, scale_by):
    """One optimizer step over a list of (ii, am, tgt) already on device."""
    opt.zero_grad(set_to_none=True)
    for ii, am, t in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(ii, am)
        (((v - t) ** 2).sum(-1).mean() * (ii.shape[0] / scale_by)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def run(label, model, opt, gen, steps, warm, bs, reset=None):
    """`reset()` is called before the arm so every arm replays the SAME batch sequence.

    Without it one shared RNG advanced across arms and each saw a different sample of buckets;
    the document arms then differed by ~4% in padded tokens per example, which is most of the
    apparent `torch.compile` gain (Codex, 2026-09-04). Arms are only comparable batch-for-batch."""
    if reset is not None:
        reset()
    torch.cuda.reset_peak_memory_stats()
    r0 = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    ntok = npad = 0
    step_t = 0.0
    for it in range(warm + steps):
        if it == warm:
            torch.cuda.synchronize(); t0 = time.time(); ntok = npad = 0; step_t = 0.0
        batches, tk, pd = gen()
        if it >= warm:
            ntok += tk; npad += pd
            torch.cuda.synchronize(); g0 = time.time()
        step(model, opt, batches, bs)
        if it >= warm:
            torch.cuda.synchronize(); step_t += time.time() - g0
    torch.cuda.synchronize(); dt = time.time() - t0
    sps = steps / dt
    out = dict(label=label, steps_per_s=round(sps, 2), examples_per_s=round(sps * bs, 1),
               real_tok_per_s=round(ntok / dt), padded_tok_per_s=round(npad / dt),
               pad_waste=round(1 - ntok / max(npad, 1), 3),
               step_wall_frac=round(step_t / dt, 3),
               peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               alloc_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0) - r0,
               gpu_hours_200m=round(200e6 / (sps * bs) / 3600, 1))
    print(f"{label:38s} {out['examples_per_s']:8.1f} ex/s  {out['padded_tok_per_s']:9d} ptok/s "
          f" pad {out['pad_waste']:.2f}  step {out['step_wall_frac']:.2f}  "
          f"peak {out['peak_gb']:5.2f}GB  retries {out['alloc_retries']:3d}  "
          f"200M->{out['gpu_hours_200m']:6.1f}h", flush=True)
    return out


def main():
    steps = int(os.environ.get("STEPS", 200)); warm = int(os.environ.get("WARM", 20))
    bs = int(os.environ.get("BS", 32))
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    pad_id = tok.pad_token_id
    model = Nano().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, fused=True)
    res, meta = {}, {}

    which = os.environ.get("CORPORA", "nqopen,triviaqa,pseudoq,documents").split(",")
    for cname in which:
        flat, offs, m = load_corpus(cname)
        tgt = load_targets(cname)
        n = offs.size - 1
        meta[cname] = dict(n=n, mean_tokens=m["mean_tokens"], role=m["role"],
                           target_shape=list(tgt.shape))
        assert tgt.shape[0] == n, f"{cname}: targets {tgt.shape[0]} vs {n}"

        # Every arm consumes this identical, pre-drawn sequence of batch draws, replayed from
        # position 0 at the start of each arm -- see `run(reset=...)`.
        n_draw = warm + steps + 8                      # +8 covers the prefetch queue's lookahead
        draws_rand = np.random.default_rng(1).integers(0, n, (n_draw, bs))
        cursor = {"i": 0}
        def _next():
            i = cursor["i"]; cursor["i"] = i + 1
            return i % n_draw
        def _reset():
            cursor["i"] = 0

        # --- path A: M9 realized -- random batch, length_chunks, blocking H2D
        def gen_m9(flat=flat, offs=offs, tgt=tgt, n=n):
            idx = draws_rand[_next()]
            batches, tk, pd = [], 0, 0
            for ch in length_chunks((offs[idx + 1] - offs[idx])):
                sub = idx[ch]
                ii, am, a, b = collate(flat, offs, sub, pad_id)
                t = torch.from_numpy(np.asarray(tgt[sub], dtype=np.float32)).to(DEV)
                batches.append((ii.to(DEV), am.to(DEV), t)); tk += a; pd += b
            return batches, tk, pd
        res[f"{cname}/m9_path"] = run(f"{cname} m9 path (random batch)", model, opt,
                                      gen_m9, steps, warm, bs, reset=_reset)

        # --- path B: length-bucketed -- one homogeneous chunk per step
        order = np.argsort((offs[1:] - offs[:-1]), kind="stable")
        nb = len(order) // bs
        draws_buck = np.random.default_rng(2).integers(0, nb, n_draw)
        def gen_bucket(order=order, nb=nb, flat=flat, offs=offs, tgt=tgt, pin=False,
                       fixed=False):
            b = int(draws_buck[_next()])
            sub = np.sort(order[b * bs:(b + 1) * bs])
            ii, am, a, p = collate(flat, offs, sub, pad_id, pin=pin, fixed=fixed)
            t = torch.from_numpy(np.asarray(tgt[sub], dtype=np.float32))
            if pin: t = t.pin_memory()
            return [(ii.to(DEV, non_blocking=pin), am.to(DEV, non_blocking=pin),
                     t.to(DEV, non_blocking=pin))], a, p
        res[f"{cname}/bucket"] = run(f"{cname} length-bucketed", model, opt,
                                     gen_bucket, steps, warm, bs, reset=_reset)

        # --- path C: bucketed + background prefetch thread + pinned memory
        _reset()                                   # workers pre-fill; reset before they start
        q = queue.Queue(maxsize=4); stop = threading.Event()
        def worker():
            while not stop.is_set():
                try: q.put(gen_bucket(pin=True), timeout=1)
                except queue.Full: pass
        th = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
        [t.start() for t in th]
        res[f"{cname}/bucket_prefetch"] = run(f"{cname} bucketed + prefetch(2) + pinned",
                                              model, opt, lambda: q.get(), steps, warm, bs)
        stop.set(); [t.join(timeout=2) for t in th]

        # --- path D: fixed power-of-two buckets + torch.compile (few shapes to compile)
        if os.environ.get("COMPILE", "1") == "1":
            _reset()
            q2 = queue.Queue(maxsize=4); stop2 = threading.Event()
            def worker2():
                while not stop2.is_set():
                    try: q2.put(gen_bucket(pin=True, fixed=True), timeout=1)
                    except queue.Full: pass
            th2 = [threading.Thread(target=worker2, daemon=True) for _ in range(2)]
            [t.start() for t in th2]
            res[f"{cname}/fixed_bucket_prefetch"] = run(
                f"{cname} fixed buckets + prefetch", model, opt, lambda: q2.get(),
                steps, warm, bs)
            cmodel = torch.compile(model, dynamic=False)
            # Explicitly compile every fixed bucket shape before timing. 30 random warm steps
            # do not guarantee coverage, and a shape compiling inside the timed window would
            # deflate the arm (Codex, 2026-09-04).
            for L in (16, 32, 64, 128, 256, 512):
                ii = torch.zeros(bs, L, dtype=torch.long, device=DEV)
                am = torch.ones(bs, L, dtype=torch.long, device=DEV)
                t = F.normalize(torch.randn(bs, 1024, device=DEV), dim=-1)
                try:
                    step(cmodel, opt, [(ii, am, t)], bs)
                except Exception as e:
                    print(f"  warm shape {L} failed: {e!r}"[:200], flush=True)
            print(f"  compiled {6} fixed shapes", flush=True)
            try:
                res[f"{cname}/fixed_bucket_compile"] = run(
                    f"{cname} fixed buckets + compile", cmodel, opt, lambda: q2.get(),
                    steps, max(warm, 30), bs)
            except Exception as e:                      # compile is an optimisation, not a gate
                res[f"{cname}/fixed_bucket_compile"] = {"error": repr(e)[:300]}
                print(f"  compile FAILED: {e!r}"[:300], flush=True)
            stop2.set(); [t.join(timeout=2) for t in th2]

    out = dict(when=time.strftime("%Y-%m-%dT%H:%M:%S%z"), bs=bs, steps=steps,
               device=torch.cuda.get_device_name(0), torch=torch.__version__,
               corpora=meta, results=res)
    p = REPO / "results" / ("m10_rate_bench_real_box.json" if len(which) > 1
                            else f"m10_rate_bench_real_{which[0]}.json")
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
