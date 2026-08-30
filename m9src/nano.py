"""nano: the distilled query tower, and its phase-1 trainer.

Architecture (m9/registry.json `objective`): HF backbone -> mean pooling over the attention mask
-> Linear(hidden, 1024, bias=True) -> L2 normalize. Loss is plain squared L2 against the frozen
teacher vector, computed in fp32 (LEAF, arXiv 2509.12539; its ablation rejected auxiliary terms,
and MSE+cosine is affine-redundant under normalized outputs).

Every knob a rule reads comes from `m9/registry.json`. Nothing here restates a registry constant
(m8/CODEMAP.md pitfall 12).
"""
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import m9base
from m9base import WORK, RESULTS

STUDENTS = {
    "bge-small-en-v1.5": {"repo": "BAAI/bge-small-en-v1.5",
                          "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "params": 33.4e6},
    "all-MiniLM-L6-v2": {"repo": "sentence-transformers/all-MiniLM-L6-v2",
                         "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "params": 22.7e6},
}


def registry():
    return json.loads((m9base.M9 / "registry.json").read_text())


class Nano(nn.Module):
    def __init__(self, student_key, out_dim=1024):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        spec = STUDENTS[student_key]
        self.key = student_key
        self.tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        self.backbone = AutoModel.from_pretrained(spec["repo"], revision=spec["revision"])
        self.head = nn.Linear(self.backbone.config.hidden_size, out_dim, bias=True)
        self.out_dim = out_dim
        self.max_seq = registry()["dose"]["max_seq"]

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(v)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.inference_mode()
    def encode_queries(self, texts, batch_size=256, prefix=""):
        """-> (n, out_dim) fp32 unit-norm, in input order. This is the serving path."""
        self.eval()
        dev = next(self.parameters()).device
        out = np.empty((len(texts), self.out_dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            b = self.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
                         max_length=self.max_seq, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                v = self(b["input_ids"], b["attention_mask"])
            v = F.normalize(v.float(), dim=-1, eps=1e-12)
            out[sel] = v.cpu().numpy()
        return out


@torch.inference_mode()
def _pooled(model, texts, prefix, batch_size=256):
    dev = next(model.parameters()).device
    h = model.backbone.config.hidden_size
    out = np.empty((len(texts), h), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        b = model.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
                      max_length=model.max_seq, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hs = model.backbone(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(hs.dtype)
        out[sel] = ((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu().numpy()
    return out


def warm_start_head(model, texts, targets, prefix=""):
    """Replace the head's random init with the closed-form ridge solution from the FROZEN
    backbone's mean-pooled outputs to the teacher targets (m9/LEDGER.md §3.2a).

    Measured at M9.0 before any arm ran (`results/m9_head_probe.json`): this alone reaches 50.8%
    of the teacher's SCREEN-3 ceiling, against 12.4% for a random head after 2,000 trained steps.
    At ~1% of LEAF's dose a random head spends a large share of the whole budget re-deriving a
    linear map that has a closed form. Identical treatment for every arm, so no contrast moves.
    """
    import warmfit
    r = registry()["warm_start"]
    lam = warmfit.selected_lambda()          # refuses unless chosen on a training-only holdout
    rng = np.random.default_rng(r["seed"])
    sel = np.sort(rng.choice(len(texts), size=min(r["n_fit"], len(texts)), replace=False))
    X = _pooled(model, [texts[i] for i in sel], prefix)
    Y = np.asarray(targets[sel], dtype=np.float32)
    Xc = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    G = Xc.T @ Xc
    scale = float(np.trace(G) / G.shape[0])
    A = np.linalg.solve(G + lam * scale * np.eye(G.shape[0], dtype=np.float32), Xc.T @ Y)
    P = Xc @ A
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    resid = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    nonpad = int(sum(len(x) for x in pretokenize(model.tok, [texts[i] for i in sel],
                                                 model.max_seq, verbose=False)))
    dev = next(model.parameters()).device
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(np.ascontiguousarray(A[:-1].T)).to(dev))
        model.head.bias.copy_(torch.from_numpy(np.ascontiguousarray(A[-1])).to(dev))
    return {"n_fit": int(sel.size), "lambda": lam, "seed": r["seed"],
            "train_objective": round(resid, 5), "prefix": prefix,
            # STAGE-0 DOSE: this phase is 60,000 supervised backbone forwards plus a solve, and it
            # is NOT part of the registered SGD dose. Reported so the retention-vs-dose curve can
            # be read against both the SGD budget and total compute (Codex pass 3, MAJOR).
            "stage0_examples": int(sel.size), "stage0_nonpad_tokens": nonpad,
            "stage0_teacher_target_accesses": int(sel.size),
            "stage0_seconds": None}


def pretokenize(tok, texts, max_len, verbose=True, label=""):
    """-> list[list[int]]. Done once per arm; per-step tokenization would cost more than the
    forward pass at these sequence lengths."""
    ids, t0 = [], time.time()
    B = 2000
    for i in range(0, len(texts), B):
        ids.extend(tok(texts[i:i + B], truncation=True, max_length=max_len,
                       add_special_tokens=True)["input_ids"])
        if verbose and (i // B) % 25 == 0 and i:
            el = time.time() - t0
            print(f"  tokenize {label} {i:,}/{len(texts):,} ({i/max(el,1e-9):.0f}/s)", flush=True)
    return ids


def collate(id_lists, rows, pad_id, device):
    batch = [id_lists[r] for r in rows]
    n = max(len(b) for b in batch)
    ii = np.full((len(batch), n), pad_id, dtype=np.int64)
    am = np.zeros((len(batch), n), dtype=np.int64)
    for k, b in enumerate(batch):
        ii[k, :len(b)] = b
        am[k, :len(b)] = 1
    return (torch.from_numpy(ii).to(device, non_blocking=True),
            torch.from_numpy(am).to(device, non_blocking=True))


def epoch_order(n, epochs, seed, partial_last=None):
    """A fresh permutation per epoch from one seeded generator; `partial_last` truncates the final
    epoch. This is the locked epoch order (m9/registry.json dose.epoch_order)."""
    rng = np.random.default_rng(seed)
    out = []
    for e in range(epochs):
        p = rng.permutation(n)
        if partial_last is not None and e == epochs - 1:
            p = p[:partial_last]
        out.append(p)
    return np.concatenate(out)


def fixed_batches(order, bs):
    """-> (flat int32 order, int32 offsets). The final batch keeps its natural, smaller size."""
    order = np.asarray(order, dtype=np.int32)
    offs = np.arange(0, len(order) + bs, bs, dtype=np.int64)
    offs = offs[offs <= len(order)]
    if offs[-1] != len(order):
        offs = np.append(offs, len(order))
    return order, offs


def token_batches(streams, steps, total_tokens, shares):
    """Token-budgeted batching (m9/LEDGER.md §3.1).

    `streams` is [(index array, token-length array indexed by those indices)] per role and
    `shares` the fraction of the total non-pad token budget each role must carry. The budget is
    tracked CUMULATIVELY, not per step: each role fills until its running total reaches its share
    of the tokens due by the end of this step, so the per-step rounding error cannot accumulate
    across 30,349 steps (Codex pass 2, BLOCKER-2).

    Guarantees, asserted rather than hoped for: exactly `steps` non-empty batches; no stream runs
    dry; realized tokens within one example of each role's target.
    """
    assert abs(sum(shares) - 1.0) < 1e-12
    flat, offs, pos = [], [0], [0] * len(streams)
    got = [0.0] * len(streams)
    for st in range(1, steps + 1):
        due = [total_tokens * sh * st / steps for sh in shares]
        for si, (idx, tlen) in enumerate(streams):
            while got[si] < due[si]:
                if pos[si] >= len(idx):
                    raise AssertionError(
                        f"stream {si} ran dry at step {st}/{steps} after {pos[si]:,} examples "
                        f"({got[si]:,.0f}/{due[si]:,.0f} tokens) -- the M9.0 token arithmetic and "
                        f"this batcher disagree, and a short arm must never look like a full one")
                j = int(idx[pos[si]])
                flat.append(j)
                got[si] += float(tlen[j])
                pos[si] += 1
        if len(flat) == offs[-1]:
            raise AssertionError(f"step {st} would be an empty batch; the LR schedule would "
                                 f"advance over nothing")
        offs.append(len(flat))
    assert len(offs) - 1 == steps
    return (np.asarray(flat, dtype=np.int32), np.asarray(offs, dtype=np.int64), pos,
            [round(g, 1) for g in got])


def lr_at(step, steps, peak, final, warmup):
    """Zero-based `step`. The denominator is `steps - warmup - 1` so the LAST EXECUTED step is the
    cosine endpoint; with `steps - warmup` the schedule stops one step short of `final` and never
    reaches it (Codex pass 2, MINOR-1)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(steps - warmup - 1, 1)
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * min(t, 1.0)))


def train_arm(arm_id, student_key, plan, cfg, eval_fn=None, device="cuda", log_every=500,
              warm_start=True, warm_texts=None, warm_prefix=""):
    """`plan`: ids (list[list[int]]), tgt ((N,1024) fp16), flat+offs (the locked batch schedule).
    `cfg`: batch_size, steps, warmup_steps, lr_peak, lr_final, seed, checkpoints, and the AdamW
    settings. Returns (run record, model)."""
    r = registry()["dose"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = Nano(student_key).to(device)
    pad_id = model.tok.pad_token_id
    ids, tgt, flat, offs = plan["ids"], plan["tgt"], plan["flat"], plan["offs"]
    steps = len(offs) - 1
    assert steps == cfg["steps"], f"{arm_id}: schedule has {steps:,} steps, cfg says {cfg['steps']:,}"
    ckpts = set(cfg["checkpoints"])

    ws = None
    if warm_start:
        model.eval()
        _t = time.time()
        ws = warm_start_head(model, warm_texts, plan["tgt"], warm_prefix)
        ws["stage0_seconds"] = round(time.time() - _t, 1)
        print(f"  {arm_id} warm-start head: {json.dumps(ws)}", flush=True)

    decay = [p for _n, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for _n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": r["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr_peak"], betas=tuple(r["betas"]), eps=r["eps"])

    hist, t0, loss_acc, nlog, tok_acc, ex_acc = [], time.time(), 0.0, 0, 0, 0
    bsz = []
    model.train()
    for step in range(steps):
        rows = flat[offs[step]:offs[step + 1]]
        assert rows.size > 0, f"{arm_id}: step {step} has an empty batch"
        lr = lr_at(step, steps, cfg["lr_peak"], cfg["lr_final"], cfg["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = lr
        ii, am = collate(ids, rows, pad_id, device)
        tok_acc += int(am.sum().item())
        ex_acc += int(rows.size)
        bsz.append(int(rows.size))
        t = torch.from_numpy(np.asarray(tgt[rows], dtype=np.float32)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(ii, am)
        v = F.normalize(v.float(), dim=-1, eps=1e-12)
        loss = ((v - t) ** 2).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), r["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_acc += float(loss.detach())
        nlog += 1

        if (step + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  {arm_id} step {step+1:,}/{steps:,} loss {loss_acc/nlog:.5f} lr {lr:.2e} "
                  f"{ex_acc/el:.0f} ex/s eta {(steps-step-1)*el/(step+1)/60:.1f}m", flush=True)
            loss_acc, nlog = 0.0, 0
        if (step + 1) in ckpts:
            rec = {"step": step + 1, "examples": ex_acc, "nonpad_tokens": tok_acc,
                   "elapsed_s": round(time.time() - t0, 1)}
            if eval_fn is not None:
                rec.update(eval_fn(model, step + 1))
                model.train()
                # The eval allocates large GPU tiles; keeping them caches the allocator into a
                # near-full 10 GB and training degrades from ~1,990 to <800 ex/s with power draw
                # falling to 160 W -- the allocator-thrash signature, not work (CLAUDE.md).
                torch.cuda.empty_cache()
            hist.append(rec)
            print(f"  {arm_id} CKPT " + json.dumps(
                {k: v for k, v in rec.items() if k != "per_component"}), flush=True)

    bsz = np.array(bsz)
    return {"arm": arm_id, "student": student_key, "steps": steps, "seed": cfg["seed"],
            "batch_size_cfg": cfg["batch_size"], "examples": ex_acc, "nonpad_tokens": tok_acc,
            "batch_size_realized": {"mean": round(float(bsz.mean()), 2), "min": int(bsz.min()),
                                    "max": int(bsz.max())},
            "tokens_per_step": {"mean": round(tok_acc / steps, 1)},
            "n_params": model.n_params(), "warm_start": ws,
            "seconds": round(time.time() - t0, 1), "history": hist}, model


def artifact_bytes(student_key):
    m = Nano(student_key)
    n = m.n_params()
    del m
    return {"student": student_key, "params": n, "fp16_bytes": n * 2,
            "fp16_mb": round(n * 2 / 1e6, 1)}


def self_test():
    """The schedule must hit the registered counts exactly, and the token batcher must land on the
    registered step count with the registered role split."""
    r = registry()["dose"]
    n = 242786
    o = epoch_order(n, r["epochs_query_only"], r["seed"])
    assert o.size == r["examples"] == n * r["epochs_query_only"], o.size
    flat, offs = fixed_batches(o, r["batch_size"])
    assert len(offs) - 1 == r["steps"], len(offs) - 1
    assert int(offs[-1]) == r["examples"]
    assert int(offs[-1] - offs[-2]) == r["examples"] - (r["steps"] - 1) * r["batch_size"]

    # token batcher: synthetic lengths matching the locked means
    mx = r["mix_arm"]
    rng = np.random.default_rng(0)
    qlen = rng.integers(8, 24, size=n).astype(np.float64)
    dlen = rng.integers(60, 130, size=200_000).astype(np.float64)
    tot_q = qlen.sum() * 16
    qorder = epoch_order(n, 16, 0)
    dorder = np.arange(int(round(0.30 * tot_q / dlen.mean() * 1.3)))
    f2, o2, pos, real = token_batches([(qorder, np.concatenate([qlen, dlen])),
                                       (dorder + n, np.concatenate([qlen, dlen]))],
                                      r["steps"], tot_q, [0.70, 0.30])
    assert len(o2) - 1 == r["steps"] and int(o2[-1]) == len(f2) == pos[0] + pos[1]
    assert abs(real[0] - 0.70 * tot_q) < 600 and abs(real[1] - 0.30 * tot_q) < 600, real
    assert np.all(np.diff(o2) > 0), "an empty batch slipped through"
    assert lr_at(0, 1000, 1e-4, 1e-5, 100) < 1e-4
    assert abs(lr_at(999, 1000, 1e-4, 1e-5, 100) - 1e-5) < 2e-7
    for k in STUDENTS:
        print(json.dumps(artifact_bytes(k)))
    print("nano self_test PASS")


if __name__ == "__main__":
    self_test()
