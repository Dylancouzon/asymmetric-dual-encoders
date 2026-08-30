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


def token_batches(streams, steps, budget, shares):
    """Token-budgeted batching for the mix arm (m9/LEDGER.md §3.1).

    `streams` is [(index array, token-length array)] per role and `shares` the per-role fraction of
    `budget` to fill each step. Each stream is consumed IN ORDER and never wraps: the schedules
    were sized at M9.0 to cover exactly `steps` steps, and running dry is an error rather than a
    silent short arm.
    """
    flat, offs, pos = [], [0], [0] * len(streams)
    for _ in range(steps):
        for si, (idx, tlen) in enumerate(streams):
            want, got = budget * shares[si], 0.0
            while got < want and pos[si] < len(idx):
                flat.append(int(idx[pos[si]]))
                got += float(tlen[idx[pos[si]]])
                pos[si] += 1
        offs.append(len(flat))
    for si, (idx, _t) in enumerate(streams):
        if pos[si] < len(idx) * 0.98:
            raise AssertionError(f"stream {si} consumed only {pos[si]:,}/{len(idx):,} -- the "
                                 f"M9.0 token arithmetic and this batcher disagree")
    return np.asarray(flat, dtype=np.int32), np.asarray(offs, dtype=np.int64), pos


def lr_at(step, steps, peak, final, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(steps - warmup, 1)
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * t))


def train_arm(arm_id, student_key, plan, cfg, eval_fn=None, device="cuda", log_every=500):
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
        if rows.size == 0:
            continue
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
            hist.append(rec)
            print(f"  {arm_id} CKPT " + json.dumps(
                {k: v for k, v in rec.items() if k != "per_component"}), flush=True)

    bsz = np.array(bsz)
    return {"arm": arm_id, "student": student_key, "steps": steps, "seed": cfg["seed"],
            "batch_size_cfg": cfg["batch_size"], "examples": ex_acc, "nonpad_tokens": tok_acc,
            "batch_size_realized": {"mean": round(float(bsz.mean()), 2), "min": int(bsz.min()),
                                    "max": int(bsz.max())},
            "tokens_per_step": {"mean": round(tok_acc / steps, 1)},
            "n_params": model.n_params(), "seconds": round(time.time() - t0, 1),
            "history": hist}, model


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
    qorder = epoch_order(n, 12, 0)[:int(round(0.70 * tot_q / qlen.mean()))]
    dorder = np.arange(int(round(0.30 * tot_q / dlen.mean())))
    budget = tot_q / r["steps"]
    f2, o2, pos = token_batches([(qorder, qlen), (dorder, dlen)], r["steps"], budget, [0.70, 0.30])
    assert len(o2) - 1 == r["steps"]
    assert 0.90 < (len(f2) / (pos[0] + pos[1])) < 1.11
    assert lr_at(0, 1000, 1e-4, 1e-5, 100) < 1e-4
    assert abs(lr_at(999, 1000, 1e-4, 1e-5, 100) - 1e-5) < 2e-7
    for k in STUDENTS:
        print(json.dumps(artifact_bytes(k)))
    print("nano self_test PASS")


if __name__ == "__main__":
    self_test()
