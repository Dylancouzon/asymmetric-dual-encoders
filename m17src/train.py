"""The one M17 training driver. Arms C / V / L / VL / VL-A, exactly as registered.

Not a launcher for `m7src/train.py`: that driver hardcodes M7's tokenizer, init and data
assumptions and calls development scoring, and `sweep.one` appends to M7 reporting files
(m17/CODEMAP.md, "Reuse hazards"). What IS reused are the primitives — `QueryTable`, ragged
bags, `occurrence_weights` for sqrt pooling, int8 quantization — bound here to the M17
tokenizer/table identity and M17 output paths.

  .venv/bin/python m17src/train.py --arm VL-A --out work/m17/runs/vla-s0
  .venv/bin/python m17src/train.py --rehearsal --out work/m17/rehearsal   # synthetic, tiny

The registry status gate is `common.require_executable`: a real run refuses while
`m17/registry.json` says `DRAFT_NOT_EXECUTABLE`. `--rehearsal` is the only bypass and says so.

What the loop implements, from `m17/registry.json`:

* warm start from the M7 `p35w-2m-s2500` **unfolded** checkpoint, lineage verified against
  `m7/FREEZE.json`, with new run ids and a fresh optimizer;
* batch composition 0.75 general / 0.25 coverage, with alias PAIRS occupying 0.125 of the batch
  (16 pairs = 32 views at batch 256), identical views in the same order in every arm;
* without-replacement sampling inside each bucket, reshuffled per pass with the run seed, and a
  per-bucket pass counter (never an overall one);
* losses: cosine to the frozen teacher query vector, listwise KL(teacher || student) over the
  cached candidate list at one fixed temperature, the inherited init anchor on EFFECTIVE folded
  rows, and the batch-normalized alias consistency term (VL-A only);
* Adam with rows and learned scalars in separate parameter groups, linear warmup then linear
  decay, both schedules predeclared;
* an overfit/divergence read every 500 steps on a fixed 2000-query held-out slice of the
  TRAINING sources (never the panel), flagged and reported, never acted on after lock;
* step-bound late snapshots (`checkpoint_averaging.checkpoint_steps`) in addition to time-bound
  recovery checkpoints, and a resume that restores optimizer, scheduler, RNG, per-bucket stream
  position and tokenizer identity;
* a JSON run record beside the checkpoints.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import WORK, freeze, registry, require_executable, sha_json, sha_text, write_json

ARMS = {
    "C":    {"vocab_extension": False, "listwise": False, "alias_consistency": False},
    "V":    {"vocab_extension": True,  "listwise": False, "alias_consistency": False},
    "L":    {"vocab_extension": False, "listwise": True,  "alias_consistency": False},
    "VL":   {"vocab_extension": True,  "listwise": True,  "alias_consistency": False},
    "VL-A": {"vocab_extension": True,  "listwise": True,  "alias_consistency": True},
}


@dataclass
class RunCfg:
    arm: str = "VL-A"
    seed: int = 0
    steps: int = 6000
    batch: int = 256
    rows_lr: float = 3e-4
    weights_lr: float = 1e-3
    warmup_steps: int = 200
    temperature: float = 0.05
    cosine_weight: float = 1.0
    listwise_weight: float = 1.0
    alias_weight: float = 0.1
    init_anchor_weight: float = 1e-3
    general_fraction: float = 0.75
    coverage_fraction: float = 0.25
    alias_views_fraction: float = 0.125
    check_every: int = 500
    heldout_queries: int = 2000
    snapshot_steps: tuple = (4500, 5250, 6000)
    checkpoint_minutes_max: float = 15.0
    device: str = "cuda"
    run_id: str = ""
    rehearsal: bool = False
    tokenizer_sha256: str = ""
    vocabulary_sha256: str = ""
    cache_sha256: str = ""
    warm_start: str = ""

    @classmethod
    def from_registry(cls, reg, arm, seed=None, **over):
        tr = reg["training"]
        d = reg["data"]
        cfg = cls(
            arm=arm,
            seed=int(tr["seed_screen"] if seed is None else seed),
            steps=int(tr["final_steps_per_fresh_run"]),
            batch=int(tr["batch"]),
            rows_lr=float(tr["rows_lr"]),
            weights_lr=float(tr["weights_lr"]),
            warmup_steps=int(tr["warmup_steps"]),
            temperature=float(tr["temperature"]),
            cosine_weight=float(tr["cosine_weight"]),
            listwise_weight=float(tr["listwise_weight_when_enabled"]),
            alias_weight=float(tr["alias_weight_when_enabled"]),
            init_anchor_weight=float(tr["init_anchor_weight"]),
            general_fraction=float(d["general_replay_fraction"]),
            coverage_fraction=float(d["targeted_coverage_fraction"]),
            alias_views_fraction=float(d["alias_query_views_fraction_of_batch"]),
            check_every=int(tr["overfit_divergence_check"]["every_steps"]),
            heldout_queries=2000,
            snapshot_steps=tuple(reg["checkpoint_averaging"]["checkpoint_steps"]),
            checkpoint_minutes_max=float(tr["checkpoint_minutes_max"]),
        )
        for k, v in over.items():
            setattr(cfg, k, v)
        if not cfg.run_id:
            cfg.run_id = f"m17-{cfg.arm.lower()}-s{cfg.seed}"
        return cfg

    def batch_shape(self):
        """(general views, unpaired coverage views, alias pairs) for one step.

        Alias PAIRS take `alias_views_fraction` of the batch as VIEWS, i.e. half that many
        pairs, and they come out of the coverage share — "half of the targeted-coverage slots
        are paired equivalent views".
        """
        alias_views = int(round(self.batch * self.alias_views_fraction))
        alias_views -= alias_views % 2
        coverage = int(round(self.batch * self.coverage_fraction)) - alias_views
        general = self.batch - coverage - alias_views
        if coverage < 0 or general < 0:
            raise ValueError(f"batch composition does not fit batch={self.batch}")
        return general, coverage, alias_views // 2


# ---- data ------------------------------------------------------------------------------

@dataclass
class Stream:
    """Without-replacement sampling inside one bucket, reshuffled per pass with the run seed.

    Passes are counted PER BUCKET, not overall: the registry's dose rule is expressed that way
    because the alias and coverage buckets are far smaller than the general one.
    """
    name: str
    population: np.ndarray
    seed: int
    pos: int = 0
    passes: int = 0
    order: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.order is None:
            self._reshuffle()

    def _reshuffle(self):
        # A STABLE per-bucket stream id: Python's `hash(str)` is salted per process, which would
        # make the same seed give different orders across a resume.
        sid = int(sha_text(self.name)[:8], 16)
        rng = np.random.default_rng([self.seed, self.passes, sid])
        self.order = self.population[rng.permutation(len(self.population))] \
            if len(self.population) else self.population
        self.pos = 0

    def draw(self, n):
        if len(self.population) == 0:
            return np.zeros(0, dtype=np.int64)
        out = []
        while n > 0:
            take = min(n, len(self.order) - self.pos)
            out.append(self.order[self.pos:self.pos + take])
            self.pos += take
            n -= take
            if self.pos >= len(self.order):
                self.passes += 1
                self._reshuffle()
        return np.concatenate(out)

    def state(self):
        return {"name": self.name, "pos": int(self.pos), "passes": int(self.passes),
                "n": int(len(self.population))}

    def load_state(self, st):
        self.passes = int(st["passes"])
        self._reshuffle()
        self.pos = int(st["pos"])


def build_streams(buckets, alias_pair_ids, seed):
    """General / unpaired-coverage / alias-pair streams from the cache's own bucket labels.

    Only the two training buckets are drawn from. Anything else the cache carries — notably the
    fixed held-out slice — is deliberately NOT a stream: the divergence read must never sample
    rows the optimizer is also seeing.
    """
    buckets = np.asarray([str(b) for b in buckets])
    pair_ids = np.asarray([str(p) for p in alias_pair_ids])
    is_alias = pair_ids != ""
    general = np.flatnonzero((buckets == "general") & ~is_alias)
    coverage = np.flatnonzero((buckets == "coverage") & ~is_alias)
    pairs = {}
    for i in np.flatnonzero(is_alias):
        pairs.setdefault(pair_ids[i], []).append(int(i))
    ordered = sorted(p for p, v in pairs.items() if len(v) == 2)
    incomplete = sorted(p for p, v in pairs.items() if len(v) != 2)
    pair_index = np.asarray([pairs[p] for p in ordered], dtype=np.int64).reshape(-1, 2) \
        if ordered else np.zeros((0, 2), dtype=np.int64)
    return ({"general": Stream("general", general, seed),
             "coverage": Stream("coverage", coverage, seed),
             "alias": Stream("alias", np.arange(len(pair_index)), seed)},
            pair_index, incomplete)


# ---- model path ---------------------------------------------------------------------------

def forward(model, ids_list, device):
    """The training forward: sqrt-count pooling, learned scalars, L2 normalize, fallback.

    Same arithmetic as `m7src/table.encode_pooled(mode='sqrt')`, but differentiable — the
    released helper is `@torch.no_grad`. Keeping one implementation here means the arm trains
    under exactly the pooling rule it will be served under.
    """
    from table import EPS, _bag_index, occurrence_weights, ragged
    flat, off, lens = ragged(ids_list, device)
    psw = occurrence_weights(ids_list, "sqrt", device=device)
    w = model.token_weights()
    if w is not None:
        psw = psw * w[flat]
    s = F.embedding_bag(flat, model.rows, off, mode="sum", per_sample_weights=psw)
    denom = torch.zeros_like(s[:, 0]).index_add_(0, _bag_index(off, lens), psw)
    mean = s / denom.clamp_min(EPS).unsqueeze(1)
    norm = mean.norm(dim=1, keepdim=True)
    fb = model.fallback_vector().to(mean.dtype)
    return torch.where(norm > EPS, mean / norm.clamp_min(EPS), fb.expand_as(mean))


def losses(model, q_s, q_t, cand_vecs, cand_mask, teacher_scores, eff_init, pair_slots, cfg,
           arm):
    """The registry's `training.loss_definition`, term by term. Returns (total, parts)."""
    parts = {}
    cos = (1.0 - (q_s * q_t).sum(1)).mean()
    parts["cosine"] = float(cos.detach())
    total = cfg.cosine_weight * cos

    if arm["listwise"]:
        neg = torch.finfo(torch.float32).min
        t_log = torch.where(cand_mask, teacher_scores / cfg.temperature, torch.full_like(
            teacher_scores, neg))
        s_log = torch.where(cand_mask,
                            torch.einsum("bd,bkd->bk", q_s, cand_vecs) / cfg.temperature,
                            torch.full_like(teacher_scores, neg))
        lt, ls = F.log_softmax(t_log, dim=1), F.log_softmax(s_log, dim=1)
        pt = lt.exp()
        kl = (pt * (lt - ls)).masked_fill(~cand_mask, 0.0).sum(1).mean()
        parts["listwise_kl"] = float(kl.detach())
        total = total + cfg.listwise_weight * kl

    if cfg.init_anchor_weight > 0:
        eff = model.rows if not model.learned_weights else \
            model.token_weights().unsqueeze(1) * model.rows
        anchor = (eff - eff_init).pow(2).mean()
        parts["anchor"] = float(anchor.detach())
        total = total + cfg.init_anchor_weight * anchor

    if arm["alias_consistency"] and pair_slots is not None and len(pair_slots):
        a, b = pair_slots[:, 0], pair_slots[:, 1]
        alias = (2.0 / q_s.shape[0]) * (1.0 - (q_s[a] * q_s[b]).sum(1)).sum()
        parts["alias"] = float(alias.detach())
        total = total + cfg.alias_weight * alias
    return total, parts


# ---- warm start -----------------------------------------------------------------------------

def load_warm_start(path, expect_vocab=None, device="cpu", rehearsal=False):
    """Load the UNFOLDED M7 checkpoint and verify its lineage before a single update.

    Refusals, all of them "do not train from this":
      * a folded release (its learned scalars are already inside the rows — folding again would
        scale every row by w twice);
      * a checkpoint distilled from another teacher or revision than the registry's;
      * a vocabulary size that does not match the base vocabulary the extension starts from.
    """
    from table import QueryTable, read_meta
    reg = registry()
    meta = read_meta(path)
    if meta.get("weights_folded"):
        raise SystemExit(f"M17 REFUSED: {path} is a FOLDED release artifact. The warm start must "
                         "be the unfolded training checkpoint; a folded table cannot resume "
                         "training and folding its scalars again double-scales every row.")
    if not meta.get("learned_weights", True):
        raise SystemExit(f"M17 REFUSED: {path} has no learned scalars; the registered warm start "
                         "is the unfolded p35w-2m-s2500 checkpoint.")
    t, rev = meta.get("teacher"), meta.get("teacher_revision")
    if not rehearsal and (t, rev) != (reg["teacher"], reg["teacher_revision"]):
        raise SystemExit(f"M17 REFUSED: warm start was distilled from {t}@{str(rev)[:12]} but the "
                         f"registry pins {reg['teacher']}@{reg['teacher_revision'][:12]}.")
    z = np.load(path)
    rows = z["rows_fp16"].astype(np.float32)
    w = z["token_weights"]
    if expect_vocab is not None and rows.shape[0] != expect_vocab:
        raise SystemExit(f"M17 REFUSED: warm start has {rows.shape[0]} rows, expected "
                         f"{expect_vocab} base rows before the extension.")
    model = QueryTable(rows, weight_init=(w if w.size else None), learned_weights=bool(w.size))
    lineage = {"path": str(path), "meta": meta, "vocab": int(rows.shape[0]),
               "dim": int(rows.shape[1])}
    if not rehearsal:
        fz = freeze()
        lineage["freeze_run_id"] = fz.get("run_id")
        lineage["freeze_training_checkpoint_sha256"] = fz.get("training_checkpoint_sha256")
    return model.to(device), lineage


def extend_model(model, new_rows, device):
    """Append new rows (and their scalars at 1.0) to a warm-started table. New ids only."""
    from table import QueryTable
    rows = model.rows.detach().cpu().numpy()
    w = model.token_weights()
    w = None if w is None else w.detach().cpu().numpy()
    rows = np.concatenate([rows, np.asarray(new_rows, dtype=np.float32)], 0)
    if w is not None:
        w = np.concatenate([w, np.ones(len(new_rows), dtype=np.float32)])
    return QueryTable(rows, weight_init=w, learned_weights=w is not None,
                      fallback_id=model.fallback_id).to(device)


# ---- the loop --------------------------------------------------------------------------------

def run(cfg: RunCfg, data, out_dir, resume=True, log=print):
    """`data` is the dict `prepare_data` returns (or the rehearsal fixture builds)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arm = ARMS[cfg.arm]
    device = cfg.device
    torch.manual_seed(cfg.seed)

    model = data["model"].to(device)
    model.train()
    eff_init = (model.token_weights().detach().unsqueeze(1) * model.rows.detach()
                if model.learned_weights else model.rows.detach()).clone()

    opt = torch.optim.Adam([
        {"params": [model.rows], "lr": cfg.rows_lr},
        *([{"params": [model.w_raw], "lr": cfg.weights_lr}] if model.learned_weights else []),
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]

    ids_all = data["ids"]                       # list[list[int]] student token ids per query
    q_t = torch.as_tensor(data["teacher_q"], dtype=torch.float32, device=device)
    bank = torch.as_tensor(data["bank"], dtype=torch.float32, device=device)
    cand = torch.as_tensor(data["candidate_ids"], dtype=torch.long, device=device)
    tsc = torch.as_tensor(data["teacher_scores"], dtype=torch.float32, device=device)
    streams, pair_index, incomplete = build_streams(data["buckets"], data["alias_pair_ids"],
                                                    cfg.seed)
    heldout = np.asarray(data["heldout_idx"], dtype=np.int64)[:cfg.heldout_queries]
    g_n, c_n, p_n = cfg.batch_shape()

    state = {"step": 0, "history": [], "flags": [], "snapshots": {}}
    ck = out / "recovery.pt"
    if resume and ck.exists():
        state = _resume(ck, model, opt, streams, cfg, log)

    def set_lr(step):
        f = min(1.0, step / max(1, cfg.warmup_steps))
        if step > cfg.warmup_steps:
            f = max(0.0, 1.0 - (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        return f

    def batch_indices():
        g = streams["general"].draw(g_n)
        c = streams["coverage"].draw(c_n)
        p = streams["alias"].draw(p_n)
        views = pair_index[p].reshape(-1) if len(p) else np.zeros(0, dtype=np.int64)
        idx = np.concatenate([g, c, views]).astype(np.int64)
        slots = None
        if len(views):
            off = len(g) + len(c)
            slots = torch.as_tensor(
                np.stack([np.arange(off, off + len(views), 2),
                          np.arange(off + 1, off + len(views), 2)], 1), device=device)
        return idx, slots

    def step_losses(idx, slots):
        q_s = forward(model, [ids_all[i] for i in idx], device)
        ii = torch.as_tensor(idx, device=device)
        cids = cand[ii]
        mask = cids >= 0
        vecs = bank[cids.clamp_min(0)]
        return losses(model, q_s, q_t[ii], vecs, mask, tsc[ii], eff_init, slots, cfg, arm)

    @torch.no_grad()
    def heldout_loss():
        model.eval()
        tot, n = 0.0, 0
        for lo in range(0, len(heldout), cfg.batch):
            sl = heldout[lo:lo + cfg.batch]
            if not len(sl):
                break
            loss, _ = step_losses(sl, None)       # alias term excluded: not a paired slice
            tot += float(loss) * len(sl)
            n += len(sl)
        model.train()
        return tot / max(1, n)

    t0 = time.time()
    last_ck = t0
    run_mean = None
    while state["step"] < cfg.steps:
        state["step"] += 1
        s = state["step"]
        lr_factor = set_lr(s)
        idx, slots = batch_indices()
        loss, parts = step_losses(idx, slots)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        run_mean = lv if run_mean is None else 0.99 * run_mean + 0.01 * lv

        if s % cfg.check_every == 0 or s == cfg.steps:
            h = heldout_loss()
            state["history"].append({"step": s, "train_running_mean": run_mean,
                                     "heldout": h, "parts": parts,
                                     "lr_factor": lr_factor,
                                     "passes": {k: v.passes for k, v in streams.items()}})
            _flag_divergence(state, log)
            log(f"  [{cfg.run_id}] step {s}/{cfg.steps} train {run_mean:.5f} heldout {h:.5f} "
                f"{ {k: round(v, 5) for k, v in parts.items()} } "
                f"{(time.time() - t0) / s * 1000:.0f} ms/step")

        if s in tuple(cfg.snapshot_steps):
            p = out / f"snapshot_{s:06d}.npz"
            _save_table(p, model, cfg, data, step=s)
            state["snapshots"][str(s)] = str(p.name)
            log(f"  [{cfg.run_id}] step-bound snapshot {p.name}")

        if (time.time() - last_ck) / 60.0 >= cfg.checkpoint_minutes_max or s == cfg.steps:
            _checkpoint(ck, model, opt, streams, state, cfg, data)
            last_ck = time.time()

    record = {
        "_schema": "m17-run-record-v1",
        "run_id": cfg.run_id, "arm": cfg.arm, "arm_definition": arm,
        "config": {**asdict(cfg), "snapshot_steps": list(cfg.snapshot_steps)},
        "registry_status": data.get("registry_status"),
        "rehearsal": cfg.rehearsal,
        "warm_start_lineage": data.get("lineage"),
        "identity": {"tokenizer_sha256": cfg.tokenizer_sha256,
                     "vocabulary_sha256": cfg.vocabulary_sha256,
                     "candidate_cache_sha256": cfg.cache_sha256},
        "batch_composition": {"general": g_n, "unpaired_coverage": c_n, "alias_pairs": p_n,
                              "batch": cfg.batch},
        "bucket_passes": {k: v.state() for k, v in streams.items()},
        "incomplete_alias_pairs": incomplete,
        "history": state["history"], "divergence_flags": state["flags"],
        "snapshots": state["snapshots"],
        "wall_clock_seconds": round(time.time() - t0, 3),
    }
    record["sha256"] = sha_json({k: v for k, v in record.items() if k != "sha256"})
    write_json(out / "run_record.json", record)
    _save_table(out / "endpoint.npz", model, cfg, data, step=state["step"])
    return record


def _flag_divergence(state, log):
    """Held-out loss up for two consecutive reads while train loss falls. Reported, not acted on."""
    h = state["history"]
    if len(h) < 3:
        return
    a, b, c = h[-3], h[-2], h[-1]
    if (c["heldout"] > b["heldout"] > a["heldout"]
            and c["train_running_mean"] < b["train_running_mean"] < a["train_running_mean"]):
        flag = {"step": c["step"], "heldout": [a["heldout"], b["heldout"], c["heldout"]],
                "train": [a["train_running_mean"], b["train_running_mean"],
                          c["train_running_mean"]],
                "note": "recorded and reported; the schedule does not change after lock"}
        state["flags"].append(flag)
        log(f"  DIVERGENCE FLAG at step {c['step']}: held-out loss rose twice while train fell")


def _save_table(path, model, cfg, data, step):
    """Unfolded training-shape table plus the M17 identity sidecar (`m7src.table.save_table`)."""
    from table import Preproc, save_table
    pre = Preproc(**data["preproc"])
    save_table(Path(path), model, pre, meta={
        "m17_run_id": cfg.run_id, "m17_arm": cfg.arm, "m17_step": int(step),
        "m17_seed": cfg.seed, "m17_rehearsal": bool(cfg.rehearsal),
        "tokenizer_sha256": cfg.tokenizer_sha256,
        "vocabulary_sha256": cfg.vocabulary_sha256,
        "candidate_cache_sha256": cfg.cache_sha256,
        "teacher": data.get("teacher"), "teacher_revision": data.get("teacher_revision"),
        "weights_folded": False})


def _checkpoint(path, model, opt, streams, state, cfg, data):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "streams": {k: v.state() for k, v in streams.items()},
                "state": state, "cfg": asdict(cfg),
                "torch_rng": torch.get_rng_state(),
                "tokenizer_sha256": cfg.tokenizer_sha256,
                "vocab": int(model.rows.shape[0])}, tmp)
    tmp.replace(path)


def _resume(path, model, opt, streams, cfg, log):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("vocab") != int(model.rows.shape[0]):
        raise SystemExit(f"M17 RESUME REFUSED: checkpoint has {ck.get('vocab')} rows, this run "
                         f"has {int(model.rows.shape[0])}; a different table is a different run.")
    if ck.get("tokenizer_sha256", "") != cfg.tokenizer_sha256:
        raise SystemExit(
            f"M17 RESUME REFUSED: checkpoint tokenizer {ck.get('tokenizer_sha256', '')[:12]!r} "
            f"!= this run's {cfg.tokenizer_sha256[:12]!r}. Schedule and step identity are only "
            "meaningful for the same tokenizer and table.")
    if ck.get("cfg", {}).get("steps") != cfg.steps:
        raise SystemExit("M17 RESUME REFUSED: the checkpoint was written under a different step "
                         "budget; do not append steps to an exhausted scheduler.")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    for k, v in streams.items():
        if k in ck["streams"]:
            v.load_state(ck["streams"][k])
    torch.set_rng_state(ck["torch_rng"].to(torch.uint8) if hasattr(ck["torch_rng"], "to")
                        else ck["torch_rng"])
    log(f"  resumed at step {ck['state']['step']} from {path.name}")
    return ck["state"]


# ---- CLI ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="VL-A", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data", default=None, help="directory holding the candidate cache and "
                                                 "the prepared training arrays")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--rehearsal", action="store_true",
                    help="synthetic tiny end-to-end rehearsal under work/m17/rehearsal")
    args = ap.parse_args(argv)

    reg = registry()
    status = require_executable(reg, args.rehearsal, what=f"arm {args.arm}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.rehearsal:
        import rehearse17
        return rehearse17.main(["--out", args.out or str(WORK / "rehearsal"),
                                "--device", device])

    if not args.data:
        raise SystemExit("--data is required for a real run: the prepared candidate cache and "
                         "training arrays (see m17src/cache.py and m17/STATUS.md step 5).")
    over = {k: v for k, v in (("steps", args.steps), ("batch", args.batch),
                              ("device", device)) if v is not None}
    cfg = RunCfg.from_registry(reg, args.arm, seed=args.seed, **over)
    data = json.loads((Path(args.data) / "prepared.json").read_text())
    data["registry_status"] = status
    out = Path(args.out or (WORK / "runs" / cfg.run_id))
    rec = run(cfg, _load_prepared(Path(args.data), data), out, resume=not args.no_resume)
    print(f"run record: {out / 'run_record.json'} ({rec['wall_clock_seconds']}s)")
    return 0


def _load_prepared(data_dir, manifest):
    """Materialize the arrays `prepared.json` points at. Kept separate so tests can bypass it."""
    d = Path(data_dir)
    z = np.load(d / "candidates.npz", allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    out = dict(manifest)
    out.update({
        "ids": json.loads((d / "student_ids.json").read_text()),
        "teacher_q": np.load(d / "teacher_q.npy"),
        "bank": np.load(d / "bank.npy"),
        "candidate_ids": arrays["candidate_ids"],
        "teacher_scores": arrays["teacher_scores"],
        "buckets": arrays["buckets"],
        "alias_pair_ids": arrays["alias_pair_ids"],
    })
    model, lineage = load_warm_start(d / manifest["warm_start"], device="cpu")
    if manifest.get("new_rows"):
        model = extend_model(model, np.load(d / manifest["new_rows"]), "cpu")
    out["model"], out["lineage"] = model, lineage
    return out


if __name__ == "__main__":
    raise SystemExit(main())
