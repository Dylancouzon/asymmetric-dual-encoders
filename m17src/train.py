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
* the MEASURED batch composition (`data.measured_dose_after_pre_lock_rule`): 204 general, 32
  unpaired coverage and 10 alias pairs = 20 views at batch 256, after the pre-lock alias
  shrink; identical views in the same order in every arm;
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
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (WORK, admit_read, admit_write, freeze, registry, require_executable, sha_array, sha_file,
                    sha_json, sha_text, sha_texts, write_json)

ARMS = {
    "C":    {"vocab_extension": False, "listwise": False, "alias_consistency": False},
    "V":    {"vocab_extension": True,  "listwise": False, "alias_consistency": False},
    "L":    {"vocab_extension": False, "listwise": True,  "alias_consistency": False},
    "VL":   {"vocab_extension": True,  "listwise": True,  "alias_consistency": False},
    "VL-A": {"vocab_extension": True,  "listwise": True,  "alias_consistency": True},
}


def resolve_dose(data_reg, batch):
    """(general, coverage, alias pairs) per batch from `measured_dose_after_pre_lock_rule`.

    At the dose's own batch the registered numbers are used verbatim (204 / 32 / 10 at 256).
    Another batch size — the pre-lock 128 fallback, or a fixture-scale rehearsal batch — is
    derived proportionally, with the `pre_lock_rule`'s floors (16 coverage views, 4 pairs)
    applied whenever they still fit. General absorbs the remainder so the sum is exact.
    """
    dose = data_reg["measured_dose_after_pre_lock_rule"]
    batch = int(batch)
    d_batch = int(dose["batch"])
    g0 = int(dose["general_views_per_batch"])
    c0 = int(dose["unpaired_coverage_views_per_batch"])
    p0 = int(dose["alias_pairs_per_batch"])
    if g0 + c0 + 2 * p0 != d_batch:
        raise ValueError(f"registry dose {g0}/{c0}/{p0} does not sum to batch {d_batch}")
    if batch == d_batch:
        return g0, c0, p0
    scale = batch / d_batch
    c = max(1, int(round(c0 * scale)))
    p = max(1, int(round(p0 * scale)))
    if 16 + 2 * 4 < batch:                      # the pre_lock_rule floors, where they fit
        c, p = max(c, 16), max(p, 4)
    g = batch - c - 2 * p
    if g <= 0:
        raise ValueError(f"batch={batch} is too small for the registered dose shape")
    return g, c, p


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
    # Batch composition is the MEASURED dose (`data.measured_dose_after_pre_lock_rule`), not
    # the superseded fractions: 204 general / 32 coverage / 10 alias pairs at batch 256.
    general_views: int = 204
    coverage_views: int = 32
    alias_pairs: int = 10
    dose_source: str = "data.measured_dose_after_pre_lock_rule"
    phase: str = ""
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
    cache_artifact_sha256: str = ""
    warm_start: str = ""

    @classmethod
    def from_registry(cls, reg, arm, seed=None, **over):
        tr = reg["training"]
        d = reg["data"]
        batch = int(over.get("batch", tr["batch"]))
        g, c, p = resolve_dose(d, batch)
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
            general_views=g,
            coverage_views=c,
            alias_pairs=p,
            check_every=int(tr["overfit_divergence_check"]["every_steps"]),
            heldout_queries=2000,
            snapshot_steps=tuple(reg["checkpoint_averaging"]["checkpoint_steps"]),
            checkpoint_minutes_max=float(tr["checkpoint_minutes_max"]),
        )
        for k, v in over.items():
            setattr(cfg, k, v)
        if not cfg.phase:
            cfg.phase = ("screen" if cfg.steps == int(tr["screen_steps"])
                         else "final" if cfg.steps == int(tr["final_steps_per_fresh_run"])
                         else "custom")
        if not cfg.run_id:
            # The phase is part of the identity: a fresh screen and a fresh final run of the
            # same arm and seed are different experiments and must not share a run directory.
            cfg.run_id = f"m17-{cfg.arm.lower()}-{cfg.phase}-s{cfg.seed}"
        return cfg

    def batch_shape(self):
        """(general views, unpaired coverage views, alias pairs) for one step.

        The registered dose, validated: the views must add up to exactly the batch.
        """
        g, c, p = int(self.general_views), int(self.coverage_views), int(self.alias_pairs)
        if min(g, c, p) < 0 or g + c + 2 * p != int(self.batch):
            raise ValueError(f"batch composition {g} general + {c} coverage + 2x{p} alias views "
                             f"does not sum to batch={self.batch}")
        return g, c, p


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
        if n and len(self.population) == 0:
            raise SystemExit(
                f"M17 REFUSED: bucket {self.name!r} is empty but the registered batch "
                f"composition asks for {n} of it. A short batch is a different dose, not a "
                "recoverable shortage.")
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
        if int(st.get("n", len(self.population))) != len(self.population):
            raise SystemExit(
                f"M17 RESUME REFUSED: bucket {self.name!r} held {st.get('n')} rows when the "
                f"checkpoint was written and holds {len(self.population)} now; the saved "
                "position would select different queries.")
        self.passes = int(st["passes"])
        self._reshuffle()
        self.pos = int(st["pos"])


TRAINING_BUCKETS = ("general", "coverage")


def build_streams(buckets, alias_pair_ids, alias_views, families, seed, need=None):
    """General / unpaired-coverage / alias-pair streams from the cache's own bucket labels.

    Only the two TRAINING buckets are drawn from. Anything else the cache carries — notably
    the fixed held-out slice — is deliberately not a stream, and an alias pair whose rows sit
    outside the training buckets is refused rather than quietly trained on: the divergence
    read must never sample rows the optimizer is also seeing.

    An alias pair must be exactly two rows, views `a` and `b`, in one query family. `need` is
    the per-batch demand `{"general": g, "coverage": c, "alias": p}`; a bucket that cannot
    supply one batch is refused here rather than producing a short batch in the loop.
    """
    buckets = np.asarray([str(b) for b in buckets])
    pair_ids = np.asarray([str(p) for p in alias_pair_ids])
    views = np.asarray([str(v) for v in alias_views])
    fams = np.asarray([str(f) for f in families])
    if not (len(buckets) == len(pair_ids) == len(views) == len(fams)):
        raise SystemExit("M17 REFUSED: bucket / pair-id / view / family arrays disagree on length")
    is_alias = pair_ids != ""
    trainable = np.isin(buckets, TRAINING_BUCKETS)
    general = np.flatnonzero((buckets == "general") & ~is_alias)
    coverage = np.flatnonzero((buckets == "coverage") & ~is_alias)

    pairs = {}
    for i in np.flatnonzero(is_alias):
        pairs.setdefault(pair_ids[i], []).append(int(i))
    bad = {}
    for pid, rows in pairs.items():
        if not all(trainable[i] for i in rows):
            bad[pid] = f"rows in buckets {sorted({buckets[i] for i in rows})}, not training rows"
        elif len(rows) != 2:
            bad[pid] = f"{len(rows)} rows, expected exactly two views"
        elif sorted(views[i] for i in rows) != ["a", "b"]:
            bad[pid] = f"views {sorted(views[i] for i in rows)}, expected ['a', 'b']"
        elif len({fams[i] for i in rows}) != 1:
            bad[pid] = f"two query families {sorted({fams[i] for i in rows})}, expected one"
    if bad:
        raise SystemExit("M17 REFUSED: malformed alias pairs " + json.dumps(bad, sort_keys=True))
    ordered = sorted(pairs)
    pair_index = np.asarray([sorted(pairs[p], key=lambda i: views[i]) for p in ordered],
                            dtype=np.int64).reshape(-1, 2) if ordered \
        else np.zeros((0, 2), dtype=np.int64)

    streams = {"general": Stream("general", general, seed),
               "coverage": Stream("coverage", coverage, seed),
               "alias": Stream("alias", np.arange(len(pair_index)), seed)}
    for name, want in (need or {}).items():
        have = len(streams[name].population)
        if want and have < want:
            raise SystemExit(
                f"M17 REFUSED: bucket {name!r} holds {have} rows but one batch needs {want}. "
                "An undersupplied bucket is a dose the pre-lock rule has to shrink, not a "
                "batch to shorten silently.")
    return streams, pair_index, {"n_pairs": len(pair_index),
                                 "n_alias_rows": int(is_alias.sum()),
                                 "views_checked": True}


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
    """The registry's `training.loss_definition`, term by term.

    Returns `(total, parts, terms)`: `parts` are the unweighted floats the history records,
    `terms` the WEIGHTED tensors as they enter the total, so the pre-lock diagnostic can take
    each term's gradient on the rows without a second forward pass.
    """
    parts, terms = {}, {}
    cos = (1.0 - (q_s * q_t).sum(1)).mean()
    parts["cosine"] = float(cos.detach())
    terms["cosine"] = cfg.cosine_weight * cos
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
        terms["listwise"] = cfg.listwise_weight * kl
        total = total + cfg.listwise_weight * kl

    if cfg.init_anchor_weight > 0:
        eff = model.rows if not model.learned_weights else \
            model.token_weights().unsqueeze(1) * model.rows
        anchor = (eff - eff_init).pow(2).mean()
        parts["anchor"] = float(anchor.detach())
        terms["anchor"] = cfg.init_anchor_weight * anchor
        total = total + cfg.init_anchor_weight * anchor

    if arm["alias_consistency"] and pair_slots is not None and len(pair_slots):
        a, b = pair_slots[:, 0], pair_slots[:, 1]
        alias = (2.0 / q_s.shape[0]) * (1.0 - (q_s[a] * q_s[b]).sum(1)).sum()
        parts["alias"] = float(alias.detach())
        terms["alias"] = cfg.alias_weight * alias
        total = total + cfg.alias_weight * alias
    return total, parts, terms


def grad_shares(model, terms):
    """Each loss term's gradient norm ON THE ROWS, and its share of the total.

    `training.alias_pre_lock_diagnostic` asks for the alias term's share of the total row
    gradient norm, and `training.optimizer.note` for the anchor's share of the row UPDATE norm.
    The update norm is NOT observable after Adam's per-parameter scaling, so what is recorded is
    the anchor's gradient share, and this note says so rather than implying the other number.
    """
    norms, live = {}, []
    for name, t in terms.items():
        if t is None or not getattr(t, "requires_grad", False):
            norms[name] = 0.0
            continue
        live.append(t)
        g = torch.autograd.grad(t, model.rows, retain_graph=True, allow_unused=True)[0]
        norms[name] = 0.0 if g is None else float(g.detach().norm())
    # The DENOMINATOR is the norm of the total row gradient, `||grad_rows sum(terms)||`, which
    # is what `training.alias_pre_lock_diagnostic` asks for (Sol step-5 P3-9). Dividing by the
    # sum of the per-term norms reported 50/50 for two exactly opposed terms whose total
    # gradient is zero. The per-term norms are kept, and their proportions are reported
    # separately as `component_norm_fraction`.
    total = 0.0
    if live:
        g = torch.autograd.grad(sum(live), model.rows, retain_graph=True, allow_unused=True)[0]
        total = 0.0 if g is None else float(g.detach().norm())
    comp = sum(norms.values())
    return {"row_grad_norms": {k: round(v, 8) for k, v in norms.items()},
            "shares": {k: (round(v / total, 6) if total else None) for k, v in norms.items()},
            "total_row_grad_norm": round(total, 8),
            "component_norm_fraction": {k: (round(v / comp, 6) if comp else None)
                                        for k, v in norms.items()},
            "sum_of_term_norms": round(comp, 8),
            "note": "each share is that term's row-gradient norm divided by the norm of the "
                    "TOTAL row gradient on this step's batch, with the registered weights "
                    "applied; shares need not sum to one, and exceed one where terms cancel. "
                    "`component_norm_fraction` is the per-term norm's share of their sum. The "
                    "anchor's share of the row UPDATE norm is not observable after Adam's "
                    "per-parameter scaling; this is its gradient share."}


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
    path = admit_read(path)
    meta = read_meta(path)
    if "weights_folded" not in meta or "learned_weights" not in meta:
        raise SystemExit(f"M17 REFUSED: {path} does not state `weights_folded` and "
                         "`learned_weights` explicitly; an unmarked artifact is not evidence "
                         "that it is the unfolded checkpoint.")
    if meta.get("weights_folded"):
        raise SystemExit(f"M17 REFUSED: {path} is a FOLDED release artifact. The warm start must "
                         "be the unfolded training checkpoint; a folded table cannot resume "
                         "training and folding its scalars again double-scales every row.")
    if not meta.get("learned_weights", True):
        raise SystemExit(f"M17 REFUSED: {path} has no learned scalars; the registered warm start "
                         "is the unfolded p35w-2m-s2500 checkpoint.")
    t, rev = meta.get("teacher"), meta.get("teacher_revision")
    # Checked in EVERY mode: the rehearsal bypasses only the status gate and the FREEZE hash.
    if (t, rev) != (reg["teacher"], reg["teacher_revision"]):
        raise SystemExit(f"M17 REFUSED: warm start was distilled from {t}@{str(rev)[:12]} but the "
                         f"registry pins {reg['teacher']}@{reg['teacher_revision'][:12]}.")
    fz = freeze()
    want_sha = fz.get("training_checkpoint_sha256")
    got_sha = sha_file(path)
    if rehearsal:
        print(f"[m17] REHEARSAL: warm-start FREEZE hash check BYPASSED for {path} "
              f"(file sha {got_sha[:12]}, m7/FREEZE.json records {str(want_sha)[:12]})")
    elif got_sha != want_sha:
        raise SystemExit(
            f"M17 REFUSED: warm start {path} hashes {got_sha[:12]} but m7/FREEZE.json's "
            f"training_checkpoint_sha256 is {str(want_sha)[:12]}. Matching teacher strings are "
            "not lineage; only these bytes are p35w-2m-s2500's unfolded checkpoint.")

    z = np.load(path)
    rows = z["rows_fp16"].astype(np.float32)
    w = z["token_weights"]
    if expect_vocab is not None and rows.shape[0] != expect_vocab:
        raise SystemExit(f"M17 REFUSED: warm start has {rows.shape[0]} rows, expected "
                         f"{expect_vocab} base rows before the extension.")
    if w.shape != (rows.shape[0],) or not np.isfinite(w).all() or (w <= 0).any():
        raise SystemExit(f"M17 REFUSED: warm start scalars have shape {w.shape} for "
                         f"{rows.shape[0]} rows, or are not finite and positive; the unfolded "
                         "checkpoint carries one positive scalar per row.")
    model = QueryTable(rows, weight_init=w, learned_weights=True)
    lineage = {"path": str(path), "meta": meta, "vocab": int(rows.shape[0]),
               "dim": int(rows.shape[1]), "checkpoint_sha256": got_sha,
               "freeze_run_id": fz.get("run_id"),
               "freeze_training_checkpoint_sha256": want_sha,
               "freeze_hash_verified": (not rehearsal) and got_sha == want_sha,
               "rehearsal_bypass": bool(rehearsal)}
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
    # The status gate belongs to the callable driver, not only to the CLI: a caller that
    # imports `run()` must meet the same bar as `python m17src/train.py`.
    require_executable(registry(), cfg.rehearsal, what=f"arm {cfg.arm} ({cfg.run_id})")
    out = Path(admit_write(out_dir))   # before anything is created under it
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
    g_n, c_n, p_n = cfg.batch_shape()
    streams, pair_index, alias_report = build_streams(
        data["buckets"], data["alias_pair_ids"], data["alias_views"], data["families"], cfg.seed,
        # Every arm DRAWS the registered alias pairs; only VL-A adds their loss term. So the
        # undersupply refusal applies to every arm, or a C run would quietly wrap a handful of
        # pairs several times inside one batch and call it the registered dose.
        need={"general": g_n, "coverage": c_n, "alias": p_n})
    heldout = np.asarray(data["heldout_idx"], dtype=np.int64)[:cfg.heldout_queries]
    _check_heldout(heldout, streams, pair_index, cfg)
    heldout_sha = _heldout_sha(heldout)

    state = {"step": 0, "history": [], "flags": [], "snapshots": {},
             "train_running": {"cosine": None, "listwise": None}}
    ck = out / "recovery.pt"
    if resume and ck.exists():
        state, eff_init = _resume(ck, model, opt, streams, cfg, eff_init, log, heldout_sha)

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
    def heldout_components():
        """The SAME components the training accumulator tracks: cosine and listwise only.

        The anchor is a regularizer on the rows, not a fit statistic, and the alias term needs
        paired slots the held-out slice does not have; including either would make the
        divergence comparison depend on a term the two sides do not share.
        """
        model.eval()
        tot = {"cosine": 0.0, "listwise": 0.0}
        n = 0
        for lo in range(0, len(heldout), cfg.batch):
            sl = heldout[lo:lo + cfg.batch]
            if not len(sl):
                break
            _, parts, _terms = step_losses(sl, None)
            tot["cosine"] += parts["cosine"] * len(sl)
            tot["listwise"] += parts.get("listwise_kl", 0.0) * len(sl)
            n += len(sl)
        model.train()
        return _monitored({k: v / max(1, n) for k, v in tot.items()})

    t0 = time.time()
    last_ck = t0
    while state["step"] < cfg.steps:
        state["step"] += 1
        s = state["step"]
        lr_factor = set_lr(s)
        idx, slots = batch_indices()
        loss, parts, terms = step_losses(idx, slots)
        # The registered pre-lock diagnostic, at step 1 and at every divergence read: each
        # term's gradient norm on the rows, from this step's own graph, no extra forward.
        read = (s == 1 or s % cfg.check_every == 0 or s == cfg.steps)
        gshares = grad_shares(model, terms) if read else None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Separate running accumulators, carried in `state` so they survive a resume.
        rm = state["train_running"]
        for key, val in (("cosine", parts["cosine"]),
                         ("listwise", parts.get("listwise_kl", 0.0))):
            rm[key] = val if rm.get(key) is None else 0.99 * rm[key] + 0.01 * val

        if read:
            tr, ho = _monitored(rm), heldout_components()
            state["history"].append({"step": s, "train_running_mean": tr["monitored"],
                                     "heldout": ho["monitored"],
                                     "train_components": tr, "heldout_components": ho,
                                     "parts": parts, "lr_factor": lr_factor,
                                     "grad_shares": gshares,
                                     "passes": {k: v.passes for k, v in streams.items()}})
            _flag_divergence(state, log)
            log(f"  [{cfg.run_id}] step {s}/{cfg.steps} train {tr['monitored']:.5f} heldout "
                f"{ho['monitored']:.5f} { {k: round(v, 5) for k, v in parts.items()} } "
                f"{(time.time() - t0) / s * 1000:.0f} ms/step")

        if s in tuple(cfg.snapshot_steps):
            p = out / f"snapshot_{s:06d}.npz"
            _save_table(p, model, cfg, data, step=s)
            state["snapshots"][str(s)] = str(p.name)
            log(f"  [{cfg.run_id}] step-bound snapshot {p.name}")

        if (time.time() - last_ck) / 60.0 >= cfg.checkpoint_minutes_max or s == cfg.steps:
            _checkpoint(ck, model, opt, streams, state, cfg, data, eff_init, heldout_sha)
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
                     "candidate_cache_sha256": cfg.cache_sha256,
                     "candidate_cache_artifact_sha256": cfg.cache_artifact_sha256},
        "batch_composition": {"general": g_n, "unpaired_coverage": c_n, "alias_pairs": p_n,
                              "batch": cfg.batch},
        "bucket_passes": {k: v.state() for k, v in streams.items()},
        "alias_pairs": alias_report,
        "heldout": {"n": int(len(heldout)), "metric": "cosine + listwise, no anchor, no alias"},
        "history": state["history"], "divergence_flags": state["flags"],
        "snapshots": state["snapshots"],
        "wall_clock_seconds": round(time.time() - t0, 3),
    }
    record["sha256"] = sha_json({k: v for k, v in record.items() if k != "sha256"})
    write_json(out / "run_record.json", record)
    _save_table(out / "endpoint.npz", model, cfg, data, step=state["step"])
    return record


def _monitored(components):
    """The divergence statistic: the two shared fit components, summed."""
    c = {k: float(v or 0.0) for k, v in components.items() if k in ("cosine", "listwise")}
    return {**c, "monitored": sum(c.values())}


def _check_heldout(heldout, streams, pair_index, cfg):
    """The registered held-out slice: 2000 unique training-source queries, disjoint from training.

    A rehearsal runs on tens of queries and says so; a real run may not silently report a
    zero-loss divergence read taken over an empty slice.
    """
    if len(heldout) == 0:
        raise SystemExit("M17 REFUSED: the held-out slice is empty; the divergence check would "
                         "report zero loss forever (registry overfit_divergence_check).")
    if len(set(heldout.tolist())) != len(heldout):
        raise SystemExit("M17 REFUSED: the held-out slice repeats queries; the registered slice "
                         "is 2000 UNIQUE queries.")
    # The alias stream's population is PAIR indices; its rows are the pair map's entries.
    training = {int(i) for k in ("general", "coverage") for i in streams[k].population}
    training.update(int(i) for i in np.asarray(pair_index).reshape(-1))
    overlap = training & set(heldout.tolist())
    if overlap:
        raise SystemExit(f"M17 REFUSED: {len(overlap)} held-out queries are also in a training "
                         "bucket; the divergence read must not see rows the optimizer sees.")
    if not cfg.rehearsal and len(heldout) != int(cfg.heldout_queries):
        raise SystemExit(f"M17 REFUSED: held-out slice has {len(heldout)} queries, the registry "
                         f"pins {cfg.heldout_queries}.")


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
    """Unfolded training-shape table plus the M17 identity sidecar (`m7src.table.save_table`).

    `save_table` keeps the unfolded rows as FP16 only. The step-bound snapshots are averaged
    later, and two genuinely different snapshots can round to the same FP16 rows, so this
    writes the FP32 unfolded rows alongside them (`rows_fp32`); `export.effective_rows` folds
    those when they are present. The FP16 array stays for the legacy readers.
    """
    from table import Preproc, save_table
    path = Path(path)
    pre = Preproc(**data["preproc"])
    save_table(path, model, pre, meta={
        "m17_run_id": cfg.run_id, "m17_arm": cfg.arm, "m17_step": int(step),
        "m17_seed": cfg.seed, "m17_rehearsal": bool(cfg.rehearsal),
        "tokenizer_sha256": cfg.tokenizer_sha256,
        "vocabulary_sha256": cfg.vocabulary_sha256,
        "candidate_cache_sha256": cfg.cache_sha256,
        "teacher": data.get("teacher"), "teacher_revision": data.get("teacher_revision"),
        "weights_folded": False, "rows_fp32_present": True})
    z = dict(np.load(path))
    z["rows_fp32"] = model.rows.detach().float().cpu().numpy()
    np.savez(path, **z)


# The configuration fields a resume must agree on: a checkpoint written under a different
# arm, seed, dose, temperature, learning rate or cache is a DIFFERENT experiment, not a run to
# continue. (`device`, `rehearsal`, `checkpoint_minutes_max` and `run_id` are not on this list.)
RESUME_BOUND_FIELDS = ("arm", "seed", "phase", "steps", "batch", "general_views",
                       "coverage_views", "alias_pairs", "warmup_steps", "temperature",
                       "cosine_weight", "listwise_weight", "alias_weight", "init_anchor_weight",
                       "rows_lr", "weights_lr", "tokenizer_sha256", "vocabulary_sha256",
                       # the recipe identity AND the artifact digest: two internally valid
                       # caches can share the recipe (P2-17)
                       "cache_sha256", "cache_artifact_sha256",
                       # the divergence read is part of the protocol: its cadence, its size and
                       # the registered snapshot window all change what a resumed run reports.
                       "check_every", "heldout_queries", "snapshot_steps")


def _heldout_sha(heldout):
    """Sha of the ORDERED held-out slice: same length, different queries is a different read."""
    return sha_array(np.asarray(heldout, dtype=np.int64))


def _checkpoint(path, model, opt, streams, state, cfg, data, eff_init, heldout_sha):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "streams": {k: v.state() for k, v in streams.items()},
                "state": state, "cfg": asdict(cfg),
                "torch_rng": torch.get_rng_state(),
                "tokenizer_sha256": cfg.tokenizer_sha256,
                "heldout_sha256": heldout_sha,
                "bound": {k: getattr(cfg, k) for k in RESUME_BOUND_FIELDS},
                "eff_init": eff_init.detach().cpu(),
                "vocab": int(model.rows.shape[0]),
                "dim": int(model.rows.shape[1])}, tmp)
    tmp.replace(path)


def _resume(path, model, opt, streams, cfg, eff_init, log, heldout_sha):
    """Restore a run — and refuse a checkpoint that belongs to a different one.

    Returns `(state, eff_init)`: the anchor initialization comes BACK from the checkpoint, so
    the anchor keeps measuring drift from where this run started rather than from wherever the
    resumed model happens to be.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    shape = (int(model.rows.shape[0]), int(model.rows.shape[1]))
    if (ck.get("vocab"), ck.get("dim")) != shape:
        raise SystemExit(f"M17 RESUME REFUSED: checkpoint table is "
                         f"{(ck.get('vocab'), ck.get('dim'))}, this run's is {shape}; a "
                         "different table is a different run.")
    if ck.get("tokenizer_sha256", "") != cfg.tokenizer_sha256:
        raise SystemExit(
            f"M17 RESUME REFUSED: checkpoint tokenizer {ck.get('tokenizer_sha256', '')[:12]!r} "
            f"!= this run's {cfg.tokenizer_sha256[:12]!r}. Schedule and step identity are only "
            "meaningful for the same tokenizer and table.")
    bound = ck.get("bound") or {}
    differ = [k for k in RESUME_BOUND_FIELDS if bound.get(k) != getattr(cfg, k)]
    if not bound:
        raise SystemExit("M17 RESUME REFUSED: the checkpoint records no bound configuration; "
                         "it predates the resume identity check and cannot be trusted to be "
                         "this experiment.")
    if differ:
        raise SystemExit(f"M17 RESUME REFUSED: the checkpoint's configuration differs on "
                         f"{differ}; recovery continues one experiment, it does not adopt "
                         "another arm, seed, dose or cache.")
    if ck.get("heldout_sha256") != heldout_sha:
        raise SystemExit(
            f"M17 RESUME REFUSED: the held-out slice hashes to {heldout_sha[:12]} but the checkpoint "
            f"was written for {str(ck.get('heldout_sha256'))[:12]}. A same-length slice of "
            "different queries makes the divergence history incomparable.")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    for k, v in streams.items():
        if k in ck["streams"]:
            v.load_state(ck["streams"][k])
    torch.set_rng_state(ck["torch_rng"].to(torch.uint8) if hasattr(ck["torch_rng"], "to")
                        else ck["torch_rng"])
    saved_init = ck.get("eff_init")
    if saved_init is None or tuple(saved_init.shape) != shape:
        raise SystemExit("M17 RESUME REFUSED: the checkpoint carries no usable anchor "
                         "initialization; recomputing it from the resumed model would silently "
                         "reset the anchor to the current rows.")
    eff_init = saved_init.to(eff_init.device).to(eff_init.dtype)
    state = ck["state"]
    state.setdefault("train_running", {"cosine": None, "listwise": None})
    log(f"  resumed at step {state['step']} from {path.name}")
    return state, eff_init


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
    ap.add_argument("--checkpoint-minutes", type=float, default=None,
                    help="recovery-checkpoint interval in minutes; may only LOWER the "
                         "registered training.checkpoint_minutes_max")
    ap.add_argument("--rehearsal", action="store_true",
                    help="synthetic tiny end-to-end rehearsal under work/m17/rehearsal")
    args = ap.parse_args(argv)

    reg = registry()
    status = require_executable(reg, args.rehearsal, what=f"arm {args.arm}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # `--rehearsal` without `--data` is the synthetic fixture world. `--rehearsal` WITH a
    # prepared directory is the pre-lock smoke of the REAL data path (m17/STATUS.md step 5):
    # it bypasses the status gate and the FREEZE hash only, and `load_warm_start` still prints
    # the checkpoint's own sha so the smoke is never mistaken for a lineage-verified run.
    if args.rehearsal and not args.data:
        import rehearse17
        return rehearse17.main(["--out", args.out or str(WORK / "rehearsal"),
                                "--device", device])

    if not args.data:
        raise SystemExit("--data is required for a real run: the prepared candidate cache and "
                         "training arrays (see m17src/cache.py and m17/STATUS.md step 5).")
    over = {k: v for k, v in (("steps", args.steps), ("batch", args.batch),
                              ("device", device)) if v is not None}
    if args.checkpoint_minutes is not None:
        # A shorter recovery interval is an operational choice (the resume smoke needs one);
        # a longer one would weaken the registered maximum, so it is refused. The interval is
        # deliberately NOT on RESUME_BOUND_FIELDS: it changes no step the optimizer takes.
        cap = float(reg["training"]["checkpoint_minutes_max"])
        # `nan` compares false against every bound, so it passed both checks and disabled
        # time-based checkpoints for the whole run (Sol step-5 P3-10).
        if not math.isfinite(args.checkpoint_minutes):
            raise SystemExit(f"M17 REFUSED: --checkpoint-minutes {args.checkpoint_minutes} is "
                             "not a finite number of minutes.")
        if args.checkpoint_minutes > cap:
            raise SystemExit(f"M17 REFUSED: --checkpoint-minutes {args.checkpoint_minutes} is "
                             f"above the registered checkpoint_minutes_max {cap}.")
        if args.checkpoint_minutes < 0:
            raise SystemExit("M17 REFUSED: --checkpoint-minutes must not be negative.")
        over["checkpoint_minutes_max"] = float(args.checkpoint_minutes)
    if args.rehearsal:
        over["rehearsal"] = True
    cfg = RunCfg.from_registry(reg, args.arm, seed=args.seed, **over)
    data = json.loads(admit_read(Path(args.data) / "prepared.json").read_text())
    data["registry_status"] = status
    out = Path(args.out or (WORK / "runs" / cfg.run_id))
    prepared = _load_prepared(Path(args.data), data, cfg, reg)
    rec = run(cfg, prepared, out, resume=not args.no_resume)
    print(f"run record: {out / 'run_record.json'} ({rec['wall_clock_seconds']}s)")
    return 0


def _check_locked_config(cfg, reg):
    """A real run uses the registered phase dose and seeds; nothing else is a registered arm."""
    tr = reg["training"]
    ok_steps = (int(tr["screen_steps"]), int(tr["final_steps_per_fresh_run"]))
    if cfg.steps not in ok_steps:
        raise SystemExit(f"M17 REFUSED: --steps {cfg.steps} is neither the registered screen "
                         f"{ok_steps[0]} nor the final {ok_steps[1]} dose.")
    if cfg.batch != int(tr["batch"]):
        raise SystemExit(f"M17 REFUSED: --batch {cfg.batch} is not the locked batch "
                         f"{tr['batch']} (the {tr['fallback_batch_before_recipe_lock_only']} "
                         "fallback is pre-lock only).")
    seeds = (int(tr["seed_screen"]), int(tr["seed_replication"]))
    if cfg.seed not in seeds:
        raise SystemExit(f"M17 REFUSED: seed {cfg.seed} is not a registered seed {seeds}.")


def _check_locked_recipe(d, sidecar, reg, fz=None):
    """The loaded cache's RECIPE must be the locked one (Sol step-5 P1-4).

    `_check_locked_config` compares steps, batch and seed; nothing compared the candidate mixes,
    K, temperature, RNG recipe version or the teacher revision and query preprocessing the cache
    was actually built under. A cache built at 31/16/16 could therefore train after the lock said
    30/17/16, because none of those inputs changes the prepared directory's stage identity.
    """
    import cache as m17cache
    parts = (sidecar.get("identity") or {}).get("parts") or {}
    cand = parts.get("candidates") or {}
    tr = reg["training"]
    fz = fz or freeze()
    want = {
        "candidate_k": int(tr["candidate_k"]),
        "candidate_mix_labeled": tr["candidate_mix_labeled"],
        "candidate_mix_query_only": tr["candidate_mix_query_only"],
        "rng": tr["candidate_construction"]["rng"],
        "rng_recipe_version": m17cache.RNG_RECIPE_VERSION,
        # the whole registered construction block (tie breaking, positive choice, backfill,
        # bank sampling): a locked change to any of it is a different cache (Sol re-check P1-4)
        "candidate_construction": tr["candidate_construction"],
    }
    got = {"candidate_k": cand.get("k"), "candidate_mix_labeled": cand.get("mix_labeled"),
           "candidate_mix_query_only": cand.get("mix_query_only"), "rng": cand.get("rng"),
           "rng_recipe_version": cand.get("rng_recipe_version"),
           "candidate_construction": cand.get("construction")}
    teacher = parts.get("teacher") or {}
    want["teacher"] = {"model": reg["teacher"], "revision": reg["teacher_revision"]}
    got["teacher"] = {"model": teacher.get("model"), "revision": teacher.get("revision")}
    pre = teacher.get("query_preprocessing") or {}
    spec = fz["encoder_spec"]
    # the COMPLETE preprocessing object the builder records (`prepare_data.teacher_preprocessing`),
    # including the Dense head, tokenizer and both precisions — not a hand-picked subset
    want["teacher_query_preprocessing"] = {
        "teacher": reg["teacher"], "revision": reg["teacher_revision"],
        "instruction": spec["query_prefix"], "max_length": int(spec["max_length"]),
        "pooling": spec["pooling"], "post_dense": spec.get("post_dense"),
        "encode_dtype": "float32", "storage_dtype": "float16", "normalized": "l2",
        "tokenizer": "the teacher's own tokenizer (" + spec["tokenizer_id"] + ")"}
    got["teacher_query_preprocessing"] = {k: pre.get(k) for k in want["teacher_query_preprocessing"]}
    extra = sorted(set(pre) - set(want["teacher_query_preprocessing"]))
    if extra:
        raise SystemExit(f"M17 REFUSED: the cache's teacher preprocessing carries fields the locked "
                         f"recipe does not know ({extra}); rebuild the prepared directory.")
    differ = sorted(k for k in want if want[k] != got[k])
    if differ:
        raise SystemExit(
            f"M17 REFUSED: the candidate cache in {d} was built under a different registered "
            f"recipe than the locked registry ({differ}): built {[got[k] for k in differ]}, "
            f"locked {[want[k] for k in differ]}. Rebuild the prepared directory with "
            "prepare_data.py; a locked recipe is not a relabelling of old lists.")


def _check_protected_screen(d, manifest, cfg):
    """A prepared directory is trainable only with a COMPLETED protected screen.

    Astra step-5 P1-3: `state: deferred_to_clock` stayed usable after the registry became
    executable. Ruling A4 allows exactly one exception — the pre-clock `--rehearsal --data`
    smoke — and requires it to say, loudly, how many unscreened new-source rows it is training
    on, so the smoke is never mistaken for admitted training.
    """
    screen = manifest.get("protected_screen") or {}
    state = screen.get("state")
    if state == "complete":
        return screen
    rows = screen.get("unscreened_new_source_rows") or {}
    if not cfg.rehearsal:
        raise SystemExit(
            f"M17 REFUSED: {d}/prepared.json records protected_screen.state={state!r}, not "
            "'complete'. A real run trains only on inputs the m10 protected screen admitted; "
            "re-run prepare_data.py --protected-screen inside the executor.")
    print(f"[m17] *** PRE-CLOCK SMOKE ON UNSCREENED DATA: protected_screen.state={state!r}. "
          f"{rows.get('pool', '?')} pool rows and {rows.get('bank', '?')} bank documents come "
          "from the deferred new source and NO protected screen has seen them. This is a "
          "rehearsal of the real data path (ruling A4), not admitted training, and it produces "
          "no registered observation. ***", flush=True)
    return screen


def _verify_prepared_hashes(d, manifest, arrays, arm):
    """Verify the recorded artifact hashes and cross-artifact alignment BEFORE the model.

    Astra step-5 P2-12: the hashes were recorded and never checked, so a same-shaped `bank.npy`,
    `teacher_q.npy`, `student_ids.json` or `new_rows.npy` was accepted, and array verification
    alone proved nothing about alignment with those files.
    """
    want = manifest.get("hashes") or {}
    if not want:
        raise SystemExit(f"M17 REFUSED: {d}/prepared.json records no artifact hashes.")
    ids = json.loads(admit_read(d / "student_ids.json").read_text())
    teacher_q = np.load(admit_read(d / "teacher_q.npy"))
    bank = np.load(admit_read(d / "bank.npy"))
    checks = [("student_ids", sha_file(d / "student_ids.json")),
              ("tokenizer", sha_file(admit_read(d / manifest["tokenizer"]))),
              ("teacher_q", sha_array(teacher_q)),
              ("bank_vector_bytes", sha_array(bank))]
    if arm["vocab_extension"] and manifest.get("new_rows"):
        checks.append(("new_rows", sha_array(np.load(admit_read(d / manifest["new_rows"])))))
    for name, got in checks:
        exp = want.get(name)
        # Sol step-5 P1-3: `if exp` let a manifest with a MISSING hash pass unverified, which
        # is the state a hand-edited or cross-build manifest is in. Every check is required.
        if not exp:
            raise SystemExit(
                f"M17 REFUSED: {d}/prepared.json records no `{name}` hash; an unverifiable "
                "artifact is not a prepared directory this driver trains from.")
        if got != exp:
            raise SystemExit(
                f"M17 REFUSED: {name} in {d} hashes {got[:12]} but prepared.json records "
                f"{str(exp)[:12]}; these are not the artifacts this data directory describes.")
    n = arrays["candidate_ids"].shape[0]
    if not (len(ids) == teacher_q.shape[0] == n):
        raise SystemExit(
            f"M17 REFUSED: {d} holds {n} candidate rows, {teacher_q.shape[0]} teacher vectors "
            f"and {len(ids)} student id lists; the three must describe the same queries.")
    if bank.shape[0] and arrays["candidate_ids"].max() >= bank.shape[0]:
        raise SystemExit(f"M17 REFUSED: {d} candidate ids reach row "
                         f"{int(arrays['candidate_ids'].max())} of a {bank.shape[0]}-row bank.")


def _check_cache_belongs_here(d, manifest, sidecar):
    """The loaded cache must be the one THIS prepared directory was built with.

    Sol step-5 P1-3: `candidates.npz` + `cache.json` copied from another build verified against
    their own sidecar and were then trained on beside this directory's student ids, teacher
    vectors and bank. The sidecar's recipe identity and artifact digest are compared with what
    `prepared.json` recorded, and the artifact INPUTS with the teacher/bank bytes actually
    present here plus the manifest's v1 digest.
    """
    want = manifest.get("hashes") or {}
    for name, got in (("cache_identity", sidecar["identity"]["sha256"]),
                      ("cache_artifact", str(sidecar.get("artifact_sha256") or ""))):
        exp = want.get(name)
        if not exp:
            raise SystemExit(f"M17 REFUSED: {d}/prepared.json records no `{name}`; the cache in "
                             "this directory cannot be tied to the manifest that describes it.")
        if got != exp:
            raise SystemExit(
                f"M17 REFUSED: the cache in {d} reports {name} {got[:12]} but prepared.json "
                f"records {str(exp)[:12]}; this is not this directory's cache.")
    inputs = sidecar.get("artifact_inputs") or {}
    if not inputs:
        raise SystemExit(f"M17 REFUSED: {d}/cache.json records no `artifact_inputs`; the "
                         "teacher, bank and v1 realizations its lists were scored from are "
                         "unknown.")
    actual = {
        "teacher_q_sha256": sha_array(np.load(admit_read(d / "teacher_q.npy"))),
        "bank_vector_bytes_sha256": sha_array(np.load(admit_read(d / "bank.npy"))),
        "bank_doc_ids_sha256": sha_texts(
            json.loads(admit_read(d / "bank_ids.json").read_text())),
        "v1_q_sha256": want.get("v1_q"),
    }
    for name, got in actual.items():
        exp = inputs.get(name)
        if not exp or not got:
            raise SystemExit(f"M17 REFUSED: {d}/cache.json does not state `{name}`; the cache "
                             "cannot be bound to the arrays in this directory.")
        if got != exp:
            raise SystemExit(
                f"M17 REFUSED: the cache in {d} was scored from {name} {str(exp)[:12]} but this "
                f"directory holds {got[:12]}; training would mix two builds.")


def _load_prepared(data_dir, manifest, cfg, reg):
    """Materialize the arrays `prepared.json` points at, arm-aware, with verified identities.

    The arm decides the table: a control arm (C/L) pointed at an expanded prepared directory
    must NOT silently get the extended table and extended ids, and an expanded arm (V/VL/VL-A)
    without new rows is not that arm at all.
    """
    d = Path(data_dir)
    arm = ARMS[cfg.arm]
    if not cfg.rehearsal:
        _check_locked_config(cfg, reg)
    if arm["vocab_extension"] and not manifest.get("new_rows"):
        raise SystemExit(f"M17 REFUSED: arm {cfg.arm} extends the vocabulary but {d} carries no "
                         "`new_rows`; an unextended table is a different arm.")
    if not arm["vocab_extension"] and manifest.get("new_rows"):
        raise SystemExit(f"M17 REFUSED: arm {cfg.arm} is a control with no vocabulary extension "
                         f"but {d} carries `new_rows`; prepare the control's own data.")

    _check_protected_screen(d, manifest, cfg)
    import cache as m17cache
    arrays, sidecar = m17cache.load(d)             # verifies the stored per-array hashes
    _verify_prepared_hashes(d, manifest, arrays, arm)
    _check_cache_belongs_here(d, manifest, sidecar)
    if not cfg.rehearsal:
        _check_locked_recipe(d, sidecar, reg)
    out = dict(manifest)
    out.update({
        "ids": json.loads(admit_read(d / "student_ids.json").read_text()),
        "teacher_q": np.load(admit_read(d / "teacher_q.npy")),
        "bank": np.load(admit_read(d / "bank.npy")),
        "candidate_ids": arrays["candidate_ids"],
        "teacher_scores": arrays["teacher_scores"],
        "buckets": arrays["buckets"],
        "alias_pair_ids": arrays["alias_pair_ids"],
        "alias_views": arrays["alias_views"],
        "families": arrays["families"],
    })
    # Identity comes from the verified artifacts, never from whatever the manifest claims.
    cfg.cache_sha256 = sidecar["identity"]["sha256"]
    # The recipe identity says which cache this is MEANT to be; two internally valid caches
    # built by different scoring paths share it. Resume binds to the artifact digest, which
    # covers the stored arrays and the teacher/bank/v1 inputs they were scored from (P2-17).
    cfg.cache_artifact_sha256 = str(sidecar.get("artifact_sha256") or "")
    if not cfg.rehearsal and not cfg.cache_artifact_sha256:
        raise SystemExit(f"M17 REFUSED: {d}/cache.json records no `artifact_sha256`; it predates "
                         "the artifact binding and cannot identify which cache bytes were used.")
    cfg.tokenizer_sha256 = sha_file(admit_read(d / manifest["tokenizer"]))
    # Controls keep the base vocabulary; they need a stable NON-EMPTY identity or the export's
    # completeness rule (a missing field is not a match) would refuse the matched control.
    cfg.vocabulary_sha256 = (sha_file(admit_read(d / manifest["new_rows"]))
                             if arm["vocab_extension"] else "base-vocab:" + cfg.tokenizer_sha256)
    cfg.warm_start = str(manifest["warm_start"])

    model, lineage = load_warm_start(d / manifest["warm_start"], expect_vocab=int(reg["base_vocab"]),
                                     device="cpu", rehearsal=cfg.rehearsal)
    if arm["vocab_extension"]:
        model = extend_model(model, np.load(admit_read(d / manifest["new_rows"])), "cpu")
    out["model"], out["lineage"] = model, lineage
    return out


if __name__ == "__main__":
    raise SystemExit(main())
