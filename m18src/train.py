"""M18 new-row-only trainer, forked from M17's numerical objective and resume loop.

The semantic change is load-bearing: inherited unfolded rows/scalars are folded once into an
immutable buffer, while only appended exact-token/placeholder rows are Parameters. There is no
optimizer state capable of moving an inherited row or scalar. The schedule is always 4,000 steps;
``stop_after`` merely pauses execution for checkpoint reads and can be raised on resume without
changing the trajectory identity.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (atomic_save_npz, admit_read, admit_write, registry, require_executable,
                    sha_array, sha_file, sha_json, write_json)

EPS = 1e-6


@dataclass
class RunCfg:
    variant: str = "T0"
    seed: int = 18001
    schedule_steps: int = 4000
    stop_after: int = 500
    batch: int = 256
    warmup_steps: int = 200
    new_rows_lr: float = 3e-4
    temperature: float = 0.05
    cosine_weight: float = 1.0
    listwise_weight: float = 1.0
    alias_weight: float = 0.1
    checkpoint_steps: tuple = (0, 250, 500, 1000, 2000, 4000)
    device: str = "cuda"
    run_id: str = ""
    rehearsal: bool = False
    tokenizer_sha256: str = ""
    preprocessing_sha256: str = ""
    vocabulary_sha256: str = ""
    cache_artifact_sha256: str = ""

    @classmethod
    def from_registry(cls, reg, variant="T0", seed=None, **over):
        tr = reg["training"]
        cfg = cls(
            variant=variant,
            seed=int(tr["seed_primary"] if seed is None else seed),
            schedule_steps=int(tr["schedule_steps"]),
            stop_after=int(tr["diagnostic_stop_after"]),
            batch=int(tr["batch"]),
            warmup_steps=int(tr["warmup_steps"]),
            new_rows_lr=float(tr["new_rows_lr"]),
            temperature=float(tr["temperature"]),
            cosine_weight=float(tr["cosine_weight"]),
            listwise_weight=float(tr["listwise_weight"]),
            alias_weight=float(tr["alias_weight"]),
            checkpoint_steps=tuple(int(x) for x in tr["checkpoint_steps"]),
        )
        for k, v in over.items():
            setattr(cfg, k, v)
        if not cfg.run_id:
            cfg.run_id = f"m18-{variant.lower()}-s{cfg.seed}"
        cfg.validate()
        return cfg

    def validate(self):
        if self.variant not in ("T0", "T1", "T2"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if not 0 <= int(self.stop_after) <= int(self.schedule_steps):
            raise ValueError("stop_after is an execution boundary within schedule_steps")
        if tuple(sorted(set(self.checkpoint_steps))) != tuple(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be unique and sorted")

    def resume_identity(self):
        # stop_after and device are intentionally absent. Raising stop_after continues the exact
        # same long schedule; changing any field here defines a different experiment.
        fields = ("variant", "seed", "schedule_steps", "batch", "warmup_steps",
                  "new_rows_lr", "temperature", "cosine_weight", "listwise_weight",
                  "alias_weight", "checkpoint_steps", "tokenizer_sha256",
                  "preprocessing_sha256", "vocabulary_sha256", "cache_artifact_sha256")
        obj = {k: (list(getattr(self, k)) if k == "checkpoint_steps" else getattr(self, k))
               for k in fields}
        return {"fields": obj, "sha256": sha_json(obj)}


class FrozenExtension(torch.nn.Module):
    """Frozen effective v1 rows plus a trainable appended block; no scalar Parameters."""

    def __init__(self, inherited_effective, new_rows, fallback_id=101):
        super().__init__()
        inherited = torch.as_tensor(inherited_effective, dtype=torch.float32).clone()
        self.register_buffer("inherited", inherited)
        self.new_rows = torch.nn.Parameter(torch.as_tensor(new_rows, dtype=torch.float32).clone())
        self.fallback_id = int(fallback_id)

    @property
    def trainable_start(self):
        return int(self.inherited.shape[0])

    @property
    def rows(self):
        return torch.cat((self.inherited, self.new_rows), dim=0)

    def forward_ids(self, ids_list):
        rows = self.rows
        out = []
        for ids in ids_list:
            if not ids:
                v = rows[min(self.fallback_id, rows.shape[0] - 1)]
            else:
                ids_t = torch.as_tensor(ids, dtype=torch.long, device=rows.device)
                uniq, counts = torch.unique(ids_t, sorted=True, return_counts=True)
                weights = counts.to(torch.float32).sqrt()
                v = (rows[uniq] * weights[:, None]).sum(0) / weights.sum().clamp_min(EPS)
            n = v.norm()
            if float(n.detach().cpu()) <= EPS:
                v = torch.zeros_like(v)
                v[0] = 1.0
            else:
                v = v / n.clamp_min(EPS)
            out.append(v)
        return torch.stack(out)

    def effective_rows(self):
        return self.rows.detach().float().cpu().numpy()


def load_warm_start(path=None, released_model=None, rehearsal=False):
    """Verify the unfolded M7 checkpoint and fold inherited scalars exactly once."""
    reg = registry()
    init = reg["models"]["unfolded_init"]
    p = Path(path or init["source_path"])
    got = sha_file(p)
    if not rehearsal and got != init["sha256"]:
        raise SystemExit(f"M18 REFUSED: unfolded init hashes {got[:12]}, expected "
                         f"{init['sha256'][:12]}")
    meta = json.loads(admit_read(p.with_suffix(".meta.json")).read_text())
    # The historical M7 metadata predates an explicit ``weights_folded`` flag. Its exact bytes
    # are nevertheless pinned by FREEZE and it declares learned weights; any unpinned input must
    # carry the newer explicit false flag.
    explicit_or_frozen = (meta.get("weights_folded") is False
                          or (got == init["sha256"] and init.get("weights_folded") is False))
    if not explicit_or_frozen or meta.get("learned_weights") is not True:
        raise SystemExit("M18 REFUSED: initialization is not verified unfolded-with-scalars")
    z = np.load(admit_read(p))
    rows = z["rows_fp16"].astype(np.float32)
    scalars = z["token_weights"].astype(np.float32)
    if scalars.shape != (rows.shape[0],) or not np.isfinite(scalars).all():
        raise SystemExit("M18 REFUSED: inherited scalar array is malformed")
    effective = scalars[:, None] * rows

    release = Path(released_model or reg["models"]["zero_v1"]["source_path"]) / "model.npz"
    zr = np.load(admit_read(release))
    released = zr["rows_fp16"].astype(np.float32)
    if effective.shape != released.shape:
        raise SystemExit("M18 REFUSED: unfolded and released v1 tables disagree on shape")
    parity = float(np.abs(effective - released).max())
    if parity > 5e-3:
        raise SystemExit(f"M18 REFUSED: unfolded→v1 row parity is {parity:.3e}")
    lineage = {"checkpoint": str(p), "checkpoint_sha256": got,
               "release_model_sha256": sha_file(release), "old_vocab": rows.shape[0],
               "dim": rows.shape[1], "effective_vs_release_fp16_max_abs": parity,
               "inherited_rows_sha256": sha_array(effective),
               "inherited_scalars_sha256": sha_array(scalars)}
    return effective, scalars, lineage


def build_model(inherited_effective, new_rows, fallback_id=101, device="cpu"):
    return FrozenExtension(inherited_effective, new_rows, fallback_id=fallback_id).to(device)


def eligible_query_indices(ids, trainable_start):
    return np.asarray([i for i, row in enumerate(ids)
                       if any(int(t) >= int(trainable_start) for t in row)], dtype=np.int64)


def _losses(model, ids, teacher_q, bank, candidate_ids, teacher_scores, temperature,
            cosine_weight, listwise_weight, alias_slots=None, alias_weight=0.0):
    q = model.forward_ids(ids)
    cosine = (1.0 - (q * teacher_q).sum(1)).mean()
    mask = candidate_ids >= 0
    cvec = bank[candidate_ids.clamp_min(0)]
    s_logits = torch.einsum("bd,bkd->bk", q, cvec) / temperature
    t_logits = teacher_scores / temperature
    neg = torch.finfo(torch.float32).min
    s_log = F.log_softmax(torch.where(mask, s_logits, neg), dim=1)
    t_log = F.log_softmax(torch.where(mask, t_logits, neg), dim=1)
    p = t_log.exp()
    listwise = (p * (t_log - s_log)).masked_fill(~mask, 0).sum(1).mean()
    total = cosine_weight * cosine + listwise_weight * listwise
    alias = torch.zeros((), device=q.device)
    if alias_slots:
        a = torch.as_tensor([x[0] for x in alias_slots], device=q.device)
        b = torch.as_tensor([x[1] for x in alias_slots], device=q.device)
        alias = (1.0 - (q[a] * q[b]).sum(1)).mean()
        total = total + alias_weight * alias
    return total, {"cosine": float(cosine.detach()), "listwise": float(listwise.detach()),
                   "alias": float(alias.detach())}


def _order(n, seed, epoch):
    return np.random.default_rng([int(seed), int(epoch), 18018]).permutation(n)


def _batch(state, n, batch, seed):
    picked = []
    while len(picked) < batch:
        order = _order(n, seed, state["epoch"])
        take = min(batch - len(picked), n - state["position"])
        picked.extend(order[state["position"]:state["position"] + take].tolist())
        state["position"] += take
        if state["position"] == n:
            state["epoch"] += 1
            state["position"] = 0
    return np.asarray(picked, dtype=np.int64)


def _alias_slots(batch_global, pair_ids):
    positions = {}
    for slot, gi in enumerate(batch_global):
        pid = str(pair_ids[int(gi)])
        if pid:
            positions.setdefault(pid, []).append(slot)
    return [tuple(v) for _, v in sorted(positions.items()) if len(v) == 2]


def _lr_factor(step, warmup, total):
    if step <= warmup:
        return step / max(1, warmup)
    return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))


def _atomic_torch_save(path, payload):
    p = Path(admit_write(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_snapshot(root, model, cfg, step, lineage, initial_new_sha):
    """Publish table+metadata as one immutable directory transaction."""
    root = Path(admit_write(root))
    if root.exists():
        meta = json.loads(admit_read(root / "meta.json").read_text())
        if (int(meta["step"]) != int(step)
                or meta["resume_identity_sha256"] != cfg.resume_identity()["sha256"]):
            raise SystemExit(f"M18 SNAPSHOT REFUSED: existing {root} has another identity")
        return meta
    stage = root.with_name(root.name + f".building-{os.getpid()}")
    if stage.exists():
        raise SystemExit(f"M18 SNAPSHOT REFUSED: stale staging directory {stage}")
    stage.mkdir(parents=True)
    rows = model.effective_rows()
    new = model.new_rows.detach().float().cpu().numpy()
    meta = {"_schema": "m18-training-snapshot-v1", "step": int(step),
            "run_id": cfg.run_id, "variant": cfg.variant, "seed": cfg.seed,
            "resume_identity_sha256": cfg.resume_identity()["sha256"],
            "tokenizer_sha256": cfg.tokenizer_sha256,
            "preprocessing_sha256": cfg.preprocessing_sha256,
            "vocabulary_sha256": cfg.vocabulary_sha256,
            "cache_artifact_sha256": cfg.cache_artifact_sha256,
            "trainable_start": model.trainable_start,
            "inherited_rows_sha256": sha_array(model.inherited.detach().cpu().numpy()),
            "inherited_scalars_sha256": lineage["inherited_scalars_sha256"],
            "new_rows_sha256": sha_array(new), "initial_new_rows_sha256": initial_new_sha,
            "new_rows_changed": sha_array(new) != initial_new_sha}
    try:
        atomic_save_npz(stage / "table.npz", rows=rows,
                        inherited=model.inherited.detach().cpu().numpy(), new_rows=new)
        write_json(stage / "meta.json", meta)
        os.replace(stage, root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return meta


def load_snapshot(path):
    p = Path(path)
    meta = json.loads(admit_read(p / "meta.json").read_text())
    z = np.load(admit_read(p / "table.npz"))
    if sha_array(z["inherited"]) != meta["inherited_rows_sha256"]:
        raise SystemExit(f"M18 SNAPSHOT REFUSED: inherited rows changed in {p}")
    if sha_array(z["new_rows"]) != meta["new_rows_sha256"]:
        raise SystemExit(f"M18 SNAPSHOT REFUSED: new-row hash mismatch in {p}")
    return z["rows"], meta


def run(cfg: RunCfg, data, out_dir, resume=True, log=print):
    require_executable(registry(), cfg.rehearsal, what=cfg.run_id, training=True)
    cfg.validate()
    out = Path(admit_write(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    model: FrozenExtension = data["model"].to(cfg.device)
    start = model.trainable_start
    inherited_sha = sha_array(model.inherited.detach().cpu().numpy())
    if inherited_sha != data["lineage"]["inherited_rows_sha256"]:
        raise SystemExit("M18 REFUSED: inherited effective rows differ from verified initialization")
    ids_all = data["ids"]
    eligible = eligible_query_indices(ids_all, start)
    if len(eligible) == 0:
        raise SystemExit("M18 REFUSED: no training query contains an active trainable row")
    if len(eligible) != len(ids_all):
        raise SystemExit(f"M18 REFUSED: prepared pool contains {len(ids_all)-len(eligible)} "
                         "zero-gradient queries; filter by the variant's active ids first")
    if len(eligible) < cfg.batch:
        raise SystemExit(f"M18 REFUSED: {len(eligible)} eligible queries < batch {cfg.batch}")

    teacher_q = torch.as_tensor(data["teacher_q"], dtype=torch.float32, device=cfg.device)
    bank = torch.as_tensor(data["bank"], dtype=torch.float32, device=cfg.device)
    cand = torch.as_tensor(data["candidate_ids"], dtype=torch.long, device=cfg.device)
    tscores = torch.as_tensor(data["teacher_scores"], dtype=torch.float32, device=cfg.device)
    pair_ids = np.asarray(data.get("alias_pair_ids", [""] * len(ids_all)), dtype=str)
    initial_new_sha = sha_array(model.new_rows.detach().float().cpu().numpy())

    opt = torch.optim.Adam([model.new_rows], lr=cfg.new_rows_lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0.0)
    state = {"step": 0, "epoch": 0, "position": 0, "history": []}
    recovery = out / "recovery.pt"
    if resume and recovery.exists():
        payload = torch.load(admit_read(recovery), map_location=cfg.device, weights_only=False)
        if payload["resume_identity"]["sha256"] != cfg.resume_identity()["sha256"]:
            raise SystemExit("M18 RESUME REFUSED: schedule/data identity changed")
        if payload["inherited_rows_sha256"] != inherited_sha:
            raise SystemExit("M18 RESUME REFUSED: inherited rows changed")
        model.new_rows.data.copy_(payload["new_rows"].to(cfg.device))
        opt.load_state_dict(payload["optimizer"])
        state = payload["state"]
        initial_new_sha = payload["initial_new_rows_sha256"]

    if state["step"] == 0:
        save_snapshot(out / "snapshots" / "step_000000", model, cfg, 0, data["lineage"],
                      initial_new_sha)
    t0 = time.time()
    while state["step"] < cfg.stop_after:
        state["step"] += 1
        step = state["step"]
        local = _batch(state, len(ids_all), cfg.batch, cfg.seed)
        slots = _alias_slots(local, pair_ids)
        ii = torch.as_tensor(local, dtype=torch.long, device=cfg.device)
        loss, parts = _losses(model, [ids_all[i] for i in local], teacher_q[ii], bank,
                              cand[ii], tscores[ii], cfg.temperature, cfg.cosine_weight,
                              cfg.listwise_weight, slots, cfg.alias_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(model.new_rows.grad).all():
            raise SystemExit(f"M18 REFUSED: non-finite new-row gradient at step {step}")
        factor = _lr_factor(step, cfg.warmup_steps, cfg.schedule_steps)
        for group in opt.param_groups:
            group["lr"] = cfg.new_rows_lr * factor
        opt.step()
        if step == 1 or step in cfg.checkpoint_steps:
            row = {"step": step, "loss": float(loss.detach()), **parts,
                   "lr_factor": factor, "epoch": state["epoch"],
                   "alias_pairs_in_batch": len(slots),
                   "inherited_rows_sha256": sha_array(model.inherited.detach().cpu().numpy())}
            if row["inherited_rows_sha256"] != inherited_sha:
                raise SystemExit(f"M18 REFUSED: inherited rows moved at step {step}")
            state["history"].append(row)
            log(f"[{cfg.run_id}] {step}/{cfg.schedule_steps} loss={row['loss']:.5f} "
                f"lr={factor:.4f}")
        if step in cfg.checkpoint_steps:
            save_snapshot(out / "snapshots" / f"step_{step:06d}", model, cfg, step,
                          data["lineage"], initial_new_sha)
        if step in cfg.checkpoint_steps or step == cfg.stop_after:
            _atomic_torch_save(recovery, {
                "resume_identity": cfg.resume_identity(), "state": state,
                "new_rows": model.new_rows.detach().cpu(), "optimizer": opt.state_dict(),
                "inherited_rows_sha256": inherited_sha,
                "initial_new_rows_sha256": initial_new_sha})

    peak = (int(torch.cuda.max_memory_allocated(cfg.device)) if str(cfg.device).startswith("cuda")
            and torch.cuda.is_available() else 0)
    record = {"_schema": "m18-run-record-v1", "run_id": cfg.run_id,
              "config": {**asdict(cfg), "checkpoint_steps": list(cfg.checkpoint_steps)},
              "resume_identity": cfg.resume_identity(), "step": state["step"],
              "eligible_queries": len(eligible), "history": state["history"],
              "inherited_rows_sha256": inherited_sha,
              "inherited_scalars_sha256": data["lineage"]["inherited_scalars_sha256"],
              "new_rows_changed": sha_array(model.new_rows.detach().cpu().numpy()) != initial_new_sha,
              "wall_clock_seconds_this_invocation": round(time.time() - t0, 3),
              "peak_vram_bytes": peak}
    record["sha256"] = sha_json({k: v for k, v in record.items() if k != "sha256"})
    write_json(out / "run_record.json", record)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=("T0", "T1", "T2"), default="T0")
    ap.add_argument("--data", required=True, help="torch-saved prepared data dictionary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    over = {"device": args.device}
    if args.stop_after is not None:
        over["stop_after"] = args.stop_after
    cfg = RunCfg.from_registry(registry(), args.variant, seed=args.seed, **over)
    data = torch.load(admit_read(args.data), map_location="cpu", weights_only=False)
    run(cfg, data, args.out, resume=not args.no_resume)


if __name__ == "__main__":
    main()
