"""The E14-HEAD patch stack: how a doc-side head reaches frozen M7 training and eval code.

`m7src` is frozen (G3), and `train.Cfg` has no doc-head knob, so the head reaches the loop by
rebinding module globals in a per-arm subprocess. Four rebindings, each with a reason:

  1. `train.infonce`      -> `e14_head.infonce_head` bound to this arm's head. The loss is a COPY,
                             not a wrapper, because the false-negative mask must stay in raw
                             teacher space -- see `e14_head`'s docstring for the reward-hacking
                             channel that closes.
  2. `torch.optim.Adam`   -> a shim that appends the head's parameter group. `train.run` builds its
                             optimizer from a literal list of dicts, so there is no injection point
                             short of the constructor. The head lands in `opt.param_groups`, which
                             means `set_lr`'s warmup/decay applies to it too -- registered, and the
                             schedule is bound into the provenance record.
  3. `dev_eval.doc_vecs`  -> the same tuple with the document array wrapped in `HeadedVecs`, a LAZY
                             slice-transforming view. Materializing headed float32 documents is
                             ~21.4 GB for HotpotQA and ~25.3 GB for the pool, over this box; the
                             wrapper transforms each chunk the scorer asks for and nothing else.
                             Registered as engineering constraint (a).
  4. `train.build_arrays` -> the holdout split, LADDER ARMS ONLY. The reported arms train on the
     + `train.encode_cached` full pair pool.

WHAT MAKES THE LADDER DEV-BLIND BY CONSTRUCTION RATHER THAN BY PROMISE. `train.run()` evaluates
`cfg.eval_components` every `eval_every` steps AND once more unconditionally at the end, and R0's
components include BOTH DENSE endpoint components. A ladder that merely promised to ignore dev
would still have observed it several times per arm, and `eval_every=0` does not help because the
final evaluation is unconditional. So in a ladder process `dev_eval.doc_vecs` and `devsuite.load`
are patched to RAISE -- no dev corpus can be loaded at all -- and `dev_eval.eval_table` is replaced
by the training-holdout InfoNCE, which is the statistic the ladder actually selects on. The
selection number and the endpoint therefore cannot be the same quantity by accident.

WHY THE HOLDOUT STATISTIC USES THE TABLE'S OWN QUERY PATH. It calls `model.encode(texts, pre,
tok=tok)` -- the one query path every consumer shares -- so the statistic is measured on the
artifact rather than on a training-time approximation of it. Its negatives are a FIXED pool sample
drawn once from a fixed seed, identical across arms and across steps, because a selection
statistic whose noise moves between arms selects on that noise.

NOTE ON THE HOLDOUT AND WHAT IT IS NOT. `p35b-2m` already distilled on every one of these queries
and used their positives in its KL candidate sets, so this is held out from E14's Phase A only. It
is a legitimate basis for choosing a learning rate; it is not a clean generalization measurement,
and no number from it belongs in a report.
"""
import hashlib
import json
import sys

import numpy as np

import m8base

REPO = m8base.REPO
WORK = REPO / "work"
RUNS = WORK / "runs"

HOLDOUT_SEED = 20260829         # fixed: every ladder arm holds out the SAME pairs
HOLDOUT_NEG = 8192              # negatives for the holdout statistic, fixed sample
HEAD_SUBCHUNK = 50_000          # rows per head application inside one scorer chunk


def _git(*args):
    import subprocess
    try:
        return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def _git_head():
    return _git("rev-parse", "HEAD") or None


def _git_dirty():
    """Whether the tree that produced this arm had uncommitted changes. Recorded, not refused:
    an arm trained from a dirty tree is not reproducible from a commit, and the reader deserves
    to know that rather than to infer it."""
    out = _git("status", "--porcelain", "--", "m8src", "m7src", "bench")
    return None if out is None else bool(out)


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


class HeadedVecs:
    """A document array that applies the head on read, one scorer chunk at a time.

    `evalkit.topk_arrays` reads `np.ascontiguousarray(doc_vecs[lo:hi])` once per chunk, so this is
    the whole interface it needs. Output dtype is fp16 -- the SAME dtype the raw cached path hands
    the scorer -- so the only difference between a headed arm and its comparator is the head, not
    a precision change riding along with it.
    """

    def __init__(self, base, head, sub=HEAD_SUBCHUNK, counters=None):
        import torch
        self._torch = torch
        self.base, self.head, self.sub = base, head, sub
        self.shape = tuple(base.shape)
        self.dtype = np.dtype("float16")
        self.reads = 0
        self.rows_read = 0
        # counted where the transform actually happens, not where the array is handed over: a
        # `doc_vecs` call proves only that the proxy was CONSTRUCTED. If the scorer never read
        # through it, the arm would have scored nothing headed and the lookup counter would look
        # exactly the same.
        self.counters = counters

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        torch = self._torch
        raw = np.ascontiguousarray(self.base[key])
        flat = raw.ndim == 1
        if flat:
            raw = raw[None, :]
        out = np.empty(raw.shape, dtype=np.float16)
        with torch.no_grad():
            for lo in range(0, len(raw), self.sub):
                hi = min(lo + self.sub, len(raw))
                t = torch.from_numpy(raw[lo:hi]).cuda().float()
                out[lo:hi] = self.head(t).to(torch.float16).cpu().numpy()
        self.reads += 1
        self.rows_read += len(raw)
        if self.counters is not None:
            self.counters["head_rows_transformed"] += len(raw)
        return out[0] if flat else out


def patch_doc_vecs(head, counters=None):
    """A replacement for `dev_eval.doc_vecs` that hands the scorer HEADED documents, lazily.

    Returns the function; the caller assigns it. Both the training driver and the scoring driver
    use this one implementation, because a second copy is a second chance to score an arm's
    documents raw.

    `dev_eval.doc_vecs` IS the choke point, and it is the only one. Every consumer in this
    tree -- `dev_eval.eval_query_vecs`, `multieval.eval_makers` (which is what `compare_full` and
    therefore the dense endpoint runs through), `fused_floor`, every reference row -- obtains
    document vectors from it, and every one of them reaches it as a module ATTRIBUTE, so a
    rebinding here is honoured at all of them. Patching `evalkit.topk_arrays` instead would NOT
    work: `multieval` does `from evalkit import topk_arrays` at import time, so `compare_full`'s
    entire scoring path would keep the original function and would silently score RAW documents
    for a head arm -- the exact failure the registration's constraint (b) is about.

    THE PROXY IS MEMOIZED ON THE UNDERLYING ARRAY, NOT ON THE COMPONENT NAME, and that is
    load-bearing rather than an optimization. `multieval.same_corpus` asserts `vecs_a is vecs_b` to
    confirm that two components sharing a corpus really do share it -- `heldout-train` and
    `heldout-longq` are both the 6.17M-row pool. A fresh proxy per component would fail that
    identity check and abort the scoring pass.
    """
    real = dev_eval_module().doc_vecs
    if getattr(real, "_e14_proxies", None) is not None:
        raise SystemExit("dev_eval.doc_vecs is ALREADY headed; patching it twice would apply the "
                         "head twice and the arm would measure normalize(H(H(d))).")
    proxies, keepalive = {}, []

    def _dv(comp):
        if counters is not None:
            counters["docvecs"] += 1
        t = real(comp)
        base = t[5]
        k = id(base)
        if k not in proxies:
            keepalive.append(base)          # so no id is recycled while a proxy holds it
            proxies[k] = HeadedVecs(base, head, counters=counters)
        return tuple(t[:5]) + (proxies[k],)

    _dv._e14_proxies = proxies
    return _dv


def dev_eval_module():
    sys.path.insert(0, str(REPO / "m7src"))
    sys.path.insert(0, str(REPO / "bench"))
    import dev_eval
    return dev_eval


class Handle:
    """What `install()` returns: the head, the patch counters, and the artifact writer."""

    def __init__(self, head, kind, head_lr, trainable, counters, holdout, arm_kind):
        self.head, self.kind, self.head_lr = head, kind, head_lr
        self.trainable, self.counters = trainable, counters
        self.holdout, self.arm_kind = holdout, arm_kind

    def assert_fired(self):
        """Every patch that MUST have run, did.

        A patch that silently fails to install produces a perfectly ordinary-looking arm that
        measures the comparator -- `b3_pool`'s `POOL PATCH NEVER FIRED` guard exists for the same
        reason, and this probe has four patches instead of one.
        """
        bad = []
        if self.counters["infonce"] == 0:
            bad.append("train.infonce was never called through the patch -- the head never "
                       "touched the loss")
        if self.counters["adam"] == 0:
            bad.append("torch.optim.Adam was never constructed through the shim -- the head's "
                       "parameters were not in the optimizer")
        if self.trainable and self.counters["adam_params_added"] == 0:
            bad.append("the shim ran but added no parameter group -- a trainable head that was "
                       "never optimized is the comparator wearing this arm's name")
        if self.arm_kind == "ladder":
            if self.counters["holdout_evals"] == 0:
                bad.append("no holdout statistic was computed -- the ladder has nothing to select "
                           "on")
            if self.counters["docvecs"] != 0:
                bad.append("a dev corpus was loaded inside a ladder process")
        else:
            if self.counters["docvecs"] == 0:
                bad.append("dev_eval.doc_vecs never ran through the head wrapper -- the in-training"
                           " dev scored RAW documents against headed training")
            if self.counters["head_rows_transformed"] == 0:
                bad.append("the head wrapper was installed but NO document row was ever "
                           "transformed through it -- the proxy was built and never read, so "
                           "nothing scored was headed")
        # Whether the head MOVED, asserted here rather than left to a later collect() pass. The
        # first version proved only that an optimizer was constructed and a parameter group added,
        # which is not the same claim: a trainable head that never moved is the comparator wearing
        # this arm's name, and a frozen head that moved is not a comparator at all.
        moved = self.head_delta()
        if self.trainable and moved == 0.0:
            bad.append("a TRAINABLE head is bit-identical to its initialization after training")
        if not self.trainable and moved != 0.0:
            bad.append(f"a FROZEN head moved by {moved:.3g} -- R0N is not the identity comparator")
        if bad:
            raise SystemExit("E14 PATCH DID NOT FIRE AS REGISTERED:\n  " + "\n  ".join(bad))

    def head_delta(self):
        """How far the head moved from its initialization, so a frozen arm can be SHOWN frozen."""
        import torch
        with torch.no_grad():
            tot = 0.0
            for k, v in self.head.state_dict().items():
                if k in self._init_state:
                    tot = max(tot, float((v - self._init_state[k]).abs().max()))
        return tot

    def snapshot_init(self):
        import torch
        self._init_state = {k: v.detach().clone() for k, v in self.head.state_dict().items()}

    def persist(self, run_id, cfg_seed, extra=None):
        """The head artifact and its provenance bindings.

        A `run_id` does not stop a stale head being paired with a table, so the record binds the
        table, the Phase-B checkpoint and the head state BY SHA256, along with the architecture,
        the lr/seed/schedule and the patch sources that produced it. Registered as engineering
        constraint (c).
        """
        import torch
        sd = self.head.state_dict()
        pt = RUNS / f"{run_id}.head.pt"
        torch.save({"kind": self.kind, "dim": next(iter(sd.values())).shape[-1],
                    "state_dict": {k: v.cpu() for k, v in sd.items()}}, pt)
        table = RUNS / f"{run_id}.npz"
        bcheck = RUNS / "p35b-2m.npz"
        rec = {
            "run_id": run_id,
            "arm_kind": self.arm_kind,
            "head": {"kind": self.kind, "trainable": self.trainable, "lr": self.head_lr,
                     "params": int(sum(p.numel() for p in self.head.parameters())),
                     "hidden_mult": None if self.kind == "lin" else 2,
                     "form": "normalize(d + f(d)), output projection zero-init"},
            "seed": cfg_seed,
            "schedule": "inherits cfg.lr_schedule/warmup_steps via opt.param_groups",
            "sha256": {
                "head_state": _sha_state(sd),
                "head_file": _sha_file(pt),
                "table": _sha_file(table) if table.exists() else None,
                "phase_b_checkpoint": _sha_file(bcheck) if bcheck.exists() else None,
                "e14_head.py": _sha_file(REPO / "m8src" / "e14_head.py"),
                "e14_patch.py": _sha_file(REPO / "m8src" / "e14_patch.py"),
                "e14_run.py": _sha_file(REPO / "m8src" / "e14_run.py"),
            },
            # CODEMAP pitfall 16: a run's meta.json records no code vintage, so nothing tells you
            # two arms were trained under different code. The source hashes above cover this
            # probe's own files; this covers everything else the arm ran through.
            "git_head": _git_head(),
            "git_dirty": _git_dirty(),
            "holdout": (None if self.holdout is None else
                        {"frac": self.holdout["frac"], "n": int(self.holdout["n"]),
                         "seed": HOLDOUT_SEED, "split_sha256": self.holdout["split_sha"],
                         "n_train_after_holdout": int(self.holdout["n_keep"]),
                         "negatives": HOLDOUT_NEG,
                         "negatives_sha256": self.holdout.get("negatives_sha"),
                         "negatives_disjoint_by_construction": (
                             "the reserve is added to train.banned_rows, which train.run removes "
                             "from the negative bank and build_arrays removes from the training "
                             "positives"),
                         "read_at": "sqrt pooling, live fp32 table (a registered proxy for the "
                                    "folded int8 endpoint)"}),
            "patch_counters": dict(self.counters),
            "head_max_abs_move_from_init": self.head_delta(),
            "_what": ("the head that was trained with this table, bound to it by sha256. A run_id "
                      "alone does not stop a stale head being paired with a table."),
        }
        if extra:
            rec.update(extra)
        (RUNS / f"{run_id}.head.json").write_text(json.dumps(rec, indent=1))
        return rec


def install(*, head_kind, head_lr, trainable, arm_kind, seed, holdout_frac=0.0, temp=0.02,
            fn_margin=0.02):
    """Rebind the four sites. Call BEFORE `sweep.one`/`train.run`, after `import m8base`.

    `arm_kind` is "reported" (full pool, headed in-training dev) or "ladder" (holdout split,
    dev-blind by construction).
    """
    import torch

    sys.path.insert(0, str(REPO / "m7src"))
    sys.path.insert(0, str(REPO / "bench"))
    import dev_eval
    import pool as poolmod
    import train

    import e14_head

    if arm_kind not in ("reported", "ladder"):
        raise ValueError(f"unknown arm_kind {arm_kind!r}")
    if arm_kind == "ladder" and not holdout_frac:
        raise ValueError("a ladder arm needs a holdout to select on")

    # THE HEAD'S INITIALIZATION MUST BE SEEDED HERE, and this is not a formality. `build_head`
    # runs before `train.run()` reaches its own `torch.manual_seed(cfg.seed)`, and the MLP head's
    # `fc1` weight and bias are RANDOM (only `fc2` is zero-init). Zero `W2` hides them in the
    # initial output but the first `W2` gradient already depends on them, so without this the
    # three MLP arms would have differed in their initialization as well as in their treatment,
    # the arms would not have been reproducible from their recorded seeds, and the seed pairing
    # the whole read rests on would have been false for MLP. `lin` is unaffected (zeros_ is
    # deterministic) -- which is exactly why this stayed invisible through a smoke that only ran
    # `lin`. Found by an adversarial review of the implementation, reproduced here before any
    # reported arm ran.
    torch.manual_seed(seed)
    head = e14_head.build_head(poolmod.DIM, kind=head_kind, device="cuda")
    if not trainable:
        for p in head.parameters():
            p.requires_grad_(False)
        head.eval()

    counters = {"infonce": 0, "adam": 0, "adam_params_added": 0, "docvecs": 0,
                "holdout_evals": 0, "head_rows_transformed": 0}
    hold = {"frac": holdout_frac} if holdout_frac else None
    h = Handle(head, head_kind, head_lr, trainable, counters, hold, arm_kind)
    h.snapshot_init()

    # ---- 1. the loss -------------------------------------------------------------------------
    def _infonce(qv, pos_v, neg_v, temp_, *a, **kw):
        counters["infonce"] += 1
        return e14_head.infonce_head(qv, pos_v, neg_v, temp_, *a, head=head, fn_space="raw", **kw)

    train.infonce = _infonce

    # ---- 2. the optimizer --------------------------------------------------------------------
    _RealAdam = torch.optim.Adam

    def _adam(params, **kw):
        counters["adam"] += 1
        groups = list(params)
        if trainable:
            groups = groups + [{"params": list(head.parameters()), "lr": head_lr}]
            counters["adam_params_added"] += 1
        return _RealAdam(groups, **kw)

    torch.optim.Adam = _adam

    # ---- 3/4. evaluation ---------------------------------------------------------------------
    if arm_kind == "reported":
        dev_eval.doc_vecs = patch_doc_vecs(head, counters)
    else:
        def _refuse(*a, **kw):
            raise SystemExit("a ladder process attempted to load a dev corpus. The ladder selects "
                             "on the training holdout and must never observe the endpoint; this "
                             "is the dev-blindness the registration requires be structural.")

        dev_eval.doc_vecs = _refuse
        import devsuite
        devsuite.load = _refuse

        # THE NEGATIVE RESERVE, MADE DISJOINT FROM TRAINING BY CONSTRUCTION.
        #
        # The first version drew the statistic's negatives uniformly from the pool. That is
        # ~32% contaminated by arithmetic alone: the negative bank is 1,997,601 of 6,169,142 pool
        # rows, and each bank row is sampled ~41 times over 2,500 steps. A trainable document head
        # can lower such a statistic by learning to demote rows it has SEEN rather than by
        # improving retrieval over the 4.17M documents it has not.
        #
        # The clean fix is available because `train.run` removes `banned_rows()` from the negative
        # bank AND `build_arrays` removes them from the training positives. Adding the reserve
        # there makes it unreachable as a training negative and as a training positive, by
        # construction rather than by measurement -- no reconstruction of `bank_ids` required.
        pool_vecs = poolmod.build()[1]
        reserve = np.sort(np.random.default_rng(HOLDOUT_SEED + 1).choice(
            pool_vecs.shape[0], HOLDOUT_NEG, replace=False))
        hold["negatives_sha"] = hashlib.sha256(reserve.tobytes()).hexdigest()
        _real_banned = train.banned_rows

        def _banned():
            arr, bset, third = _real_banned()
            merged = np.union1d(np.asarray(arr, dtype=np.int64), reserve)
            return merged, set(bset) | set(int(x) for x in reserve), third

        train.banned_rows = _banned

        # the holdout split, installed on build_arrays so training can never draw these pairs
        stash = {}
        _real_ba = train.build_arrays

        def _ba(cfg, index, side_index=None):
            q_texts, pos_idx, hn_idx, src_id, srcs = _real_ba(cfg, index, side_index=side_index)
            n = len(q_texts)
            perm = np.random.default_rng(HOLDOUT_SEED).permutation(n)
            nh = int(round(holdout_frac * n))
            hold_i, keep_i = np.sort(perm[:nh]), np.sort(perm[nh:])
            stash["full_q"] = q_texts
            stash["keep"] = keep_i
            stash["hold_q"] = [q_texts[i] for i in hold_i]
            stash["hold_pos"] = np.array([pos_idx[i][0] for i in hold_i], dtype=np.int64)
            stash["hold_allpos"] = [pos_idx[i] for i in hold_i]
            stash["hold_i"] = hold_i
            hold["n"], hold["n_keep"] = nh, len(keep_i)
            hold["split_sha"] = hashlib.sha256(hold_i.tobytes()).hexdigest()
            print(f"  [e14] holdout {nh} pairs of {n} (seed {HOLDOUT_SEED}), "
                  f"split sha {hold['split_sha'][:12]}", flush=True)
            return ([q_texts[i] for i in keep_i], [pos_idx[i] for i in keep_i],
                    [hn_idx[i] for i in keep_i], src_id[keep_i], srcs)

        train.build_arrays = _ba

        # The teacher encode of the training queries is cached under a name carrying the pair
        # COUNT and a hash of the texts, so a 98% split would miss M7's cache and re-encode ~331K
        # queries through a 400M teacher. Encode the FULL set (a cache hit) and take the subset.
        _real_ec = train.encode_cached

        def _ec(name, texts, **kw):
            if name.startswith("trainq-") and "full_q" in stash:
                full = np.asarray(_real_ec(f"trainq-{len(stash['full_q'])}", stash["full_q"], **kw))
                # the held-out queries' TEACHER vectors, which the false-negative mask needs
                stash["hold_tq"] = full[stash["hold_i"]]
                return full[stash["keep"]]
            return _real_ec(name, texts, **kw)

        train.encode_cached = _ec

        # THE STATISTIC, built to match the objective rather than to be convenient.
        #
        # Four things the first version got wrong, all found by review before any reported arm:
        #   * negatives overlapped the training bank -> now the reserve above, disjoint by
        #     construction, with its id-set hash bound into the arm's provenance;
        #   * it pooled `mean` while the endpoint reads `sqrt` -> now sqrt;
        #   * it disabled the own-positive, all-positive and false-negative masks, on the stated
        #     ground that masks would "drift between arms". THAT REASONING WAS WRONG: the id masks
        #     are index comparisons and the false-negative mask is computed in RAW teacher space,
        #     so all three are arm-INDEPENDENT. Omitting them rewarded a head for demoting a
        #     query's own siblings. All three are restored, `fn_margin` at the arm's own value.
        #   * (not repaired, declared) it reads the live fp32 table while the endpoint reads the
        #     folded int8 release artifact. Folding mid-training would mean running the release
        #     path on an unfinished artifact every eval; this is registered as an explicit fp32
        #     proxy, and it is one more reason this statistic is descriptive and no longer selects.
        sqrt_pre = None

        def _eval_table(model, pre, components=None, tok=None):
            nonlocal sqrt_pre
            counters["holdout_evals"] += 1
            if sqrt_pre is None:
                from dataclasses import asdict as _asdict
                from table import Preproc as _Preproc
                sqrt_pre = _Preproc(**{**_asdict(pre), "pool_mode": "sqrt"})
            n = len(stash["hold_q"])
            with torch.no_grad():
                qv = torch.from_numpy(
                    model.encode(stash["hold_q"], sqrt_pre, tok=tok)).cuda().float()
                pos_v = torch.from_numpy(
                    np.ascontiguousarray(pool_vecs[stash["hold_pos"]])).cuda().float()
                neg_v = torch.from_numpy(np.ascontiguousarray(pool_vecs[reserve])).cuda().float()
                tq = torch.from_numpy(np.ascontiguousarray(stash["hold_tq"])).cuda().float()
                mx = max(len(p) for p in stash["hold_allpos"])
                allpos = np.full((n, mx), -1, dtype=np.int64)
                for r, ps in enumerate(stash["hold_allpos"]):
                    allpos[r, :len(ps)] = ps
                loss = float(e14_head.infonce_head(
                    qv, pos_v, neg_v, temp, tq, pos_v, fn_margin, head=head, fn_space="raw",
                    neg_pool_idx=torch.from_numpy(reserve).cuda(),
                    pos_pool_idx=torch.from_numpy(stash["hold_pos"]).cuda(),
                    all_pos_idx=torch.from_numpy(allpos).cuda()))
            # negated: `macro` is read as higher-is-better everywhere in this harness
            return {"holdout-infonce": {"0": -loss}}

        dev_eval.eval_table = _eval_table

    return h
