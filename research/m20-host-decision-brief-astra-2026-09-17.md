# M20 execution-host decision — validation brief, Astra, 2026-09-17

You are validating a **decision**, not reviewing code. M20's implementation already passed two
review rounds (`m20/REVIEW_TRIAGE.md`: Astra NO-GO with seven P1s, all fixed; Fable GO). The
reserved access is still **UNSPENT**.

## The situation

The registered execution pin (`m13/RESERVED_EXECUTION.md`, `scripts/m13_reserved_cloud.py`) names
one retained Runpod A100, `k3aee2m68765em`, STOP-only, $1.6636111111/h including storage. On
2026-09-16 the controller passed every preflight and Runpod refused the resume: **"There are not
enough free GPUs on the host machine to start this pod."** It has now refused ~76 times over 6
hours. Nothing was spent: wallet delta $0.00, no `m8-reserved-spent` tag, all three pods `EXITED`.
Receipt: `results/m13_reserved_cloud_attempt1_nocapacity_2026-09-16.json`.

Runpod has plenty of capacity elsewhere, including a fresh A100 SXM 80GB at $1.39/h. The shortage
is specific to the host the stopped pod is pinned to. M13 already lost two hosts to this condition.

The owner wants it unblocked and has approved "your best recommendation", subject to this
validation.

## The three options

1. **Fresh Runpod pod** (A100 SXM 80GB $1.39/h, or a consumer-class card at $0.34–0.69/h).
2. **The local RTX 3080 box**, the machine this repository lives on.
3. **Keep waiting** on the retained host.

## My recommendation: option 2, the local box

The reasoning I want you to attack:

1. **The local environment is not a substitute for the validated one — it is the validated one.**
   `m14/HANDOFF.md` records the M13 release-evidence stack as Torch 2.8.0+cu126, Transformers
   4.57.6, Sentence Transformers 5.7.0, Tokenizers 0.22.2, NumPy 2.3.5, Datasets 5.0.1. The local
   `.venv` reports exactly those, plus `bm25s` 0.3.11 and `PyStemmer` 3.1.0, which are the versions
   `m7/FREEZE.json`'s `bm25_run_keys` pin. A fresh pod is the only option of the three that
   rebuilds this stack, and `HARNESS.md` says `m7/requirements.lock.txt` is "not a validated
   fresh-install recipe for today's complete harness".
2. **It already reproduces the registered numbers.** On this box, `m20/SMOKE.md` records the full
   BEIR-15 path on SciFact reproducing `m12/six_dbsf.json`'s `dense`, `bm25` and `dbsf@100` to
   machine precision and M13's registered six-set Nano, bge-small and LEAF values to the rounding
   of their published six decimals. `results/m20_dbsf_reproduction.json` records three zero deltas.
3. **Speed is not the A100's advantage here.** The registered contract is fp32 with TF32 disabled.
   `results/m20_tower_rate_benchmark.json` measures Stella at 87–98 documents/second on the 3080;
   `results/m13_encode_benchmark.json` measured 101/second on the A100.
4. **It dissolves three open problems at once**: the $1,000 ceiling (stage C had ~6% headroom), the
   $348 wallet against a ~$309 expected bill, and the capacity block itself.
5. **Cost of being wrong is low.** Every stage is resumable and only stage B is protected, and its
   per-system atomic outputs resume under `reserved.crash` (R23).

The price is about 7.5 days of the owner's GPU: roughly 52 hours for the reserved four and 130 for
BEIR-15, from the measured rates.

## What I am asking you to validate

Answer these directly. Essential-only: correctness of the reasoning, evidence or access-discipline
damage, and material risks I have missed. No style or wording findings. If you think the
recommendation is right, say so plainly.

1. **Is the environment claim true and material?** Check the version evidence yourself. Does
   rebuilding the stack on a fresh pod actually risk moving a reported number, or am I overstating
   it? Note `m7src/fusion.py:_pkg_versions` makes a changed BM25 version visible rather than silent.
2. **Does running on the local box violate a registered pin or owner ruling?** `m13/RULINGS.md`
   R19 says "the A100 execution is pushed to M14"; `m13/RESERVED_EXECUTION.md` names the retained
   pod and its cost cap; `CLAUDE.md` says "Box limitations must not reshape an arm; both E arms run
   together on the cloud GPU" and "Check the actual runtime before scheduling GPU work". Does any
   of that forbid local execution of a descriptive evaluation, or does it need a new dated ruling?
   Does the choice of host touch any estimand, weight, threshold, partition or release rule?
3. **What discipline is lost by not running `scripts/m13_reserved_cloud.py`?** Locally, stages A
   and B would be invoked directly (`m8src/pre_encode.py`, then `m13src/score13.py
   --reserved-only`). Enumerate the gates that only exist in the controller, and say which of them
   matter when there is no pod and no bill. `m13src/reserved_transaction.py:preflight` is the other
   half of the gating.
4. **Is the memory risk the right thing to worry about, and is it the only one?** The box has 25 GB
   of RAM and 534 GB free on `/`. BM25 must index MS MARCO's 8.8M documents and FEVER's 5.4M;
   `m7src/fusion.py` says HotpotQA's 5.23M was "the single most expensive repeated step on this
   box". I am measuring peak RSS at two corpus sizes before committing. Is anything else on this
   box a hard blocker rather than a slowdown — VRAM at 10 GB, the 512-token contract, disk?
5. **Is there a better option I have not considered?**

## Hard access rules

- **Do not read** `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`,
  `work/dev/cqadup-english.json`, `work/m9reserve`, `work/lotte`, or any reserved qrels cache. The
  reserved four are FEVER, DBpedia-entity, cqadup-android, cqadup-english.
- **No recursive or repo-wide content searches** across `results/` or `work/`. Open only the files
  below; name any extra file you need rather than globbing for it.
- Read-only. Run nothing that writes. Reading installed package versions with
  `.venv/bin/python -c "import importlib.metadata …"` and `nvidia-smi` is permitted and expected.
- End your report with the exact list of files you opened.

## Files to open

`CLAUDE.md`; `HARNESS.md`; `instructions-m20.md`; `m13/RULINGS.md` (R19, R20, R22, R23);
`m13/RESERVED_EXECUTION.md`; `m14/HANDOFF.md`; `m20/REGISTRATION.md`; `m20/STATUS.md`;
`m20/SMOKE.md`; `m20/REVIEW_TRIAGE.md`; `m8src/pre_encode.py`; `m13src/reserved_transaction.py`;
`m13src/reserved_support.py`; `m20src/roster.py`; `m20src/beir15.py`;
`scripts/m13_reserved_cloud.py`; `m7src/fusion.py`; `results/m20_tower_rate_benchmark.json`;
`results/m13_encode_benchmark.json`; `results/m20_dbsf_reproduction.json`;
`results/m13_reserved_cloud_attempt1_nocapacity_2026-09-16.json`; `m7/FREEZE.json` (the `fusion`
key only).

Return a verdict: **ENDORSE**, **ENDORSE WITH CONDITIONS** (list them), or **REJECT** (name the
better option).
