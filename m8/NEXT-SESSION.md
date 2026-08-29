# M8 worklist (rewritten 2026-08-29, after Phase 0 and the five owner rulings)

**Read `m8/STATUS.md` first, then `m8/LEDGER.md` (the protocol authority), then
`m8/registry.json` (the executable half). This file is the remaining work only — everything
finished lives in LEDGER/RESULTS/EXPLORED/CODEMAP and is not repeated here.**

**Nothing is blocked on Dylan.** All five decisions are ruled (STATUS has the table, LEDGER §15 the
reasoning). Two rulings created work rather than closing it — items 1 and 2.

## In order

1. **`E14-HEAD` — the milestone's main bet, and runnable now.** Registered, bar frozen at 0.0040 on
   B3's two scalars. An MLP head over the teacher's **cached** document vectors, trained jointly
   with the table; the transformer is never re-run, so it costs a training run rather than millions
   of forward passes. **It must be nonlinear** — a linear doc-side map is provably absorbable into
   the table and would measure nothing while producing a number.
   **Brief a review on the design before running it.** B3 is the evidence that this pays: two
   reviews caught fatal flaws pre-run, and the fix to the primary contrast is visibly what saved the
   result — the retired contrast came out negative on dense.
   **Do not let a null close E14.** An MLP on the final vector cannot recover what the tower
   discarded, so a null is weak evidence about `E14-LORA`, and the registry says so.

2. **`E10-REMEDY` — the shadow, per Dylan's per-question ruling.** Seven surviving slices,
   ~14,034 queries (`writing/dev`, `recreation/dev`, `recreation/test`, `science/dev`,
   `technology/dev`, `lifestyle/dev`, `lifestyle/test`). Three stay dead on community overlap.
   In order: drop leaked **queries and near-duplicate documents** → **re-screen requiring zero
   hits** → hash-pin as a protected partition → add to `protected_filter`'s index → **regenerate
   the fit list** (the current 337,981 predates these) → register the use limit.
   **Review the remediation before trusting it**: a shadow that is quietly still contaminated is
   worse than no shadow, because it gives false reassurance right before the one-shot access.

3. **B6-pre with `--head mlp`.** Cheap, and it gates whether anything `E14-HEAD` produces is
   shippable: E3 requires the head to fuse into ONE doc ONNX file as plain nodes, and the PASS on
   record used `--head linear`.

4. **The crossed B×A seed design.** Six A legs, ~30 minutes. The three chains on disk are the
   diagonal of a 3×3; crossing the three B checkpoints against three A seeds separates B variance,
   A variance given B, and their interaction. **The cheapest thing that would turn the noise-floor
   convention into an actual bound** — the current floor aliases both legs, which biases it
   *downward*, the anti-conservative direction.

5. **`D-FINEWEB` (E13).** Bar unfrozen, so the guard refuses it. Now the more interesting data
   lever, because B3 closed the Phase-A pair-count route and this is a **B-side** one — it feeds the
   pseudo-query pool, and ~12% of the table's 30,522 rows are never updated in training.
   Two strings attached: E13's standing ruling is *measure first, ship-decide later*, and **"a
   wrapper tag — including our own — is not a licence"**; and this arm **triggers the deferred
   patent question** (a general web crawl is the one planned source that could contain patent text),
   so settle that before web-crawl data enters training.

6. **`m8src/freeze.py` and `final_run.py`, then their suites** (§4.4 gap list). The largest
   engineering left, weeks from being needed. The tests cannot precede the modules.

7. **`E14-LORA`** — registered and refused, TBD bar by design. Only after `E14-HEAD` reports, and it
   needs a fresh ruling: the bill (doubled 10.12M pre-encode, hours of pool re-encoding per arm, the
   stella derived-weights licence check, a forced C2 redefinition) is **not** covered by "measure it
   small first".

## Session rules (they have earned their keep every time)

Sonnet subagents for mechanical work; Codex and Fable adversarially for anything about to become
expensive or permanent. Commit and push after every completed item. **Smoke every new path** and
**check the rate**, not just that something is running. `setsid nohup` for anything long — harness
interrupts killed plain background waiters twice on 2026-08-29 while the detached training survived
untouched.
