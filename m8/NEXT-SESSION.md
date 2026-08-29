# M8 worklist

**Read `m8/STATUS.md`, then `m8/LEDGER.md` §15's audit entry (2026-08-29), then the `D2` row in
`m8/registry.json`. This file is the remaining work only.**

Re-routed 2026-08-29 by a milestone audit: `D2` is the primary lever and everything else is
deferred behind it. One item needs Dylan and is at the top of STATUS; nothing else is blocked.

## 1. `D2` — the milestone's primary lever. Registered, not yet started.

The registry row is the authority. Order of work:

1. **Coverage precondition first — it is cheap and it can disqualify a vocabulary.** Train both
   tokenizers (65,536 / 131,072, multi-word merges), tokenize the pinned dev queries and the pool,
   report the update histogram. A vocabulary with >20% never-updated dev-reachable rows does not
   run.
2. **Smoke both chains at ~90 steps** before any full arm (G5). New code path, no execution
   history — that is where the bug is.
3. **Selection**: one seed per vocabulary, read on the wikipedia + heldout groups only. The bar's
   endpoint (out-of-domain macro) is never read here.
4. **Reported set**: the finalist at 3 seeds against R0. **R0's three chains already exist** as
   §23's diagonal — `m8nf-seed0` and `m8nfb-seed{1,2}-{b,a}`. Verify they read as full chains
   before use; do not substitute an A-leg-only arm.
5. **Bar 0.00519 dense** (chain floor, adopted in the audit amendment). Fused and worst-group are
   descriptive.

**Kill criterion, and it must be allowed to fire.** One continuation at doubled steps if the
adequacy gate trips, with a paired doubled-step R0. Then the class closes. No third budget, no
vocabulary escalation after a number, no re-tokenized retry.

## 2. `NF-CROSSED-FUSED` — optional, cheap, do it if D2 is not ready to launch

Nine fused scoring passes over cells already on disk, no training. If it lands before `D2` runs, a
§15 amendment promotes `D2`'s fused read to a barred endpoint and restores the intersection-union
type-I protection `B3` and `E14-HEAD` had.

## 3. `E10` — time-boxed, and it does not gate `D2`

The remedy artifact stays unpinned; the review's BLOCKERs stand (`m8/LEDGER.md` §15, "E10
REOPENED"). A shadow can stop a bad artifact shipping but may never be a selection surface, so it
cannot raise the number. **Either** rebuild it properly — union screen across roles, an independent
acceptance detector, a canary that proves acceptance can fail, length-adaptive matching — **or**
declare M8 ran without a shadow and record that as a stated limitation. A third rebuild attempt is
not worth a session.

## 4. Deferred until a lever clears — do not start these

`freeze.py` / `final_run.py` (§4.4's gap list), all recipe/data probes (`B8`, `B9`, `B13`, `B14`,
`B15`, `B6`, `D-GENRE`, `D-SYNTH`, `R-LIST`, `R-PHASE`), `D-FINEWEB` (pool-varying, bar not
computable), `R1-ASSEMBLY`, `S-SELECT`, `E14-LORA`. Reasons are on each registry row's
`plan_status`. **No new floors, guards or registry machinery until `D2` has a number.**

If `D2` misses, the one named alternate is `B10`/`pool_mode` — a redesigned arm, never a revival
of M7's dead one, with its own bar. If that misses too, the pre-committed exit applies: M8 closes
as a measurement and the reserved four stay unspent.

## Session rules

Sonnet subagents for mechanical work; **Codex adversarially before anything becomes permanent** —
it has returned BLOCKERs on an implementation, a STOP on a shadow, and two over-claims in an
already-committed write-up. **Commit and push before launching guarded work**, and treat
`m8/LEDGER.md` and `m8/registry.json` as frozen for the duration of any guarded run (CODEMAP 24).
Smoke every new path; check the RATE, not just that it runs. `setsid nohup` for anything long.
`git status m7/ results/` after anything out of `m7src/`.
