# Quick adversarial review: M8's plan before we hand off to a new session

Be adversarial and be BRIEF. I want blockers, not an essay. If the plan is sound, say so in a line
and spend your effort on the one thing most likely to be wrong.

**READ-EXCLUSION, binding.** Do NOT read, grep, cat or otherwise open:
`results/frozen_eval/untouched-*`, any `*fever*qrels*` / `*dbpedia*qrels*` HF cache, or
`work/m9reserve/`. These are held-out confirmatory sets. A previous review dumped two of them into
its context via a repo-wide grep. **Name the files you read**; prefer targeted reads over
repo-wide searches. If you need something from a protected set, say so instead of reading it.

Repo `/home/dylan/asymetric-dual-encoders`, branch `m8-planning`, pushed. Read: `m8/STATUS.md`,
`m8/NEXT-SESSION.md`, `m8/LEDGER.md` §5 and §15, `m8/registry.json` (rows `D2`, `D2-PRE`,
`E14-LORA`, `NF-CROSSED-FUSED`, and `ship_rule`), `m8src/decide.py`, `m8src/probe_guard.py`.

## What changed since your last review (all of your BLOCKERs were adopted)

1. `D2` repaired: **sum** init not mean; pool-expansion clause removed; coverage gate rebuilt on
   occurrence mass + a performance condition; `NF-CROSSED-FUSED` mandatory; deterministic tokenizer.
2. `D2-PRE` registered ahead of `D2` — closed-form, cross-fitted, four arms at equal row budget,
   with a registered reversal (if an additive n-gram arm wins, D2 stands down).
3. §13's "n-gram rows superseded by D2, no auto-revival" withdrawn.
4. Exit gated on `D2`, `B10`, `B8`, `R-LIST` all missing plus a re-run of the standing directive.
5. **NEW, and the biggest change: the owner ruled M8 may ship a better SYSTEM.** Ship-rule
   condition 4 now accepts a `qualifying_system` (document-side) key alone; E11/G4-4 superseded.
   `E14-LORA` (LoRA/adapter/last-block on stella) is authorised to be registered and measured.
   Bounds kept: query side stays a pure lookup table; tower must be DERIVED from stella, never
   trained from scratch; report must decompose the win.
6. **The LEDGER was compressed 2,694 → ~1,470 lines** (§15 1,070→~165, §17–23 420→118).

## Attack these four, in priority order

1. **The condition-4 amendment.** Made quickly, under an owner ruling, on a protocol that exists to
   stop exactly this kind of late widening. Does it create a path where a v2 "ships" without the
   artifact this project is about having improved? Are the invariants I kept (pure-lookup-table
   query side; tower derived not from-scratch; report decomposes the win) sufficient, checkable,
   and actually checked anywhere? I deliberately did NOT put the first two in
   `decide.qualifying_table` because they are artifact properties, not config-key facts — is
   "asserted at freeze" good enough when `freeze.py` **does not exist yet**?
2. **Did the LEDGER compression drop anything BINDING?** This is the risk I introduced today and
   cannot audit from inside. The originals are in git (`git show HEAD~3:m8/LEDGER.md`) and the
   pre-compression §15 and §17–23 are recoverable from history. Diff old against new and tell me
   whether any *rule, bar, withdrawn claim, or owner ruling* was lost — I do not care about lost
   prose. Be specific: quote what is missing.
3. **`E14-LORA`'s shape, before it is registered.** Its full row is unwritten — that is next
   session's first task, so this is the moment to get the design attacked rather than the
   registration. What is the right bar, comparator and staging? Specifically: (a) if the document
   tower changes, what is the honest comparator — the same table against stock-stella documents, or
   a jointly retrained table? (b) does the existing chain floor (σ_chain 0.00153 → bar 0.00519)
   even apply when the document space moves, or does that need its own floor? (c) what is the
   cheapest dev-scale arm that could produce a trustworthy sign?
4. **`D2-PRE`'s validity as a screen.** It routes a five-chain spend on closed-form ridge fits.
   `B17` fired a registered branch on a closed-form fit and was disowned for measuring its own fit
   set. Is cross-fitting enough to make `D2-PRE` a trustworthy router, or am I about to repeat
   `B17` with a bigger consequence?

## Constraints

- Verify in the repo; quote `file:line`. Say VERIFIED or INFERRED for each finding.
- BLOCKER / MAJOR / MINOR. Skip MINORs unless they are cheap and real.
- Do not write to any file.
- If you think we are ready to hand off to a fresh session, say that plainly too.
