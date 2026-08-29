# M8 worklist (rewritten 2026-08-29, after E14-HEAD reported and E10 was reopened)

**Read `m8/STATUS.md` first, then `m8/LEDGER.md` (the protocol authority), then
`m8/registry.json` (the executable half). This file is the remaining work only.**

**Nothing is blocked on Dylan.** His open rulings this session: M9 picks its own document tower on
measurement with the PAIR as a tie-break broken only on CI-resolved loss; re-encoding is
acceptable; patents are dropped as a held-out domain.

## What just closed

- **`E14-HEAD`: NO SURVIVOR.** Both heads harm (dense −0.0244 / −0.0293 vs a +0.0040 bar). The
  patch stack measured as a clean null. Read the CORRECTED write-up in `m8/RESULTS.md` — two
  over-claims were withdrawn after review, and the corrected statements are narrower than the
  first version. `E14-LORA` stays registered and REFUSED with a TBD bar; this result is an INPUT
  to that bar, not a licence to write it.
- **The `paths_guard` holes are closed** (reserved CORPUS cache dirs and the `mteb/` name
  spelling). Tests hard-code the routes rather than deriving them from the constant under test.

## In order

1. **`E10` — REOPENED, and this is the largest piece of real work left.** `m8/LEDGER.md` §15
   (2026-08-29, "E10 REOPENED") has the full disposition; the artifact may NOT be pinned, served
   or grandfathered. What the rebuild needs, in dependency order:
   - **Screen each shadow query against the UNION of protected queries AND documents, and each
     shadow document against the union of protected documents AND queries**, across every
     protected family the policy names — not just the four CQADupStack corpora. This is the
     BLOCKER: the current screen compared roles, and two verbatim leaks survived it.
   - **An INDEPENDENT acceptance detector.** The existing one may remove, but cannot certify
     itself: remediation builds the exact complement of its own hits, so the re-screen provably
     returns zero. Use StackExchange IDs/URLs, migration and cross-site links, character
     shingles/containment — and **re-read the SERIALIZED files**, not the in-memory survivors.
   - **A canary test that deliberately leaves or reinserts a known leak and proves acceptance
     FAILS.** Without it, "0 hits" is unfalsifiable.
   - **Length-adaptive matching.** Document detection needs 8 shared entries from bottom-32
     sketches, so a document under 15 normalized words can never be a hit. Freeze thresholds on
     adversarial fixtures BEFORE rerunning.
   - Then, and only then: `freeze_lotte.py pin` (which must bind the hashes of what was SCREENED,
     accept only the registered seven-slice set, and be the sole loader), the `paths_guard`
     partition entry, and the fit list — noting `build_fitlist()` currently reads S0's EMPTY
     `kept` and would silently omit LoTTE entirely.
   - **Correlation with the exam is still unmeasured** and "uncorrelated" is not credible: a
     preregistered candidate panel, rank-correlated across LoTTE dev slices, unused CQADupStack
     subforums and non-reserved entity proxies, keeping LoTTE TEST slices untouched.

2. **The next capacity lever — recommendation: bigram rows trained through the forward.** The
   project's own physics (CLAUDE.md standing directive #4) says only n-gram rows and
   multiplicity-dependent pooling add capacity. B3 closed data volume; E14 closed cheap
   document-side re-shaping. M7 killed the CLOSED-FORM bigram fit at −0.0301 with a specific
   diagnosis — the teacher-ward correction undoes the A-phase gains — which does NOT transfer to
   rows learned by the contrastive objective, and M7's own note leaves the joint-retrain
   escalation open with its own pre-registration. Likely registrable against the EXISTING A-leg
   floor, unlike D-FINEWEB.
   **Write it with a kill criterion that can actually fire.** This is the SECOND lever whose cheap
   version failed and whose expensive escalation is "open" (E14 is the other). That pattern must
   not become a way to never take no for an answer: if the joint version fails, the class closes.

3. **`D-FINEWEB` (E13) — its patent trigger is GONE, but it needs a measurement, not paperwork.**
   Its bar reads `TBD-noise-floor` and the floor does not exist: every arm so far shares one
   pseudo-query pool, so §23's crossed design explicitly does not bound a POOL-VARYING lever. A
   different draw is a different ~925K-span text set and a fresh teacher encode per seed — ~2.3 h
   before the arm is registrable at all. E13's standing ruling is *measure first, ship-decide
   later*, and "a wrapper tag — including our own — is not a licence".

4. **`m8src/freeze.py` and `final_run.py`, then their suites** (§4.4 gap list). The largest
   engineering left; weeks from being needed. The tests cannot precede the modules.

5. **`E14-LORA`** — registered and refused, TBD bar by design. Needs a fresh ruling: the bill
   (doubled 10.12M pre-encode, hours of pool re-encoding per arm, the stella derived-weights
   licence check, a forced C2 redefinition) is not covered by "measure it small first". E14-HEAD's
   result weakens the case FOR it without closing it.

## Session rules (they earned their keep again today)

Sonnet subagents for mechanical work; **Codex adversarially for anything about to become
permanent** — two reviews today returned 5 BLOCKERs on an implementation and a STOP on a shadow,
and a third caught two over-claims in a write-up that had ALREADY been committed. Brief them
BEFORE a conclusion becomes permanent, not after.
**Commit and push before launching guarded work, and treat `m8/LEDGER.md` and `m8/registry.json`
as frozen for the duration of any guarded run** (CODEMAP 24 — the write-time check fails after the
whole job has run). **Smoke every new path**, and check the RATE, not just that something runs.
`setsid nohup` for anything long. **Run `git status m7/ results/` after anything out of `m7src/`.**
