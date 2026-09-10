# Stage 1 review triage — 2026-09-10

Source: `research/m13-codex-impl-review-2026-09-10.md` (Part A re-checks the design review's 13
findings; Part B is 13 new findings, B1–B8 P1). Rule applied (`m11/STATUS.md` §Scope note): keep a
finding where the fix also catches a plausible accident by one researcher on one box; drop the
malicious-actor fortress. Fixes go in two passes with disjoint files, then a re-review.

## Fix — scoring transaction (`score13.py`, `access13.py`, `rehearse13.py`, tests)

| Finding | Fix |
|---|---|
| B1 `--preflight-only` dispatches into a continuation | check `preflight_only` before any dispatch; delete the pre-tag payload-check option (R1 settled it) |
| B2/B3/A3 + ruling R11 | write a BEGIN-bound run manifest BEFORE the tag (checkpoint sha, system, code identity, registry sha, BEGIN commit, expected datasets, attempt counter); every persisted row and every continuation/recovery validates against it; a bridge failure is persisted as terminal for that dataset and is never retried; zero-score continuation admissible under R11's conditions; the remote tag must point at the BEGIN commit |
| A1 query text and qrels unauthenticated inside | hash the loaded query texts and qrels against the manifest fields that exist, on the exact objects scored |
| A2 frozen-cache-only | pin stella id, revision and dtype in `Config`; require every shard present and recorded before calling `encode_cached`; never encode; persist the consumed cache identity |
| A4 six-set files opened outside the phase | one loader function for `results/frozen_eval/<six>` that asserts the tag phase; remove the option that allowed pre-tag reads |
| A12 evidence_for ×4 | independent try/except per conjunct; persist the error; a failed unreachable conjunct is omitted, a reached failure is `unscorable` per the registry |
| B13 reserved failure → success | when the reserved batch is required and not runnable, the run ends `INCOMPLETE_RESERVED` with a nonzero exit; the six-set decisions stay durable. M9 mode becomes an explicit refusal ("not wired until recipe decisions are fixed"), not a half-working path |
| B12 offline | the rehearsal CLI sets HF offline and `local_files_only=True` |

## Fix — build controller (`build13.py`, `build_lock.py`, `trainer10.py`, `corpus_loader.py`, tests)

| Finding | Fix |
|---|---|
| B4 extension completion not crash-consistent | persist an in-flight phase record before training the extension; publish macro, checkpoint hash and completion in one atomic state write; reconcile on restart before any decision |
| B5 extension restarts AdamW and desynchronizes the mix phase | carry the optimizer state into the extension; the trainer takes a global step offset for `mix_window`/data position and a local step for the LR; the complete evaluation history is passed so the kill rule sees it; extension seeds derive from (seed, j) — RNG state is not carried, disclosed |
| B6 gate accepts `{}`, veto never applied | gate record must carry `executed: true`, `decision`, both checkpoint shas and the E1 verdict hash; a veto selects bs32; the verdict's registry hash must equal the live registry |
| B7 ceiling caps estimates | refuse a mandatory plan above the ceiling; refuse `--benchmark` once cycle 1 has started; before each extension, spend = elapsed wall-clock × billed price (the runtime proxy for the bill, disclosed) plus outstanding mandatory reserve plus the next cycle must fit under the ceiling |
| B8 failed export still `complete` | export or parity failure ends `FROZEN_UNVERIFIED`, never `complete`; finalization is resumable |
| B9 ragged tail dropped forever | the tail documents form a final short batch each epoch, so every eligible document is presented once per epoch (R5); tested on a non-divisible pool |
| B10 validator/fingerprint gaps | pin `extension_examples`, require `require_all_forms`, validate the document-policy subfields; the fingerprint hashes `corpus_loader.py` and `data10.py` too |
| B11 default artifacts changed | emit `epoch_shuffle`, `document_policy` and `n_losses` only when the feature is on, so the default manifest and checkpoint schema are byte-identical again; add non-divisible, two-extension, publication-window and >200-step sidecar tests |

## Dropped, and why

- **Isolated checkout with copied source for the rehearsal (B12, design #11).** Git is the source
  of truth for what ran; the fixture exercises the code path, not provenance. A copied checkout adds
  a second thing that can drift. Offline mode is kept.
- **Carrying torch RNG state across an extension (B5).** The data order is a pure function of the
  global step; only dropout draws differ, disclosed in the record. Optimizer state IS carried.
- **Billed cost from the provider inside the controller (B7).** Not readable at runtime; the
  elapsed × price proxy is recorded and the bill reconciles it in the allocation table.
- **Generic M9 executor now (B13).** Wired later, behind the recipe-decision gate, as a thin entry
  point sharing the scoring helpers; today it refuses.
