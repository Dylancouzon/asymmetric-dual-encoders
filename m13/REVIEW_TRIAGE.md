# Review triage — 2026-09-10

## Stage 2: the LoTTE gate script and the deferred DEV-6 read (Astra, `research/m13-codex-gate-review-2026-09-10.md`)

All nine findings accepted. Fixes in `lotte_gate13.py`, `dev6_from_checkpoint.py`, `run_arm.py`,
`build13.py` and their tests; rulings R17 and R18 (`m13/RULINGS.md`); registration amendment.

| Finding | Fix |
|---|---|
| 1 P1 read repeatable after a crash | exclusive receipt (O_EXCL) binding every input before the first LoTTE open; per-slice outputs persisted; plain re-run refuses; `--recover` completes the same read under the identical identity, reading only unfinished slices |
| 2 P1 mutable arm records stand in for the manifest | `--write-manifest` materialises the lock's second manifest commit as `m13/LOTTE_GATE_MANIFEST.json`; the gate requires it tracked, unmodified and field-equal to the live records; arm name, seed and registry binding checked; `build13.check_gate_manifest` binds the gate record to it |
| 3 P1 hash-then-load reopens the file | bytes read once, hashed, deserialised from the same buffer (gate and DEV-6 filler) |
| 4 P1 slice hashes authenticate nothing | R18: `results/m8_lotte_pin.json` required; five hashes and counts compared per slice; duplicate qrel rows and positives refuse. Correction: no pin existed; the M8 pin step is run on the day |
| 5 P1 resumed caches unverified | `encode_cached(verify=True)` |
| 6 P1 refusal leaks qids | count only |
| 7 P2 DEV-6 fill repeatable after a crash | attempt line before the read; `attempts_including_this` disclosed (a development read; disclosed, not refused) |
| 8 P2 `--dev6 defer` on family F | refused for every family but E |
| 9 P2 runbook handoffs and commands | `git pull` steps, interpreter prefixes, manifest and pin steps (`m13/EXECUTION.md`) |
| unverified: quantile method | R17 pins `inverted_cdf` in the registration; the gate refuses without the field |
| unverified: code identity at the end | computed at preflight into the receipt and again before the record; a change refuses |

Second review (Sol, `research/m13-codex-gate-rereview-2026-09-10.md`): NO-GO, seven P1s. All accepted.

| Finding | Fix |
|---|---|
| 1 P1 concurrent `--recover` while the first process reads | non-blocking exclusive `flock` under `work/lotte/gate13/` held for the whole attempt; recovery needs the lock |
| 2 P1 registration accepted any numpy quantile method | equality to `inverted_cdf` required (R17 admits nothing else) |
| 3 P1 recipe fields only required to exist | student, layers, head, objective and mix compared against `m13/build_config.json`'s registered knobs; the manifest carries the full recipe |
| 4 P1 tokenizer and backbone configuration loaded by repository name | the dependency identity (repo, tokenizer and config hashes) is recorded in the manifest and re-derived at load; a mismatch refuses. `nano10` still pins no revision; that is M10 code, disclosed |
| 5 P1 manifest committed but not pushed; pin unchecked | both must be tracked, unmodified, committed and on a remote branch; the pin's counts are required |
| 6 P1 build trusts the gate's `decision` enum | `check_gate` recomputes the veto rule from the recorded bootstrap under the registered margin; the gate record and the manifest must be committed and pushed (`REQUIRE_COMMITTED`) |
| 7 P1 receipt directory not fsynced | directory fsynced after the receipt and after every atomic write |
| 8 P2 recovery trusts persisted slice content | slice digests journaled when written; recovery accepts only a journaled digest, the pin's hashes, the right counts and finite in-range rows |
| 9 P2 two runbook commands without the interpreter | prefixed |
| 10 P3 `check_gate_manifest` reported the wrong commit | reports the gate's `manifest_commit` and verifies it against the live `git log` |

## Stage 1 review triage

**Superseded in part by rulings R13–R16 (same day):** extension cycles and post-tag continuation are
DELETED rather than fixed. B4, B5 and B7 fall away with the extensions; B2/B3's continuation and the
R11 zero-score path fall away with R14. Everything else in the two tables stands.

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
