codex
Not ready for a cold handoff. The handoff files currently direct the wrong work.

- **BLOCKER — VERIFIED:** condition 4 can ship a system whose lookup table did not improve—or regressed behind a tower gain. Modified towers force C2’s system comparison (`m8/registry.json:666`), and `decide.ship()` has no table-quality or artifact-invariant guard (`m8src/decide.py:177-213,216-235`). “Pure lookup” preserves architecture, not table improvement. Freeze-time assertion is not enough: `freeze.py` does not exist (`m8/STATUS.md:83-85`), and `decide.py` explicitly declines enforcement (`m8src/decide.py:187-192`). Require machine-checked artifact provenance and explicitly decide whether table non-regression is required.

- **BLOCKER — VERIFIED:** cold-start state is contradictory. STATUS still asks the now-resolved system/table question (`m8/STATUS.md:7-11`); NEXT says do not start E14-LORA (`m8/NEXT-SESSION.md:81-86`); its row simultaneously says “pending,” `TBD`, comparator `R0`, and “authorised…bar 0.00519” (`m8/registry.json:455-472`).

- **MAJOR — VERIFIED:** compression lost binding content:
  - Missing final patent ruling: **“`D-FINEWEB`’s patent trigger is REMOVED”** (`HEAD~3:m8/LEDGER.md:1519-1525`). Current text wrongly resurrects the trigger (`m8/LEDGER.md:1250-1251`).
  - Missing E14 rule: **“THE LEARNING RATE IS NOW PRE-REGISTERED AT 1e-3 AND THE LADDER SELECTS NOTHING”** (`HEAD~3:m8/LEDGER.md:1998-2007`). Current registry again says the ladder selects (`m8/registry.json:438`).
  - T1’s withdrawn “independent reproduction” claim (`HEAD~3:m8/LEDGER.md:2510-2517`) is compressed back into “the lesson reproduces” (`m8/LEDGER.md:1436-1438`).

- **BLOCKER — INFERRED:** E14’s honest primary comparator is a fresh stock-stella tower **with its table jointly retrained under the identical recipe**, not frozen R0 and not one unchanged table across moving spaces. R0 is secondary. The existing 0.00519 floor covers B×A variation under a fixed document tower (`m8/LEDGER.md:1449-1462`); it does not cover a new document-training leg. Measure/register a D×downstream floor first. Cheapest credible sign: one fixed LoRA configuration, train-only fitting, two OOD corpora only, three paired end-to-end seeds against the stock-tower retraining control.

- **BLOCKER — VERIFIED:** cross-fitting fixes B17’s own-fit leakage, but D2-PRE still has an unfrozen router: “clearly positive,” “plausibly above,” “adequate,” and “no material” are undefined (`m8/registry.json:549-554`). `probe_guard` checks only presence/placeholders, not executability (`m8src/probe_guard.py:96-117`). Lock numeric thresholds, folds, train-only feature selection, arm multiplicity, and the additive-over-D2 reversal margin before running.

Files read: `m8/STATUS.md`, `m8/NEXT-SESSION.md`, `m8/LEDGER.md` at HEAD and `HEAD~3`, `m8/registry.json`, `m8src/decide.py`, `m8src/probe_guard.py`. No protected files read; no files written.
tokens used
125,442
Not ready for a cold handoff. The handoff files currently direct the wrong work.

- **BLOCKER — VERIFIED:** condition 4 can ship a system whose lookup table did not improve—or regressed behind a tower gain. Modified towers force C2’s system comparison (`m8/registry.json:666`), and `decide.ship()` has no table-quality or artifact-invariant guard (`m8src/decide.py:177-213,216-235`). “Pure lookup” preserves architecture, not table improvement. Freeze-time assertion is not enough: `freeze.py` does not exist (`m8/STATUS.md:83-85`), and `decide.py` explicitly declines enforcement (`m8src/decide.py:187-192`). Require machine-checked artifact provenance and explicitly decide whether table non-regression is required.

- **BLOCKER — VERIFIED:** cold-start state is contradictory. STATUS still asks the now-resolved system/table question (`m8/STATUS.md:7-11`); NEXT says do not start E14-LORA (`m8/NEXT-SESSION.md:81-86`); its row simultaneously says “pending,” `TBD`, comparator `R0`, and “authorised…bar 0.00519” (`m8/registry.json:455-472`).

- **MAJOR — VERIFIED:** compression lost binding content:
  - Missing final patent ruling: **“`D-FINEWEB`’s patent trigger is REMOVED”** (`HEAD~3:m8/LEDGER.md:1519-1525`). Current text wrongly resurrects the trigger (`m8/LEDGER.md:1250-1251`).
  - Missing E14 rule: **“THE LEARNING RATE IS NOW PRE-REGISTERED AT 1e-3 AND THE LADDER SELECTS NOTHING”** (`HEAD~3:m8/LEDGER.md:1998-2007`). Current registry again says the ladder selects (`m8/registry.json:438`).
  - T1’s withdrawn “independent reproduction” claim (`HEAD~3:m8/LEDGER.md:2510-2517`) is compressed back into “the lesson reproduces” (`m8/LEDGER.md:1436-1438`).

- **BLOCKER — INFERRED:** E14’s honest primary comparator is a fresh stock-stella tower **with its table jointly retrained under the identical recipe**, not frozen R0 and not one unchanged table across moving spaces. R0 is secondary. The existing 0.00519 floor covers B×A variation under a fixed document tower (`m8/LEDGER.md:1449-1462`); it does not cover a new document-training leg. Measure/register a D×downstream floor first. Cheapest credible sign: one fixed LoRA configuration, train-only fitting, two OOD corpora only, three paired end-to-end seeds against the stock-tower retraining control.

- **BLOCKER — VERIFIED:** cross-fitting fixes B17’s own-fit leakage, but D2-PRE still has an unfrozen router: “clearly positive,” “plausibly above,” “adequate,” and “no material” are undefined (`m8/registry.json:549-554`). `probe_guard` checks only presence/placeholders, not executability (`m8src/probe_guard.py:96-117`). Lock numeric thresholds, folds, train-only feature selection, arm multiplicity, and the additive-over-D2 reversal margin before running.

Files read: `m8/STATUS.md`, `m8/NEXT-SESSION.md`, `m8/LEDGER.md` at HEAD and `HEAD~3`, `m8/registry.json`, `m8src/decide.py`, `m8src/probe_guard.py`. No protected files read; no files written.
EXIT=0
