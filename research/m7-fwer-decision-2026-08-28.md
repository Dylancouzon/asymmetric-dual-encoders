# The tier rule's familywise error claim — a decision for Dylan

**Status: OPEN. It must close BEFORE the freeze**, because it is a registered rule and a registered
rule may only move before its numbers exist. Nothing else is blocked by it.

Written 2026-08-28 in response to the pre-freeze one-shot-path review
(`research/m7-codex-onepath-2026-08-28.md`, MAJOR 5), which said the ledger's "family bounded at
α = 0.025" claim was not delivered by the procedure. The review was right that the claim as written
was unearned. It was wrong about the size — but only measurement could show that, and the
measurement did not exist when the review was made.

---

## 1. What the rule is

A tier win in the final run requires **all three** legs, one-sided, over a family of three
comparisons (C1 released int8 table > lr-dense-pertask · C2 > BM25 · C3 released system >
OpenSearch doc-v3-gte):

1. **Holm-corrected sign-flip rejection** at family α = 0.025 — so the smallest p must clear
   0.025/3 = 0.008333 to reject at rank 1;
2. **raw paired bootstrap CI lower bound > 0** (a 2.5% one-sided endpoint);
3. **raw one-sided lower bound at the Bonferroni level α/3 = 0.008333 > 0**, from the same draws.

Leg 3 was added earlier on 2026-08-28, before any confirmatory number existed, precisely because
three separate 2.5% intervals are not a familywise 2.5%.

## 2. The objection, and its arithmetic

`boot.signflip` is **exact under the SHARP null** — that per-query differences are exchangeable in
sign, i.e. the two systems are identical per query. The claim the report wants to make is about the
**weak null**: the macro mean difference is ≤ 0. Real nDCG differences are skewed and bounded, so
the two nulls come apart.

The repo already measured that gap for the sign-flip leg alone
(`results/m7_signflip_weaknull.json`, S = 1000): at nominal 0.025 it rejects at **0.038** and
0.023; at the Bonferroni level 0.008333 it rejects at **0.013** and 0.008.

The review's arithmetic follows in one step. Three comparisons, each of whose decisive leg runs at
up to 0.013 under the weak null, union-bound the family at

> 3 × 0.013 = **0.039**, not 0.025.

That is a correct **bound**. It is loose in two specific ways, and both matter here:

* the rule is a **conjunction** — its per-comparison rate is at most the smallest leg's, and the two
  bootstrap legs had never been measured under the weak null at all;
* the three comparisons **share one candidate system and one query sample**, so their rejections are
  positively dependent, and a union bound over-counts exactly that overlap.

## 3. So it was measured

`m7src/tier_rule_calibration.py` → `results/m7_tier_rule_calibration.json`.

**Construction.** The same weak null as the committed sign-flip suite, extended to the family: take
a stand-in candidate A and the three real comparators, centre each (comparison, dataset) per-query
difference vector to mean zero — so all three weak nulls are true *by construction* while the skew,
the boundedness and the per-dataset n are the real ones — then resample queries within each dataset.
**The same resampled query indices feed all three comparisons**, which is what makes them dependent,
exactly as in the final run. The full three-leg rule is then replayed as `final_run.py` implements
it, Holm and all.

Two stand-ins, because the answer should not depend on which one is used: `bge-small-en-v1.5` (a
small symmetric transformer) and `lr-dense-websearch` (a zero-query-compute lookup table, the closer
architectural analogue to ours).

Fidelity, stated rather than assumed: R = 2,000 sign-flip replicates instead of the run's 100,000
(the setting the committed weak-null suite already uses), B = 10,000 bootstrap replicates (the real
value), and one shared randomization draw per dataset reused across the three comparisons — which is
not an approximation at all, since `boot` seeds every call with SEED = 0 and the real run therefore
reuses the same draws too. A self-check asserts the vectorized legs reproduce `boot.signflip` and
`boot.paired` exactly before any simulation runs.

**Result, S = 1,000 (binomial SE ≈ 0.005):**

| | family-wise rejection rate | per-comparison conjunction |
|---|---|---|
| stand-in `bge-small-en-v1.5` | **0.019** | 0.012 · 0.006 · 0.006 |
| stand-in `lr-dense-websearch` | **0.025** | 0.007 · 0.010 · 0.012 |

Per-leg family rates, for the record: Holm(sign-flip) 0.021 / 0.029, the plain CI>0 leg 0.063 /
0.079, the Bonferroni lower-bound leg 0.022 / 0.026.

**A higher-precision run at S = 4,000 (SE ≈ 0.0025) is in flight; these numbers are the S = 1,000
ones and will be superseded, not contradicted, by it.**

Three things this settles.

* **The rule is not running at 0.039.** It runs at **0.019–0.025** against a nominal 0.025. The
  union bound was loose by roughly a factor of two, and the reason is measurable: the individual
  conjunctions are 0.006–0.012, and their union is far below their sum because the three comparisons
  share a query sample.
* **It is not conservative either.** One stand-in lands exactly on 0.025. The rule is approximately
  correctly sized under a realistic weak null, not comfortably inside it.
* **The plain 2.5% CI leg is the loose one** (0.063 / 0.079 at family level, ≈ 0.03 per comparison).
  Legs 1 and 3 — both operating at 0.008333 — are what actually binds. Leg 3 was worth adding.

## 4. The options

### (a) Narrow the stated inferential claim to the null actually tested

Say, in the report and in the ledger: *the p-value is exact under the sharp null of per-query sign
exchangeability; under the weak null the same procedure's measured familywise rate on our data
shapes is 0.019–0.025 against a nominal 0.025.*

- **Cost**: the sharp null is a stronger hypothesis than "the macro difference is ≤ 0", so rejecting
  it is formally "these systems are not identical, in the positive direction" rather than "A's macro
  exceeds B's". In practice, with a directional statistic, every reader will read the second — which
  is exactly why the distinction has to be written down rather than relied on.
- **Benefit**: nothing changes, no rule moves, and the claim becomes one the procedure actually
  supports. The measurement in §3 is then quoted as the answer to "but what about the weak null?".
- **Risk**: it reads as a hedge if the measurement is not quoted alongside it. It should always be.

### (b) Replace the decisive leg with a weak-null-valid procedure

The standard fix is a **studentized** sign-flip (Janssen 1997): flip signs on `mean(d)/sd(d)` rather
than on `mean(d)`, which is asymptotically valid under the weak null.

**It was measured in the same run, on the same simulations.** Substituting the studentized leg gives
a familywise rate of **0.020 / 0.023** against the current rule's 0.019 / 0.025 — inside the
Monte-Carlo error of each other. Per-comparison the two agree to ≤ 0.002 everywhere.

- **Cost**: a new statistic on the one-shot path, a new implementation to test, and a registered rule
  changed days before the freeze.
- **Benefit**: measurably **none** at our data shapes. The theoretical guarantee is nicer; the
  realized error rate is the same.
- **Verdict**: I do not recommend it. Changing the decisive leg of an irreversible decision for a
  guarantee that buys 0.002 of measured error rate is motion, not rigour.

### (c) State the measured rate, and keep the rule

Keep all three legs exactly as registered, and replace the ledger's unearned sentence — "the rule
bounds the family at 0.025" — with the measurement: *under the sharp null the Holm family-wise rate
is 0.013 (`m7_signflip_calibration.json`); under a realistic weak null the whole three-leg rule
measures 0.019–0.025 against a nominal 0.025 (`m7_tier_rule_calibration.json`); a union bound over
the marginal legs alone would say 0.039, and the measured rate is lower because the three
comparisons share a query sample.*

- **Cost**: none to the procedure. The report carries one more paragraph and one more artifact.
- **Benefit**: the strongest honest claim available, with its own falsification attached, and it
  answers the reviewer's objection with a number instead of an argument.
- **Note**: (c) *is* (a) plus the measurement. They are not exclusive; (c) is how I would write (a).

## 5. What I recommend, and why

**(c).** The reviewer's objection was that a claim was made without evidence. The right response to
that is evidence, not a different procedure. The evidence says the rule is approximately correctly
sized on our actual data shapes, and that the alternative procedure would not change the outcome.

Concretely, if you agree, I will:

1. replace the ledger's familywise sentence with the measured rates, both nulls, both artifacts;
2. state in the report that the p-value is exact under the sharp null, that the confirmatory claim
   is about the weak null, and that the gap was measured rather than assumed;
3. leave `final_run.py` untouched.

**What would change my recommendation**: if the S = 4,000 run comes back materially above 0.025 —
say 0.032 or higher, which S = 1,000 cannot exclude — then the rule is genuinely anti-conservative
and (b) becomes worth its cost, or leg 3 moves to a stricter level. I will report that number either
way before anything is written into the ledger.

## 6. The honest caveats on the measurement itself

- It uses **stand-in candidates**, because our own system has no six-set numbers yet and must not
  have any until the final run. The stand-ins were chosen to bracket the architecture; the null is
  built from real per-query difference distributions, so the skew and boundedness are ours.
- **S = 1,000 gives SE ≈ 0.005**, so 0.025 and 0.032 are not separated by this run. That is what the
  S = 4,000 run is for.
- It is a **type-I calibration of a procedure**, reading only the frozen comparator per-query nDCG
  vectors already committed in `results/perquery.json` — no qrels, no corpora, and no number
  produced by our own system. Same access class as the two committed sign-flip suites, and it is not
  a six-set access.
- The measurement is of the rule **as implemented**. If any leg changes, it must be re-run.
