# M8 status

**Stage: Phase 0 largely executed, overnight 2026-08-29.** LEDGER **v2** is live
(`m8/LEDGER.md` + `m8/registry.json`), gated by six adversarial reviews. **All three noise floors
are now measured** (dense, fused, and the B-leg one that was the last gap-list item), five probes
have run, the teacher question is answered, B3 has been redesigned twice and is running, and every
result carries a registration stamp. **No M8 training candidate exists yet. No protected set has been scored. The
reserved four are untouched.**

---

## WAKE-UP NOTE — five decisions, in the order they block work

**1. E14 — doc-side co-adaptation. The biggest unopened lever, and it needs your ruling.**
LightRetriever's table works because its document encoder was *co-trained to be reachable by a bag
of token vectors*. M7 fit a bag to a **frozen** doc space never trained to be bag-compatible —
which is, literally, what M7's own verdict "the remaining gap is architectural" describes. The fix
is to LoRA/fine-tune the document tower jointly with the table. **Your rulings survive it**: the
query side stays pure lookup (E1), it is training-time not index-time (E5), one doc-side ONNX file
(E3). **The bill, so you are not agreeing to a hidden one:** it breaks document-vector sharing
with frozen M7, so the 10.12M-document reserved pre-encode is paid **twice**; it needs a licence
check that stella permits released derived weights; and it **forces C2 to be redefined**, since
"the M8 table against the *same* frozen document vectors" stops being possible once the doc tower
moves — and C2 is your E11 ruling. **A literature sweep found nobody has measured this**
(`research/m8-planning/literature-2026-08-29.md`): LightRetriever's own ablations never freeze the
document tower, and EmbedDistill runs the inverse experiment with a *transformer* query side. So
E14 is an open experiment — unguided, and correspondingly novel if it works. **I have not opened
it.**

**2. P(ship), before the milestone spends a week.** `results/m8_power.json`, calibrated on real
paired per-query vectors. Reserved-four macro SE **0.00209**, 95% half-width **0.0041**, **MDE
0.0068**.

| scenario | true C1 effect | P(ship) |
|---|---|---|
| structural target | +0.020 | **0.84** |
| modest | +0.010 | **0.80** |
| recipe-only | +0.005 | **0.21** |
| M7 repeat (its post-gate transfer was 0.000 ± 0.005) | 0.000 | **0.002** |
| dense lags fused (strict C2 binds) | +0.020 / +0.006 | **0.57** |

*An earlier version of this table read 0.67 / 0.57 / 0.15 / 0.002 / 0.46 and was wrong: the
simulator still carried the planning draft's guard constants after §5 had been given its measured
ones, overstating the six-set guard's false-veto rate ~40×. Found by adversarial review; the
simulator now reads the registry instead of restating it.*

**A recipe-only programme ships with probability ~0.2; a repeat of M7's measured transfer ships
essentially never.** That is the case for spending the milestone on capacity, and the LEDGER says
so in the protocol (§7) so the budget cannot drift back.

**3. E10 — the shadow. IT REOPENS: all ten LoTTE slices reject.**
`results/m8_lotte_overlap.json`, 5.25M documents screened. Three of the ten fail on **community
intersection with protected sets** (english, physics, android + softwareengineering) — those are
dead under any remedy. The other seven fail only on **query leakage**, 2–15 fingerprint matches
per ~2,000 questions. Exact matches concentrate almost entirely in the three community-overlapping
slices; the clean-community seven are nearly all fingerprint-*near*.

**I did not relax the bar after watching it bite.** But the tension is yours to resolve: §3's
standing rule **R1 removes the ITEM**, while S0's newer bar drops the whole **SLICE**. Under a
per-question remedy, seven slices survive with ~2,000 questions each. A third option is measured:
`results/m8_shadow_alternatives.json` — the **eight unused CQADupStack subforums** (323,488
documents, 8,961 queries), already licence-cleared, no contamination against the reserved pair,
but the same benchmark family as two reserved sets. **Also: LoTTE's `search` queries are
non-commercial-research-only** (GooAQ licence); I used forum queries only, which needs no ruling.
**Your call: lose the shadow, authorise the per-question remedy, or substitute the subforums.**

**4. Two licence/provenance rulings.**
- **harrier** is `microsoft/harrier-oss-v1-0.6b` — Microsoft, so "OK with justification", **not**
  disqualified. But it has **three** blockers now: undisclosed training data (your ruling);
  **last-token pooling** that `m7src/teacher.py` raises on, so it needs new code; and **no
  published retrieval-only number** to sanity-check a screen against. My read: not worth your
  ruling yet.
- **HUPD** (USPTO reserve) is tagged **CC-BY-NC-SA-4.0**, stricter than previously recorded. Our
  standing rule says a wrapper tag cannot restrict public-domain text (37 CFR 1.71) — but that is
  a legal interpretation and I am not inferring it. The stronger citation-based construction needs
  a **PatentsView API key** you must request.

**5. The E12 comparator's bill.** LR-dense-websearch means pushing 10.12M documents through a
1.5B-parameter Qwen on a 10 GB card — plausibly more GPU time than all of Stage R.
`instructions-m8.md` already sanctions published numbers as labelled context. **Pre-agreeing the
fallback beats discovering the collision in week three.**

---

## What was measured tonight

**The B-LEG noise floor — the last floor, and it closes §4.4's gap list**
(`results/m8_noise_floor_bleg.json`, LEDGER §15). Both existing floors held the Phase-B checkpoint
FIXED and varied only the Phase-A seed. `R-PHASE` and every pool-or-init lever flow through the B
leg instead, and no bar could read them, because the A-leg floor holds constant the very leg they
perturb. Three full B→A chains varying only the seed now settle it.

**The B leg adds essentially nothing.** A-leg floor 0.00095–0.00227 across the four endpoints;
B-leg floor **0.00070–0.00218**. A whole extra 16,000-step seed-dependent phase does not widen the
instrument, so R-PHASE and the pool/init levers read the same 0.0040 planning minimum as
everything else. **One exception now binds**: at `int8/mean`, worst-group and out-of-domain macro
take **0.004369** (2 × floor), not 0.0040. *The honest caveat: this compares two
max-over-three-pairwise statistics at K = 3, and the ranges overlap almost entirely — the claim is
"the B leg does not visibly inflate the floor", not that the two are equal.*

**B3 — the probe was rebuilt twice tonight, both times before any arm ran, and is now running.**
Its original lever was synthetic ICT augmentation. An adversarial review of the *arm definition*
killed it: "equal updates AND equal exposure" over-constrains a fixed batch (three constraints,
two free variables), my proposed batch-scaling fix did not even deliver equal *influence* under
mean reduction, and the ICT shortcut survives sentence removal because the positive's teacher
vector is precomputed over the **full document** — the cheap repair does not work. Above all,
adding synthetic pairs at fixed compute measures whether spending Phase-A budget on ICT helps, not
whether Phase A is short of pairs. Retired as registry row `B3-ICT`, refused, reasoning kept.

**What replaced it**: nested subsets of the *real* pair pool at {0.25, 0.50, 0.75, 1.00} with
updates, batch, negatives and the Phase-B checkpoint all held, so total draws are 1,280,000 in
every arm and only the count of distinct pairs moves. A second review then found the replacement
had **no computable verdict** — its fused endpoint named a quantity that does not exist (the two
held-out dev components carry row indices, not text, so they have no fused read) while citing a
floor measured on a different one, its dense endpoint was two numbers with no conjunction rule, and
the whole bar lived in prose. All fixed before scoring: the scalars are pinned, the bar runs as
code (`m8src/b3_decide.py`) with **a test per branch**, and the primary contrast moved from
1.00-vs-0.75 to **1.00-vs-0.50** because at fixed draws the fractions are epochs and the last
quarter is the least-powered segment of a concave curve — the design could have cleared its own
manipulation check at twice the bar while the primary sat under it, and then declared "not
starved". `no_survivor` is narrowed too: `p35b-2m` already distilled on every training query, so a
null here says nothing about B-side pair levers. **Three free arms**: the f=1.00 recipe already
exists at three seeds as the floor arms, verified config-identical.

**T1 — the teacher question is answered: NO SWAP** (`results/m8_t1_decision.json`, LEDGER §21).
These are the **first measurements of two teachers M7 closed "on arithmetic, not merit"**.

| candidate | published tower | best λ | dev macro | Δ vs incumbent | 95% CI |
|---|---|---|---|---|---|
| **stella-400M-v5** (re-probed) | MTEB-Ret 58.97 | 0.01 | **0.3438** | — | — |
| granite-r2 | BEIR 53.1 | 0.01 | 0.2915 | −0.052 | [−0.066, −0.039] |
| gte-modernbert-base | BEIR **55.33** | 0.01 | 0.2349 | −0.109 | [−0.123, −0.094] |

All optima interior; both losses CI-resolved at 5–11× the swap penalty, so condition 1 fails and
the expensive off-family read is never bought. **The frame reproduces M7's own number**: the
incumbent re-probed scores 0.3438 against M7's recorded 0.3439, through a new solver, a new init
builder and a regenerated fit list. And **the tower again fails to order the table** — gte has the
higher published score and the lower table. That is an n=2 sign anecdote on self-reported card
figures, so it *corroborates* M7's eight-candidate Spearman-0.000 result rather than reproducing
it; I had called it an independent reproduction and that is withdrawn.

**B2 — the KL term is dead, measured on BOTH sides. H2 confirmed** (LEDGER §19). The teacher's
target over its 32-candidate set is, for the median training query, a delta function
(**4.73e-07 nats** against a ln(32)=3.466 ceiling; max-probability median 1.000000; **84.3%** of
queries below 1e-4). And the half that makes it a measurement rather than an inference: the
**shipped M7 table ranks the positive first in 99.75% of queries**, so **the KL term's own median
value is 1.08e-07 nats**. The loss term was computed on the artifact that ships, not deduced from
the target's shape. Top-200 distractors raise the target's median six orders of magnitude.

*Two corrections a review forced, both recorded: the first version drew distractors from a
contiguous pool prefix where training draws a seeded random sample — and the fix made the recipe
look **worse**, not better (median 5.65e-07 → 4.73e-07); and the student side did not exist at all,
so "carries no information" was an inference. It is now a number.* It still does **not** say a
listwise objective wins — that is `R-LIST`'s question and its bar is unfrozen, so the guard
refuses it.

**B7 — PASSED, and it reopened two doors** (§18, §18b). A Gram-free preconditioned solver does
65,536 rows in 51 iterations / 10 s / 4.4 GB where the dense fp64 Gram would be 34 GB, and 131,072
rows in 17 s against 137 GB. Its registered real-data precondition passed with **identical dev
macro at all four λ**. Honest note: at the 30,522-row control the *direct* solve is faster at small
λ — CG's value is existing above 50,368 rows at all.

**Noise floors — measured; B3's bar frozen at 0.0040** (§4.7, §4.7b). Dense floor 0.00095–0.00227,
fused floor 0.00059–0.00066 — tighter, because part of the fused score comes from a deterministic
BM25 run with no seed at all. The planning minimum binds almost everywhere. **A lever that clears
the fused bar but not the dense one has not shown a table improvement.** *(The two are not a
like-for-like ratio — the fused macro is over four components and the dense endpoints over six.)*

**Fit list regenerated** (§3.3): 337,981 kept of 338,076. The 95 removals are **all** from
M9-reserve; screening against six+dev+reserved alone removes zero, because M7 had already applied
R1 against exactly that index.

**§17 — the short-query story, twice corrected.** H3's premise does not survive. Nor did my first
replacement for it: ArguAna carries **99.7%** of the within-dataset length variance, so that claim
was ArguAna-only and is **withdrawn**. **Fragmentation survives — and survives every single-dataset exclusion**, which is the check the
first version skipped: it had run leave-one-out only against the dataset that threatened the claim
I disliked, not against nfcorpus, which carries 53% of the variance behind the claim I kept. Run
for all six, the worst case is +0.0369 at **t = 3.28** and every exclusion stays positive. The
table falls ~0.05 nDCG further behind per +1.0 subwords-per-word, because the *teacher* pulls ahead
while the table stays flat. No published match for that asymmetry.

**B17 — its registered branch fired and is DISOWNED** (§20). The ≤0.40 branch ("the class caps in
domain") triggered at 0.1999 — but the same class fitted on 350K general queries scores 0.3439, so
what B17 measured is its own 957-query fit set, exactly as its pre-registered caveat warned. The
rule is **not** amended. Disowned in the direction that costs me a fourth witness for D2/D1.

**A near-miss, found by review.** `work/dev/cqadup-{android,english}.json` held the complete
corpora **and qrels** of two reserved confirmatory sets. Any dev script calling
`devsuite.load("cqadup-android")` would have scored one silently. **Nothing scored them.** Now a
protected kind, along with the HF `*-qrels` caches and the `load_dataset` network route (§15).

## Infrastructure built

| item | artifact |
|---|---|
| LEDGER v2 + machine-readable registry; full manifest key schema; teacher-swap side-effect rule | `m8/LEDGER.md`, `m8/registry.json` |
| Executable ship rule + 11 checks, mostly its refusals | `m8src/decide.py`, `m8src/test_decide.py` |
| Guards G1/G2 hardened against four concrete routes; 26 checks | `m8src/paths_guard.py`, `m8src/probe_guard.py`, `m8src/test_guards.py` |
| Rule audit — diffs each result's registry **from git at that result's commit**, so a bar that moved after a number is a BLOCKER | `m8src/rule_audit.py` |
| Gram-free solver; teacher screens; init at the true vocabulary; entropy probe; both floors | `m8src/blockcg.py`, `teacher_screen.py`, `init_m8.py`, `b2_entropy.py`, `noise_floor.py`, `fused_floor.py` |

`./run_m8_tests.sh` runs everything committed. **`m8/CODEMAP.md` now carries four "things that
must move with the encoder"** where M7 had two — the two new ones are silent failures.

## Next

`m8/NEXT-SESSION.md` has the remaining worklist. Nothing is blocked except by the five decisions
above; the largest open engineering items are B6-pre (the doc-side ONNX fuse gate, which gates D1),
the B-leg noise floor, and the two one-shot test suites in LEDGER §4.4's gap list.

## File contract

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | This. Stage, decisions, results. | always, first |
| `m8/LEDGER.md` | Binding protocol: rules, bars, verdicts, amendments, measured findings. | before any decision |
| `m8/registry.json` | The executable half of §9. `probe_guard` reads this, not the prose. | before any run |
| `m8/NEXT-SESSION.md` | Remaining worklist. | at session start |
| `m8/RESULTS.md` / `m8/EXPLORED.md` / `m8/CODEMAP.md` | runs / closed avenues / modules and pitfalls. | as needed |
| `research/m8-planning/*` | Archival record: 7 reviews, the literature sweep, the challenger Specs. | on demand |

Every number carries an artifact pointer; no file restates another; a future session cold-starts
from STATUS + LEDGER + registry alone.
