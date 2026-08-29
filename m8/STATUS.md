# M8 status

**Stage: Phase 0 executing, overnight 2026-08-29.** LEDGER **v2** is live
(`m8/LEDGER.md` + `m8/registry.json`), gated by two adversarial reviews of v1 — **Codex: BLOCK,
9 BLOCKER / 9 MAJOR / 3 MINOR**, and a Fable scientific-judgment pass. All findings actioned; the
map is LEDGER §16. Guards, the executable decision rule and the power simulation are committed and
their tests pass. **No M8 training has run. No protected set has been scored.**

---

## WAKE-UP NOTE FOR DYLAN — five decisions, in the order they block work

**1. E14 — doc-side co-adaptation. The biggest lever nobody had named, and it needs your ruling.**
The Fable review's headline: LightRetriever's table works because its document encoder was
*co-trained to be reachable by a bag of token vectors*. M7 fit a bag to a **frozen** doc space that
was never trained to be bag-compatible. That is, literally, what M7's own verdict "the remaining
gap is architectural" describes. The fix is to LoRA/fine-tune the document tower jointly (or
alternately) with the table. **Your rulings survive it**: the query side stays a pure lookup (E1),
it is training-time not index-time (E5), and it ships as one doc-side ONNX file (E3).
**What it costs, so you are not agreeing to a hidden bill:** it breaks document-vector sharing
with frozen M7, so the 10.12M-document reserved pre-encode is paid **twice** (~20.7 GB and tens of
GPU-hours per system); it needs a licence check that stella permits released derived weights; and
it **forces C2 to be redefined**, because "the M8 table against the *same* frozen document
vectors" stops being possible once the doc tower moves — and C2 is your E11 ruling. It also
reopens the "frozen off-the-shelf document tower" premise, which CLAUDE.md explicitly lists as
revisitable with arithmetic and your sign-off.
**A literature sweep run tonight says nobody has measured this** (`research/m8-planning/literature-2026-08-29.md`):
LightRetriever's own ablation table never freezes its document tower — every row trains it fully —
and EmbedDistill (arXiv 2301.12005) runs the inverse experiment with a small *transformer* query
side, never a bag of embeddings. So E14 is an open experiment, not a question we can settle by
reading: unguided, and correspondingly novel if it works. **I have not opened it. Yes/no is yours.**

**2. P(ship) — the number you asked the protocol to produce before it spends a week.**
`results/m8_power.json`, calibrated on real paired per-query vectors. Reserved-four macro SE
**0.00209**, 95% half-width **0.0041** (the plan's prior guess was 0.005 — good agreement).
**Minimum detectable effect 0.0068.**

| scenario | true C1 effect | P(ship) |
|---|---|---|
| structural target | +0.020 | **0.84** |
| modest | +0.010 | **0.80** |
| recipe-only | +0.005 | **0.21** |
| M7 repeat (post-gate transfer was 0.000 ± 0.005) | 0.000 | **0.002** |
| dense lags fused (strict C2 binds) | +0.020 / +0.006 | **0.57** |

*Correction, same night: an earlier version of this table read 0.67 / 0.57 / 0.15 / 0.002 / 0.46.
It was wrong. The simulator still carried the planning-draft's guard constants (six-set margin
0.005, SE 0.006, three homogeneous worst-groups) after §5 had been given its measured ones
(margin 0.0075, near-sibling SE 0.0026–0.0032, four datasets whose SEs differ fourfold). It
overstated the six-set guard's false-veto rate about fortyfold, and that guard is the dominant
term. Found by an adversarial review of the rewrite; the simulator now READS the registry instead
of restating it.*

Read the bottom two rows: **a recipe-only programme ships with probability ~0.2, and a repeat of
M7's measured transfer ships essentially never.** That is the case for spending the milestone on
capacity (D2/D1, and E14 if you open it) rather than on recipe knobs, and the LEDGER now says so
in the protocol (§7) so the budget cannot quietly drift back. The remaining drag on the good
scenarios is the qualifying-v2-table condition (85% assumed) and, at the null, the worst-group
guard — which vetoes ~23% of truly-equal candidates, mostly on DBpedia's 400 queries. That is the
deliberate safe-error direction, not a defect.

**3. LoTTE as the shadow (E10) — IT REOPENS. The full screen is done and every slice rejects.**
`results/m8_lotte_overlap.json`, 5.25M documents screened in 19 minutes. Three findings, and an
alternative the plan never named.

(a) **Licence splits.** LoTTE's passages and package are CC BY-SA 4.0, but its `search` queries
are **non-commercial-research-only** (inherited GooAQ licence, quoted verbatim in LEDGER §2.3).
I took the conservative route, which needs no ruling and improves the instrument: **forum queries
only** — also the better analogue to CQADupStack.

(b) **All ten slices reject, and the failures split cleanly in two.**

| | slices | why |
|---|---|---|
| hard reject | `writing/test`, `science/test`, `technology/test` | their StackExchange communities **literally include the protected sets**: english, physics, android + softwareengineering |
| reject on query leakage only | the other seven | 2–15 fingerprint matches per ~2,000 forum questions (0.1–0.75%) |

The structure is informative: **exact** query matches concentrate almost entirely in the three
community-overlapping slices (111, 13, 34), while the seven clean-community slices are nearly all
fingerprint-**near** matches (2–12 each, one exact). Two identical question titles across
different StackExchange sites are not evidence of contamination — "what is the difference
between X and Y" is not a leak.

(c) **I did not relax the bar after watching it bite.** But note the tension, which is yours to
resolve: the ledger's standing decontamination rule **R1 removes the ITEM** on query overlap,
while S0's newer, narrower bar drops the whole **SLICE**. Everywhere else in this project a
contaminated item is deleted, not its source. Under a per-question remedy, **seven slices survive
with ~2,000 questions each** — a perfectly usable shadow. The alternative is computed and in the
JSON, labelled DESCRIPTIVE / NOT ADOPTED.

(d) **An alternative nobody named, now measured** (`results/m8_shadow_alternatives.json`).
CQADupStack has **twelve** subforums. This project uses four — programmers and physics as dev,
android and english as two of the reserved four. **Eight have never been touched**: gaming, gis,
mathematica, stats, tex, unix, webmasters, wordpress = **323,488 documents and 8,961 queries**,
almost exactly the reserved four's query count. They are CC BY-SA 3.0 under the licence this
project already verified from the primary source, they need no download beyond what just ran, the
loader already exists, and they carry **no contamination against the reserved android/english**,
being different subforums. The honest objection: they are the same benchmark family as two of the
four reserved sets, so a weaker independence check than a separate corpus — though LoTTE is also
StackExchange, and the shadow is a once-crossed non-regression gate, not a selection instrument.

**Your call, three options:** keep the slice-level bar and lose the shadow (the pipeline loses its
STOP gate); authorise the per-question remedy and keep LoTTE; or substitute the eight unused
CQADupStack subforums. I have not chosen — E10 is your ruling and S0's registration says the
session does not substitute a shadow.

**4. Two licence/provenance rulings on reserved and candidate assets.**
- **harrier-oss-v1-0.6b** (teacher challenger). Identified tonight as **`microsoft/harrier-oss-v1-0.6b`**
  — so on the vendor rule it is Microsoft, "OK with justification", **not** disqualified. But it
  now carries **three** blockers, not one: training data **undisclosed** (the contamination black
  box — your ruling); it uses **last-token pooling**, which `m7src/teacher.py` does not implement
  and raises on, so screening it needs new code rather than a new config row; and it publishes
  **no retrieval-only number** at all, only a mixed-task MTEB v2 overall of 69.0, so a screen
  result would have nothing to be sanity-checked against. My read: not worth your ruling until
  the other three challengers have been screened.
- **HUPD** (the buildable USPTO reserve): its HuggingFace card is tagged **CC-BY-NC-SA-4.0**,
  more restrictive than the "CC-BY" previously recorded. Under our standing rule a wrapper tag is
  not a licence and cannot restrict public-domain text (37 CFR 1.71) — but that is a *legal
  interpretation*, and per your own instruction I am not inferring it. Also: the stronger
  citation-based USPTO construction needs a **PatentsView API key**, which requires your signup.

**5. The E12 comparator's bill.** You approved scoring bge-small and LR-dense-websearch inside the
access as descriptive context. bge-small is cheap. **LR-dense-websearch means pushing 10.12M
documents through a 1.5-billion-parameter Qwen on a 10 GB card** — plausibly more GPU time than
all of Stage R's training combined, bought for one descriptive row. `instructions-m8.md` already
sanctions published numbers as labelled context. **Pre-agreeing the fallback now beats discovering
the collision in week three.**

### Four things you should see, that are not questions

**B2 found a real mechanism, and it is the best news of the night.** Objective A's KL term asks the
student to match the teacher's distribution over the query's positive plus 31 distractors drawn
*uniformly* from two million documents, at temperature 0.02. Measured on 4,000 TRAIN queries:
**for the median query that distribution is a delta function to seven decimal places** — entropy
5.65e-07 nats against a ln(32) = 3.466 ceiling, teacher max-probability median exactly 1.0, and
**82.9% of queries below 1e-4 nats**. One of the two training signals is, for five queries in six,
carrying no information at all. Drawing distractors from the teacher's own top-200 instead raises
the median entropy **six orders of magnitude**, to a quarter of the ceiling.

That is a concrete, measured mechanism for why M7's recipe programme transferred ~0.000 — and it
is a different class of lever from knob-tuning. It does **not** yet say a listwise objective wins;
that is `R-LIST`'s question and its bar is still unfrozen, so the guard refuses it.
`results/m8_b2_entropy.json`, LEDGER §19.



**B7 passed, and it reopens two doors at once.** The dense fp64 Gram is what closed granite-r2 and
gte-modernbert in M7 "on arithmetic, not merit" — 20.3 GB at 50,368 rows against an 18 GB budget.
A Gram-free preconditioned solver now does **65,536 rows in 51 iterations, 10 seconds, 4.4 GB**,
and 131,072 rows in 17 seconds — where the dense Gram would have been 34 GB and 137 GB. It agrees
with the direct solve to 4.6e-7. So **D2's 64–128K vocabulary is computable, and so is a teacher
screen for challengers that do not share stella's WordPiece vocabulary — which is all four of
them.** `results/m8_b7_solver.json`, LEDGER §18.

**D2's one existing precedent points the wrong way, and you should know that before the week is
spent.** The only published vocabulary-size ablation on a *static / bag-of-words* retriever is VDR
(ICLR 2024, arXiv 2212.07699): 30K → 110K rows moved BEIR nDCG@10 **44.5 → 42.6**, a small
regression. It is confounded — the two vocabularies also swapped English BERT for multilingual
BERT, which the authors themselves blame — so it does not close D2. But it is the closest evidence
that exists and it is negative. Positive precedent is thinner: multi-word tokens help e-commerce
retrieval on a *contextual* model. **Neither a known success nor a known failure**: the week is not
pre-empted, and it is not de-risked either. Recorded in LEDGER §8 so D2's registration is read
against it rather than against an assumption.

**The short-query story was wrong, and so was my first correction of it.** H3's premise ("best on
the longest queries, worst on the shortest") is a between-dataset reading of six points, and it
does not survive: the per-dataset length slopes flip sign, and ArguAna's own retention *declines*
across its length quartiles. But my first replacement claim — "within datasets, longer queries are
relatively worse, t = 3.0" — was itself an ArguAna-only result: **ArguAna carries 99.7% of the
within-dataset length variance**, and excluding it the slope is t = 1.51, unresolved. Withdrawn.
**What survives is fragmentation**, and it survives the same attack: ArguAna holds only 2.2% of
the fragmentation variance, and excluding it the slope is +0.045, **t = 4.71** — unchanged. The
table falls ~0.05 nDCG further behind its teacher per +1.0 subwords-per-word, because the *teacher*
pulls ahead on fragmented queries while the table stays flat. That is the channel a multi-word
tokenizer reaches, and it is why B7 was promoted. LEDGER §17.

**A near-miss, found by review rather than by accident.** `work/dev/cqadup-android.json` and
`work/dev/cqadup-english.json` held the **complete corpora and qrels of two of the four reserved
confirmatory sets**, materialized on 2026-08-26 when the untouched-final pair was defined. Any M8
dev script calling `devsuite.load("cqadup-android")` would have scored a reserved set silently.
**Nothing scored them** — no M8 evaluation had run. They are now a protected kind, along with the
HuggingFace `*-qrels` caches and the `load_dataset` network route. LEDGER §15.

## Done tonight

**Protocol.** LEDGER v2 + `m8/registry.json`, after two adversarial gates on v1 (Codex: BLOCK,
9/9/3) and two Fable passes, the second of which found four v1 fossils that had survived the
rewrite into the *executable* layer — the "wrong number, no error" class. All actioned; the map is
LEDGER §16, the amendments are §15, and the obligations still open are a **GAP LIST** in §4.4
rather than a "DONE" heading that lies.

| item | artifact |
|---|---|
| LEDGER v2 + machine-readable registry (23 probes; **7 runnable, 16 refuse themselves**), full manifest key schema, teacher-swap side-effect rule | `m8/LEDGER.md`, `m8/registry.json` |
| Executable ship rule — every threshold a number, reading the registry rather than restating it | `m8src/decide.py` |
| **11 checks on the rule, mostly its refusals**; establishes by measurement the `paired_dep` reduction §4.1 only asserted | `m8src/test_decide.py` |
| Joint power simulation → MDE 0.0068, P(ship) table above | `m8src/power.py`, `results/m8_power.json` |
| Guards G1 + G2, hardened against four concrete routes; **26/26 checks pass** | `m8src/paths_guard.py`, `m8src/probe_guard.py`, `m8src/test_guards.py` |
| **Rule audit** — for every stamped result it fetches the registry *from git at that result's commit* and diffs the bar against today's, so a registration that moved after a number existed is a BLOCKER | `m8src/rule_audit.py`, `results/m8_rule_audit.json` |
| **B7 PASSED** — Gram-free preconditioned solver: 64K rows in 51 its / 10 s / 4.4 GB against a 34 GB dense Gram | `results/m8_b7_solver.json` |
| **S0** — LoTTE overlap screen, 5.25M documents, 19 min: all ten slices reject | `results/m8_lotte_overlap.json` |
| Protected-query filter, 80,954 queries over four partitions | `results/m8_protected_filter.json` |
| Shadow alternatives measured (8 unused CQADupStack subforums) | `results/m8_shadow_alternatives.json` |
| Retention decomposition + fragmentation attribution, both rewritten after review showed the first versions overclaimed | `results/m8_retention_decomposition.json`, `results/m8_fragmentation_attribution.json` |
| Serial GPU/RAM/disk schedule | `results/m8_schedule.json` |
| LoTTE acquired with provenance; M9-reserve inventories (EUR-Lex complete, USPTO sampled) | `work/lotte/`, `work/m9reserve/` |

**Two defects the smokes caught before they cost anything**, which is what smokes are for:
`M7_ENCODER` defaults to M7's *pre-swap* bge-base teacher, so every noise-floor arm died on a
teacher mismatch — now pinned in `m8base`; and `sweep.one` catches its own exception and returns
`None`, so a failed arm exits 0 and my driver ran all five after the first had failed.

**A clean replication, unlooked for:** the noise floor's seed-0 arm reproduces M7's shipped
candidate's proxy macro to all sixteen digits (0.5105689103506673). The floor's frame is the
released artifact's frame.

## Running / next

Noise floor (5 arms trained → full-suite scoring → floor per precision × pool-mode × endpoint),
then the fused floor, then B7's real-data verification. **Wave-1 probes stay refused until their
bars are frozen** — `probe_guard` enforces that, and today 16 of 23 registry rows refuse
themselves.

**Left for the next session, specified and unblocked:** the T1 teacher screens (B7 removed the
arithmetic that blocked them; the four challengers need new encoder `Spec`s, `validate_encoder`
passes, and Dylan's harrier ruling), the fit-list regeneration run, and the four gap-list
obligations in LEDGER §4.4.

## File contract

| file | contract | read when |
|---|---|---|
| `m8/STATUS.md` | ONE screen plus the wake-up note. Stage, running, blocked. | always, first |
| `m8/LEDGER.md` | Binding protocol. Rules, bars, verdicts, amendments. | before any decision |
| `m8/registry.json` | The executable half of §9. `probe_guard` reads this, not the prose. | before any run |
| `m8/EXPLORED.md` / `m8/RESULTS.md` | dead ends / runs, one row each. | as needed |
| `research/m8-planning/*` | Archival planning record (6 reviews). Point at it, never restate. | on demand |

Rules: every number carries an artifact pointer; no file restates another; a future session
cold-starts from STATUS + LEDGER + registry alone.
