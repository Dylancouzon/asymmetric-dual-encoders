# M10 feasibility review — 2026-09-04 (second review of the day)

Dylan: *"is our goal feasible? Are there any gaps, weaknesses, or avenues to improve? … No
over-engineering or things that are not defensible towards public scrutiny. If there are prior
decisions that you want to reopen, ask me."* Reviewer: Fable (this session), full read of the M10
files, M9's record and the 2026-09-04 Fable review; two Sonnet literature sweeps and two
source-verification sweeps (web-only, no repo reads, no sub-subagents). Read-exclusion: no
reserved, LoTTE or six-set query/qrels file was opened; the only six-set numbers used are the M7
comparator rows already observed in `results/perquery.json` and `results/m7_final_run.json`.
Dispositions: `instructions-m10.md` §Amendment 2026-09-04b (B1–B6) and decisions 11–13.

## 1. Verdict

**C1a (release, avg-6) is reachable if coverage works. C1b (release, clean-4) and C2a (aim, avg-6)
are the real contest and sit at ~92% uniform retention — right where M9 stood on the one form it
covered. C2b (aim, clean-4) needs 95.3% and no published precedent reaches it at our teacher gap.**
The plan should run; it should also say this in advance, which it now does (B1).

Arithmetic (`results/m10_conjunct_arithmetic.json`, comparator rows only):

| conjunct | pass point | uniform retention of the ceiling |
|---|---|---|
| C1a release avg-6 | 0.5143 | **89.5%** |
| C2a aim avg-6 | 0.5256 | 91.5% |
| C1b release clean-4 | 0.5185 | **91.9%** |
| C2b aim clean-4 | 0.5372 | **95.3%** |

Per dataset, to *equal* bge-small: scifact 91.4%, nfcorpus 83.0%, fiqa 72.9%, arguana 94.7%,
scidocs 85.7%, trec-covid 92.0%. At uniform 92%, fiqa (a disclosed stella training set) supplies
73% of nano's avg-6 margin over bge-small — the reason clean-4 is the headline, and the reason the
headline is harder than the avg-6 aim. LEAF beats the ceiling on trec-covid (0.8301 vs 0.8234).

## 2. What the literature says about the teacher gap (Sonnet sweep, primary sources)

| system | teacher → student | gap | loss | retention |
|---|---|---|---|---|
| LEAF `leaf-ir-asym` (2509.12539) | arctic-embed-m-v1.5 109M/768d → MiniLM-L6 23M | 4.7× | ‖e‖₂ | **97.7%** BEIR (SciFact 97.8, NFCorpus 98.9, TREC-COVID 94.9, SCIDOCS 91.6) |
| 2306.11550 | BERT-base 110M → 1/2/4 of its own layers | — | L2, 8M MS MARCO queries | 86.1 / 92.5 / 96.2% |
| EmbedDistill (2301.12005) | BERT-base 110M → 6-layer / 4-layer | 1.6× / 10× | score KD + L2 + labels | ~99% / 95–97% |
| DistilVDR (2608.10636) | Qwen3-VL-Emb-8B 4096d → 70M | 116× | cosine | **86.9%** (79–95% by domain) |
| CARE (2604.10937) | Qwen3-8B → 305M | 27× | MSE + Asym-InfoNCE | 86.5% |
| mxbai-edge-colbert (2510.14880) | stella-1.5B → 17M/32M | — | L2, multi-vector | teacher score not reported |
| **ours** | stella-400M 1024d → ≤35M | **11.4×** | L2 | M9: 93.8% covered form, 50–71% uncovered |

No paper distils gte-large / e5-large / bge-large / arctic-l / stella into a ≤50M student by
regression and reports retention: **our pair is uncharted.** The two large-gap cases sit at ~87%;
the small-gap cases at 96–98%; M9's 93.8% on its covered form sits between them where the gap
predicts. LEAF's per-dataset floor is SCIDOCS at 91.6% — even LEAF's clean-4 retention (≈95.8%) is
barely above C2b's demand. Reading: **the in-distribution ceiling PLANNING §13 calls "open" is
most likely the capacity-gap effect the literature shows, not a recipe defect.** What would move
it: a per-token nonlinear head (B3, the one compute lever inside the cap), D-COV, more dose on
covered forms, and — outside M10 — document-side co-adaptation (M16). TAKD (1902.03393) and
2605.31191 (image classification) both say student capacity caps KD gains; LEAF argues embedding
regression onto a fixed space is easier than the logit-KD those results describe. Neither settles it
for us; the screen's G-MLP and D-COV arms are the test.

## 3. Gaps found, and what was done

| # | gap | disposition |
|---|---|---|
| G1 | **The COV screen could not see the headline.** All four clean-4 sets are scientific/biomedical; COV had consumer-health, StackExchange, legal, finance. The A3−A2 contrast on harvested *scientific* text would be judged on surfaces blind to it, and the plan conceded those forms are "tested only by the six-set transaction" — i.e. the most important data decision for the headline was to be made blind | **B4**: constructed `arxiv-title` COV family (2,000 held-out real titles → own abstract among 100K; qrels by construction; licence-clean; protected before harvest), plus `ctgov-title` if ClinicalTrials.gov's terms verify. A form-retention surface, disclosed as such, never a claim |
| G2 | **"A nonlinear head has no serving path" was wrong** for per-token heads | **B3**: proven — fastembed reproduces `1152→512→GELU→1024` per token to min-cos 0.99999988, zero custom ops, 34.48M (`results/m10_head_mlp_parity_box.json`). Arm G-MLP replaces G-768 |
| G3 | **No four-conjunct release rule.** M9's table had C1/C2; with C1a/C1b under gatekeeping, the likely outcome (C1a pass, C1b fail) had no registered consequence | **B2**: table registered; the ship decision is **decision 11** (default: ship, disclosed) |
| G4 | **Feasibility never stated.** The mandate registered C2b without saying the evidence puts it out of reach, or that C1b is harder than C2a | **B1**: §Goal carries the arithmetic and the literature ceiling before any nano number exists |
| G5 | **Biomedical training coverage is thin by rule, not by availability.** Three headline sets are biomedical; PubMed / CORD-19 / NutritionFacts are excluded as source families (M7 rule written for a table trained on document text); Wikipedia-medical seeds, arXiv q-bio and (if licensed) ClinicalTrials.gov / DailyMed are what remains | **decision 12** — Dylan's; §4 below |
| G6 | Query-asset target: bge-small + three-layer head ≈70.8 MB fp16 vs M9's 70 MB target; nobody had done the arithmetic for the wider head | **B6**: recorded; M9's rule (logged, measured justification) applies at the lock |
| G7 | The DEV-6 recipe pre-screen read DEV-6 twice for defaults the screen re-decides | **B5**: dropped |

**Checked and sound:** the data thesis and family A's three-outcome rule; A6's ordering; the
generation contract and A8's gates; fixed-sequence gatekeeping (C1a → C1b is the order that tests
the most in the likely scenarios, since C1b needs more retention than C1a); the LEAF-regime
optimizer defaults; the dose and extension rule; D-COV's motivation — it is exactly the expected
squared *score* error over the document distribution, `E_d[((s−t)·d)²] = (s−t)ᵀΣ_d(s−t)`, with the
centred covariance correct because `q·μ_d` is constant across documents and cannot change a ranking.

**Considered and not changed:** replacing the ≤2-decision confirmation design with a three-seed
anchor noise floor (saves ≈10 GPU-hours; weakens a reviewed rule for a K=3 σ estimate); cutting
G-1536 or B-50/50; a wider-hidden student via truncated bge-base with a factorised embedding (fits
the cap on paper, has no pretrained checkpoint, no precedent, a new export — over-engineering);
initialising the student's token embeddings from stella's (breaks the pretrained backbone).

## 4. The biomedical question (decision 12) — the one reopen worth Dylan's time

Three of the four headline sets are biomedical (`nfcorpus`, `trec-covid`, `scifact`); M7's
contamination map excludes PubMed, PMC, CORD-19 and NutritionFacts as *source families* from
training, seeding **and COV**. That rule was written for a table trained on document text. Two
web sweeps (primary clauses quoted; HF tags not accepted as evidence) established:

**4a. Training text — the reopen I raised and then withdrew.** PubMed titles / PubMedQA questions
as biomedical query-form text looked like the most direct lever (LEAF's own source, 272K PubMedQA).
It fails on **licence**, before contamination: NLM — *"NLM does not claim the copyright on the
abstracts in PubMed; however, journal publishers or authors may"* and users *"are expected to
adhere to the terms and conditions asserted by the copyright holder"*
(ncbi.nlm.nih.gov/home/about/policies). No affirmative grant → the 2026-09-04 rule's "no licence"
class, fully out. PubMedQA's MIT licence covers the repository, not the abstracts. Europe PMC is a
mirror of PubMed/PMC. **Withdrawn; not re-proposable without a licence change.**

**4b. Clean biomedical text that DOES exist (no reopen needed; §Data updated):**

| source | clause (primary) | commercial + derivative | overlap with excluded families | use |
|---|---|---|---|---|
| arXiv metadata | *"free to use descriptive metadata … under CC0 1.0"* (info.arxiv.org/help/api/tou.html); full text NOT granted | yes (titles, abstracts) | none | harvest: title, claim forms; `arxiv-title` COV. q-bio is small (~3K/yr) |
| MedlinePlus, government-authored Health Topics | *"Works produced by the federal government are not copyrighted"*; A.D.A.M. encyclopedia and ASHP monographs are third-party copyright, *"must be authorized in writing"* (medlineplus.gov/about/using/usingcontent) | yes for the government half only | none with the six; **same source as MedQuAD** (COV MedicalQA) → fingerprint-screen, disclose same-source | harvest / seeds: consumer-health |
| CDC pages | *"not subject to copyright, is in the public domain"*, except contractor/licensed content (cdc.gov/other/agencymaterials.html) | yes | same MedQuAD caveat | consumer-health |
| ClinicalTrials.gov | NLM parent policy (government works); the site's own terms page is JS-rendered and its clause **could not be read** | likely, **unverified** | none | `ctgov-title` COV and biomedical title/summary harvest **only if the clause is recorded at M10.0-d** |
| Wikipedia medicine (WikiProject Medicine, ~30K+ articles) | CC BY-SA | yes | none | seeds (already approved) |

Out: DailyMed (manufacturer-authored labels, no clean grant), OpenAlex (CC0 index, PubMed-sourced
abstracts), bioRxiv/medRxiv (per-article licences; COVID preprints are inside CORD-19), WHO
(CC BY-NC-SA), Cochrane (restricted), NICE (UK-only commercial grant).

**4c. Validation-only: CUREv1 as a COV family — this is decision 12.** arXiv 2412.06954, HF
`clinia/CUREv1`, MTEB revision `3bcf51c9…`, **CC BY-NC 4.0** (admissible for validation under
Dylan's 2026-09-04 rule). **2,000 real English queries** written by doctors and nurses (50 layman +
150 expert in each of 10 disciplines; none LLM-generated). Corpus: 244,600 passages chunked from
51,083 **full-text** open-access articles (PMC Open Access Subset, Nature, BioMedCentral) — so it is
PubMed-family by M7's map. Qrels: three levels; clinicians seeded relevance, the candidate pools were
**annotated by Qwen 2.5 72B**, ≈9.9 relevant + 30.4 partially relevant per query; nDCG@10. The paper
carries **no decontamination statement** against NFCorpus / CORD-19 / SciFact / SCIDOCS.

Why admit it: it is the only real-query biomedical retrieval surface with an affirmative grant, and
without it the screen sees biomedical questions only through MedQuAD's templated consumer-health
set. Why it needs Dylan: it reopens the "excluded from COV" clause of the source-family rule for a
selection surface. The precedent is already in the protocol — CQADupStack-programmers/physics are
DEV while cqadup-android/english are reserved, same family, different split — and the guard is the
same: fingerprint screen against the six's documents, removal counts published, never a claim.
Caveats to disclose if admitted: LLM-annotated pools; full-text passages, so an abstract in
`nfcorpus` or CORD-19 could sit inside a CUREv1 passage (the 8-gram screen catches it).
Alternatives considered: PublicHealthQA (172 English CDC/WHO COVID FAQ queries; WHO half is NC;
MTEB pins revision `main`; COVID topic overlaps `trec-covid`) — too small and too close; BioASQ
(500 test queries over 14.9M PubMed abstracts, registration-gated, not redistributed) — too large to
encode and PubMed abstracts outright; DORIS-MAE (100 queries) and BIRCO-Clinical-Trial (50) — too
small; ChemTEB — NQ/HotpotQA questions, which are in our training pool.

## 5. What would change the verdict

- **Upward:** G-MLP or D-COV moving covered-form retention from ~94% toward 96–97% at screen dose
  (read on `nq-250k`, the named per-arm diagnostic); or decision 12(b) plus the constructed
  scientific surface showing biomedical forms retained near the covered level.
- **Downward:** A3−A2 failing to resolve on the constructed scientific family — then the harvested
  scientific text does not buy scidocs-form retention and the headline rests on the generator; or the
  real-data trainer rate landing near M9's 10% pipeline efficiency, which makes the screen a
  fortnight of box time and forces the cloud for everything.
