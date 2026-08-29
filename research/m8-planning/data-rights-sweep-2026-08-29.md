# M8 data-rights + shadow-dev sweep (Sonnet, 2026-08-29)

*Primary-source licensing review for new training corpora (Task A) and shadow-dev / M9-reserve
eval candidates (Task B). Project standard applied: no affirmative licence at a primary source =
OUT; a hub wrapper tag is not evidence. Everything the agent could not verify is flagged.*

## Task A — training-data rights (verdicts)

- **USPTO patent full text — IN (public domain).** Verified at the regulation, not a wrapper:
  37 CFR 1.71(d)&(e), 1.84(s) — specifications/claims are generally not subject to copyright.
  Bulk: data.uspto.gov/bulkdata (note: Open Data Portal requires account sign-in from 2026-06-18).
  HF mirrors (HUPD CC-BY, AllenAI ODC-BY) are over-claims on PD text — cite 37 CFR, not their tags.
  Zero overlap with the six / reserved four.
- **EUR-Lex — IN (with one stated gap).** Commission Decision 2011/833/EU: reuse for commercial or
  non-commercial purposes; exceptions (private individuals, third-party works) noted. The decision
  is SILENT on TDM/ML training — an unconditioned reuse grant, not an affirmative TDM clearance;
  record that nuance. Corpora: NLP-AUEB/eurlex (EURLEX57K, CC-BY-SA-4.0) or nlpaueb/multi_eurlex
  (CC-BY-4.0) — mirror licences agree with the primary. Zero eval overlap. Genre: legal/regulatory.
- **US federal (CFR, Congressional, court opinions) — IN.** 17 U.S.C. §105; CourtListener/FreeLaw
  bulk opinions marked "free of known copyright restrictions" (their CC BY-ND covers the editorial
  layer only, not opinion text). Zero eval overlap.
- **PMC-OA — CONDITIONAL (E8).** The Commercial-Use-Allowed subset is real and cleanly filterable
  via oa_file_list.csv (CC0/CC-BY/CC-BY-SA/CC-BY-ND groups; ~3.4M articles total in PMC-OA).
  BUT: NFCorpus is PubMed-derived and CORD-19 (TREC-COVID's corpus) draws substantially on PMC-OA —
  the exact PMID overlap is NOT web-resolvable and must be measured locally (NFCorpus ~10K PMIDs ∩
  the commercial-subset file list) before any decision. Also gates future PubMed-family eval
  candidates (BioASQ/PubMedQA) via the R2 rule.
- **arXiv — OUT except a filtered minority.** The default arXiv submission licence grants arXiv a
  non-exclusive DISTRIBUTION right and explicitly cannot authorize third-party reuse; the Kaggle
  "CC0" tag covers Cornell's compiled metadata table, not abstract-text copyright. Only per-paper
  CC-BY/CC0 opt-ins (a minority, ~18% ballpark, approximate) are usable — and they inherit SciDocs
  adjacency (S2ORC citation graph). Per our own wrapper-tag rule: OUT as a bulk source.
- **SEC EDGAR — OUT.** Filing prose is authored by private companies and retains copyright; only
  SEC's own material is PD. (The sweep explicitly corrected an initial wrong PD impression.)
- **Stack Overflow post-2024 — OUT** (no-LLM-training clickwrap, confirmed unchanged); this also
  taints benchmarks built from post-2024 dumps (FreshStack).
- **HackerNews — OUT.** The MIT licence is on the API client code; the YC ToU makes no affirmative
  third-party reuse grant for HN posts.
- **OpenAlex — metadata genuinely CC0, but abstracts are deliberately NOT distributed as plaintext
  ("legal constraints"; inverted-index only)** — reconstructing plaintext reintroduces exactly the
  risk they're avoiding. Not a clean text source.
- **DBpedia/Wikipedia reconciliation:** DBpedia-entity short abstracts ARE Wikipedia lead sections
  verbatim — the mechanism behind the ledger's already-disclosed 9.32% TRAIN-document overlap.
  Nothing new to disclose; the figure is mechanistically sound.

## Task B — shadow-dev / M9-reserve candidates

| candidate | licence | verdict |
|---|---|---|
| **LoTTE** | CC BY-SA 4.0 over the 2021 StackExchange dump (pre-clickwrap, same footing as CQADupStack) | Best ready-made option: clean licence, right size (400–2,100 q / 100K–2M docs per topic, 5 topics). Caveat: StackExchange-family (different subforums from CQADupStack and from the reserved pair). Usable ONLY under a written reading that "out-of-family" means "not literally CQADupStack". |
| FreshStack | CC-BY-SA label over an Oct-2024 SO dump — permissive tag over clickwrap-restricted data | OUT (wrapper problem in reverse) |
| SciQ | CC BY-NC | OUT |
| TREC Robust04/News/CAsT | LDC restricted | OUT |
| BRIGHT | no licence at primary source; majority StackExchange anyway | OUT |
| RAR-b | 12 sub-licences, reformatted MCQ, poor genre fit | not pursued |
| CoIR | CC BY-SA, but code modality | flag only |
| BioASQ/PubMedQA | licence unclear / covers labels not text; PubMed family | CONDITIONAL on the E8/PMC decision (clean only if PMC-OA is REJECTED for training) |
| MLDR-en | mC4-derived | OUT (C4 exclusion) |
| MIRACL-en | already a TRAIN source + >5M docs | doubly OUT |

**Ranked:** shadow-dev → LoTTE (with the family-reading caveat put to Dylan) — and the sweep is
explicit that finding NO other clean ready-made candidate is a real gap (COLIEE promising on genre,
no licence stated; organizer contact would be needed). M9 reserve → build-our-own retrieval sets
over **EUR-Lex** (EURLEX57K) and **USPTO** full text: the cleanest rights stories available and
genuinely out-of-family, at the cost of constructing queries/qrels ourselves (which must then be
done by a frozen, documented, pre-registered procedure to be defensible).

## Unreached primary sources (explicit)

NFCorpus/TREC-COVID ↔ PMC-OA-commercial PMID overlap (local check required); BRIGHT licence;
BioASQ licence; COLIEE licence (organizer contact); ESCI data-vs-code Apache-2.0 scope (already an
open ledger item, unchanged); arXiv CC-subset ∩ SciDocs overlap (unquantified).
