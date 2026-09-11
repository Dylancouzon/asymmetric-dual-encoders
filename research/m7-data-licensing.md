# M7 training-data licensing sweep (2026-08-25, Sonnet web sweep, licenses verified on primary sources)

Context: Qdrant training and publicly releasing a retrieval model (Apache 2.0, commercial use). Decision made with Dylan 2026-08-25: **clean stack, MS MARCO excluded from training** — relaxed 2026-09-04 for validation use, see the rule-change section at the end of this file.

## Per-dataset table

| Dataset | Size | License | Verified where | Commercial-release risk | Contamination vs our six | Notes |
|---|---|---|---|---|---|---|
| MS MARCO | 8.8M passages / 500K+ pairs | "non-commercial research purposes only" | https://microsoft.github.io/msmarco/Notice.html | **High** — explicit clause, terminates on violation | Low (Bing web-derived) | IBM Granite explicitly excluded it for this reason. **Excluded by decision.** |
| Natural Questions | 307K | CC BY-SA 3.0 | https://github.com/google-research-datasets/natural-questions/blob/master/LICENSE | Low, share-alike caveat (untested on model weights) | Low | Wikipedia-based. |
| TriviaQA | 650K QA pairs | Apache 2.0 (code and data per README) | https://github.com/mandarjoshi90/triviaqa | Low for the QA pairs | Low | Evidence docs keep their own copyright; use the pairs, don't redistribute raw evidence. |
| SQuAD v1/v2 | 100–150K | CC BY-SA 4.0 | https://huggingface.co/datasets/rajpurkar/squad | Low, share-alike caveat | Low | Wikipedia-sourced. |
| HotpotQA | 113K | CC BY-SA 4.0 | https://huggingface.co/datasets/hotpotqa/hotpot_qa | Low, share-alike caveat | Low | Wikipedia-sourced. |
| FEVER | 185K claims | CC BY-SA 3.0 | https://huggingface.co/datasets/fever/fever | Low, share-alike caveat | Low | Use the primary card, not mirrors (some mirrors mislabeled GPL/NC). |
| Quora Question Pairs | 404K | No open license (Quora ToS; HF/Kaggle tags "unknown"/"other") | https://www.kaggle.com/c/quora-question-pairs/data | **High** — no affirmative grant | Low | Excluded from training AND from the dev suite: no terms analysis supports even eval use inside commercial model development (Codex finding, 2026-08-25). |
| PAQ | 65M QA pairs | Data CC BY-SA; generation code CC BY-NC | https://github.com/facebookresearch/PAQ/blob/main/README.md | Medium | Low | Using released pairs OK (share-alike); regenerating more is NC-restricted. |
| GooAQ | 5M+ | **Contradictory**: LICENSE Apache 2.0, README "no commercial purposes"; unresolved issue since 2023 | https://github.com/allenai/gooaq/issues/4 | **High** | Low | Skip until AllenAI resolves. |
| ELI5 | 270K threads | Undetermined (Pushshift/Reddit legal status; Reddit 2023 API terms restrict training) | https://huggingface.co/datasets/defunct-datasets/eli5 | **High** | Low | Skip. |
| WikiAnswers (Paralex) | 18M pairs | No clear grant from Answers.com; HF mirror's MIT tag is the uploader's claim | http://knowitall.cs.washington.edu/paralex/ | **High** | Low | Don't trust re-uploader SPDX tags. |
| StackExchange dumps | tens of M | Content CC BY-SA 4.0, but 2024 download terms include a no-LLM-training clickwrap | https://archive.org/details/stackexchange · https://devclass.com/2024/07/30/stack-exchange-restricts-access-to-dump-of-user-contributed-data-as-critics-complain-license-permits-reuse-for-any-purpose/ | **High for new downloads** | **FiQA** (finance sites) | Skip. |
| Amazon ESCI | 1.8M+ query-product judgments | Apache 2.0 | https://github.com/amazon-science/esci-data/blob/main/LICENSE | Low | None (e-commerce) | Clean, released for this purpose. |
| MIRACL | ~726K pairs, 18 langs | Apache 2.0 packaging over Wikipedia CC BY-SA | https://huggingface.co/datasets/miracl/miracl-corpus | Low-Medium (attribution obligations on underlying text) | Low | Arctic-embed 2.0 precedent. |
| Mr. TyDi | ~48K queries, 11 langs | Apache 2.0 | https://github.com/castorini/mr.tydi/blob/main/LICENSE | Low | Low | Wikipedia-based. |
| S2ORC | 81M papers | Current release ODC-By 1.0; papers keep publisher copyright | https://github.com/allenai/s2orc/blob/master/README.md | Medium | **High: SciDocs, SciFact are built from it** | Excluded for contamination regardless of license. |
| sentence-transformers/embedding-training-data | ~1B+ pairs | No collection-level license; explicit "user's responsibility" disclaimer | https://huggingface.co/datasets/sentence-transformers/embedding-training-data | **High as a blanket source** | Includes S2ORC pairs | Format reference only; go dataset-by-dataset. |
| nomic-ai contrastors data | 235M pairs | Gated; mix includes Reddit, StackExchange, S2ORC, PAQ | https://github.com/nomic-ai/contrastors | High as a bundle | High (S2ORC etc.) | Research reproducibility release, not licensing-clean. |
| Kubernetes documentation (kubernetes/website, English `content/en/docs` only) | 1,718 markdown files at the pinned revision (counts/bytes in `results/m17_k8s_source_manifest.json`) | **CC BY 4.0** for the whole repository. The repo's single `LICENSE` is the verbatim CC BY 4.0 legalcode, opening "Attribution 4.0 International". The site footer string is the publisher's own statement: `The Kubernetes Authors \| Documentation Distributed under <a href="https://git.k8s.io/website/LICENSE">CC BY 4.0</a>` (`i18n/en/en.toml`, `[main_documentation_license]`), beside `Copyright © Kubernetes a Series of LF Projects, LLC. For website terms of use, trademark policy and other project policies please see https://lfprojects.org/policies/` (`[main_copyright_notice]`). **Correction to the earlier draft row: there is no Apache-2.0 code split.** A recursive tree listing of the pinned commit (15,566 paths) contains exactly one licence file, the root `LICENSE`; `content/en/examples` has no separate grant, so the example manifests are CC BY 4.0 like the prose. | Read 2026-09-11 from `https://raw.githubusercontent.com/kubernetes/website/main/LICENSE` (396 lines, CC BY 4.0 legalcode) and `.../main/i18n/en/en.toml`; canonical legalcode https://creativecommons.org/licenses/by/4.0/legalcode; repo copy https://github.com/kubernetes/website/blob/main/LICENSE. Revision pinned: **`17133089068629ec12ca15c1bdf36a60d2671a74`** (`git ls-remote https://github.com/kubernetes/website.git refs/heads/main`, 2026-09-11). | **Low, with a mandatory attribution obligation.** CC BY 4.0 permits commercial use and adapted material; §3(a) requires attribution, a licence link and an indication of modification. Two carve-outs quoted from the legalcode: §2(b)(2) "Patent and trademark rights are not licensed under this Public License", and §2(a)(6) "No endorsement". Kubernetes/CNCF/Linux Foundation marks are therefore NOT licensed — the release may not imply endorsement or use the marks as branding. No share-alike clause (unlike the CC BY-SA rows), so derived weights carry no copyleft. Attribution artifact written and to ship with derived weights: `research/m17-k8s-attribution.md`. | Screened pre-clock through the approved fingerprint interfaces, candidate-side index streamed against protected-side text as in `m10src/cov_screen.py::screen`: `m7src/decontam.Inverted(...).match(text, MIN_SHARE=8)` over the extracted k8s documents, streaming the **development suite** (`m7src/devsuite.COMPONENTS` = nq-250k, hotpotqa, cqadup-programmers, cqadup-physics) documents via `decontam.stream_dev_component_docs` / `stream_cqa_dev_docs` and their queries. **Measured 2026-09-11 over 5,553,821 dev documents and 12,772 dev queries: 0 exact matches, 5 near-matched k8s documents (0.302% of 1,653)** — all from the document side (nq-250k 1, hotpotqa 4), none from any dev query. Flagged paths are listed in `results/m17_k8s_source_manifest.json` and must be dropped or held out before training. **Six-set and reserved-four coverage is deferred, not waived**: the only interfaces reaching them (`decontam.stream_six_docs`, `decontam.protected_queries` / `protected_query_index`, hence `m10src/protected10.build` + `protected10.hits`, and the `m9src/extended_screen.py` CLI) materialize protected payloads in-process, which pre-clock development may not do. That screen belongs inside the M17 executor, which must run `protected10.build()` before importing `m9base` and admit these documents through `protected10.hits(text, idx)` before any becomes a training input, vocabulary-discovery context or candidate-bank row. | **Admitted for download** as of 2026-09-11 (M17 step 2a) — licence row complete; admitted in principle by Dylan 2026-09-11, `m17/LEDGER.md` A2. Acquisition route: `git clone --filter=blob:none` of `https://github.com/kubernetes/website.git` then `git checkout 17133089068629ec12ca15c1bdf36a60d2671a74`, into gitignored `work/m17/sources/kubernetes-website`; submodules (the Docsy theme) not fetched; English `content/en/docs` only; never a scrape of the rendered kubernetes.io site. **Acquisition terms restricting training: none found.** Finding: an anonymous git clone of a public GitHub repository presents no clickwrap and no separate download agreement, unlike the 2024 StackExchange dump terms; GitHub's ToS binds the account holder's use of the service, not the licence of the cloned content, and the repository grants CC BY 4.0 outright. No robots/ToS gate is crossed because the rendered site is not fetched. Remaining obligations are attribution (above) and the trademark/endorsement carve-outs. Use is capped by the registry's `new_source_share_max_of_training_queries` (10% of processed views per run) and the per-domain row cap. Same checklist applies to any second project-documentation source. |

## Precedents from released models

- **Granite Embedding (IBM)**: card states training on "publicly available paired data with permissive, enterprise-friendly licenses, IBM-internal paired data, and IBM-generated synthetic data" and explicitly **did not use MS MARCO** as research-only. Released Apache 2.0. Closest precedent for what Qdrant wants. (Verified in substance via search results referencing arXiv 2502.20204 and IBM's card language; verbatim quote not pulled — read the HF README directly before citing in customer-facing text.)
- **Qwen3-Embedding** (arXiv 2506.05176): large-scale synthetic stage generated by Qwen3-32B (Apache 2.0 generator → outputs commercially reusable). The clean synthetic precedent.
- **Arctic-Embed 2.0** (arXiv 2412.04506): mC4, CC News, Wikipedia pretraining; MIRACL + prior Arctic English data (which includes MS MARCO) finetuning. No synthetic generation disclosed.
- **E5-mistral** (arXiv 2401.00368): 500K synthetic examples from GPT-3.5/GPT-4 — OpenAI terms restrict training competing models with outputs. The counterexample to avoid.
- **Gecko** (arXiv 2403.20327): FRet synthetic dataset, not released.
- **BGE / C-Pack** (arXiv 2309.07597): C-MTP scrape + public sets incl. MS MARCO; per-source licensing not disclosed. MIT on weights is not proof of a clean mix.

## Synthetic route

Generate queries over affirmatively licensed seed corpora with a permissively licensed generator (Qwen3-family precedent). No lab released their synthetic query sets; we generate our own. Generator must not be a proprietary API whose terms restrict training competing models.

Two caveats added after the Codex review (2026-08-25):
- **FineWeb/C4 are NOT approved seed corpora.** A permissive dataset wrapper does not grant rights to every scraped page underneath it. Approved seeds: Wikipedia (CC BY-SA) and the corpora of already-approved datasets. Anything else needs a rights review and Dylan's sign-off first.
- **An Apache-2.0 generator is necessary, not sufficient.** Weights licensing does not settle memorization, upstream training-data claims, or similarity of outputs to copyrighted sources. Required: document the generator's terms separately from seed rights, run memorization/near-duplicate filters on outputs against both seeds and the six benchmarks, retain per-query seed provenance, and do not redistribute the synthetic set without review.

## Contamination map (corpus-level — the binding rule for all training data)

| Source corpus | Contaminates | Why |
|---|---|---|
| S2ORC / Semantic Scholar | SciDocs, SciFact | Both built directly from it |
| PubMed | NFCorpus, TREC-COVID | NFCorpus is NutritionFacts + PubMed; CORD-19 is PubMed-derived |
| NutritionFacts.org and mirrors | NFCorpus | NFCorpus's non-PubMed half; excluding PubMed alone leaves it exposed |
| CORD-19 | TREC-COVID | TREC-COVID's corpus IS CORD-19 |
| StackExchange finance / Reddit finance | FiQA-2018 | FiQA is drawn from those posts |
| args.me / idebate | ArguAna | ArguAna is built from idebate.org counter-arguments |
| Wikipedia (general) | none of the six | None of the six are Wikipedia-sourced |

## Decided stack (Dylan 2026-08-25, tightened after the Codex and Opus reviews same day)

Approved: Amazon ESCI, TriviaQA (QA pairs only — evidence documents keep their own copyright, so it feeds distillation, not contrastive positives), MIRACL + Mr. TyDi (English), self-generated synthetic queries per the synthetic-route rules above. Everything in the High-risk rows is excluded.

**CC BY-SA: one position for the whole family** (Opus finding M2 — the earlier split was inconsistent: MIRACL/Mr.TyDi/Wikipedia-synthetic are the same CC BY-SA text under permissive packaging as the NQ/SQuAD/HotpotQA/FEVER pairs). The recorded position: training on CC BY-SA text with model-card attribution is acceptable; weights are not treated as derivative works. Under it, the QA pair sets are admissible alongside the already-approved Wikipedia-derived sources. **Dylan confirmed this position 2026-08-25** — NQ, SQuAD, HotpotQA, and FEVER are approved for training, with attribution recorded in the model card.

**Evaluation-use standard** (Opus finding M3, tightened by Codex round 2): dev/eval datasets need an affirmative open license too — model selection and fusion fitting are model development by the same standard that excluded Quora, and a wrapper tag is not evidence (the same error this file rejects for training data). Before the dev-suite freeze, the M7 session records source-level license evidence (primary URL + revision) for each dev and untouched-final set: NQ and HotpotQA are verified above; Climate-FEVER, DBpedia-entity, and CQADupStack (Apache packaging over the 2014 CC BY-SA StackExchange dump, which predates the 2024 no-LLM-training clickwrap; eval-only use here) still need their rows verified and recorded. Quora, with no license at all, stays out of training and eval.

**Fingerprint decontamination scope** (Codex round 2): exact + near-duplicate filtering runs TRAIN↔DEV, TRAIN↔the-six, and TRAIN↔UNTOUCHED-FINAL, with removal counts logged per pair of partitions.

Decontamination is fingerprint-level, not name-level: on top of the source-corpus exclusion list, every training pair and synthetic seed gets exact + near-duplicate filtering (character n-gram / MinHash) against all six benchmarks' queries and documents, with removal counts logged per benchmark. Web-derived text can mirror any excluded source under a different name.

## Rule change 2026-09-04 (Dylan): non-commercial licences are admissible for VALIDATION, not training

*"MS-MARCO is allowed for validation. Not training."* Generalised by Dylan the same message to
**any dataset with similar licensing** — MS MARCO is the largest instance, not a special case.

**Three classes, and the rule reaches only the first.**

| class | examples here | status |
|---|---|---|
| **Affirmative grant, restricted to non-commercial research** | MS MARCO, PAQ's NC generation code, GooAQ (README clause) | **Validation/eval/diagnostics ALLOWED. Training still forbidden.** |
| **No affirmative grant at all** | Quora QQP, WikiAnswers/Paralex, ELI5, StackExchange new downloads (no-LLM-training clickwrap), `sentence-transformers/embedding-training-data`, nomic contrastors | **Still fully OUT.** "No licence" is not "non-commercial licence" — there is nothing to rely on for either use. QQP's eval exclusion (Codex, 2026-08-25) stands unchanged. Reopening needs an explicit Dylan ruling. |
| **Excluded for contamination, licence irrelevant** | S2ORC (SciDocs, SciFact built from it), StackExchange **finance** — money.SE, FiQA's source (other StackExchange sites stay admissible for validation under the dataset-level standard, as CQADupStack has been since M7; corrected 2026-09-04 after the Opus pass, which found the earlier "StackExchange (FiQA)" would have excluded BRIGHT and CQADupStack) | **Still OUT of training AND validation.** The rule changes nothing; the reason was never the licence. |

**What "validation" means, operationally.** Measuring, diagnosing, screening, dev-surface use, and
reporting numbers. **Not**: gradient signal, distillation targets, mined negatives, or seeds for
synthetic generation — a synthetic query seeded from an NC corpus is NC-derived training text and is
forbidden. The released artifact must remain derivable from the clean stack alone.

**The training exclusion is unchanged and still priced** at +0.0058 [-0.0015, +0.0131] on the six
(`results/m7_cleanstack_tax.json`) — not what M7's miss is made of. `m7src/freeze.py`
`NON_COMMERCIAL_SOURCES` and the `sources-research/` quarantine in `m7src/mix.py` stay exactly as
they are: they guard *released weights*, which this rule does not touch. A validation cache must
never be written under `work/train/sources/` (a canonical `msmarco` file retroactively refuses the
frozen artifact, `mix.py:22-25`).

**Two confounds any MS MARCO validation row must carry.**

1. **Comparator home advantage.** stella, bge-small, arctic, LEAF and the OpenSearch models all
   train on MS MARCO; `zero` and `nano` do not. A comparative number on MS MARCO is biased
   **against us** — a win there is strong evidence, a loss is uninterpretable. Prefer it for
   within-system diagnostics (query-form coverage, weight transfer) over head-to-head bars.
2. **Decontamination reverses direction.** `m7src/decontam_msmarco.py` checks MS MARCO *training*
   rows against protected queries. Validation use needs the opposite sweep: MS MARCO validation
   queries against our approved training sources, or the surface is contaminated by our own mix.

Commercial-use risk of the reporting itself is low: MS MARCO is a BEIR/MTEB member and commercial
vendors publish those rows routinely. Recorded as Dylan's ruling, not an inference.
