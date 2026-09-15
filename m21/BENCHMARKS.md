# Canonical benchmark tables

This is the single public table set for the M21 research preview. README, the Zero and Nano
model cards, and `research/constella-in-plain-english.md` should reproduce these tables verbatim
or link here. Values are copied from committed evidence; none were recomputed, re-averaged, or
re-partitioned for this file.

## Reading rules

- The headline partition is **clean-4**: NFCorpus, SCIDOCS, SciFact, and TREC-COVID. Those four
  columns come first below. ArguAna and FiQA are always shown beside them and carry `†` because
  Stella discloses training/evaluation contact with both datasets. “Clean-4” means no disclosed
  Stella contact; it is not a causal contamination estimate. Source: `instructions-m15.md` and
  `results/m10_final_run.json`.
- Retrieval-quality values are nDCG@10 from the frozen **exact-search** harness. No ANN recall,
  Qdrant latency, or Edge latency appears in a quality table. Source: `m13/STATUS.md` and
  `m11/release/MODEL_CARD.md`.
- Dense/lexical systems and fused systems are separate tables. The registered M7 convex-fusion
  contrast and the later Qdrant-runnable DBSF result are also kept distinct.
- `ESTABLISHED` means the registered superiority claim passed its registered decision rule, or,
  for Zero C1, that the registered analysis established the opposite direction. `UNESTABLISHED`
  means superiority was not established. It never means parity, equivalence, or a tie.

## Exact-search retrieval quality: dense and lexical systems

| system | NFCorpus **(clean-4)** | SCIDOCS **(clean-4)** | SciFact **(clean-4)** | TREC-COVID **(clean-4)** | ArguAna† | FiQA† | committed source |
|---|---:|---:|---:|---:|---:|---:|---|
| constella-zero (int8) | 0.3124 | 0.1677 | 0.6101 | 0.5490 | 0.5916 | 0.3728 | `results/m7_final_run.json`, identified as the result of record by `m7/STATUS.md`; displayed in `m11/release/MODEL_CARD.md` |
| constella-nano | 0.363080 | 0.217710 | 0.721097 | 0.787116 | 0.623296 | 0.477765 | persisted score rows named by `m13/STATUS.md`; displayed there and in `m14/MODEL_CARD.md`; decision record `results/m10_final_run.json` |
| bge-small-en-v1.5 | — | — | — | — | — | — | absolute cells are not present in the allowlisted committed files; see [Discrepancies](#discrepancies) |
| LEAF asym | — | — | — | — | — | — | absolute cells are not present in the allowlisted committed files; see [Discrepancies](#discrepancies) |
| BM25 | 0.3180 | 0.1565 | 0.6791 | 0.6099 | 0.4878 | 0.2532 | `results/m7_final_run.json`, identified as the result of record by `m7/STATUS.md`; displayed in `m11/release/MODEL_CARD.md` |
| Stella teacher, symmetric | 0.4134 | 0.2395 | 0.7796 | 0.8234 | 0.6369 | 0.5536 | `results/m7_final_run.json`, identified as the result of record by `m7/STATUS.md`; displayed in `m11/release/MODEL_CARD.md` |

† Stella-disclosed training/evaluation contact; excluded from the clean-4 headline. Source:
`m13/STATUS.md`, `m14/MODEL_CARD.md`, and `instructions-m15.md`.

The dashes are intentional missing evidence, not zeroes. The allowlisted evidence reports Nano’s
per-dataset differences against bge-small and LEAF, but deriving either comparator’s absolute
score by subtraction is forbidden. Until a committed result file containing those absolute rows
is admitted as evidence, this table must not invent them.

## Exact-search retrieval quality: fusion

Fusion is not a query-encoder-only comparison. It combines constella-zero with BM25 and is kept
outside the dense/lexical table for that reason.

| fused system | NFCorpus **(clean-4)** | SCIDOCS **(clean-4)** | SciFact **(clean-4)** | TREC-COVID **(clean-4)** | ArguAna† | FiQA† | all-six macro | clean-4 macro | committed source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| constella-zero + BM25, Qdrant DBSF, prefetch 100 | 0.3442 | 0.1850 | 0.7173 | 0.7184 | 0.5800 | 0.3872 | 0.4887 | 0.4912 | `m12/six_dbsf.json`, named and reproduced in `m12/FINDINGS.md` |
| constella-zero + BM25, convex0 `w=0.8`, prefetch 1000 | 0.3497 | 0.1881 | 0.7068 | 0.7018 | 0.5975 | 0.4026 | 0.4911 | 0.4866 | `results/m7_final_run.json`, identified by `m7/STATUS.md`; displayed in `m11/release/MODEL_CARD.md`; clean-4 macro in `m12/FINDINGS.md` |

† Stella-disclosed training/evaluation contact. The DBSF row is a post-M7 product-policy
override and prospective deployment evidence, not the operator used by M7 C3. The convex0 row is
the registered M7 operator of record and is not implemented by Qdrant. No confidence interval was
computed for DBSF versus convex0, so their observed differences establish neither superiority nor
equivalence. Source: `m12/FINDINGS.md`.

## Registered contrasts

### Nano final-run fixed sequence

The `95% reporting interval` is not the gate. The registered decision statistic is the one-sided
2.5% lower bound, with each conjunct tested at one-sided alpha 0.025 in fixed sequence. All values
below are copied from `results/m10_final_run.json`; the public six-decimal renderings also appear
in `m13/STATUS.md` and `m14/MODEL_CARD.md`.

| contrast | partition | orientation | point delta | one-sided 2.5% lower bound | 95% reporting interval | sign-flip p | status |
|---|---|---|---:|---:|---:|---:|---|
| C1b | **clean-4** | Nano − bge-small | 0.017647740632012167 | 0.003673798316062068 | [0.003673798316062068, 0.031698888096276336] | 0.0064099359006409935 | **ESTABLISHED** |
| C1a | all six | Nano − bge-small | 0.027448720131874858 | 0.017270563193075145 | [0.017270563193075145, 0.037594566875687525] | 9.99990000099999e-06 | **ESTABLISHED** |
| C2a | all six | Nano − LEAF asym | 0.016181116478486347 | 0.0065043256929021645 | [0.0065043256929021645, 0.025962418134693854] | 0.0005399946000539995 | **ESTABLISHED** |
| C2b | **clean-4** | Nano − LEAF asym | -0.0010630231295977424 | -0.014456023597166766 | [-0.014456023597166766, 0.012369626557096195] | 0.5594744052559475 | **UNESTABLISHED** |

The clean-4 Nano-versus-LEAF result is **UNESTABLISHED**. It is not parity, equivalence, or a
tie. Source: `results/m10_final_run.json`, `m13/STATUS.md`, and `m14/MODEL_CARD.md`.

### Zero M7 confirmatory family

These are the rounded public values committed in `m7/STATUS.md`, which identifies
`results/m7_final_run.json` as the result of record. C2 is judged against its Holm threshold, not
against the unadjusted p-value alone.

| contrast | partition | orientation | point delta | raw 95% interval | sign-flip p | registered threshold | status |
|---|---|---|---:|---:|---:|---:|---|
| C1 | all six | Zero int8 − LightRetriever dense (0.4583) | -0.0243 | [-0.0405, -0.0086] | 0.997 | registered C1 rule | **ESTABLISHED BELOW THE BAR** |
| C2 | all six | Zero int8 − BM25 (0.4174) | +0.0165 | [+0.0017, +0.0311] | 0.0149 | Holm 0.0083 | **UNESTABLISHED** |
| C3 | all six | Zero + BM25 convex0 − OpenSearch (0.4868) | +0.0043 | [-0.0063, +0.0151] | 0.219 | registered C3 rule | **UNESTABLISHED** |

Zero therefore did **not** confirmatorily beat BM25: M7 C2 failed the registered Holm threshold.
Likewise, C3 did not establish superiority or equivalence. Source: `m7/STATUS.md` and
`instructions-m15.md`.

M7 also reports the pre-registered clean-4 sensitivity contrasts as descriptive: C1 `-0.0443`
with interval `[-0.0675, -0.0212]`; C2 `-0.0311` with interval
`[-0.0517, -0.0109]`; and C3 `-0.0107` with interval `[-0.0262, +0.0043]`.
No p-values for those descriptive rows are reported in the allowlisted source, so they are not
promoted into the confirmatory table. Source: `m7/STATUS.md`.

## Serving cost

These are synthetic query latencies, not workload estimates. Protocol: three fresh processes per
model, batch one, four CPU threads; hydration includes loader imports, verification, and load but
excludes interpreter startup; first inference is reported separately; warm timing uses five
warmups and twenty synthetic batch-one samples per length; OS disk cache was not flushed. Source:
`results/m13_serving_costs.json`.

The displayed values are the committed medians-across-three-trials table copied verbatim from
`m14/HANDOFF.md` and `m14/MODEL_CARD.md`; the raw trials and byte counts are in
`results/m13_serving_costs.json`.

| model | hydration | first query | warm 20-word p50 | peak RSS | model assets | committed source |
|---|---:|---:|---:|---:|---:|---|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB | `results/m13_serving_costs.json`; rendering in `m14/HANDOFF.md` |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB | `results/m13_serving_costs.json`; rendering in `m14/HANDOFF.md` |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB | `results/m13_serving_costs.json`; rendering in `m14/HANDOFF.md` |

This is query-encoder serving cost only. It is not retrieval quality, ANN recall, Qdrant search
latency, or end-to-end system latency. The frozen Stella document tower still runs when documents
are indexed. Source: `results/m13_serving_costs.json` and `m14/MODEL_CARD.md`.

## Mandatory training-dose disclosure

Nano saw exactly **199,999,721 training examples**. Do not round this to 200M. The nominal
schedule was 200,000,000, but the executed dose recorded by both `dose_run_examples` and
`training.examples` is 199999721. Source: `results/m13_build_record.json`; public rendering in
`m13/STATUS.md`, `m14/HANDOFF.md`, and `m14/MODEL_CARD.md`.

## Discrepancies

This audit was taken against the public text **as it stood before M21's edits**. Items 3-11 were
fixed in commits `d1b4063`..`124155d` and are marked RESOLVED below; they are retained as the
record of what was wrong and how it was corrected, not as open defects. Items 1 and 12 are carried
limitations. Dispositions: `m21/STATUS.md`.

1. **The required absolute bge-small and LEAF dataset rows are absent from the allowlist.**
   `results/m10_final_run.json` contains Nano-minus-comparator deltas and identifies the frozen
   comparator source, while `m13/STATUS.md` and `m14/MODEL_CARD.md` publish Nano’s absolute rows.
   None of the allowlisted committed files publishes the absolute per-dataset bge-small or LEAF
   nDCG@10 values. Subtracting the deltas from Nano would violate the no-recomputation mandate.
   The two required rows therefore remain visibly unpopulated above. This is a source-coverage
   blocker for a fully populated canonical table, not permission to derive the values.

2. **RESOLVED — Nano absolute-score provenance.** The six Nano values are the equal-weight means
   of the committed per-query rows in `results/m10_final_scores/<dataset>.json`, each carrying
   `system = "nano-dense"` and the freeze/registry/comparator hashes of the registered run. This
   was verified by reproduction: `scifact.json` has n=300 and mean 0.721097, matching the published
   row exactly. The same values are published in `m13/STATUS.md` and `m14/MODEL_CARD.md`. The
   aggregate is the registered statistic over committed rows, not a re-derivation.

3. **RESOLVED — two different fused results were both called “fused.”** Every public fusion number now names convex0 or DBSF with its prefetch depth. README and the plain-English
   page lead with `0.4911`, the M7 convex0 all-six macro, while the deployed Qdrant recommendation
   is DBSF at `0.4887`. Their clean-4 macros also differ: convex0 `0.4866`, DBSF `0.4912`.
   These are different operators and prefetch depths, not replicate disagreement. Public text
   must name the operator whenever it gives a fusion number. Sources: `README.md`,
   `research/constella-in-plain-english.md`, `m11/release/MODEL_CARD.md`, and `m12/FINDINGS.md`.

4. **RESOLVED — unresolved results were called ties.** The page now says superiority is unresolved and claims no equivalence. It calls M7 C3 (`+0.0043`, interval
   `[-0.0063, +0.0151]`) a “statistical tie” and calls the DBSF/convex observations a tie despite
   saying no confidence interval was computed for the latter. Those statements do not establish
   equivalence. Canonical wording is **UNESTABLISHED superiority; no equivalence claim**. Sources:
   `research/constella-in-plain-english.md`, `m7/STATUS.md`, `m12/FINDINGS.md`, and
   `instructions-m15.md`.

5. **RESOLVED — the planned 200M dose was presented as executed.** Plan and execution are now distinct, with 199,999,721 as the executed figure. It says the build
   controller pins exactly `200,000,000`, labels the build `200M`, and defines the glossary dose as
   exactly `200M`. The committed build record instead reports exactly `199999721` executed
   examples, rendered publicly as `199,999,721`. The plan and execution must remain distinct.
   Sources: `research/constella-in-plain-english.md`, `results/m13_build_record.json`,
   `m13/STATUS.md`, and `m14/MODEL_CARD.md`.

6. **RESOLVED — stale in-progress Nano numbers.** The build narrative is past tense with the registered results. Its opening table says the
   real build is “Running,” with roughly `24` hours remaining, about `$100` spent, and `94%` of the
   teacher so far; later it says M13 has `170` tests green and Nano’s six-set access is unspent.
   M13 is now closed, `m13/STATUS.md` reports `267` M13 tests passed, and the six-set run is
   complete. These operational numbers are stale snapshots, not benchmark evidence. Sources:
   `research/constella-in-plain-english.md` and `m13/STATUS.md`.

7. **RESOLVED — two Zero cost protocols were mixed.** Public text uses the common three-model protocol and names it. The Zero card reports `0.38 ms`
   end-to-end on one CPU core and `0.22 s` hydration, while the later common serving protocol
   reports `0.1119 ms` warm p50 on four threads and `0.2618 s` hydration. README also abbreviates
   the query time as about `0.1 ms`. These values are not interchangeable; only the common
   three-model protocol belongs in the canonical serving table. Sources:
   `m11/release/MODEL_CARD.md`, `README.md`, `results/m13_serving_costs.json`, and
   `m14/HANDOFF.md`.

8. **RESOLVED — Zero asset sizes used different definitions.** Each figure now states what it measures. README describes the release as
   `94 MB`; the Zero card calls the int8 query asset `31.8 MB`; the common serving table reports
   `90.1 MiB` of measured assets. The card’s file table shows that `model.npz` is the roughly
   `94 MB` artifact while the int8 ONNX path is roughly `31 MB`; the serving JSON measures the
   NumPy/tokenizers bundle. These are not one canonical asset quantity and must retain their
   labels. Sources: `README.md`, `m11/release/MODEL_CARD.md`, and
   `results/m13_serving_costs.json`.

9. **RESOLVED — the architecture sentence was internally inconsistent.** Corrected in the plain-English page. Original finding:
   The sentence said bge-small is a `6`-layer encoder while also
   naming feature layers `12`, `8`, and `4`.** The committed build record reports the exported
   feature-layer list `[12, 8, 4]` and total parameters `34540672`; it does not support the
   `6`-layer description. Source: `research/constella-in-plain-english.md` and
   `results/m13_build_record.json`.

10. **RESOLVED — Zero's rows are cited to their committed source.** Original finding:
    They were displayed in a card rather than in an
    allowlisted result JSON.** `m11/release/MODEL_CARD.md` gives the Zero, BM25, Stella, and convex
    fusion rows, and `m7/STATUS.md` explicitly identifies `results/m7_final_run.json` as their
    result of record. That JSON was not in this task’s read allowlist. The rows are therefore
    traceable to a named committed result, but were not independently re-read here.

11. **RESOLVED — Nano vector-storage figures assumed float64 serving.** FastEmbed now returns fp32 natively, so a 1024-d query vector is 4,096 bytes and the float64 arithmetic is removed from the cards.
    `m14/HANDOFF.md` says Nano emits fp32 and gives `4.096 GB` per million vectors. The later Nano
    card says native FastEmbed output is float64: `8,192` bytes per query and `8.192 GB` per
    million, reduced to `4,096` bytes and `4.096 GB` only after an explicit fp32 cast. The serving
    cost JSON records the intended fp32 width (`4096` bytes) but does not establish the native
    served dtype. Any storage figure must state whether the cast was applied. Sources:
    `m14/HANDOFF.md`, `m14/MODEL_CARD.md`, and `results/m13_serving_costs.json`.

12. **Several public-page measurements lie outside this canonical table and lack a per-number
    committed-result pointer in the allowlisted documents.** They must remain findings rather than
    be copied into a benchmark row until their result files are named and checked: README’s edge
    shard change (`466 MB` to `1.82 GB`) and padding diagnostic (about `500` PAD rows and cosine
    `0.35`); the plain-English page’s M9 component scores/retentions, M10 screen table, M11 parity
    claim (`4.5e-8` over `1,024` queries), M17 score table, M18 score/recall table, and M19 judgment
    counts (`1,376` results and `60` undecidable). Source locations for the claims are `README.md`
    and `research/constella-in-plain-english.md`; neither document attaches those individual
    values to a committed result file in the material this task was permitted to inspect.
