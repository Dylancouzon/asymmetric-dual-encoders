# Sol review — M20 benchmark artifact and M22 readiness (2026-09-23)

Model `gpt-5.6-sol`, reasoning effort xhigh, `--sandbox read-only`. Verdict **GO WITH CONDITIONS**.
No BEIR-15, reserved-four, R1 or R2 result withdrawn.

Brief: `work/m20/logs/sol_artifact_brief.md`. Transcript: `work/m20/logs/sol_artifact_review.log`.

Access audit: 31 bounded exec commands. One `rg`, scoped to a single allowlisted file. No recursive
search over `results/` or `work/`; no read of `results/frozen_eval/untouched-*`, reserved qrels
caches or `work/m9reserve`; **no web search naming a reserved dataset or forum** — the rule added
after the previous review's incident held.

## What was fixed in response (all before M22 starts)

- **P1 — mandatory labels.** The page emitted only `reserved` and `teacher-disclosed`, hiding
  `comparator-training`, `M7-dev`, `source-family-contact-with-arguana`, `licence-unverified` and
  the licence roles. Every row now carries its full registered contact set and licence role, and the
  captions state that `comparator-training` is a separate concern from teacher overlap and is NOT
  excluded from the clean subset.
- **P1 — arithmetic.** "FEVER alone contributes +0.243 of that gap" conflated a per-dataset gap with
  a contribution to a four-row mean. Corrected: the disclosed-row mean gap is `+0.0550`; FEVER's
  per-dataset gap is `+0.2431`, contributing `+0.0608`, while the other three contribute `−0.0058`.
- **P2 — precision.** Arrays were 10dp roundings and R2's contrast literals were pre-rounded to 4dp.
  All three arrays are now spliced at full precision, and the verifier checks bit-exact equality.
- **P2 — external comparison.** "match published BEIR figures" overstated BM25 (0.2256 vs 0.228).
  Now: bge-small *matches* the official card's 0.40833, BM25 is *close to* 0.228 without matching.
- **P2 — stale handoff.** `m20/STATUS.md` still said stage C "is running now" and listed it as a
  pending five-day open item; `ROADMAP.md` still said "stage C next". All corrected.
- **Debt discharged.** The published HTML is committed at `research/artifacts/m20-benchmark-evidence.html`
  (sha256 `47c29e0162923e5ab454df5d96078b9f0478c9ac5935db19e230e98b958dfbfb`) rather than living only
  under gitignored `work/`.

## What it confirmed

- The 176 committed score rows reproduce `results/m20_beir15_run.json` exactly; recomputation from
  per-query scores differed only by floating summation order, at most `3.72e-14`.
- The clean-11 subset is the registered operational meaning of "no disclosed teacher overlap";
  DBpedia and CQADupStack should **not** be excluded merely for being reserved.
- CQADupStack is not double-counted: one row of the fifteen, each forum carrying one twelfth.
- Presenting both aggregates does not itself reweight or overturn R1 — but it must not be copied
  into M22 as a registered result.
- No standing rule forbids a private external artifact carrying comparator or reserved-score
  absolutes; the prohibition is specifically about **model cards**.

## M22 readiness — resolvable without an owner ruling

I had expected the mandate's "card carries the BEIR-15 tables" to contradict the standing rule that
comparator per-dataset absolutes stay off cards. It does not. The resolution:

- publish Nano / Zero / document-tower rows, with all contact labels and the FEVER caveat;
- preserve the registered contrast deltas;
- do **not** publish bge/LEAF per-dataset absolutes, nor the artifact-only aggregate/rank analysis;
- an owner ruling is needed **only** if M22 wants the full comparator matrix on a card.

Object-storage unavailability blocks M20 stage D and M22's storage-retirement recommendation. It
does **not** block card preparation, staging, verification or release. Deletion of the three pods
still requires explicit owner approval even after both archives and Hub bytes verify.

---

# Verdict: GO WITH CONDITIONS

The committed M20 results are sound, and M22 may proceed once the published page and handoff documents are corrected. No BEIR-15, reserved-four, R1, or R2 table result needs withdrawal.

## Findings

- **P1 — The page omits mandatory contact and licence labels.** The registration requires every table row to carry all contact labels and any licence role ([registry](/home/dylan/asymetric-dual-encoders/m20/beir15_registry.json:543), [registration](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md:103)). The page emits only `reserved` and `teacher-disclosed` flags ([HTML](/home/dylan/asymetric-dual-encoders/work/m20/logs/m20-benchmarks-artifact.html:409)). It therefore hides material facts including comparator-training on MS MARCO/NQ/HotpotQA/Quora, M7 development contact, source-family contact, and evaluation-only/licence-unverified status. This matters directly to the clean-11 bge comparison. Republish with the complete `contact` and `licence_role` fields.

- **P1 — “FEVER alone contributes +0.243 of that gap” is mathematically wrong.** The four disclosed-row mean gap is `0.547872 − 0.492868 = 0.055004`. FEVER’s per-dataset gap is `+0.243106`, but its contribution to that four-row mean is `+0.060777`; the other three contribute `−0.005773` ([claim](/home/dylan/asymetric-dual-encoders/work/m20/logs/m20-benchmarks-artifact.html:264)). Correct it to “FEVER’s per-dataset gap is +0.243,” or give the actual macro contribution.

- **P2 — The embedded arrays are not exact copies.** All 120 BEIR cells and all 32 RESERVED cells are correct roundings to ten decimal places, but none is bit-exact to the JSON; maximum error is `4.98e-11` and `4.90e-11`, respectively. R1 is embedded at full precision, while R2 is pre-rounded to four decimals ([HTML](/home/dylan/asymetric-dual-encoders/work/m20/logs/m20-benchmarks-artifact.html:409), [contrasts](/home/dylan/asymetric-dual-encoders/work/m20/logs/m20-benchmarks-artifact.html:434)). Visible four-decimal values, aggregates, ranks, and shifts are unchanged, but the page’s “reproduce to 1e-12” claim is not true of its own literals. Splice the full-precision values during republication.

- **P2 — The BM25 external comparison is proximity, not a match.** M20’s `0.2256` differs from the published `0.228` by `0.0024`; they do not even round identically to three decimals. The bge comparison is fine: M20 `0.4082` and the official model-card value `0.40833` both render as `0.408`. Replace “match published BEIR figures” with “is close to” for BM25 ([page](/home/dylan/asymetric-dual-encoders/work/m20/logs/m20-benchmarks-artifact.html:382), [BEIR paper](https://arxiv.org/pdf/2104.08663), [official bge card](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5e62ea33e012fda8c02802b906664c915ebd1bb1/README.md)).

- **P2 — The post-stage-C handoff remains internally stale.** The top of `m20/STATUS.md` correctly says stage C is complete, but its execution section still says stage C “is running now” ([STATUS](/home/dylan/asymetric-dual-encoders/m20/STATUS.md:139)), and its owner-open-items section still calls stage C a pending five-day run ([STATUS](/home/dylan/asymetric-dual-encoders/m20/STATUS.md:306)). `ROADMAP.md` likewise says “stage C next” and “M22 after M20” ([ROADMAP](/home/dylan/asymetric-dual-encoders/ROADMAP.md:25)). Remove those stale premises before handing control to M22.

## Checks that passed

- The 176 committed per-system score rows reproduce `results/m20_beir15_run.json` exactly at the stored `mean_ndcg10` level. Direct recomputation from their per-query scores differed only by floating summation order, at most `3.72e-14`.
- The RESERVED array uses the correct sources: FEVER/DBpedia table values and the two forum values from `cqadupstack.per_member`.
- All R1 values and intervals match exactly. All R2 displayed values are the correct four-decimal rounding.
- All-15 and clean-11 means, ranks, and shifts are correct. Nano/bge are `0.5081 < 0.5171` over 15 and `0.5137 > 0.5059` over 11.
- Excluding exactly FiQA, ArguAna, FEVER, and Climate-FEVER is the registered operational meaning of “no disclosed teacher overlap.” DBpedia and CQADupStack should not be excluded merely because they are reserved; comparator-training is a separate label, not part of this teacher-clean definition.
- CQADupStack is not double-counted. It appears once among the 15 datasets, after an equal mean over its 12 forums. Each forum therefore has one-twelfth of one dataset’s macro weight, exactly as registered.
- Presenting both aggregates does not, by itself, reweight or overturn R1. The page makes the post-hoc descriptive status and non-inferential interpretation sufficiently explicit. It must not be copied into M22 as a registered result, however: M22 says nothing is recomputed or repartitioned.
- The two DBSF examples are numerically correct; the tendency is not monotonic but is supportable as a qualitative descriptive observation.
- The bge-over-stella reversal occurs exactly on FEVER, HotpotQA, and Climate-FEVER.
- SciFact, NFCorpus, and the `1,978 shards / 131.1 GiB / one damaged / hash reproduced` statements match their committed receipts.

## External publication and M22

There is no standing rule in the inspected material forbidding a private external artifact containing comparator or reserved-score absolutes. The prohibition is specifically against comparator per-dataset absolutes on model cards. Calling the page “internal evidence” does not itself create access control, but no underlying rule was violated; scores are not closed queries or qrels.

M22’s table instructions are resolvable without an owner ruling:

- Publish the Nano/Zero/document-tower rows where applicable, with all contact labels and the FEVER caveat.
- Preserve the registered contrast deltas.
- Do not publish bge/LEAF per-dataset absolutes or the artifact-only aggregate/rank analysis.
- An owner ruling is needed only if M22 wants the full comparator matrix on a card.

Object-storage unavailability blocks M20 stage-D completion and M22’s storage-retirement recommendation. It does **not** block card preparation, staging, verification, or release. Even after both archives and Hub bytes are verified, deletion still requires explicit owner approval.

Before M22 starts, correct the two P1 page defects and the stale STATUS/ROADMAP handoff. The full-precision splice should be included in that republication. As reproducibility debt, preserve a versioned copy or recorded hash of the published HTML; the current copy is ignored under `work/` and is not Git-bound.
