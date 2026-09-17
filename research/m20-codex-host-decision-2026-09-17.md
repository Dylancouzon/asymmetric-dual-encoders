# M20 execution-host validation — Astra (gpt-6-astra, xhigh), 2026-09-17

Read-only validation against `research/m20-host-decision-brief-astra-2026-09-17.md`.
Verdict **ENDORSE WITH CONDITIONS**; all conditions are discharged in `m13/RULINGS.md` R24 and
`m20/REGISTRATION.md` §8. Access log audited: every file it opened is on the brief's list, no
protected payload, nothing written. Its report follows verbatim.

---
**ENDORSE WITH CONDITIONS.** The local box is my preferred execution host, provided its memory and GPU checks pass before reserved access is spent. The matching software and existing reproduction evidence support that choice. They do not yet establish that the complete workload fits, and switching hosts requires a dated amendment to the execution pin.

This was a read-only decision validation. I ran package-metadata queries, `nvidia-smi`, resource queries, and arithmetic from the supplied receipts. I did not run evaluators, tests, model imports, downloads, or memory benchmarks. No protected payload was opened and nothing was written.

1. **The environment claim is true for the documented package versions, and materially favors local execution.**

   Installed versions match the brief:

   | Package | Installed |
   |---|---|
   | Torch | `2.8.0+cu126` |
   | Transformers | `4.57.6` |
   | Sentence Transformers | `5.7.0` |
   | Tokenizers | `0.22.2` |
   | NumPy | `2.3.5` |
   | Datasets | `5.0.1` |
   | bm25s | `0.3.11` |
   | PyStemmer | `3.1.0` |

   I also checked the remaining release-stack versions in the handoff: ONNX `1.22.0`, ONNX Runtime `1.29.0`, and FastEmbed `0.8.0` all match. Python is `3.12.14`. The four `fusion.selected_on.bm25_run_keys` entries in `m7/FREEZE.json` all record the installed BM25 and PyStemmer versions.

   This establishes continuity with the documented release stack. It does not establish complete environment identity: package versions alone do not authenticate drivers, binary builds, model caches, or numerical behavior across GPUs.

   The reproduction evidence is useful and appropriately limited. The DBSF receipt records zero deltas for all three checks on both SciFact and NFCorpus. The SciFact end-to-end smoke records agreement with the prior dense/BM25/DBSF values and differences consistent with the precision of the published Nano/comparator table. I inspected those receipts; I did not recompute their scores or open the underlying comparison files.

   Rebuilding on a fresh pod introduces avoidable setup and compatibility risk, particularly because `HARNESS.md` explicitly disclaims the historical lockfile as a complete fresh-install recipe. But rebuilding does **not inherently change a reported number**: reproducing package versions, model bytes, execution settings, and the existing open-data checks can control that risk. An environment can also be reproduced deliberately instead of resolving current package releases.

   **There is an important qualification to the brief’s BM25-version argument.** `_pkg_versions()` participates in `cache_key()`, but `fusion.bm25_run()` computes that key only when `cache_path` is supplied. Both M20 scoring paths call it without a cache path, and their BM25 score records include configuration and depth but not package versions. Consequently, the cache mechanism does **not** automatically expose a changed BM25 version on these M20 paths. Record and verify the installed versions explicitly for execution and continuation. This strengthens the case for retaining the present environment. See [fusion.py](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:150).

2. **Local execution needs a dated host amendment; it need not change the experiment.**

   The “both E arms” cloud instruction concerns the registered training comparison. It does not independently prohibit this descriptive evaluation on a local GPU.

   Nevertheless, the retained-host requirement is explicit elsewhere: R19 names A100 execution, R22 specifies a cloud session, `instructions-m20.md` requires the reserved transaction on the retained A100, and `m13/RESERVED_EXECUTION.md` identifies its execution arrangement. “Descriptive” does not waive those instructions.

   Record the owner’s already-approved recommendation as a dated operational amendment before execution. It should identify the local host, replace the cloud-specific controller/cost/STOP deliverables with applicable local receipts, and state how stage deadlines apply locally. Preserve the original registration in history.

   That amendment need not change any estimand, NDO-3 weight, threshold, partition, bootstrap constant, alpha, or release rule. Preserve model revisions, prompts, 512-token truncation, fp32 compute, disabled TF32 matmul, fp16 document storage, exact retrieval, and the full roster. Preserve the completed six-set evidence.

   Retained volumes remain STOP-only. Choosing local execution provides no authority to delete them.

3. **The direct scorer retains the scientific transaction, but two bare commands would omit material readiness checks.**

   `reserved_transaction.py` retains substantial protection: clean/pushed-tree checks; roster, dataset, bootstrap and disk checks; authentication of the prior trigger and checkpoint; complete pre-encode verification and shard hashes; spent-tag handling; durable BEGIN before protected access; bounded continuation; authenticated completed-system outputs; and completion publication.

   The following controller responsibilities would need explicit disposition:

   | Controller responsibility | Treatment for local execution |
   |---|---|
   | Exact pushed `origin/main` execution; deployment/checkpoint agreement; preservation of existing controller receipts and logs | Preserve an explicit execution commit and artifact identities locally. Transaction checks overlap, but checking the configured upstream is not identical to the controller’s fresh `origin/main` check. |
   | Compilation, reserved-support tests, and roster identity assertion before encoding | Retain readiness evidence for the exact execution version. A roster mismatch must be caught before the tag; its later scoring-time assertion is insufficient protection. |
   | Required `PASSED` DBSF reproduction receipt | Retain and bind it to the execution receipt. The transaction preflight does not enforce this prerequisite. |
   | Corpus-loader preflight, followed by the offline corpus-loader check after stage A | Retain. Completed vector shards do not prove that BM25 can load the required corpus through the offline path. |
   | Pre-tag query-model exercise, pinned Zero snapshot availability, and Nano dependency validation | Retain. These catch unavailable or incompatible models before spending access. |
   | Explicit cache/device/thread settings and offline Hub, Transformers, and Datasets settings for scoring | Preserve the intended execution environment; invoking the scorer directly does not itself supply the controller’s environment. |
   | Disk admission before pre-encode | Retain a local capacity calculation. The controller requires approximately 166 GB initially to accommodate reserved vectors while preserving the later 120 GB floor. |
   | Shared stage-A deadline, projection arguments, separate stage timeouts, process monitoring and exit checks | Record suitable local scheduling limits and monitoring. Omitting `--stage-deadline` disables the pre-encode projection. Do not silently discard registered caps. |
   | Result retrieval, completion checks, controller receipt and cost receipt | Remote transfer becomes unnecessary; durable completion verification and a local execution receipt remain necessary. The scorer still performs its own publication. |
   | Pod identity, storage/price checks, wallet cover, cloud allocation checks, resume/SSH handling, and STOP confirmation | Remove from the local workload’s launch path through the amendment. Continue accounting for retained storage and preserve STOP-only authority. |

   These can be a small explicit launch checklist or wrapper; they do not require another transaction framework. The source comparison is [the controller](/home/dylan/asymetric-dual-encoders/scripts/m13_reserved_cloud.py:191) against [transaction preflight](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:161).

   **If the capacity waiter remains active, stop it before local execution.** Otherwise a delayed cloud resume can race the local run or incur unintended charges. Recheck the durable spent-tag state immediately before launch. I did not inspect live cloud state or remote tags during this review.

4. **RAM is the leading unresolved feasibility risk, but it is not the only one.**

   Live resource queries reported 25 GiB RAM, approximately 23 GiB available, 7 GiB swap with 1.3 GiB already used, 534 GiB free on `/`, and 438 GiB free on `/mnt/d`.

   `nvidia-smi` failed with **“GPU access blocked by the operating system.”** This prevents live GPU certification from this sandbox; it does not demonstrate that the owner’s ordinary runtime lacks CUDA access. The launch environment must establish actual GPU access.

   **BM25:** The implementation materializes corpus text, tokenizes it, and builds the index in memory. Peak memory includes those overlapping objects, index-construction intermediates, retrieval arrays, and Python result dictionaries. Document count alone is an inadequate predictor: token counts, text lengths, and vocabulary also matter.

   Two-size measurements are the right first step, but use the actual BM25 path and representative corpus text, measure the complete peak, and extrapolate with headroom for the rest of the process. Similar document counts do not make HotpotQA proof that FEVER fits. MS MARCO’s larger count also does not automatically make it the highest-memory corpus.

   **Crucially, stage A does not presently build BM25 indexes.** Despite the registration’s stage-order description, the controller runs neural pre-encoding, while reserved BM25 indexing occurs in `_score_bm25` after the tag. The offline corpus preflight checks loading and counts, not indexing memory. Therefore the resource feasibility check must precede the tag rather than relying on a successful stage A. See [reserved_support.py](/home/dylan/asymetric-dual-encoders/m13src/reserved_support.py:305).

   **VRAM and length:** Ten GB is not an established hard blocker, but the supplied smoke does not certify it for worst-case shapes. Document and query SentenceTransformer paths use batches of 256; Stella uses a 32,768-token batch budget. The A100 benchmark’s passages averaged about 153 tokens. A maximum of 512 in that sample does not establish feasibility for batches dominated by 512-token inputs.

   Stage C also constructs all query encoders together. Running separate `--corpora` invocations does not make that path one-model-at-a-time within each invocation. Exercise its actual residency and exact-search path, not just isolated document encoders. Include result construction and the reserved bootstrap in host-memory planning.

   If smaller inference batches or search chunks are needed, establish and validate them before the tag. Do not solve memory pressure by reducing the 512-token contract, changing precision, approximating search, dropping systems, or changing the corpus.

   **Disk:** The registered document volumes imply approximately **147.36 GB of raw fp16 vectors across all three towers**: 44.02 GB reserved and 103.34 GB additional BEIR work. That is comfortably below current free space, so vectors alone are not a blocker. Downloads, extracted/cache copies, corpus text, archive staging, temporary files, and destination copies still require a complete peak-space budget. Preserve the registered 120 GB scoring floor.

   **Time:** Applying the two local measurements to the registered document volumes gives approximately **41.5–47.1 hours for reserved encoding alone** and **97.4–110.5 hours for additional BEIR encoding alone**. Thus 182 hours is plausible planning arithmetic, not an established end-to-end duration. Downloads, hashing, BM25, query encoding, exact search, archive work, and interruptions remain outside those encode-rate estimates.

   The A100 Stella measurements are about 13–15% faster than the corresponding local measurements. That supports “no demonstrated large encoding advantage”; it does not establish equal total runtime or erase the A100’s memory advantage.

   **Recovery and persistence:** R23 makes interruption recoverable under fixed identities; it does not make a deterministic OOM harmless. An incomplete reserved system can require repeating work across its four datasets, and changing the tagged code or registry afterward is not an admissible routine fix. Keep the execution checkout and environment stable, monitor the multi-day process, and protect the exported reserved archive payloads promptly after stage B. Their loss cannot be repaired by an ordinary archive rerun that reopens protected originals.

5. **No better unconditional option is established by the supplied evidence.**

   Keep the local box as the first choice. If representative memory or GPU checks fail, the fallback should be a fresh host with enough measured RAM/VRAM, the same pinned environment and artifacts, and the existing open-data reproduction checks. Select it for demonstrated workload fit; GPU rental price alone does not resolve the memory problem.

   Continued waiting has no demonstrated completion bound. The receipt verifies one capacity refusal and a $0 wallet delta over that short attempt; it does not independently verify the brief’s later retry count or establish that hours of waiting are free.

   Local execution sharply reduces incremental compute expenditure, but it does **not dissolve every cost or completion dependency**. The retained stopped volumes still bill. Using the receipt’s conservative storage formula, their combined storage is about $0.21736/hour, or **$39.56 over 182 hours**. That is historical receipt-based arithmetic, not a fresh provider quote. Object storage, transfers, and local electricity also remain outside the “no GPU rental” statement.

   Object-storage provisioning and verification remain required for M20 completion. As registered, that dependency need not block stages A–C.

The conditions for endorsement are:

- Record and push the dated local-host amendment, preserving scientific and access rules and explicitly resolving cloud-specific deadlines and receipts.
- Preserve the applicable controller checks above, explicitly record/verify package versions, stop any outstanding capacity waiter, and recheck access state at execution.
- Establish live CUDA access and sufficient RAM, VRAM, and disk for representative worst-case execution before spending the tag. Complete any required operational adjustments beforehand.
- Preserve R23 continuation identities, durable results and reserved archive exports, and finish both verified archive destinations before declaring M20 complete.

I endorse the host choice subject to those conditions. The unresolved memory measurements and blocked live GPU check prevent an unconditional launch endorsement.

Files opened, exactly:

- [research/m20-host-decision-brief-astra-2026-09-17.md](/home/dylan/asymetric-dual-encoders/research/m20-host-decision-brief-astra-2026-09-17.md)
- [CLAUDE.md](/home/dylan/asymetric-dual-encoders/CLAUDE.md)
- [HARNESS.md](/home/dylan/asymetric-dual-encoders/HARNESS.md)
- [instructions-m20.md](/home/dylan/asymetric-dual-encoders/instructions-m20.md)
- [m13/RULINGS.md](/home/dylan/asymetric-dual-encoders/m13/RULINGS.md)
- [m13/RESERVED_EXECUTION.md](/home/dylan/asymetric-dual-encoders/m13/RESERVED_EXECUTION.md)
- [m14/HANDOFF.md](/home/dylan/asymetric-dual-encoders/m14/HANDOFF.md)
- [m20/REGISTRATION.md](/home/dylan/asymetric-dual-encoders/m20/REGISTRATION.md)
- [m20/STATUS.md](/home/dylan/asymetric-dual-encoders/m20/STATUS.md)
- [m20/SMOKE.md](/home/dylan/asymetric-dual-encoders/m20/SMOKE.md)
- [m20/REVIEW_TRIAGE.md](/home/dylan/asymetric-dual-encoders/m20/REVIEW_TRIAGE.md)
- [m8src/pre_encode.py](/home/dylan/asymetric-dual-encoders/m8src/pre_encode.py)
- [m13src/reserved_transaction.py](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py)
- [m13src/reserved_support.py](/home/dylan/asymetric-dual-encoders/m13src/reserved_support.py)
- [m20src/roster.py](/home/dylan/asymetric-dual-encoders/m20src/roster.py)
- [m20src/beir15.py](/home/dylan/asymetric-dual-encoders/m20src/beir15.py)
- [scripts/m13_reserved_cloud.py](/home/dylan/asymetric-dual-encoders/scripts/m13_reserved_cloud.py)
- [m7src/fusion.py](/home/dylan/asymetric-dual-encoders/m7src/fusion.py)
- [results/m20_tower_rate_benchmark.json](/home/dylan/asymetric-dual-encoders/results/m20_tower_rate_benchmark.json)
- [results/m13_encode_benchmark.json](/home/dylan/asymetric-dual-encoders/results/m13_encode_benchmark.json)
- [results/m20_dbsf_reproduction.json](/home/dylan/asymetric-dual-encoders/results/m20_dbsf_reproduction.json)
- [results/m13_reserved_cloud_attempt1_nocapacity_2026-09-16.json](/home/dylan/asymetric-dual-encoders/results/m13_reserved_cloud_attempt1_nocapacity_2026-09-16.json)
- [m7/FREEZE.json](/home/dylan/asymetric-dual-encoders/m7/FREEZE.json) — only the `fusion` key was extracted and inspected.
