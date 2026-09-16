# M20 planning inputs (2026-09-16)

Facts gathered in the planning session so the execution session does not repeat the lookups.
Nothing here is a registration; `m20/REGISTRATION.md` and `m20/beir15_registry.json` are written
and pushed by the execution session before observation. Values marked *expected* are from the BEIR
paper and must be verified against the download.

## Verified state

- No `m8-reserved-spent` tag locally or on origin; the only spent tags are `m7-six-spent`,
  `m9-six-spent`, `m10-six-spent`.
- `m13src/reserved_support.py` hardcodes `SYSTEMS = ("nano-dense", "bge-small-en-v1.5",
  "leaf-ir-asym")` and the report derivation assumes exactly those three. The extension is code,
  not configuration.
- `scripts/m13_reserved_cloud.py` is tracked on `main`. The artifact worktree `work/m13cloud` sits
  at `480f7a7`, 34 commits behind `main`; the controller refuses to run unless HEAD equals
  `origin/main`, so advance it as part of execution.
- Local disk: `/` 535 GB free; `D:` 438 GB free.

## Roster (eight systems)

| system | query side | document side | new work |
|---|---|---|---|
| `nano-dense` | frozen M13 checkpoint | Stella `ffeb2b7e` shards | inherited |
| `bge-small-en-v1.5` | bge-small `5c38ec7c` | same model | inherited |
| `leaf-ir-asym` | `MongoDB/mdbr-leaf-ir` `4262131b` | Arctic-M v1.5 `e58a8f75` | inherited |
| `zero-dense` | released `constella-zero` | Stella shards (reused) | R20; query encode only |
| `stella-query` | Stella `ffeb2b7e` query tower with its query prompt | Stella shards (reused) | R22; query encode only |
| `bm25` | `bm25s` Lucene defaults `k1=1.2, b=0.75`, English stopwords, stemmer, `s > 0` filter and self-hit drop as in `m7src/fusion.py` | corpus text | R22; CPU index per corpus |
| `zero+bm25 dbsf@100` | `m12src/qfusion.dbsf` over both runs truncated to 100 | — | R22; derived |
| `nano+bm25 dbsf@100` | same operator | — | R22; derived |

Fusion inputs must be truncated to depth 100 *before* DBSF (mu and sigma are computed over the
prefetch); the M12 self-exclusion condition applies to the DBSF rows. Reproduce one six-set DBSF
number from `m12/six_dbsf.json` with the extended code before it runs on new data.

## BEIR-15 candidates

Eight datasets are fully covered by the six-set and the reserved transaction; CQADupStack is
covered for two of twelve forums; six are new. HF revisions were read on 2026-09-16 (`HfApi.dataset_info`);
HF wrapper licence tags are **not** licence evidence (M7 ledger) — the primary-source evidence in
`m7/LEDGER.md` §"Source-level licence evidence" governs.

| dataset | source | revision | docs / queries (expected) | contact label | status |
|---|---|---|---:|---|---|
| SciFact, NFCorpus, SCIDOCS, TREC-COVID | six-set | frozen | — | clean | scored (M13) |
| FiQA, ArguAna | six-set | frozen | — | teacher-disclosed | scored (M13) |
| FEVER | `BeIR/fever` | `426aea4d` | 5,416,568 / 6,666 | reserved; teacher-disclosed; comparator-training | reserved four |
| DBpedia-entity | `BeIR/dbpedia-entity` | `6029bd92` | 4,635,922 / 400 | reserved | reserved four |
| CQADupStack (12 forums; dataset score = mean of the 12 forum nDCG@10 values, BEIR convention; per-forum rows also reported) | `mteb/cqadupstack-*` | android `e03f271e`, english `fa6afc4c`, gaming `1c976986`, gis `080d0534`, mathematica `fc1af0b6`, physics `9c2faaa0`, programmers `339629ee`, stats `6b474f6f`, tex `2b6f1e27`, unix `3197b091`, webmasters `c691d4af`, wordpress `6117bc48` | ≈ 457k / 13,145 total | android/english reserved; programmers/physics M7-dev; others clean | 2 reserved, 10 new (≈ 394k docs) |
| MS MARCO | `BeIR/msmarco` | `a918e0d1` | 8,841,823 / 6,980 (dev) | comparator-training; validation-only licence role (primary source: microsoft.github.io/msmarco/Notice.html, "non-commercial research purposes only"; recorded in `research/m7-data-licensing.md` row 1 and its 2026-09-04 validation-use rule change) | new |
| NQ | `BeIR/nq` | `b7253e6c` | 2,681,468 / 3,452 | M7-dev (250k subset); comparator-training | new |
| HotpotQA | `BeIR/hotpotqa` | `a7e8bab2` | 5,233,329 / 7,405 | M7-dev; comparator-training | new |
| Touché-2020 | `BeIR/webis-touche2020` | `7ebed360` | 382,545 / 49 | source-family contact with ArguAna; licence disclosed | new |
| Quora | `BeIR/quora` | `54934c6f` | 522,931 / 10,000 | comparator-training; **no primary-source licence — evaluation-only row (R22)** | new |
| Climate-FEVER | `BeIR/climate-fever` | `d4edc15f` | 5,416,593 / 1,535 | teacher-disclosed (FEVER corpus); **no affirmative licence — evaluation-only row (R22)**; corpus is FEVER's Wikipedia, so reuse the reserved FEVER shards after hash verification | new queries; corpus shared |

Excluded: BioASQ, Signal-1M, TREC-NEWS, Robust04 (not freely redistributable). Quora and
Climate-FEVER are included under R22 as evaluation-only rows because they are part of the standard
public benchmark; their licence status is disclosed in every table and they remain out of training,
targets, negatives and generation seeds in every role.

New Stella document encodes: ≈ 18.0M passages (MS MARCO, NQ, HotpotQA, Touché, Quora, ten
CQADupStack forums) on top of the reserved ≈ 10.1M; Climate-FEVER adds queries only if its corpus
hashes match FEVER's. Arctic-M (LEAF documents) and bge-small encode the same
corpora; both are much cheaper than Stella. Historical local caches under `work/enc*` may hold
Stella HotpotQA vectors from M7 dev; reuse only if hash-bound to the same revision and fp16
contract, otherwise re-encode.

## Cost sketch

Reserved cap stands at 55.2 h × $1.6636111111/h ≈ $92. BEIR-15's encode is ≈ 1.75× the reserved
volume; register its own cap in `m20/REGISTRATION.md` from the measured reserved rate
(`results/m13_encode_benchmark*.json`) rather than this sketch. Working expectation: total M20
compute well under half the $1,000 ceiling. Three stopped 500 GB volumes bill continuously; the
archive (deliverable 5) is what makes their retirement possible in M22.

## Archive sizing

fp16 1024-d Stella vectors are 2 KB per document: ≈ 57 GB for all ≈ 28M public-BEIR documents.
Raw corpora, queries and qrels compress to roughly 25–30 GB. Both targets receive the same hashed
tarballs; `results/m20_archive_manifest.json` lists every file, size and SHA-256, and re-hash at
both targets is the verification. Proposed exact destinations, to be fixed in the registration:
local `/mnt/d/constella-archive/beir15/` (Windows `D:\constella-archive\beir15\`); object storage
bucket and prefix **to be named by the owner** before the run (provider and account are not in the
repo). Credentials never enter git, logs or the pod image.

## Identities the registration must pin (Astra review, 2026-09-16)

- `constella-zero`: the Hub **weights** revision and per-file SHA-256 actually loaded (card
  revision `0e9cd89e` is not the weights revision; resolve it from `m11/release/` records).
- `stella-query`: the literal Stella s2p query prompt string, max length and truncation side, and
  the fp32-on-CUDA / TF32-off contract matching the document encode.
- `bm25` and DBSF parameters as in the roster table; the depth-100 truncation order.
- The crash rule: `reserved.crash` governs the reserved transaction (owner ruling R23,
  2026-09-16); R14 applied to the six-set run only.

## Order of work

1. Registration and registry amendment; commit and push.
2. Executor extension with tests on synthetic fixtures; DBSF reproduction check against M12.
3. Astra review, fix, Sol review (sequential, essential-only, allowlisted files, reserved
   read-exclusion, access log audited).
4. Advance `work/m13cloud` to `origin/main`; check pod state, price and wallet.
5. One cloud session: reserved transaction first (tag, four datasets, eight systems), then
   BEIR-15 remaining corpora, then archive pull; pod STOP in `finally`.
6. Results, receipts, `m10_final_run.json` → `COMPLETE`, ledger boundary note, `m20/STATUS.md`.
