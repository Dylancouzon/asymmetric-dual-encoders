# Hostile-reviewer pass on `m15/PAPER.md` (Fable), 2026-09-17

Read-only, inside the allowlist. The review's verdict: "there is a paper, but not the one drafted."
Its strongest points are recorded below with my disposition. The full text is in the session record.

## Accepted and fixed in draft v2

| Finding | What was wrong | Fix |
|---|---|---|
| Artifact sizes contradict across sections | 6.1 reports zero 90.1 MiB against nano 132.3 MiB; 6.3 reported zero 270.1 MB against nano 46.1 MB. Both orderings inverted, and the two rows measure different packagings from different milestones | v2 states both packagings with their definitions and says the ordering depends on which one you mean. The one-sided claim is withdrawn |
| The edge index is synthetic | `m9/RESULTS.md`: "One synthetic 1M x 1024 index. Latency and architecture only" | v2 says so in the section and in the limitations |
| The 50-to-100x and the 1.11-to-3.28x come from different harnesses | Inside the experiment that produced the collapse, the encoder-only gap was 0.234 ms against 1.173 ms, about 5x | v2 reports the same-harness collapse, 5x to 1.96x, and keeps the cross-harness numbers apart |
| The 1.11x lower bound is Docker noise | The source says to treat sub-millisecond Docker differences as noise | v2 quotes the range with that condition |
| The LEAF comparison is not on one index | LEAF-asym encodes documents with `snowflake-arctic-embed-m-v1.5` (109M, 768-d) and queries with `mdbr-leaf-ir` (23M). Its teacher ceiling on the six is 0.5264 against Stella's 0.5744 | v2 discloses it in Section 5.1 and reframes the contrast as system against system |
| Retention reframes the LEAF result | The student retains 0.9256 of its teacher; LEAF-asym retains 0.979 of its own | v2 reports both in the frontier table and says which way it points |
| Section 1 paragraph 3 is false as written | pyNIFE and arXiv 2306.11550 both freeze the document encoder, and LEAF's asym mode exists to keep the teacher's index | v2 uses the one defensible sentence from `NOVELTY.md` |
| Depth-10 inversion is dev-only | `m12/FINDINGS.md`: tiers 1 and 2 are dev only, four components | v2 labels it |
| "The gap is architectural" overstates M8 | `m8/FINDINGS.md`: closure was an owner decision not to spend, and the strongest lead was never run | v2 states what the evidence supports |
| The clean-4 headline registration date differs by family | The partition was pre-registered for both; the headline designation predates the student's numbers and postdates the table's six-set run of 2026-08-28 | v2 dates it per family |
| The frontier is never drawn | Contribution 1 promises it; the draft scattered the rows | v2 adds the frontier table, with derived cells labelled as derived |
| Buried results | The fused table system reaching the inference-free band with no query-side network; the 19x rescore trap; the subword-fragmentation correlation; the inert objective at 99.75% | v2 promotes all four |
| Index bytes are missing from the cost story | Per one million documents: bge-small 0.77 GB at 384-d, OpenSearch postings about 1.4 GB, LEAF and arctic-m 1.54 GB at 768-d, LightRetriever 3.07 GB at 1536-d | v2 adds the ladder and Stella's place in it |

## Accepted, recorded, not fixed by writing

- **The Stella query tower's latency is unmeasured.** It is the swap a team with a Stella index
  would actually make, and the paper cannot price it. Named in Section 9 as an open measurement.
- **No recall is reported for any index configuration.** The quantization quality claim is withdrawn
  until the deferred benchmark runs.

## Rejected or downgraded

- **"C5 is deployment hygiene, cut it."** Kept, shortened. A silent 0.35 cosine from a default
  tokenizer setting is exactly the failure a practitioner hits, and it is the cost side of the
  swap claim. It moves out of the contribution list.
- **"C4b absorbability is only a lemma."** Kept as a short subsection with its pooling condition.
  It is the reason a whole class of levers was skipped, and stating it saves the next team the sweep.
- **"C2 should be a row, not a contribution."** Kept as a result with the confound disclosed. The
  reviewer's own reframing, that the win tracks the teacher gap, is now in the paper.
