# M17 — better zero (parked idea register)

Previously M16; moved 2026-09-10. No schedule or compute committed. Read `m8/FINDINGS.md` and
`m8/EXPLORED.md` before proposing a probe. Full ideas, arithmetic and reopening conditions remain
in `research/archive/m10-cleanup-2026-09-10/instructions-m16.md`.

| Lead | Status / trade-off |
|---|---|
| Document-side LoRA co-adaptation (`E14-LORA`) | Leading unrun capacity experiment; needs budget and explicit reopening of the frozen index premise |
| Train teacher for distillability / full co-training | Larger follow-ups; would give up drop-in compatibility with stella's existing index |
| Hard-candidate listwise distillation (`R-LIST`) | Open; prior uniform-bank loss was nearly inert, not a clean test of this class |
| Multiplicity-dependent pooling (`B10`) | Adds capacity; evidence weak, not a demonstrated improvement |
| Remove instruction prefix (`B4`) | Cheap hypothesis; measure it, do not promote algebra about offsets to a quality guarantee |
| Tokenizer-first joint distillation (`B5`) | Different from failed additive rows on frozen weights; budget table bytes explicitly |
| Fusion-aware table training, transferred from M12 | Needs aligned per-query candidate/lexical scores, one corpus, operator-specific loss and a new noise floor; currently unfunded |

**Do not revive cosine-versus-squared-L2 as a normalized-output loss lever.** They differ by a
constant scale; the old B3 recommendation was withdrawn. The pyNIFE retention comparison is one
exposed dataset with different teachers and ANN search, not a transferable +0.035 forecast.
Do not repeat absorbable centering/whitening, fixed-row additive n-grams or the failed M8 probes
without their recorded reopening conditions. M12 already supplies the DBSF deployment recipe.

Any new experiment requires scope, budget and a pre-observation protocol. It must not interfere
with M13's frozen-teacher nano experiment or consume its protected access by assumption.
