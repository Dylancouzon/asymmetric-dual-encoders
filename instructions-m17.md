# M17 — Constella Zero v1.1 planning

**Active for planning, 2026-09-11 (Dylan).** Plan better vocabulary across query domains,
especially cloud/software terms such as S3 and k8s, and a modest retrieval improvement within a
future **72 hours on the local RTX 3080**. This session is planning only; small local checks and
cheap Luna research agents are authorized. Keep the document tower/index and Nano/M13 unchanged.
No rule exception, training execution, publication or protected evaluation access is implied.

Start at `m17/STATUS.md`, `m17/PLANNING.md`, `m17/registry.json` (draft), and `m17/CODEMAP.md`.
The recommendation is a bounded, jointly trained vocabulary extension and one hard-candidate
distillation comparison. Static late interaction is researched for compatibility, not adopted.

Previously M16; moved 2026-09-10. Read `m8/FINDINGS.md` and
`m8/EXPLORED.md` before proposing a probe. Historical ideas, arithmetic and reopening conditions remain
in `research/archive/m10-cleanup-2026-09-10/instructions-m16.md`.

## Historical idea register — not the active execution queue

| Lead | Status / trade-off |
|---|---|
| Document-side LoRA co-adaptation (`E14-LORA`) | Historical unrun capacity experiment; outside the current frozen-index request |
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
