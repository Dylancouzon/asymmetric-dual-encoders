# Project roadmap

Reset 2026-09-10 on Dylan's request and extended by later owner-directed work. This changes scope
and navigation, not results, registered constants or access rules. **M20 is the next execution milestone**; it is not authorized to execute until a new session
follows `instructions-m20.md`. M19 closed at its development-judgment boundary with confirmation
sealed and released Zero v1 retained.

| Milestone | Deliverable | State / entry point |
|---|---|---|
| M0–M6 | Baselines, methodology, first Edge prototype | Closed; `research/m1-m6-findings.md` |
| M7 | Frozen zero table and confirmatory measurement | Closed, missed dense bar; `m7/STATUS.md` |
| M8 | Table-improvement probes | Closed; `m8/FINDINGS.md` |
| M9 | First nano build | Stopped/frozen, final close-out pending M13; `m9/STATUS.md` |
| M10 | Data, harness preparation and box recipe screen | Closed at this scope; `m10/STATUS.md` |
| M11 | Zero/document models and serving ports | Closed in repo record; `m11/STATUS.md` |
| M12 | Qdrant fusion audit | Closed; `m12/FINDINGS.md` |
| M13 | Cloud E, build, final evaluation and cost frontier | Planned historical track, not current execution; `m13/STATUS.md`, `instructions-m13.md` |
| M14 | Nano research preview (own Hub account, custom FastEmbed branch) | **Closed 2026-09-15**, published at revision `6bb167dc`; `m14/STATUS.md` |
| M15 | Whitepaper and evidence package | After measurements, nano optional; `instructions-m15.md` |
| M16 | Image-model scoping | Unscheduled; `instructions-m16.md` |
| M17 | Zero v1.1: vocabulary and modest quality improvement | **Closed 2026-09-12, negative result** (`no_survivor`: no screen arm met eligibility vs v1; zero v1 stays shipped); `m17/STATUS.md` |
| M18 | Internal Qdrant project-memory system and specialized Zero | **Closed 2026-09-13** (`SYSTEM_READY` + `ENCODER_NO_IMPROVEMENT`); `m18/STATUS.md` |
| M19 | Deterministic short-query Zero feasibility | **Closed 2026-09-13** as `SYSTEM_READY` + `ENCODER_INCONCLUSIVE`; development result label-sensitive, confirmation sealed, Zero v1 retained; `m19/STATUS.md` |
| M20 | Reserved four, official release, upstream FastEmbed PR, BEIR-18 | Next execution milestone; not authorized until a session follows `instructions-m20.md` |
| M21 | Research-preview polish: cards, one benchmark table, README, plain-English page, Nano float64, FastEmbed PR-readiness | **Closed 2026-09-15**; cards published, `constella-research-preview` branch pushed; `m21/STATUS.md` |

## Migration map

| Previous scope / number | Current owner |
|---|---|
| M10.0–M10.2 completed data / box screening | M10, closed |
| Pending E arms, M10.3 build, M10.4 judging, LoTTE, costs | M13 |
| M13 nano release | M14 preview (R20), M20 official release |
| M14 reserved four, BEIR-18, upstream FastEmbed PR | M20 (R20, 2026-09-15) |
| M14 paper | M15 |
| M15 image model | M16 |
| M16 better-zero ideas | M17 |

Historical documents and review-finding IDs keep their old numbers; the original future mandates
are in `research/archive/m10-cleanup-2026-09-10/`. Current root mandates use the new numbers.
Code (`m10src`), results, registries, freezes and spent tags retain their experiment identities.

M13 follows one ordered runbook: prepare executors → cloud E → applicable **pre-build LoTTE**
gate → build/freeze → final quality/cost measurement. No protected access follows merely from
merging M10. M9 close-out must wait until every recipe decision is fixed.

Harness improvements are ordinary maintenance (`HARNESS.md`), not another milestone. Optional
CURE/MS MARCO reads, new losses, doc2query aliases and teacher co-adaptation do not block the build.
Project audit: `PROJECT_STATUS.md`. Research lessons: `m10/FINDINGS.md`. Cross-milestone
post-mortem of M17–M19: `research/m17-m19-postmortem-2026-09-13.md`.
