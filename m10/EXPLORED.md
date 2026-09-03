# M10 closed avenues, each with what would reopen it

The register is `m10/PLANNING.md` §7 (considered and rejected, with reopening conditions) and
`m10/COV_CANDIDATES.md` (refused selection components). This file lists what changed after the
plan was written, so a future session does not re-derive it.

| avenue | why closed | reopens if |
|---|---|---|
| FineWeb, any role | ruled 2026-09-01: documents need reserved-set fingerprints that do not exist; seeding adds a rights review for topics Wikipedia and the pool already carry | family A wins on forms yet COV shows a topic gap Wikipedia cannot seed |
| A student above 35M | hard cap, Dylan 2026-09-01 | never |
| A nonlinear head | no fastembed serving path; width comes from linear multi-layer pooling instead (parity proven, `results/m10_head_width_parity_mac.json`) | fastembed gains a post-pooling head slot and the linear width proves insufficient |
| Claude (Sonnet/Haiku) as the synthetic-query generator | Anthropic's terms restrict using the Services to train competing models; the M7 licensing rule requires an open-weights generator with a pinned revision; a usage window is not a batch pipeline | never |
| Climate-FEVER as a claim-form selection surface | no licence at the primary source (the HF tag is a wrapper); same finding as M7 | a licence statement appears at github.com/tdiggelm/climate-fever-dataset |
| LegalBench ConsumerContractsQA | CC BY-NC 4.0 at the primary source | never |
| Reserved-set document fingerprints | an unspent held-out access in disguise (Codex pass 1 B2) | never inside M10 |
| Spending M9's LoTTE read on the M9 candidate | would burn the only fresh surface on a candidate that misses | never |
| The RTX 3080 box as M10's execution target | Dylan 2026-09-01: "M10 will be done on a GPU budget, if allowed, or not at all." A LEAF-scale build is ≈ 10 days on it; the 50M dose it forced was a box artifact | never for M10 |
| A 50M-example build with an 83.4M cap | set by the box's wall-clock; LEAF's dose is 201M on an easier target. Now 200M, extension capped by budget | never — the dose is budget-bound |
| Qwen3-8B 4-bit and the Qwen3-4B fallback as generation artifacts | needed only for a 10 GB card; the A100 serves the 8B in bf16. A Mac mlx-lm 4-bit pass may prototype prompts and produces nothing that enters the smoke record, the data, or the manifest | the GPU has under 40 GB |
| The COV resolution check as a screen-cutting rule | Codex pass 7: its comparators were family F's backbones and it made a COV-read decision before selection. Demoted to a descriptive resolution number with non-candidate comparators (e5-small-v2, gte-small) | never |
