# M8 runs

One row per run. Detail belongs in the run JSON; never restate a number a `results/m8_*.json`
already holds. Written by hand until a sweep driver exists.

| run | what | outcome | artifact |
|---|---|---|---|
| `m8_power` | joint power simulation of the full ship rule | macro SE 0.00209, MDE 0.0068, P(ship) 0.67/0.57/0.15/0.002/0.46 across the five registered scenarios | `results/m8_power.json` |
| `m8_retention_decomposition` | descriptive re-read of M7's final run: what is the query-side loss made of? | the short-query premise fails within datasets; **fragmentation** is the consistent channel (+0.050 gap per +1.0 subwords/word, t=4.6) | `results/m8_retention_decomposition.json` |
| `m8_schedule` | ridge control timing + serial GPU/RAM/disk plan | reserved-4 pre-encode 20.7 GB fp16 per system | `results/m8_schedule.json` |
| `S0` (smoke, 2K docs/slice) | LoTTE overlap screen | every slice DROPs; 3 on community intersection with reserved sets, 7 on query leakage | `results/m8_lotte_overlap.SMOKE.json` |
| `S0` (full) | LoTTE overlap screen, all 5.25M docs | running | `results/m8_lotte_overlap.json` |
| `blockcg` (smoke) | block-CG vs the direct ridge solve | agreement 2.4e-8 relative Frobenius | — |
