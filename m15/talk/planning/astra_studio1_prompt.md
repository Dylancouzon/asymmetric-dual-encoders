New round: the owner asked to redesign the demo as an interactive "workload studio": a set of sliders, knobs and switches to set serving preferences, live traffic, latency, load and nDCG, the option to split traffic across encoders at the same time, and the ability to represent actual workloads. Files: m15/talk/demo/studio.py (server) and m15/talk/demo/studio.html (screen). Attached: screenshot. Driven scenario output (same machine, Apple M5 Pro, 14 worker threads, one Qdrant Edge shard of 2.68M NQ passages, judged NQ queries, 100 ms p95 target):

stella 40: p95 75 ms, nDCG 0.627, 1.9 cores, 59 CPU-s per 1k queries
stella spike 600: served 81/s, p95 6.9 s, 13.6 cores, 168 CPU-s/1k
tiered split 10/30/60 at 40: p95 22 ms, nDCG 0.592, 0.4 cores
tiered split at 600: served 460/s, p95 3.2 s in every lane (Stella's 60/s share ties up the shared workers: head-of-line blocking)
auto at 600: served 642/s, p95 83 ms, nDCG 0.602, 3.4 cores, 5.2 CPU-s/1k, all Nano
auto on the daily wave: p95 75 ms
Earlier findings that still apply: Edge search is serialized in this one-shard process (about 1,070 wide searches/s, 6,000 narrow); 600/s is robust (Stella fails, Nano and wide Zero hold), 2,000/s (Nano and wide Zero fail, narrow Zero holds); 1,000/s is a knife edge.
Constraints: laptop only, about 4 minutes on stage, hybrid audience, speaker drives it, must be truthful, at-a-glance readable from the back of the room, no unnecessary labels.
Deliver concisely: (1) does this achieve the owner's goal (interactive, real workloads, intuitive)? what confuses or is missing? (2) the 3-5 most valuable improvements, concrete (controls to add or remove, what to show, presets that represent real workloads such as an agent fan-out burst, a tiered free/paid product, a nightly reindex-free model swap); (3) a 4-minute stage path through the studio with one line per beat; (4) honesty traps. Read only the two named files. No recursive searches; never read results/frozen_eval/untouched-*, work/m9reserve, or reserved qrels.
