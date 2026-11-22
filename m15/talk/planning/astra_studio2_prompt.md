Review round for the workload studio, focused on AUTO ROUTING, which the owner will mostly demo. Files: m15/talk/demo/studio.py, m15/talk/demo/studio.html (read both). Attached: screenshot after quiet -> spike -> quiet -> floor raised to 0.56.
Auto policy now: quality first. Settings below the nDCG floor are never used (nDCG per setting comes from the measured NQ sweep). Among allowed settings, take the best quality whose estimated service time plus max(this request's queue wait, smoothed recent wait) fits 60% of the p95 target; estimates are live EWMAs that decay toward quiet-time cost when unused (half-life 5 s), and a pick writes the decayed value back. If nothing fits, take the allowed setting with the lowest quiet-time cost (highest capacity). Default target 150 ms, floor 0.50.
Stress results (laptop, 14 workers, one serialized Edge shard; p95 is the max over 4 one-second samples):
OK quiet 40/s                         served     42 p95 max      86 nDCG 0.627 floor 0.50 mix {'stella': 1.0}
OK spike 600/s                        served    657 p95 max      93 nDCG 0.568 floor 0.50 mix {'nano': 1.0}
OK back to 40/s                       served     43 p95 max     111 nDCG 0.621 floor 0.50 mix {'stella': 0.29, 'nano': 0.71}
OK 1000/s floor .50                   served   1005 p95 max     134 nDCG 0.576 floor 0.50 mix {'nano': 1.0}
-- 2000/s floor .50 (expect over)     served   1410 p95 max    1093 nDCG 0.562 floor 0.50 mix {'nano': 1.0}
-- 2000/s floor .47                   served   2010 p95 max     100 nDCG 0.490 floor 0.47 mix {'nano': 0.23, 'zero': 0.77}
OK 40/s floor .50                     served     44 p95 max      98 nDCG 0.625 floor 0.50 mix {'stella': 1.0}
OK 40/s floor .62                     served     34 p95 max      89 nDCG 0.641 floor 0.62 mix {'stella': 1.0}
OK wave 40-600                        served    585 p95 max      22 nDCG 0.579 floor 0.50 mix {'stella': 0.01, 'nano': 0.99}
OK bursts                             served     40 p95 max      85 nDCG 0.617 floor 0.50 mix {'stella': 1.0}
OK agent tasks                        served    327 p95 max     126 nDCG 0.576 floor 0.50 mix {'nano': 1.0}
OK quiet again                        served     44 p95 max     117 nDCG 0.601 floor 0.50 mix {'stella': 1.0}
Owner goals: interactive, represents real workloads, intuitive at a glance, live continuous traffic with controls that apply immediately, nDCG floor with priority over latency, auto routing as the centerpiece; truthful.
Push for improvements, concisely: (1) anything in the auto policy that is wrong, unstable, or would draw a fair objection from search engineers (and the fix); (2) the 3-5 changes with the most value for an auto-routing demo (what to show so the audience SEES the router's decisions and the tradeoff; controls; presets); (3) a 3-4 minute stage path built around auto with one line per beat; (4) any honesty trap. Read only the two named files. No recursive searches; never read results/frozen_eval/untouched-*, work/m9reserve, or reserved qrels.
