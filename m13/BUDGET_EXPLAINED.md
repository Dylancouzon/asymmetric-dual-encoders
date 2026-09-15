# Budget allowance versus expected training cost

The published ~$721 project figure is a conservative admission allowance, not an
expected bill. Source: results/m13_build_allocation.json and
results/m13_preflight_retry_allocation.json. The original allocation explicitly calls
it a conservative ceiling, retains completed E/LoTTE lines alongside actual spend,
and includes both hourly running storage and a 30-day disk reserve.

| Component | Approximate allowance |
|---|---:|
| Full build and recovery window, 144 hours | $239.56 |
| Other modelled stages (benchmark, E, LoTTE, reserved encoding, export/final) | $138.89 |
| 30-day reserve for three retained 500 GB disks/container storage | $309.00 |
| Initial paid spend and two setup hours | $18.75 |
| Original total | $706.20 |
| Additional cumulative recovery allowance at latest admission | $15.16 |
| Latest preflight-retry admission total | $721.36 |

Measured speed around1,054 examples/second gives 52.71hours for200M examples.
At$1.59/hour GPU plus conservative running storage, this is ~$87.69 for TRAINING
ONLY. It excludes new setup, evaluation, finalization, stopped-disk charges and any
recipe changes that alter throughput. It is not an $88 estimate for side evaluations.
The CPU-only M9 COV diagnostic adds no Runpod charge and does not launch training.

Last read-only balance check: $464.5132273978 against tracked funding baseline$505,
i.e.$40.4867726022 paid so far. This snapshot ages as the active Pod/storage bill.
Round two is financially plausible but has no new allocation or launch authorization;
its full costs would be reconciled first. Keep the existing active-run caps unchanged.
Idle-Pod retirement remains a pending task in RESUME.md. A reconciled expected total
must distinguish paid costs, remaining stages and storage duration; do not present the
admission allowance as an expected-spend forecast.
