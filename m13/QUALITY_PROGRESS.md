# M13 first scheduled quality evaluation

Evaluation: `mid1041666`, reported step **1,041,666**, approximately **33.3M examples**.

Metric: nDCG@10 on the admitted COV surface; units averaged within families, then four families weighted equally.

| Family | M13 | Teacher | Teacher retention |
|---|---:|---:|---:|
| BRIGHT | 0.184328 | 0.219125 | 84.12% |
| consumer-health | 0.713016 | 0.750712 | 94.98% |
| finance | 0.325429 | 0.372625 | 87.33% |
| legal | 0.856656 | 0.884456 | 96.86% |
| **Overall** | **0.519857** | **0.556730** | **93.38%** |

This is the first quality point; there is no earlier M13 quality evaluation for a trend comparison. Training resumed afterward. COV was used in recipe selection, so this is development evidence, not fresh final validation. It does not establish superiority over M9 on a different suite.

Artifact: `results/m13_cov_mid1041666.json`, copied byte-for-byte from the training host and verified against its remote SHA256:
`1f8627196f45bd9cd9feadaa559c93dd1743223327c55d80288445366be9f689`.

Teacher baseline: `results/m10_cov_teacher_ceiling.json`. Execution commit: `ee1bd6617c4fec7f4c97c73e7c6eed52f66ff50a`. No DEV-6, final six-set, reserved or LoTTE evaluation was performed for this checkpoint.
