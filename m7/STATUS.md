# M7 status

**Stage:** bring-up (step 1–2 of the mandated order)
**Updated:** 2026-08-25

## Machine (confirmed)

RTX 3080, **10 GB VRAM** (not 12) · 25 GB RAM · 16 cores · 946 GB free on ext4 (`/`, WSL2) · driver 610.53 / CUDA UMD 13.3, nvcc 12.6 → torch cu126 wheels.

## Running

- Python 3.12 venv + torch cu126 install.

## Last result

None yet.

## Next step

1. Finish env, verify `torch.cuda.is_available()`.
2. `scripts/validate_perquery.py`, re-download the six, verify `eval_manifest.json` corpus hashes.
3. Named harness-validation cells (bge-small ArguAna 0.6034, bge-small SciFact 0.7127, bm25 FiQA 0.2532, each ≤0.003).
4. Encode-throughput benchmark on 10K docs → wall-clock + RAM/disk budgets.

## Open blockers

None. Nothing needed from Dylan yet (HF release go is the only pending item, far off).
