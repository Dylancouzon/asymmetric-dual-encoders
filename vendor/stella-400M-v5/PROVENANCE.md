# Vendored remote code: `NovaSearch/stella_en_400M_v5`

Copied byte-identically from the Hugging Face snapshot at revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` on the date of this commit. These are the modules the model's
`auto_map` names, i.e. the code that runs under `trust_remote_code=True`.

**Why vendored.** The document side of this artifact's every reproduction executes this
code. A pinned revision does pin it today, but a Hub repo can be rewritten, gated or
deleted, and nothing here should depend on that not happening.

**Licence.** The model card declares MIT; `modeling.py` and `configuration.py` carry
Apache-2.0 headers (Copyright 2024 The GTE Team Authors and Alibaba Group; portions
NVIDIA Corporation). Both permit redistribution with attribution, which this file is.
No modifications have been made; hashes are in `results/m7_teacher_code_pin.json`.

**Not imported by this repo.** Loading still goes through `transformers` at the pinned
revision; this copy exists so a third party can reconstruct the doc tower if the Hub
copy ever changes. `teacher_code.verify()` asserts the two are byte-identical.
