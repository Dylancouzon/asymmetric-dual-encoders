"""T1's challenger encoder `Spec`s, registered into `encoders.REGISTRY` at RUNTIME.

WHY AT RUNTIME. `m7src/encoders.py` is frozen for M8 (G3), and adding a `Spec` there would be an
edit to it. Inserting into the dict at import time gets the same effect with none of the cost: the
registry is a plain dict, `encoders.active()` reads `M7_ENCODER` and looks it up, so a screen that
imports this module before touching anything else sees the challenger. Nothing under `m7src/`
changes.

EVERY FIELD BELOW IS FROM A PRIMARY SOURCE, not a plausible default -- see
`research/m8-planning/challenger-specs-2026-08-29.md` for the evidence per field. A wrong `Spec`
does not crash; it silently encodes a different model, which is why `validate_encoder.py` is
mandatory before any encode (m7/CODEMAP.md).

THREE THINGS THAT DIFFER FROM EVERY EXISTING REGISTRY ENTRY, and each is the point of T1:
  * `tokenizer_id` and `vocab` are NOT the 30,522-row BERT WordPiece every current entry shares.
    That is exactly why these candidates were closed on arithmetic in M7 and why B7 had to exist
    (LEDGER §18).
  * `cls_id` is 50281, not 101. `table.py` hardcodes `CLS_ID = 101` as a module default but takes
    it as a constructor PARAMETER, so `m8src/teacher_screen.py` passes `spec.cls_id` explicitly.
    Inheriting the default would put a silently wrong vector behind every degenerate empty query.
  * the released table's size changes with the vocabulary: 50,368 x 768 = **38.68 MB** int8 for
    both models here, against the incumbent's 31.3 MB and the 233 MB cap.

ONLY THE TWO ModernBERT CANDIDATES ARE REGISTERED HERE. They are the registered CG-frame controls,
they need no `trust_remote_code`, and their `tokenizer.json` files are byte-identical to each
other -- so they share one bag matrix and sit in ONE frame, which makes their comparison the
cleanest thing T1 can measure. The other two are deliberately absent:
  * `stella_en_1.5B_v5` needs `trust_remote_code` (same-repo, so `revision` does pin the code) and
    has NO usable sequence-start row: its `config.json` and `tokenizer_config.json` disagree
    (`bos_token_id` 151643 against `bos_token: null`). Its fallback row must be registered before
    it is screened.
  * `microsoft/harrier-oss-v1-0.6b` uses LAST-TOKEN pooling, which `m7src/teacher.py` does not
    implement and raises on; and its training data is undisclosed, which needs Dylan's ruling
    before adoption. Two blockers, one of them code.
"""
import encoders
from encoders import Spec

CHALLENGERS = {
    "granite-r2": Spec(
        name="granite-r2",
        repo="ibm-granite/granite-embedding-english-r2",
        revision="47ea694b257b703fee9253d75c2b1f2985180498",
        dim=768,
        pooling="cls",                       # 1_Pooling/config.json pooling_mode_cls_token: true
        query_prefix="",                     # no prompt in the card or config_sentence_transformers
        doc_prefix="",
        trust_remote_code=False,             # ModernBertModel is native to transformers 4.57.6
        post_dense=None,                     # modules.json: Transformer -> Pooling only
        tokenizer_id="modernbert-bpe-50368",
        cls_id=50281,
        vocab=50368,                         # len(tokenizer) == config vocab_size, no padding gap
        notes="CG-frame control. BEIR(15) 53.1 per its own card. Apache-2.0. int8 table 38.68 MB. "
              "Closed in M7 on arithmetic (50,368-vocab fp64 Gram = 20.3 GB), reopened by B7.",
    ),
    "gte-modernbert-base": Spec(
        name="gte-modernbert-base",
        repo="Alibaba-NLP/gte-modernbert-base",
        revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        dim=768,
        pooling="cls",
        query_prefix="",                     # config_sentence_transformers.json prompts: {}
        doc_prefix="",
        trust_remote_code=False,
        post_dense=None,
        tokenizer_id="modernbert-bpe-50368",  # byte-identical tokenizer.json to granite-r2
        cls_id=50281,
        vocab=50368,
        notes="CG-frame control. BEIR(15) 55.33 per its own card. Apache-2.0. int8 table 38.68 MB. "
              "Shares granite-r2's tokenizer byte for byte, so the two share a bag matrix.",
    ),
}


def register():
    """Insert the challengers. Refuses to shadow an existing name -- silently replacing a
    registered Spec would make every cached vector under that name mean something else."""
    for name, spec in CHALLENGERS.items():
        if name in encoders.REGISTRY and encoders.REGISTRY[name] is not spec:
            raise AssertionError(
                f"{name!r} is already in m7src's REGISTRY. Refusing to shadow it: every encode "
                f"cache keyed on that name would silently change meaning.")
        encoders.REGISTRY[name] = spec
        encoders.BY_REPO[spec.repo] = spec
    return sorted(CHALLENGERS)


register()
