"""Teacher-init rows at the TOKENIZER's true size, for encoders where that is not `vocab_size`.

THE BUG THIS EXISTS FOR, and why M7 could not have seen it. `m7src/init_table.teacher_rows` sizes
the table with `tok.vocab_size`, and `get_init`'s `vocab=` argument is honoured only for the
`random` init. For every model in M7's registry that is correct, because all ten ship stock BERT
WordPiece where

    tok.vocab_size == len(tok) == config.vocab_size == 30,522.

ModernBERT breaks that identity:

    tok.vocab_size  = 50,280     the base BPE vocabulary
    len(tok)        = 50,368     base + 88 special/added tokens
    config.vocab_size = 50,368   the actual embedding matrix

and **`[CLS]` is token 50,281 — inside the gap**. A 50,280-row table therefore has no row for the
token that `Preproc(add_special_tokens=True)` puts at the front of EVERY query, and none for
`[SEP]` either. It is not a rounding detail: it is a table that cannot represent its own inputs.

So this is a third thing that must move with the encoder, alongside the two `m7/CODEMAP.md`
already names (anything assuming dim 768, and `table.py`'s `CLS_ID`): **anything assuming
`tok.vocab_size == len(tok)`**.

WHAT THIS DOES. Exactly what `teacher_rows` does -- forward `[cls] + prefix_ids + [token] + [sep]`
for every token id and take the pooled, projected, normalized output -- with `V = len(tok)`
instead of `tok.vocab_size`. It is a faithful copy at the right size, not a redesign; `m7src` is
frozen for M8 (G3) so the copy lives here, and it falls straight through to `get_init` whenever
the identity does hold, so the incumbent's path is untouched and its cached init is still reused.
"""
import numpy as np

import m8base

INIT = m8base.WORK / "init"
INIT.mkdir(parents=True, exist_ok=True)


def true_vocab(tok):
    """The number of rows a table must have to index every id this tokenizer can emit."""
    return max(len(tok), tok.vocab_size)


def needs_m8_init(tok):
    return true_vocab(tok) != tok.vocab_size


def teacher_rows(pre, batch=256, device=None):
    """`m7src/init_table.teacher_rows`, verbatim in construction, sized by `len(tok)`."""
    import torch
    import encoders
    from init_table import spec_tag
    from table import get_tokenizer
    from teacher import load_post_dense, load_teacher, pool_project_normalize

    device = device or m8base.device()
    sp = encoders.active()
    tok = get_tokenizer()
    V = true_vocab(tok)
    p = INIT / f"{spec_tag()}-teacher-{pre.fingerprint()}-v{V}.npy"
    if p.exists():
        return np.load(p)

    _, model = load_teacher(dtype=torch.float32, device=device)
    dense = load_post_dense(sp, device)
    pre_ids = tok(pre.prefix, add_special_tokens=False)["input_ids"] if pre.prefix else []
    cls, sep = tok.cls_token_id, tok.sep_token_id
    if cls is None or sep is None:
        raise AssertionError(
            f"{sp.name}: tokenizer has cls_token_id={cls!r} sep_token_id={sep!r}. The init's "
            f"context is [cls] + prefix + [token] + [sep]; without both, the row for every token "
            f"would be built in a different context than the queries it must serve. Register an "
            f"explicit context for this encoder before screening it.")
    out = np.empty((V, sp.dim), dtype=np.float32)
    for lo in range(0, V, batch):
        hi = min(lo + batch, V)
        seqs = [[cls] + pre_ids + [t] + [sep] for t in range(lo, hi)]
        ids = torch.tensor(seqs, dtype=torch.long, device=device)
        att = torch.ones_like(ids)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=att).last_hidden_state
        v = pool_project_normalize(h, att, sp.pooling, dense)
        if v.shape[1] != sp.dim:
            raise AssertionError(f"{sp.name}: init rows are {v.shape[1]}-d but Spec.dim is "
                                 f"{sp.dim}")
        out[lo:hi] = v.detach().cpu().numpy()
    np.save(p, out)
    return out


def get_init(kind, pre, **kw):
    """Drop-in for `init_table.get_init` that is correct when the identity fails.

    Falls through to M7's function whenever `tok.vocab_size == len(tok)`, so the incumbent reuses
    its existing cached init byte for byte and nothing about M7's path changes."""
    import init_table
    from table import get_tokenizer
    tok = get_tokenizer()
    if not needs_m8_init(tok) or kind != "teacher":
        return init_table.get_init(kind, pre, **kw)
    return teacher_rows(pre)
