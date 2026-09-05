"""M10 generation client -- the §Data contract, decisions 14 and 15.

Server: vLLM OpenAI-compatible, `Qwen/Qwen3-8B-AWQ` at the pinned revision, thinking disabled.
Sampling: temperature 0.8, top_p 0.95, per-request `seed = blake2b-64(seed_passage_id)`.
The reply must parse as one JSON list of exactly n strings (`forms.parse`, strict); one retry on
a contract failure, then the seed is dropped.

Contract points the 2026-09-04 Codex pass forced to be explicit, all recorded in `m10/LEDGER.md` §1:
  * the 64-bit digest is used whole (big-endian), masked to 63 bits only because the OpenAI `seed`
    field is a signed integer -- it is not reduced mod 2**31-1;
  * the RETRY seed is `blake2b-64(passage_id + b"#retry")`, still deterministic in the passage id.
    An identical seed would reproduce the identical failing sample and make the retry a no-op;
  * `max_tokens` is 60 (400 for `argument`/`conversational`) PER QUERY, so a request for n queries
    gets n x that. 60 tokens for five 25-60-word `howto` posts is unsatisfiable -- see the Tier-2
    record in `m10/LEDGER.md` §3;
  * exact-duplicate removal is corpus-wide: pass the same `seen` set across every call;
  * `MODEL`/`REVISION` are settable so the registered bf16 fallback can run the same code path,
    and `health()` asserts the server actually serves the model this client names.
"""
import hashlib, os, time
from concurrent.futures import ThreadPoolExecutor

import requests

import forms

MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen3-8B-AWQ")
REVISION = os.environ.get("GEN_REVISION", "4da05a8edb55c6046cce958586c33b61da07bb79")
BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000/v1")
LONG_FORMS = ("argument", "conversational")
TEMPERATURE, TOP_P = 0.8, 0.95
TOK_PER_QUERY, TOK_PER_QUERY_LONG = 60, 400
_MASK63 = (1 << 63) - 1


def seed_of(passage_id, retry=False):
    """Per-request seed, deterministic in the seed passage id (contract)."""
    b = str(passage_id).encode() + (b"#retry" if retry else b"")
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big") & _MASK63


def max_tokens_for(form, n):
    """PER QUERY, times n -- the contract's 60/400 is one query's budget."""
    return n * (TOK_PER_QUERY_LONG if form in LONG_FORMS else TOK_PER_QUERY)


def health(base=BASE, model=None, timeout=10):
    """-> served model ids. Asserts the server serves the model this client will request."""
    r = requests.get(base + "/models", timeout=timeout)
    r.raise_for_status()
    ids = [m["id"] for m in r.json()["data"]]
    want = model or MODEL
    assert want in ids, f"server serves {ids}, not {want!r} -- the client would be rejected"
    return ids


def _one(form, passage, pid, n, base, timeout):
    """One seed -> (queries|None, contract_ok, retried, out_tokens, failure_text, finish).

    `finish` is the server's `finish_reason`, or "transport" when every attempt failed to reach a
    reply. Truncation ("length") must be distinguishable from a badly worded prompt: the first
    is a token-budget bug, the second is what the gate is for.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": forms.SYSTEM},
                     {"role": "user", "content": forms.prompt(form, passage, n=n)}],
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "max_tokens": max_tokens_for(form, n),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out_tok, last, transport, fin = 0, None, False, None
    for retry in (False, True):
        body["seed"] = seed_of(pid, retry=retry)
        try:
            r = requests.post(base + "/chat/completions", json=body, timeout=timeout)
            r.raise_for_status()
            blob = r.json()
            out_tok += blob.get("usage", {}).get("completion_tokens", 0)
            ch = blob["choices"][0]
            fin = ch.get("finish_reason")
            text = ch["message"]["content"] or ""
        except Exception as e:
            # A malformed reply is a contract failure like any other: retry once, then drop the
            # seed. It must never take the whole run down (Codex 2026-09-04). But a TRANSPORT
            # failure is not evidence about the prompt (Fable 2026-09-04): it is reported
            # separately so a dead server cannot burn a prompt revision or drop a form.
            last, transport = repr(e)[:200], True
            continue
        transport = False
        got = forms.parse(text, n)
        if got is not None:
            return got, True, retry, out_tok, None, fin
        last = text[:200] + (f"  [finish_reason={fin}]" if fin else "")
    return None, False, True, out_tok, last, ("transport" if transport else fin)


def generate(form, seeds, n=5, base=BASE, workers=32, timeout=600, label="", seen=None):
    """seeds: [(passage_id, passage_text)] -> dict with queries, rates and per-seed counts.

    `seen`: a caller-owned set of normalized query strings, shared across every call so exact
    deduplication is corpus-wide rather than per-call. Mutated in place.
    """
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda s: _one(form, s[1], s[0], n, base, timeout), seeds))
    dt = time.time() - t0
    if seen is None:
        seen = set()
    queries, n_emitted = [], 0
    for (qs, _o, _r, _t, _e, _f), (pid, _p) in zip(rows, seeds):
        if qs is None:
            continue
        for q in qs:
            n_emitted += 1
            k = " ".join(q.split()).lower()
            if k and k not in seen:
                seen.add(k)
                queries.append({"query": q.strip(), "form": form, "seed_id": pid})
    ok = [r for r in rows if r[1]]
    reached = [r for r in rows if r[5] != "transport"]
    out_tok = sum(r[3] for r in rows)
    res = dict(form=form, model=MODEL, revision=REVISION, n_seeds=len(seeds), n_per_seed=n,
               max_tokens=max_tokens_for(form, n),
               contract_ok=len(ok), contract_rate=round(len(ok) / max(len(reached), 1), 4),
               contract_rate_all_seeds=round(len(ok) / max(len(seeds), 1), 4),
               reached=len(reached),
               retried=sum(1 for r in rows if r[2]), seconds=round(dt, 1),
               out_tok=out_tok, out_tok_per_s=round(out_tok / max(dt, 1e-9), 1),
               n_queries=len(queries), exact_dupes=n_emitted - len(queries),
               transport_failures=sum(1 for r in rows if r[5] == "transport"),
               truncated=sum(1 for r in rows if r[5] == "length"),
               finish_reasons={k: sum(1 for r in rows if r[5] == k)
                               for k in {r[5] for r in rows} if k},
               first_failures=[r[4] for r in rows if not r[1]][:3],
               queries=queries)
    if label:
        print(f"{label:26s} contract {res['contract_rate']:.3f}  {res['n_queries']:5d} queries  "
              f"{res['out_tok_per_s']:7.1f} out tok/s  {res['seconds']:6.1f}s", flush=True)
    return res
