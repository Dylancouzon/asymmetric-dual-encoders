"""The frozen teacher document/query encoder -- whichever one is selected.

Which model, how it pools, and what prompt it wants all come from `encoders.py`; set M7_ENCODER
to switch. Pooling used to be hardcoded to CLS here, so a mean-pooled teacher (stella) could not
be run at all, and the cache key asserted "cls-l2" as a literal regardless of what was run.

Encodes are cached as sharded fp16 .npy under work/enc/<key>/, keyed on (model, revision, dtype,
tokenizer identity, POOLING, prefix, max_length, corpus hash); shards make a long corpus encode
resumable after a crash. The key for bge-base is byte-identical to the pre-registry one so the
existing ~22 GB of encodes stay valid -- `m7src/test_encoders.py` pins that.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import encoders
from _paths import DEVICE, WORK, empty_cache

SPEC = encoders.active()
# Module-level names kept so the dozen existing call sites need no change.
TEACHER = SPEC.repo
TEACHER_REV = SPEC.revision
QUERY_PREFIX = SPEC.query_prefix
DOC_PREFIX = SPEC.doc_prefix
ENC = WORK / "enc"
SHARD = 50_000  # texts per shard file


def sha_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


_CACHE = {}
_DENSE = {}


def load_post_dense(spec, device=DEVICE):
    """Load the post-pooling Dense module a Spec declares, or None.

    stella's published pipeline is Transformer -> Pooling(mean) -> Dense_1024 -> normalize. Encoding
    without that Dense yields a different model from the one its MTEB score describes, so this is
    not optional polish -- it is loader fidelity, the same failure class as the M6 gate's BLOCKER 1.
    """
    if not spec.post_dense:
        return None
    key = (spec.repo, spec.revision, spec.post_dense, device)
    if key not in _DENSE:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg = json.loads(Path(hf_hub_download(spec.repo, f"{spec.post_dense}/config.json",
                                              revision=spec.revision)).read_text())
        if cfg.get("activation_function", "").rsplit(".", 1)[-1] not in ("Identity",):
            raise NotImplementedError(f"{spec.name} Dense uses a non-identity activation "
                                      f"{cfg['activation_function']!r}; add it before encoding")
        w = load_file(hf_hub_download(spec.repo, f"{spec.post_dense}/model.safetensors",
                                      revision=spec.revision))
        W = w["linear.weight"].to(device).float()
        b = w.get("linear.bias")
        _DENSE[key] = (W, None if b is None else b.to(device).float())
    return _DENSE[key]


def load_teacher(model_id=TEACHER, revision=TEACHER_REV, dtype=torch.float32, device=DEVICE):
    key = (model_id, revision, str(dtype), device)
    if key not in _CACHE:
        sp = encoders.by_repo(model_id)
        kw = {"trust_remote_code": True} if sp.trust_remote_code else {}
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, **kw)
        # config_kwargs belong on the CONFIG, not on from_pretrained: transformers 4.57 forwards
        # unknown from_pretrained kwargs straight to the model's __init__, which raises.
        if sp.config_kwargs:
            cfg = AutoConfig.from_pretrained(model_id, revision=revision, **kw,
                                             **sp.config_kwargs)
            model = AutoModel.from_pretrained(model_id, revision=revision, config=cfg,
                                              dtype=dtype, **kw).to(device).eval()
        else:
            model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype,
                                              **kw).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def release_teacher(model_id=TEACHER, revision=TEACHER_REV, dtype=torch.float32, device=DEVICE):
    """Evict one memoized model and free its VRAM. load_teacher's cache is right for a pipeline
    that uses one encoder and wrong for a loop over five of them (teacher_probe)."""
    if _CACHE.pop((model_id, revision, str(dtype), device), None) is not None:
        empty_cache()


def pool_project_normalize(h, attention_mask, pooling, dense):
    """The ONE implementation of "hidden states -> embedding": pool by the encoder's convention,
    apply its post-pooling Dense if it has one, L2 normalize. Returns a fp32 CUDA tensor.

    It is a named function because it had started to exist in three places -- encode_batch,
    teacher_probe (deleted) and init_table.teacher_rows, which hardcoded CLS and dropped the Dense
    entirely. A table whose ROWS are built with different pooling from the DOCUMENTS they are
    scored against is wrong in a way no shape check catches.
    """
    if pooling == "cls":
        v = h[:, 0]
    elif pooling == "mean":
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    v = v.float()
    if dense is not None:
        W, bias = dense
        v = v @ W.T + (0.0 if bias is None else bias)
    return torch.nn.functional.normalize(v, dim=-1)


@torch.no_grad()
def encode_batch(tok, model, texts, max_length=512, device=DEVICE, pooling="cls", dense=None):
    """Tokenize, forward, pool/project/normalize. Returned fp32 on CPU."""
    b = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    h = model(**b).last_hidden_state
    return pool_project_normalize(h, b["attention_mask"], pooling, dense).cpu().numpy()


def _order_by_length(tok, texts, max_length):
    n_tok = [min(len(tok(t, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]), max_length)
             for t in texts]
    order = np.argsort(np.array(n_tok), kind="stable")
    return order, n_tok


@torch.no_grad()
def encode(texts, prefix="", max_length=512, batch_tokens=16384, model_id=TEACHER,
           revision=TEACHER_REV, dtype=torch.float32, device=DEVICE, verbose=False):
    """Length-bucketed dynamic batching; returns (N, dim) fp32 normalized in input order."""
    tok, model = load_teacher(model_id, revision, dtype, device)
    spec = encoders.by_repo(model_id)
    pooling, dense = spec.pooling, load_post_dense(spec, device)
    order, n_tok = _order_by_length(tok, [prefix + t for t in texts], max_length)
    # The output width is the Dense's out_features when there is one, NOT the backbone's hidden
    # size. They coincide for stella's square 1024->1024 head, but an MRL head (e.g. 1024->256)
    # would silently write 256 values into 1024-wide rows.
    dim = model.config.hidden_size if dense is None else int(dense[0].shape[0])
    if dim != spec.dim:
        raise AssertionError(f"{spec.name}: encode width {dim} != Spec.dim {spec.dim}; the "
                             f"registry and the loaded modules disagree")
    out = np.empty((len(texts), dim), dtype=np.float32)
    i, t0, done = 0, time.time(), 0
    while i < len(order):
        # grow the batch until padded token count would exceed batch_tokens
        longest, j = 0, i
        while j < len(order):
            L = max(longest, n_tok[order[j]])
            if (j - i + 1) * L > batch_tokens and j > i:
                break
            longest, j = L, j + 1
        idx = order[i:j]
        out[idx] = encode_batch(tok, model, [prefix + texts[k] for k in idx], max_length, device,
                                pooling=pooling, dense=dense)
        done += len(idx)
        if verbose and (done % 20000 < len(idx)):
            print(f"    {done}/{len(texts)} @ {done/(time.time()-t0):.0f} texts/s", flush=True)
        i = j
    return out


DT = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def cache_key(name, prefix, max_length, model_id, revision, corpus_sha, dtype, spec=None):
    # The key must change whenever the vectors would, so POOLING and the tokenizer identity come
    # from the registry rather than being asserted as bge's literals. The pre-registry version
    # hardcoded "cls-l2"/"bert-wordpiece-30522", so a mean-pooled encode would have been stored
    # under a key claiming CLS. For bge-base the resulting blob is unchanged.
    # `spec` is passed explicitly only by test_encoders, which replays keys for encodes made by
    # OTHER encoders and resolves each one's Spec from its own meta.json. Everything else resolves
    # from the repo, where the active encoder disambiguates a shared repo.
    spec = spec or encoders.by_repo(model_id)
    blob = json.dumps({"name": name, "prefix": prefix, "max_length": max_length, "model": model_id,
                       "revision": revision, "pooling": spec.pooling_key,
                       "corpus_sha256": corpus_sha,
                       "encode_dtype": DT[dtype], "store_dtype": "fp16",
                       "tokenizer": spec.tokenizer_id,
                       # Added CONDITIONALLY so bge-base's blob stays byte-identical and the ~22 GB
                       # of existing encodes remain valid. post_dense also covers a future MRL
                       # truncation, since that is a different Dense directory and hence a
                       # different output dimension -- which is what would otherwise collide.
                       **({"post_dense": spec.post_dense} if spec.post_dense else {}),
                       # Also conditional, same reason. These change the attention kernel, so two
                       # encodes that differ only here are not interchangeable -- e.g. dropping the
                       # overrides after installing xformers must not silently reuse these vectors.
                       **({"config_kwargs": dict(sorted(spec.config_kwargs.items()))}
                          if spec.config_kwargs else {})},
                      sort_keys=True)
    return f"{name}-{DT[dtype]}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}", blob


# Per-call encode provenance, keyed by the caller's `name`. The cache lives under a gitignored
# work/ tree whose bytes nothing verified: `encode_cached` skipped any shard that merely EXISTED
# and `_combined` trusted `combined.f16` on its byte SIZE alone, so the document vectors behind a
# confirmatory number came from mutable, unauthenticated files (Codex one-shot-path review
# 2026-08-28, MAJOR 4). The cache KEY binds the inputs; this binds the OUTPUT bytes, and the final
# run records what it consumed. Read it after an `encode_cached` call.
PROVENANCE = {}


def sha_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _read_shard_manifest(d):
    p = d / "shards.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"shards": {}}


def _write_shard_manifest(d, man):
    p = d / "shards.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    tmp.rename(p)


def encode_cached(name, texts, prefix="", max_length=512, model_id=TEACHER, revision=TEACHER_REV,
                  batch_tokens=32768, device=DEVICE, verbose=True, dtype=torch.float16,
                  verify=False):
    """Shard-resumable cached encode. Returns (N, dim) fp16 memmap-backed array.

    Every shard this function writes is hashed into `<cache>/shards.json`, and the summary lands
    in `PROVENANCE[name]`. `verify=True` re-hashes every pre-existing shard and the stitched
    `combined.f16` before returning, and ABORTS on a mismatch rather than re-encoding: on the
    one-shot path a cache whose bytes changed under us is a fact to surface, not to paper over.
    A shard that predates the manifest is recorded trust-on-first-use and reported as such --
    never counted as verified.
    """
    key, blob = cache_key(name, prefix, max_length, model_id, revision, sha_texts(texts), dtype)
    d = ENC / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(blob)
    n_shards = (len(texts) + SHARD - 1) // SHARD
    man = _read_shard_manifest(d)
    tofu, verified, written = [], [], []
    for s in range(n_shards):
        p = d / f"shard_{s:05d}.npy"
        sid = f"{s:05d}"
        if p.exists():
            rec = man["shards"].get(sid)
            if rec is not None and rec.get("shard_size") not in (None, SHARD):
                # SHARD is not in the cache key, so changing it re-slices the same corpus into
                # different files. Treat a layout change as a stale cache rather than stitching a
                # prefix of the old shards (Codex review #4, MAJOR 10).
                raise SystemExit(
                    f"ENCODE CACHE REFUSED: {d} was written with SHARD={rec['shard_size']} but "
                    f"this process uses SHARD={SHARD}. The shard layout is not part of the cache "
                    "key; delete the cache directory and re-encode.")
            if rec is None:
                man["shards"][sid] = {"bytes": p.stat().st_size, "sha256": sha_file(p),
                                      "trusted_on_first_use": True}
                tofu.append(sid)
            elif rec.get("trusted_on_first_use"):
                tofu.append(sid)
            elif verify:
                got = sha_file(p)
                if got != rec["sha256"]:
                    raise SystemExit(
                        f"ENCODE CACHE REFUSED: {p} does not match the hash recorded when it was "
                        f"written ({got[:12]} vs {rec['sha256'][:12]}). The cache is gitignored "
                        "and mutable; delete the shard (and combined.f16) to force a re-encode, "
                        "and do not reuse this cache for a confirmatory number until you know why "
                        "it changed.")
                verified.append(sid)
            continue
        lo, hi = s * SHARD, min((s + 1) * SHARD, len(texts))
        if verbose:
            print(f"  [{name}] shard {s+1}/{n_shards} ({hi-lo} texts)", flush=True)
        v = encode(texts[lo:hi], prefix, max_length, batch_tokens, model_id, revision,
                   dtype=dtype, device=device, verbose=verbose)
        np.save(p.with_suffix(".tmp.npy"), v.astype(np.float16))
        p.with_suffix(".tmp.npy").rename(p)
        man["shards"][sid] = {"bytes": p.stat().st_size, "sha256": sha_file(p),
                              "rows": int(hi - lo), "shard_size": SHARD,
                              "trusted_on_first_use": False}
        written.append(sid)
        # Persist after EVERY shard. Writing the manifest only at the end meant a crash partway
        # through a long encode left hundreds of shards with no records, which the next
        # verify=True call adopts as trust-on-first-use and then refuses -- turning a resumable
        # job into a full re-encode (Codex review #4, MAJOR 10).
        _write_shard_manifest(d, man)
    if verify and tofu:
        raise SystemExit(
            f"ENCODE CACHE REFUSED: {len(tofu)} of {n_shards} shards under {d} predate hash "
            "recording, so their bytes cannot be authenticated -- and this call asked for "
            "verification. Delete the cache directory and re-encode; a confirmatory number may "
            "not rest on bytes nothing ever checked.")
    out = _combined(d, n_shards, len(texts), man, verify=verify)
    _write_shard_manifest(d, man)
    PROVENANCE[name] = {
        "cache_key": key, "n_shards": n_shards, "n_rows": len(texts),
        "shards_written_now": len(written), "shards_verified": len(verified),
        "shards_trusted_on_first_use": len(tofu), "verify_requested": bool(verify),
        "shard_sha256": {sid: man["shards"][sid]["sha256"] for sid in sorted(man["shards"])
                         if int(sid) < n_shards},
        "combined_sha256": man.get("combined", {}).get("sha256"),
        "_note": ("every shard behind these vectors was written or re-hashed by this process"
                  if verify and not tofu else
                  "shards marked trust-on-first-use predate hash recording and are NOT verified"
                  if tofu else "shard hashes recorded; not re-verified this call"),
    }
    return out


def _combined(d, n_shards, n_rows, man=None, verify=False):
    """Return the encode as a read-only memmap over one contiguous file.

    np.concatenate over the shards would materialize the whole encode in RAM -- 8 GB for the
    5.23M-doc HotpotQA dev component, on every call. The shards stay as the resumable unit; this
    is a one-time stitch, and the combined file is what every reader mmaps afterwards.

    The stitch used to be accepted on byte SIZE alone, which cannot tell one 10.7 GB file from
    another. It is now rebuilt whenever the shard hashes it was built from have changed, and
    re-hashed under `verify`.
    """
    parts = [np.load(d / f"shard_{s:05d}.npy", mmap_mode="r") for s in range(n_shards)]
    if n_shards == 1:
        return parts[0]
    dim = parts[0].shape[1]
    comb = d / "combined.f16"
    man = man if man is not None else _read_shard_manifest(d)
    src = [man["shards"][f"{s:05d}"]["sha256"] for s in range(n_shards)
           if f"{s:05d}" in man["shards"]]
    rec = man.get("combined") or {}
    if comb.exists() and comb.stat().st_size == n_rows * dim * 2 and not rec:
        # A stitch that predates hash recording. Adopt it trust-on-first-use rather than
        # rebuilding every pre-existing cache in the tree (10.7 GB for HotpotQA alone) -- and
        # LABEL it, so `verify` below refuses it instead of treating it as authenticated.
        rec = man["combined"] = {"n_rows": n_rows, "dim": int(dim),
                                 "bytes": comb.stat().st_size, "from_shard_sha256": src,
                                 "sha256": sha_file(comb), "trusted_on_first_use": True}
    stale = (not comb.exists()
             or comb.stat().st_size != n_rows * dim * 2
             or rec.get("from_shard_sha256") != src
             or rec.get("n_rows") != n_rows)
    if not stale and verify:
        if rec.get("trusted_on_first_use"):
            raise SystemExit(
                f"ENCODE CACHE REFUSED: {comb} predates hash recording, so it cannot be shown to "
                "be the stitch of these shards -- and this call asked for verification. Delete it "
                "to force a verified rebuild.")
        got = sha_file(comb)
        if got != rec.get("sha256"):
            raise SystemExit(
                f"ENCODE CACHE REFUSED: {comb} does not match the hash recorded when it was "
                f"stitched ({got[:12]} vs {str(rec.get('sha256'))[:12]}). Delete it to force a "
                "rebuild from the shards.")
    if stale:
        tmp = d / "combined.tmp"
        mm = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(n_rows, dim))
        off = 0
        for p in parts:
            if p.shape[1] != dim:
                raise SystemExit(f"ENCODE CACHE REFUSED: shard width {p.shape[1]} != {dim} in {d}")
            mm[off:off + len(p)] = p
            off += len(p)
        mm.flush()
        del mm
        if off != n_rows:
            # A stitch that covered only part of the file used to be written anyway, hashed, and
            # then pass every later verification with an unwritten zero tail
            # (Codex review #4, MAJOR 10).
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"ENCODE CACHE REFUSED: shards under {d} supply {off} rows, not "
                             f"{n_rows}; delete the cache directory and re-encode.")
        tmp.rename(comb)
        man["combined"] = {"n_rows": n_rows, "dim": int(dim), "bytes": comb.stat().st_size,
                           "from_shard_sha256": src, "sha256": sha_file(comb),
                           "trusted_on_first_use": False}
    return np.memmap(comb, dtype=np.float16, mode="r", shape=(n_rows, dim))
