"""A8 quality gate 2 -- distribution overlap against real queries (instructions-m10.md:480-486).

For each of the 12 registered query forms that has real text TODAY (5 of them: title, keyword,
claim, factoid, product), report its mean cosine to its nearest MS MARCO dev queries (k=1, k=5)
and the two-sample energy distance to a MS MARCO subsample, plus two floors: the MS MARCO
self-baseline (two disjoint real-query subsamples against each other) and the form's own
within-form nearest-neighbour cosine. **Action: none -- a disclosed diagnostic.**

MS MARCO is admissible here under the 2026-09-04 rule change (validation only, never a seed,
target, negative or gradient) -- `research/m7-data-licensing.md` "Rule change 2026-09-04". Its
files and vectors live ONLY under `work/m10msmarco/`, never under `work/train/sources/` (that path
would retroactively refuse a released artifact, `m7src/freeze.py:NON_COMMERCIAL_SOURCES`) and
never in `work/m10targets/` (the query-target cache that trains the student).

The five forms that exist today, and where their text and vectors come from:
- title / keyword / claim: `work/m10harvest/harvest_train.jsonl`, vectors read (read-only) from
  the M10 query-target cache (`m10src/targets10.TargetCache`) by content hash.
- factoid: the union of the PAQ build sample (`work/m10paq/paq_build.jsonl`, vectors from the same
  target cache) and the M9 pool's factoid-labelled rows (text AND vectors both from
  `corpus_loader.load_segments(["m9-pool"])`, so the two never drift apart) -- the instructions
  name both as factoid sources, so both are drawn from and pooled before the 5,000 sample
  (ambiguity noted, reading taken: one "factoid" row, sampled from the union).
- product: the M9 pool's product-labelled rows (`esci-us`), same segment machinery.

The seven generated forms (howto, argument, finance, health, comparison, yesno, conversational)
do not exist yet and are listed as `forms_pending`; `--forms` re-runs this script against them
later without recomputing the MS MARCO side (the MS MARCO vector cache is content-addressed and
skipped once built).

CLI:
    .venv/bin/python m10src/a8_gate2.py                    # the five forms that exist today
    .venv/bin/python m10src/a8_gate2.py --forms howto ...  # add generated forms once built
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np

MSMARCO_DIR = REPO / "work" / "m10msmarco"
MSMARCO_TSV = MSMARCO_DIR / "msmarco_queries.dev.tsv"
MSMARCO_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
VECS_DIR = MSMARCO_DIR / "vecs"
REPORT = REPO / "results" / "m10_a8_gate2.json"

FORMS_TODAY = ("title", "keyword", "claim", "factoid", "product")
FORMS_PENDING = ("howto", "argument", "finance", "health", "comparison", "yesno", "conversational")

N_MSMARCO_SAMPLE = 50_000
N_FORM_SAMPLE = 5_000
K_LIST = (1, 5)
SEED = 0


def assert_not_training_path(p):
    """MS MARCO validation vectors/text must never live where a training path would read them."""
    parts = Path(p).resolve().parts
    if "train" in parts and "sources" in parts:
        raise SystemExit(f"REFUSED: {p} is under work/train/sources/ -- MS MARCO is validation-"
                         f"only (research/m7-data-licensing.md 'Rule change 2026-09-04') and must "
                         f"never reach a path a training source loader reads.")
    return p


def sha_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- MS MARCO download --

def ensure_msmarco_queries():
    """Download queries.dev.tsv from the official Microsoft blob if not already present."""
    assert_not_training_path(MSMARCO_TSV)
    if MSMARCO_TSV.exists():
        return
    import tarfile
    import urllib.request
    MSMARCO_DIR.mkdir(parents=True, exist_ok=True)
    tgz = MSMARCO_DIR / "_queries.tar.gz"
    print(f"downloading {MSMARCO_URL}", flush=True)
    urllib.request.urlretrieve(MSMARCO_URL, tgz)
    with tarfile.open(tgz) as t:
        member = t.getmember("queries.dev.tsv")
        with t.extractfile(member) as fh, open(MSMARCO_TSV, "wb") as out:
            out.write(fh.read())
    tgz.unlink()


def load_msmarco_queries():
    ensure_msmarco_queries()
    qids, texts = [], []
    with MSMARCO_TSV.open(encoding="utf-8") as fh:
        for line in fh:
            qid, q = line.rstrip("\n").split("\t", 1)
            qids.append(qid)
            texts.append(q)
    return qids, texts


# --------------------------------------------------------------------------- MS MARCO encoding --

def msmarco_manifest():
    ensure_msmarco_queries()
    return {"source_url": MSMARCO_URL,
            "file": "queries.dev.tsv (from queries.tar.gz)",
            "sha256": sha_file(MSMARCO_TSV),
            "licence": "MS MARCO: non-commercial research purposes only -- admissible here as "
                       "VALIDATION under the 2026-09-04 rule change; never a seed, target, "
                       "negative or gradient (research/m7-data-licensing.md).",
            "seed": SEED}


def encode_msmarco_sample(n=N_MSMARCO_SAMPLE):
    """-> (sample_texts, unit-norm fp32 (n, dim) vectors), cached under work/m10msmarco/vecs/.

    Content-addressed on (source sha256, seed, n, encoder identity): re-running with the same
    inputs reads the cache back rather than re-encoding, and a form-only re-run (`--forms`) never
    touches this path at all once it exists.
    """
    import m9base                              # noqa: F401  pins M7_ENCODER, installs the guard
    import teacher
    import targets10

    VECS_DIR.mkdir(parents=True, exist_ok=True)
    qids, all_texts = load_msmarco_queries()
    src_sha = sha_file(MSMARCO_TSV)
    ident = targets10.identity()
    key_blob = json.dumps({"source_sha256": src_sha, "seed": SEED, "n": n, "identity": ident},
                          sort_keys=True)
    tag = hashlib.sha256(key_blob.encode()).hexdigest()[:12]
    meta_p = VECS_DIR / f"sample-{tag}.meta.json"
    keys_p = assert_not_training_path(VECS_DIR / f"sample-{tag}.keys.u8")
    vecs_p = assert_not_training_path(VECS_DIR / f"sample-{tag}.vecs.f16")

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(all_texts), size=n, replace=False)
    sample_texts = [all_texts[i] for i in sample_idx]

    if meta_p.exists():
        m = json.loads(meta_p.read_text())
        if m["tag"] == tag and m["n"] == n:
            keys = np.fromfile(keys_p, dtype=np.uint8).reshape(n, 16)
            vecs = np.fromfile(vecs_p, dtype=np.float16).reshape(n, ident["dim"])
            want_keys = targets10.keys_of(sample_texts)
            if np.array_equal(keys, want_keys):
                return sample_texts, normalize(np.asarray(vecs, dtype=np.float32)), m

    t0 = time.time()
    v = teacher.encode(sample_texts, prefix=ident["prefix"], max_length=ident["max_length"],
                       batch_tokens=32768, verbose=True)
    el = time.time() - t0
    a = np.asarray(v, dtype=np.float32)
    nrm = np.linalg.norm(a, axis=1)
    if not (0.99 < nrm.min() and nrm.max() < 1.01):
        raise SystemExit(f"MS MARCO target norms {nrm.min():.4f}..{nrm.max():.4f}")
    keys = targets10.keys_of(sample_texts)
    keys.tofile(keys_p)
    a.astype(np.float16).tofile(vecs_p)
    rep = {"tag": tag, "n": n, "source_sha256": src_sha, "seed": SEED, "identity": ident,
          "encode_seconds": round(el, 1), "texts_per_s": round(n / max(el, 1e-9), 1)}
    meta_p.write_text(json.dumps(rep, indent=1))
    print(f"encoded {n:,} MS MARCO queries in {el:.0f}s = {n/max(el,1e-9):,.0f} texts/s", flush=True)
    return sample_texts, normalize(a), rep


def normalize(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.clip(n, 1e-12, None)


# --------------------------------------------------------------------------- form samples -------

def _sample_from_cache(texts, forms, want_form, cache, n=N_FORM_SAMPLE, seed=SEED):
    """-> (sample texts, unit-norm vectors) for `want_form`, read read-only from `cache`
    (targets10.TargetCache) by content hash. Never appends."""
    idx = [i for i, f in enumerate(forms) if f == want_form]
    if not idx:
        return None, None
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
    sample = [texts[i] for i in sel]
    rows = cache.rows_for(sample)
    miss = int((rows < 0).sum())
    if miss:
        raise SystemExit(f"{want_form}: {miss} of {len(sample)} texts have no target in "
                         f"{cache.dir}; run targets10.py first")
    return sample, normalize(np.asarray(cache.vecs()[rows], dtype=np.float32))


def load_form_samples(want_forms, n=N_FORM_SAMPLE, seed=SEED):
    """-> {form: (texts, vecs)}, skipping (with a printed reason) any form this build cannot
    load. `want_forms` is the set actually requested this run."""
    import targets10
    import corpus_loader as CL

    out, skipped = {}, {}
    cache = targets10.TargetCache()
    m9_segs_cache = {}

    def m9_pool_segments():
        if "segs" not in m9_segs_cache:
            segs, _man = CL.load_segments(["m9-pool"], verbose=False)
            m9_segs_cache["segs"] = segs
        return m9_segs_cache["segs"]

    harvest_path = REPO / "work" / "m10harvest" / "harvest_train.jsonl"
    if harvest_path.exists():
        h_texts, h_forms = CL._rows_from_jsonl(harvest_path)
        for form in ("title", "keyword", "claim"):
            if form in want_forms:
                t, v = _sample_from_cache(h_texts, h_forms, form, cache, n, seed)
                if t is None:
                    skipped[form] = f"no rows of form {form!r} in {harvest_path}"
                else:
                    out[form] = (t, v)
    else:
        for form in ("title", "keyword", "claim"):
            if form in want_forms:
                skipped[form] = f"{harvest_path} does not exist"

    if "factoid" in want_forms:
        factoid_texts, factoid_vecs = [], []
        paq_path = REPO / "work" / "m10paq" / "paq_build.jsonl"
        if paq_path.exists():
            p_texts, p_forms = CL._rows_from_jsonl(paq_path, default_form="factoid")
            rows = cache.rows_for(p_texts)
            miss = int((rows < 0).sum())
            if miss:
                raise SystemExit(f"factoid (PAQ): {miss} of {len(p_texts)} texts have no target")
            factoid_texts += p_texts
            factoid_vecs.append(normalize(np.asarray(cache.vecs()[rows], dtype=np.float32)))
        else:
            skipped.setdefault("factoid", "")
            skipped["factoid"] += f"PAQ build sample missing at {paq_path}; "
        try:
            # texts AND vectors both come from the M9-pool segments, so the two stay aligned
            # (`_m9_texts()` is the unscreened/undeduped count and does not match `load_segments`,
            # which re-screens and dedups -- pulling one from each mismatched by ~59K rows).
            for seg in m9_pool_segments():
                fnames = [CL.FORMS[i] for i in seg.forms]
                sel = [i for i, f in enumerate(fnames) if f == "factoid"]
                if sel:
                    factoid_texts += [seg.texts[i] for i in sel]
                    vecs_seg = normalize(np.asarray(seg.array[seg.rowmap[sel]], dtype=np.float32))
                    factoid_vecs.append(vecs_seg)
        except Exception as e:
            skipped.setdefault("factoid", "")
            skipped["factoid"] += f"M9 pool factoid load failed: {e}; "
        if factoid_texts and factoid_vecs:
            all_vecs = np.concatenate(factoid_vecs, axis=0)
            assert len(factoid_texts) == all_vecs.shape[0], (len(factoid_texts), all_vecs.shape)
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(factoid_texts), size=min(n, len(factoid_texts)), replace=False)
            out["factoid"] = ([factoid_texts[i] for i in sel], all_vecs[sel])
            skipped.pop("factoid", None)
        elif "factoid" not in skipped:
            skipped["factoid"] = "no factoid text found in PAQ build sample or the M9 pool"

    if "product" in want_forms:
        try:
            found = False
            for seg in m9_pool_segments():
                fnames = [CL.FORMS[i] for i in seg.forms]
                sel = [i for i, f in enumerate(fnames) if f == "product"]
                if sel:
                    texts_p = [seg.texts[i] for i in sel]
                    vecs_p = normalize(np.asarray(seg.array[seg.rowmap[sel]], dtype=np.float32))
                    rng = np.random.default_rng(seed)
                    pick = rng.choice(len(texts_p), size=min(n, len(texts_p)), replace=False)
                    out["product"] = ([texts_p[i] for i in pick], vecs_p[pick])
                    found = True
            if not found:
                skipped["product"] = "no product-form rows in the M9 pool"
        except Exception as e:
            skipped["product"] = f"M9 pool product load failed: {e}"

    for f in want_forms:
        if f not in out and f not in skipped:
            skipped[f] = "not yet built (a generated form; see forms_pending)"
    return out, skipped


# --------------------------------------------------------------------------- metrics ------------

def nn_cosine(query_vecs, pool_vecs, k_list=K_LIST, exclude_self=False):
    """Mean cosine of every row of `query_vecs` to its k nearest rows of `pool_vecs` (Euclidean on
    unit vectors == cosine order). `exclude_self=True` drops the i==j match when the two arrays
    are literally the same array (the within-form density reference)."""
    sim = query_vecs @ pool_vecs.T  # cosine, since both are unit-norm
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    kmax = max(k_list)
    kmax = min(kmax, sim.shape[1])
    part = np.partition(sim, -kmax, axis=1)[:, -kmax:]
    part_sorted = np.sort(part, axis=1)[:, ::-1]
    out = {}
    for k in k_list:
        k = min(k, part_sorted.shape[1])
        out[f"mean_cos_k{k}"] = float(part_sorted[:, :k].mean())
    return out


def _mean_pairwise_dist(x, y, unit_norm=True):
    """Mean Euclidean distance over all pairs of rows of x and y. On unit-norm vectors
    |a-b| = sqrt(2 - 2 cos(a,b)), which turns an (n, m, dim) materialization into one (n, m)
    matmul -- the naive broadcast form is O(n*m*dim) memory (100+ GB at n=m=5000, dim=1024) and
    was never actually run that way."""
    if unit_norm:
        cos = x @ y.T
        d2 = np.clip(2.0 - 2.0 * cos, 0.0, None)
        return float(np.sqrt(d2).mean())
    return float(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean())


def energy_distance(x, y):
    """Two-sample energy distance on Euclidean distance (unit vectors): 2E|X-Y| - E|X-X'| -
    E|Y-Y'|. Nonnegative; 0 iff the samples are drawn from the same distribution (in population;
    on finite identical samples it is exactly 0)."""
    dxy = _mean_pairwise_dist(x, y)
    dxx = _mean_pairwise_dist(x, x)
    dyy = _mean_pairwise_dist(y, y)
    return float(2 * dxy - dxx - dyy)


# --------------------------------------------------------------------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", nargs="+", default=list(FORMS_TODAY),
                    help="forms to score this run (default: the five that exist today)")
    ap.add_argument("--n-msmarco", type=int, default=N_MSMARCO_SAMPLE)
    ap.add_argument("--n-form", type=int, default=N_FORM_SAMPLE)
    ap.add_argument("--report", default=str(REPORT))
    a = ap.parse_args()

    man = msmarco_manifest()
    ms_texts, ms_vecs, enc_rep = encode_msmarco_sample(n=a.n_msmarco)
    man["encode"] = enc_rep

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ms_texts))
    ms_main = ms_vecs[perm[:a.n_form]]
    ms_base_a = ms_vecs[perm[a.n_form:2 * a.n_form]]
    ms_base_b = ms_vecs[perm[2 * a.n_form:3 * a.n_form]]

    baselines = {
        "msmarco_self": {
            **nn_cosine(ms_base_a, ms_base_b, K_LIST),
            "energy_distance": energy_distance(ms_base_a, ms_base_b),
            "n": int(a.n_form),
            "what": "two disjoint MS MARCO dev subsamples against each other -- the floor",
        }
    }

    samples, skipped = load_form_samples(set(a.forms), n=a.n_form)
    rows = {}
    for form, (texts, vecs) in samples.items():
        row = nn_cosine(vecs, ms_vecs, K_LIST)
        row["energy_distance_vs_msmarco"] = energy_distance(vecs, ms_main[:len(vecs)])
        own = nn_cosine(vecs, vecs, K_LIST, exclude_self=True)
        row["own_form_nn"] = own
        row["n"] = len(texts)
        rows[form] = row

    out = {
        "_what": "A8 quality gate 2 -- distribution overlap of each query form against real "
                "MS MARCO dev queries. Action: none, a disclosed diagnostic "
                "(instructions-m10.md:480-486).",
        "manifest": man,
        "seed": SEED,
        "k": list(K_LIST),
        "n_form_sample": a.n_form,
        "n_msmarco_sample": a.n_msmarco,
        "forms": rows,
        "skipped": skipped,
        "baselines": baselines,
        "forms_pending": list(FORMS_PENDING),
    }
    Path(a.report).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.report}")
    for form, row in rows.items():
        print(f"  {form:12s} n={row['n']:5d}  nn_k1={row['mean_cos_k1']:.4f}  "
              f"nn_k5={row['mean_cos_k5']:.4f}  energy={row['energy_distance_vs_msmarco']:.4f}  "
              f"own_nn_k1={row['own_form_nn']['mean_cos_k1']:.4f}")
    for form, why in skipped.items():
        print(f"  SKIPPED {form}: {why}")


if __name__ == "__main__":
    main()
