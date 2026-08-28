"""Full-suite comparison of any set of release artifacts against one baseline, in ONE corpus pass.

The generic form of what `dev_audit.py` did for the lever chain. Everything decision-bearing on
dev goes through here: released `QueryTable` path, each artifact's own frozen preprocessing rule
(including its pooling mode), fp16 and int8, ordinary and dependence-preserving statistics side by
side, unrounded per-query values dumped and hashed, pin and code identity recorded.

It exists because the alternative -- running `compare_release.py` once per candidate -- re-reads
the 6.17M-row pool and the 5.23M-row HotpotQA corpus per candidate, and because two comparisons
made in separate processes cannot be pooled into one dependence-aware statement.

Usage: compare_full.py <tag> <baseline_run_id> <run_id>[:<pool_mode>] ...
       compare_full.py attrib p35w-2m-s2500 p4x-nopseudo-a p4x-pseudo500k-a
"""
import gzip
import hashlib
import json
import sys
import time
from dataclasses import asdict

import torch

import boot
import dev_audit
import dev_eval
import encoders
import multieval
from _paths import REPO, WORK
from table import Preproc, ensure_release, get_tokenizer, load_table, read_meta


def load(run_id, pool_mode=None, device="cuda"):
    rel = ensure_release(WORK / "runs" / f"{run_id}.npz", device=device)
    meta = read_meta(rel)
    assert meta.get("weights_folded"), f"{run_id}: not a release-shape artifact"
    pre = Preproc(**meta["preproc"])
    if pool_mode:                      # an explicit override is a labelled experiment, not a default
        pre = Preproc(**{**meta["preproc"], "pool_mode": pool_mode})
    return rel, pre, {q: load_table(rel, variant=q, device=device) for q in ("fp16", "int8")}


def main(tag, baseline, candidates, smoke=False):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    spec = encoders.active()
    comps = dev_audit.SMOKE_COMPS if smoke else dev_eval.dev_components()
    pin = dev_audit.verify_pin(dev_eval.dev_components(), pool_bytes=not smoke)

    names = [baseline] + candidates
    loaded, makers = {}, {}
    for spec_str in names:
        rid, _, mode = spec_str.partition(":")
        rel, pre, models = load(rid, mode or None, device=device)
        key = spec_str
        loaded[key] = {"run_id": rid, "release": rel.name, "sha256": dev_audit.sha_file(rel),
                       "preproc": asdict(pre)}
        for q, m in models.items():
            makers[f"{key}|{q}"] = (lambda mm, pp: (lambda c, texts: mm.encode(texts, pp, tok=tok))
                                    )(m, pre)
        print(f"  {key}: {pre}", flush=True)

    per = multieval.eval_makers(makers, components=comps,
                                max_docs=200_000 if smoke else None)

    base_key = names[0]
    out = {"_what": "full pinned dev suite, released QueryTable path, each artifact under its own "
                    "frozen preprocessing rule",
           "_status": "exploratory dev SELECTION evidence; only the three frozen-test comparisons "
                      "are confirmatory (review #3 MAJOR 1)",
           "encoder": asdict(spec), "components": comps, "baseline": base_key,
           "artifacts": loaded,
           "macros_unrounded": {t: multieval.macro(p) for t, p in per.items()},
           "per_component_unrounded": {t: multieval.means(p) for t, p in per.items()},
           "comparisons": {}}
    for key in names[1:]:
        for q in ("fp16", "int8"):
            out["comparisons"][f"{key}_vs_{base_key}|{q}"] = boot.both_ways(
                per[f"{key}|{q}"], per[f"{base_key}|{q}"])

    dump = {"components": comps, "encoder": asdict(spec),
            "per_query": {t: {c: {qq: float(v) for qq, v in d.items()} for c, d in p.items()}
                          for t, p in per.items()}}
    dpath = REPO / "results" / f"m7_devperquery_{tag}.json.gz"
    raw = json.dumps(dump, sort_keys=True).encode()
    with gzip.GzipFile(filename=str(dpath), mode="wb", mtime=0) as f:
        f.write(raw)
    out["per_query_dump"] = {"path": dpath.name,
                             "payload_sha256": hashlib.sha256(raw).hexdigest(),
                             "file_sha256": dev_audit.sha_file(dpath)}
    out["pin_evidence"] = pin
    out["code_identity"] = dev_audit.code_identity()
    out["seconds"] = round(time.time() - t0, 1)
    (REPO / "results" / f"m7_compare_full_{tag}.json").write_text(json.dumps(out, indent=1))

    print(f"\nbaseline {base_key}: {out['macros_unrounded'][base_key + '|fp16']:.4f}")
    for key in names[1:]:
        c = out["comparisons"][f"{key}_vs_{base_key}|fp16"]
        d = c["dependence_preserving"]
        print(f"  {key}: {out['macros_unrounded'][key + '|fp16']:.4f}  "
              f"dep delta {d['paired']['delta']:+.4f} ci {d['paired']['ci95']} "
              f"p={d['signflip']['p']:.4f}   (ordinary ci {c['ordinary']['paired']['ci95']})")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2:], smoke="--smoke" in sys.argv)
