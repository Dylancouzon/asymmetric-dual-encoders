"""The one screen statistic and the one decision function (m9/LEDGER.md §4.2).

Stratified paired bootstrap over a weighted component macro: align identical qids per component
AGAINST THE PINNED MANIFEST, take per-query nDCG@10 differences, resample n_d queries with
replacement within each component, combine component means at the surface's weights, and read an
empirical quantile. Resample indices are drawn once per (surface, seed, B) and shared across every
contrast on that surface.

Deliberately not `m7src/boot.py`: boot exposes one-sided lower bounds only at 2.5% and 0.8333%
(M=1 and M=3), and the teacher rule needs 1.25% (Bonferroni over two challengers). Same algorithm,
the quantile and the weights are parameters.

Two things Codex forced. `align()` checks the pinned qid-SET hash for every component, so two
systems cannot quietly agree to drop the same hard queries (pass 1, BLOCKER-5) -- and the check is
on the sorted set, which is what the statistic actually consumes, not on load order. And the
decision threshold is ONE registered number measured from 2,031 historical dev contrasts; the
earlier `max(0.0051, 2F)` form was withdrawn because a two-seed range is not a sigma estimate
(pass 2, MAJOR-8).
"""
import json

import numpy as np

import m9base
from m9base import M9, RESULTS


def reg():
    return json.loads((M9 / "registry.json").read_text())


def constants():
    return json.loads((RESULTS / "m9_lock_constants.json").read_text())


def surface(name):
    s = reg()["surfaces"][name]
    comps = list(s["components"])
    w = s["weights"]
    if isinstance(w, str):
        w = {c: 1.0 / len(comps) for c in comps}
    tot = sum(w[c] for c in comps)
    return comps, {c: w[c] / tot for c in comps}


def align(a, b, comps, check_manifest=True):
    """-> [(component, delta array in sorted-qid order)]. Refuses a missing component and refuses
    a qid SET that does not match the M9.0 manifest. Order is imposed here (sorted), so the
    manifest field this reads is `qids_sha256`, not `qids_ordered_sha256`; the ordered hash pins
    the loader's order and is checked in `lock_constants`, not here."""
    man = constants()["qid_manifest"] if check_manifest else {}
    out = []
    for c in comps:
        if c not in a or c not in b:
            raise AssertionError(f"component {c!r} missing from one side -- refuse to shrink")
        qids = sorted(set(a[c]) & set(b[c]))
        if len(qids) != len(a[c]) or len(qids) != len(b[c]):
            raise AssertionError(
                f"{c}: qid mismatch, |A|={len(a[c])} |B|={len(b[c])} |A&B|={len(qids)}")
        if check_manifest:
            m = man[c]
            if len(qids) != m["n_queries"]:
                raise AssertionError(f"{c}: {len(qids)} queries, manifest pins {m['n_queries']}")
            h = _sha_list(qids)
            if h != m["qids_sha256"]:
                raise AssertionError(f"{c}: qid set hash {h[:12]} != manifest {m['qids_sha256'][:12]}")
        out.append((c, np.array([a[c][q] - b[c][q] for q in qids], dtype=np.float64)))
    return out


def _sha_list(items):
    import hashlib
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()


CHUNK = 1000   # bootstrap replicates per block


def indices(sizes, B, seed):
    """A resample PLAN, not the draws. Materialising every block at once is 3.2 GB for DEV-6 on a
    25 GB box that is also holding document memmaps, and holding them in a list defeats the point
    of chunking (Codex pass 2 MAJOR-10, pass 3). The blocks are regenerated on demand from the
    same seed, so two contrasts handed the same plan see draw-for-draw identical resamples."""
    return {"sizes": list(sizes), "B": B, "seed": seed}


def _blocks(idx):
    rng = np.random.default_rng(idx["seed"])
    B = idx["B"]
    for b in range(0, B, CHUNK):
        k = min(CHUNK, B - b)
        yield {c: rng.integers(0, n, size=(k, n), dtype=np.int32) for c, n in idx["sizes"]}


def macro(per_component, surface_name):
    comps, w = surface(surface_name)
    means = {c: float(np.mean(list(per_component[c].values()))) for c in comps}
    return float(sum(w[c] * means[c] for c in comps)), means


def contrast(a, b, surface_name, *, quantile, idx=None, weights=None, check_manifest=True):
    st = reg()["statistic"]
    comps, w = surface(surface_name)
    if weights is not None:
        tot = sum(weights[c] for c in comps)
        w = {c: weights[c] / tot for c in comps}
    aligned = align(a, b, comps, check_manifest)
    sizes = [(c, d.size) for c, d in aligned]
    if idx is None:
        idx = indices(sizes, st["B"], st["seed"])
    assert idx["sizes"] == sizes, f"resample draws were made for {idx['sizes']}, not {sizes}"
    B = idx["B"]

    point = float(sum(w[c] * d.mean() for c, d in aligned))
    parts = []
    for blk in _blocks(idx):
        acc = np.zeros(next(iter(blk.values())).shape[0], dtype=np.float64)
        for c, d in aligned:
            acc += w[c] * d[blk[c]].mean(axis=1)
        parts.append(acc)
        del blk
    reps = np.concatenate(parts)
    return {"surface": surface_name, "weights": w, "point": point,
            "per_component": {c: float(d.mean()) for c, d in aligned},
            "n_per_component": {c: int(d.size) for c, d in aligned},
            "quantile": quantile, "quantile_method": st["quantile_method"],
            "lower_bound": float(np.quantile(reps, quantile, method=st["quantile_method"])),
            "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "B": B, "seed": st["seed"]}


def seed_sensitivity(arm1, arm1b):
    """|macro(anchor) - macro(seed replica)| on DEV6, aligned through the pinned manifest.

    REPORTED, never read by a rule. Codex pass 2, MAJOR-8: a single absolute difference between
    two seeds is one half-normal draw, not an estimated sigma; it can sit near zero under large
    real seed variance or inflate a threshold arbitrarily. Calling it a noise floor would repeat
    m8/CODEMAP.md pitfall 18 with K=2 instead of K=3.
    """
    comps, _ = surface("DEV6")
    aligned = align(arm1, arm1b, comps)          # forces both through the manifest check
    m1, _ = macro(arm1, "DEV6")
    m2, _ = macro(arm1b, "DEV6")
    return {"delta": abs(m1 - m2), "macro_anchor": m1, "macro_replica": m2,
            "n_aligned": {c: int(d.size) for c, d in aligned}, "K": 2,
            "status": "REPORTED ONLY -- no rule reads this"}


def mde():
    """The single registered minimum detectable effect. One number, fixed at M9.0."""
    return float(reg()["rules"]["mde"]["value"])


def rank_stable(hist_a, hist_b, surface_name):
    """Sign agreement of the contrast at the two REGISTERED late checkpoints.

    Reads them by step id, not by position: an arm that lost its final checkpoint would otherwise
    have checkpoints 2 and 3 silently substituted for 3 and 4 (Codex pass 2, MAJOR-10).
    """
    want = reg()["dose"]["checkpoints"][-2:]

    def at(hist, step):
        for h in hist:
            if h["step"] == step and surface_name in h.get("macros", {}):
                return macro(h["per_component"], surface_name)[0]
        return None

    got = [(at(hist_a, s_), at(hist_b, s_)) for s_ in want]
    if any(x is None or y is None for x, y in got):
        return {"checked": False, "stable": False, "required_steps": want,
                "reason": "an arm is missing one of the registered late checkpoints on this surface"}
    d3, d4 = (got[0][0] - got[0][1]), (got[1][0] - got[1][1])
    return {"checked": True, "required_steps": want,
            "stable": bool(np.sign(d3) == np.sign(d4) and d4 != 0),
            "delta_ckpt3": d3, "delta_ckpt4": d4}


def decide(kind, a, b, *, hist_a=None, hist_b=None, idx=None):
    """kind in {'teacher_swap','student','prompt','mix'} -> the registered verdict."""
    rule = reg()["rules"][kind]
    sname = rule["surface"]
    thr = rule["margin"] if isinstance(rule["margin"], (int, float)) else mde()
    res = contrast(a, b, sname, quantile=rule["quantile"], idx=idx)
    res["rule"] = {"kind": kind, **rule}
    res["threshold"] = thr
    res["margin_met"] = res["point"] >= thr
    res["bound_positive"] = res["lower_bound"] > 0
    ok = res["margin_met"] and res["bound_positive"]

    if kind == "teacher_swap":
        # direction stability across the three registered weightings (Codex MAJOR-1)
        comps, _ = surface(sname)
        alt = {"equal": {c: 1.0 for c in comps},
               "query_pooled": {c: constants()["qid_manifest"][c]["n_queries"] for c in comps}}
        res["sensitivity"] = {k: contrast(a, b, sname, quantile=rule["quantile"], idx=idx,
                                          weights=v)["point"] for k, v in alt.items()}
        signs = {np.sign(res["point"])} | {np.sign(v) for v in res["sensitivity"].values()}
        res["direction_stable"] = len(signs) == 1
        ok = ok and res["direction_stable"]
    elif rule.get("rank_stability"):
        # SCREEN-3: the only surface with four checkpoints, since DEV-6 is now read once at the
        # end (m9/registry.json dose.checkpoint_surfaces).
        res["rank_stability"] = rank_stable(hist_a, hist_b,
                                            reg()["rules"]["rank_stability_surface"])
        ok = ok and res["rank_stability"]["stable"]

    res["pass"] = bool(ok)
    res["action"] = "adopt challenger" if ok else f"default: {rule['default']}"
    res["scope"] = reg()["rules"]["scope"]
    return res


def self_test():
    """Every conjunct must be independently reachable, or the rule is really only one rule."""
    man = constants()["qid_manifest"]
    comps6, _ = surface("DEV6")
    rng = np.random.default_rng(7)

    def make(shift, noise=0.085, comps=None):
        comps = comps or comps6
        a, b = {}, {}
        for c in comps:
            n = man[c]["n_queries"]
            base = rng.random(n)
            # qids must match the pinned manifest, so borrow the real ones
            qs = _qids(c)
            a[c] = {q: float(base[i] + shift + noise * rng.standard_normal())
                    for i, q in enumerate(qs)}
            b[c] = {q: float(base[i]) for i, q in enumerate(qs)}
        return a, b

    ck = reg()["dose"]["checkpoints"][-2:]

    def H(per):
        return [{"step": s_, "per_component": per, "macros": {"DEV6": {}, "SCREEN3": {}}}
                for s_ in ck]

    big_a, big_b = make(0.02)
    h, hb = H(big_a), H(big_b)
    d = decide("student", big_a, big_b, hist_a=h, hist_b=hb)
    assert d["pass"] and d["threshold"] == mde(), d["threshold"]
    n_a, n_b = make(0.0)
    d0 = decide("student", n_a, n_b,
                hist_a=H(n_a), hist_b=H(n_b))
    assert not d0["pass"]
    # sub-margin but resolved: fails on the MARGIN, not the bound
    s_a, s_b = make(0.002, noise=0.004)
    d1 = decide("student", s_a, s_b,
                hist_a=H(s_a), hist_b=H(s_b))
    assert d1["bound_positive"] and not d1["margin_met"] and not d1["pass"], d1["point"]
    assert mde() == reg()["rules"]["mde"]["value"]
    # rank instability alone kills an adoption
    d2 = decide("student", big_a, big_b,
                hist_a=[{"step": ck[0], "per_component": big_b, "macros": {"DEV6": {}}},
                        {"step": ck[1], "per_component": big_a, "macros": {"DEV6": {}}}],
                hist_b=[{"step": ck[0], "per_component": big_a, "macros": {"DEV6": {}}},
                        {"step": ck[1], "per_component": big_b, "macros": {"DEV6": {}}}])
    assert not d2["pass"] and not d2["rank_stability"]["stable"]
    # teacher rule runs on the family-weighted 3-component surface with its sensitivity row
    t_a, t_b = make(0.02, comps=surface("SCREEN3")[0])
    dt = decide("teacher_swap", t_a, t_b)
    assert dt["pass"] and set(dt["sensitivity"]) == {"equal", "query_pooled"}
    assert abs(sum(dt["weights"].values()) - 1.0) < 1e-12
    assert dt["weights"]["nq-250k"] == 0.5
    print(json.dumps({"big": round(d["point"], 5), "null_pass": d0["pass"],
                      "small_point": round(d1["point"], 5), "small_pass": d1["pass"],
                      "rank_unstable_pass": d2["pass"], "teacher_pass": dt["pass"],
                      "teacher_point": round(dt["point"], 5)}, indent=1))
    print("screen_stats self_test PASS")


_QC = {}


def _qids(comp):
    if comp not in _QC:
        import dev_eval
        _doc_ids, _dt, q_ids, _qt, _qr, _dv = dev_eval.doc_vecs(comp)
        _QC[comp] = sorted(str(x) for x in q_ids)
    return _QC[comp]


if __name__ == "__main__":
    self_test()
