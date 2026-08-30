"""The one screen statistic and the one decision function (m9/LEDGER.md §4.2).

Stratified paired bootstrap over a weighted component macro: align identical qids per component
AGAINST THE PINNED MANIFEST, take per-query nDCG@10 differences, resample n_d queries with
replacement within each component, combine component means at the surface's weights, and read an
empirical quantile. Resample indices are drawn once per (surface, seed, B) and shared across every
contrast on that surface.

Deliberately not `m7src/boot.py`: boot exposes one-sided lower bounds only at 2.5% and 0.8333%
(M=1 and M=3), and the teacher rule needs 1.25% (Bonferroni over two challengers). Same algorithm,
the quantile and the weights are parameters.

Two things Codex forced. `align()` checks the pinned ordered-qid hash, so two systems cannot
quietly agree to drop the same hard queries (BLOCKER-5). And the decision threshold is
`max(0.0051, 2*F)` with F measured by the seed-replica arm, not M8's table-specific 0.004
(MAJOR-2 / MAJOR-4).
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
    a qid set that does not match the M9.0 manifest."""
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


def indices(sizes, B, seed):
    rng = np.random.default_rng(seed)
    return {c: rng.integers(0, n, size=(B, n)) for c, n in sizes}


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
    B = st["B"]
    if idx is None:
        idx = indices([(c, d.size) for c, d in aligned], B, st["seed"])
    else:
        B = int(next(iter(idx.values())).shape[0])
        for c, d in aligned:
            assert idx[c].shape == (B, d.size), f"{c}: idx {idx[c].shape} vs n={d.size}"

    point = float(sum(w[c] * d.mean() for c, d in aligned))
    reps = np.zeros(B, dtype=np.float64)
    for c, d in aligned:
        reps += w[c] * d[idx[c]].mean(axis=1)
    return {"surface": surface_name, "weights": w, "point": point,
            "per_component": {c: float(d.mean()) for c, d in aligned},
            "n_per_component": {c: int(d.size) for c, d in aligned},
            "quantile": quantile,
            "lower_bound": float(np.quantile(reps, quantile, method="linear")),
            "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "B": B, "seed": st["seed"]}


def seed_floor(arm1, arm1b):
    """F = |macro(m9s1) - macro(m9s1b)| on DEV6. The measured training-noise term of the MDE."""
    m1, _ = macro(arm1, "DEV6")
    m2, _ = macro(arm1b, "DEV6")
    return abs(m1 - m2)


def mde(F=None):
    r = reg()["rules"]["mde"]
    floor = float(r["formula"].split(",")[0].split("(")[1])
    return floor if F is None else max(floor, 2.0 * F)


def rank_stable(hist_a, hist_b, surface_name):
    """Sign agreement of the contrast at the last two checkpoints (m9/LEDGER.md §4.2)."""
    def m(h):
        return macro(h["per_component"], surface_name)[0]
    if len(hist_a) < 2 or len(hist_b) < 2:
        return {"checked": False, "stable": False, "reason": "fewer than two checkpoints"}
    d3 = m(hist_a[-2]) - m(hist_b[-2])
    d4 = m(hist_a[-1]) - m(hist_b[-1])
    return {"checked": True, "stable": bool(np.sign(d3) == np.sign(d4) and d4 != 0),
            "delta_ckpt3": d3, "delta_ckpt4": d4}


def decide(kind, a, b, *, F=None, hist_a=None, hist_b=None, idx=None):
    """kind in {'teacher_swap','batch_size','student','prompt','mix'} -> the registered verdict."""
    rule = reg()["rules"][kind]
    sname = rule["surface"]
    thr = rule["margin"] if isinstance(rule["margin"], (int, float)) else mde(F)
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
        res["rank_stability"] = rank_stable(hist_a, hist_b, sname)
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

    big_a, big_b = make(0.02)
    h = [{"per_component": big_a}, {"per_component": big_a}]
    hb = [{"per_component": big_b}, {"per_component": big_b}]
    d = decide("student", big_a, big_b, F=0.001, hist_a=h, hist_b=hb)
    assert d["pass"] and d["threshold"] == 0.0051, d["threshold"]
    n_a, n_b = make(0.0)
    d0 = decide("student", n_a, n_b, F=0.001,
                hist_a=[{"per_component": n_a}] * 2, hist_b=[{"per_component": n_b}] * 2)
    assert not d0["pass"]
    # sub-margin but resolved: fails on the MARGIN, not the bound
    s_a, s_b = make(0.002, noise=0.004)
    d1 = decide("student", s_a, s_b, F=0.001,
                hist_a=[{"per_component": s_a}] * 2, hist_b=[{"per_component": s_b}] * 2)
    assert d1["bound_positive"] and not d1["margin_met"] and not d1["pass"], d1["point"]
    # a measured seed floor RAISES the threshold
    assert mde(0.01) == 0.02 and mde(0.0001) == 0.0051
    # rank instability alone kills an adoption
    d2 = decide("student", big_a, big_b, F=0.001,
                hist_a=[{"per_component": big_b}, {"per_component": big_a}],
                hist_b=[{"per_component": big_a}, {"per_component": big_b}])
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
