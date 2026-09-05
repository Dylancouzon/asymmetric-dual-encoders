"""The M10 screen lock: `m10/screen_registry.json` is the authority, and this validates it.

M9's lesson, copied deliberately (`m9src/final_stats.py`): *"Constants come from the registry.
Prose is not authoritative; the registry is."* A rule that reads a number reads it from here, so
a mandate sentence and a running arm cannot disagree silently.

What this refuses, and why each one has bitten something before:
- a contrast naming an arm that does not exist, or an arm no contrast reads;
- a contrast count that is not thirteen (the Bonferroni denominator is 13; a fourteenth contrast
  silently changes every bound -- a Codex pass corrected exactly that count on 2026-09-04);
- statistics that disagree with `m10src/cov_macro`'s implementation or with the measured
  resolution number's artifact;
- a family with no outcome->action entry (an unresolved contrast with no registered default is a
  decision taken after seeing the number).
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))
REGISTRY = REPO / "m10" / "screen_registry.json"
RESOLUTION = REPO / "results" / "m10_cov_resolution.json"


def cfg():
    return json.loads(REGISTRY.read_text())


def validate(r=None):
    """-> list of problems; empty means the lock is coherent."""
    import cov_macro
    r = r or cfg()
    bad = []
    arms, contrasts, alias = r["arms"], r["contrasts"], r["anchor_aliases"]

    trained = [a for a, v in arms.items() if not str(v.get("note", "")).startswith("not trained")]
    if len(trained) != r["trained_arms_expected"]:
        bad.append(f"{len(trained)} trained arms, registry expects {r['trained_arms_expected']}")
    if len(contrasts) != r["statistics"]["bootstrap"]["n_contrasts"]:
        bad.append(f"{len(contrasts)} contrasts vs a Bonferroni denominator of "
                   f"{r['statistics']['bootstrap']['n_contrasts']}")

    def known(name):
        return name in arms or name in alias
    for cid, c in contrasts.items():
        for side in ("a", "b"):
            if not known(c[side]):
                bad.append(f"contrast {cid}: '{c[side]}' is neither an arm nor an anchor alias")
    # descriptive contrasts count as readers: A1 exists only for A2-A1, which is registered
    # descriptive, and an arm trained for no contrast at all is a wasted 2 GPU-hours
    everything = list(contrasts.values()) + list(r["descriptive_contrasts"].values())
    read = {c[s] for c in everything for s in ("a", "b") if c.get(s)}
    for a, v in arms.items():
        if a in read or a in alias or v.get("family") == "F":
            continue
        bad.append(f"arm {a} is trained but no contrast reads it")

    fams = {v["family"] for v in arms.values() if "family" in v}
    for f in fams:
        if f not in r["outcome_to_action"]:
            bad.append(f"family {f} has no outcome->action entry")
    if set(r["order"]) != fams:
        bad.append(f"order {r['order']} does not cover the families {sorted(fams)}")

    st = r["statistics"]
    if tuple(sorted(st["families"])) != tuple(sorted(cov_macro.FAMILIES)):
        bad.append(f"registry families {st['families']} != cov_macro.FAMILIES "
                   f"{list(cov_macro.FAMILIES)}")
    b = st["bootstrap"]
    if abs(b["quantile"] - b["alpha_family"] / b["n_contrasts"]) > 1e-15:
        bad.append("quantile is not alpha_family / n_contrasts")
    if b["quantile_method"] != "inverted_cdf":
        bad.append("quantile_method is not the registered inverted_cdf")
    if RESOLUTION.exists():
        got = json.loads(RESOLUTION.read_text())
        if abs(got["resolution_distance"] - st["measured_resolution_distance"]) > 5e-7:
            bad.append(f"registry resolution {st['measured_resolution_distance']} != artifact "
                       f"{got['resolution_distance']}")
        if got["MDE_registered"] != st["MDE"]:
            bad.append("MDE disagrees with the resolution artifact")
        if not got["mde_below_resolution"]:
            bad.append("artifact says the MDE is at or above the resolution distance; the "
                       "registry's note says otherwise")
    return bad


if __name__ == "__main__":
    problems = validate()
    for p in problems:
        print("PROBLEM:", p)
    print("LOCK OK" if not problems else f"{len(problems)} PROBLEMS")
    sys.exit(1 if problems else 0)
