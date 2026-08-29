"""The refusals that must happen. LEDGER G1 + G2, tested both ways.

`m7/CODEMAP.md`: "a suite nobody runs is documentation, which is how test_freeze_guard.py stayed
broken for two days after the teacher swap." These run in `./run_m8_tests.sh`.

Two halves, because either alone fails open:
  RUNTIME  -- the process-wide open() guard actually refuses, for every protected kind, through
              every path a caller might use (builtins.open, io.open, os.open, numpy, pathlib).
  STATIC   -- no unclaimed file under m8src/ even MENTIONS a protected path, so a future module
              cannot quietly acquire the capability by being refactored into an allowlisted one.
"""
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import m8base
import paths_guard
import probe_guard

REPO = m8base.REPO
M8SRC = REPO / "m8src"
FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILED.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:                                    # noqa: BLE001
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def _refused(fn):
    try:
        fn()
    except paths_guard.ProtectedPathRefusal:
        return True
    except FileNotFoundError:
        raise AssertionError("guard let the call through to the filesystem (FileNotFoundError "
                             "means it was NOT refused)")
    return False


# ---------------------------------------------------------------- G2 runtime ----------------

def t_untouched_refused_builtin():
    p = REPO / "results" / "frozen_eval" / "untouched-fever.json"
    assert p.exists(), "fixture missing: the reserved payload should exist and stay unopened"
    assert _refused(lambda: open(p)), "builtins.open of a reserved payload was NOT refused"


def t_untouched_refused_os_open():
    p = REPO / "results" / "frozen_eval" / "untouched-dbpedia-entity.json"
    assert _refused(lambda: os.open(p, os.O_RDONLY)), "os.open was NOT refused"


def t_untouched_refused_pathlib():
    p = REPO / "results" / "frozen_eval" / "untouched-cqadup-android.json"
    assert _refused(p.read_text), "Path.read_text was NOT refused"


def t_untouched_refused_json_and_numpy():
    p = REPO / "results" / "frozen_eval" / "untouched-cqadup-english.json"
    assert _refused(lambda: json.load(open(p))), "json.load(open(...)) was NOT refused"
    assert _refused(lambda: np.load(p, allow_pickle=False)), "np.load was NOT refused"


def t_untouched_refused_relative_and_dotdot():
    """A relative path, or one routed through '..', must classify the same. The guard resolves
    before matching for exactly this reason."""
    rel = os.path.relpath(REPO / "results" / "frozen_eval" / "untouched-fever.json", Path.cwd())
    assert _refused(lambda: open(rel)), f"relative path {rel} was NOT refused"
    dotted = REPO / "m8src" / ".." / "results" / "frozen_eval" / "untouched-fever.json"
    assert _refused(lambda: open(dotted)), "'..'-routed path was NOT refused"


def t_lotte_and_m9_refused():
    for kind, sample in (("lotte", REPO / "work" / "lotte" / "anything.json"),
                         ("m9reserve", REPO / "work" / "m9reserve" / "eurlex_inventory.json")):
        assert _refused(lambda s=sample: open(s)), f"{kind} payload was NOT refused"


def t_six_and_manifest_allowed():
    """The guard must NOT over-block. The six are development-informed and already scored; the
    manifest is metadata. Blocking them would push callers into disabling the guard."""
    for p in (REPO / "results" / "frozen_eval" / "scifact.json",
              REPO / "results" / "eval_manifest.json",
              REPO / "results" / "m7_final_run.json"):
        assert p.exists(), f"fixture missing: {p}"
        with open(p) as fh:
            fh.read(16)


def t_reserved_work_dev_alias_refused():
    """THE 2026-08-29 INCIDENT. work/dev/cqadup-{android,english}.json hold the reserved sets'
    corpora AND their qrels. A protected partition is defined by its CONTENT, not by where one
    copy of it happens to live."""
    for name in ("cqadup-android.json", "cqadup-english.json"):
        p = REPO / "work" / "dev" / name
        assert p.exists(), f"fixture missing: {p} (if it was deleted, say so in the ledger)"
        assert _refused(lambda q=p: open(q)), f"{name} -- reserved qrels -- was NOT refused"


def t_reserved_work_dev_alias_via_devsuite_refused():
    """The route that would actually have happened: devsuite.load('cqadup-android')."""
    import devsuite
    assert _refused(lambda: devsuite.load("cqadup-android")), \
        "devsuite.load('cqadup-android') reached a reserved set"


def t_dev_components_still_loadable():
    """And the guard must not break the dev suite it sits next to."""
    p = REPO / "work" / "dev" / "cqadup-physics.json"
    if p.exists():
        with open(p) as fh:
            fh.read(16)


def t_hf_cache_reserved_qrels_refused():
    for d in paths_guard.RESERVED_HF_CACHE_PREFIXES:
        probe = paths_guard.HF / d / "default" / "anything.arrow"
        assert _refused(lambda q=probe: open(q)), f"HF cache {d} was NOT refused"


def t_hf_cache_reserved_CORPORA_refused():
    """The routes an adversarial review found LIVE and unguarded on 2026-08-29.

    The suite previously iterated `RESERVED_HF_CACHE_DIRS`, which held only the two `-qrels`
    spellings -- so it asserted exactly the cases that already worked and could never have found
    these. Hard-coded here on purpose: a test that derives its cases from the same constant the
    code reads cannot detect that the constant is incomplete.
    """
    for d in ("BeIR___fever", "BeIR___dbpedia-entity",
              "mteb___cqadupstack-android", "mteb___cqadupstack-english"):
        probe = paths_guard.HF / d / "default" / "0.0.0" / "shard.arrow"
        assert _refused(lambda q=probe: open(q)), f"HF cache {d} was NOT refused"


def t_hf_cache_beir_cqadupstack_config_dir_refused():
    """The BeIR spelling puts the subforum in a CONFIG directory, so the repo dir never says it."""
    for cfg in ("android", "english"):
        probe = paths_guard.HF / "BeIR___cqadupstack" / cfg / "0.0.0" / "shard.arrow"
        assert _refused(lambda q=probe: open(q)), f"BeIR cqadupstack/{cfg} was NOT refused"


def t_hf_cache_non_reserved_cqa_allowed():
    """And the guard must NOT swallow the DEV subforums, or every dev eval starts refusing."""
    for cfg in ("physics", "programmers"):
        probe = paths_guard.HF / "BeIR___cqadupstack" / cfg / "0.0.0" / "shard.arrow"
        assert paths_guard.classify(probe) is None, f"dev subforum {cfg} was wrongly protected"


def t_load_dataset_mteb_name_spelling_refused():
    """`mteb/cqadupstack-android` carries the subforum in the NAME, not the config."""
    for name in ("mteb/cqadupstack-android", "mteb/cqadupstack-english"):
        for cfg in (None, "queries", "corpus"):
            assert _refused(lambda n=name, c=cfg: paths_guard.check_dataset(n, c)), \
                f"check_dataset({name!r}, {cfg!r}) was NOT refused"


def t_load_dataset_dev_subforum_allowed():
    for name, cfg in (("mteb/cqadupstack-physics", "queries"),
                      ("BeIR/cqadupstack", "programmers")):
        assert paths_guard.check_dataset(name, cfg) is None, \
            f"dev component {name}/{cfg} was wrongly refused"


def t_load_dataset_network_route_refused():
    """load_dataset('BeIR/fever-qrels') touches no guarded PATH. Downloading the labels fresh is
    the same contact as opening the frozen payload."""
    for name, cfg in (("BeIR/fever-qrels", None), ("BeIR/dbpedia-entity", "queries"),
                      ("BeIR/cqadupstack", "android")):
        try:
            paths_guard.check_dataset(name, cfg)
        except paths_guard.ProtectedPathRefusal:
            continue
        raise AssertionError(f"load_dataset guard let {name}/{cfg} through")
    # and it must not over-block a dev component
    assert paths_guard.check_dataset("BeIR/cqadupstack", "programmers") is None
    assert paths_guard.check_dataset("BeIR/scifact", "corpus") is None


def t_pre_encode_exemption_is_narrow():
    """m8src.pre_encode may load the reserved CORPORA and nothing else. The dangerous case is a
    qrels repo, which needs NO config -- so "no config" must not be waved past."""
    import importlib
    pg = importlib.reload(paths_guard) if False else paths_guard
    saved = pg._claim
    try:
        pg.__dict__["_claim"] = "m8src.pre_encode"
        assert pg.check_dataset("BeIR/fever", "corpus") == "untouched_corpus_only", \
            "the pre-encode must be able to load the reserved corpora"
        for bad, cfg in (("BeIR/fever-qrels", None), ("BeIR/fever-qrels", "corpus"),
                         ("BeIR/fever", None), ("BeIR/fever", "queries"),
                         ("BeIR/dbpedia-entity", "queries")):
            try:
                pg.check_dataset(bad, cfg)
            except pg.ProtectedPathRefusal:
                continue
            raise AssertionError(f"pre_encode was allowed {bad!r} config={cfg!r}")
        # and it still cannot open a reserved PAYLOAD by path
        assert _refused(lambda: open(REPO / "results" / "frozen_eval" / "untouched-fever.json")), \
            "pre_encode opened a reserved payload"
    finally:
        pg.__dict__["_claim"] = saved


def t_symlink_alias_refused():
    """A symlink whose name mentions none of the hints must still classify: the guard resolves
    BEFORE it looks at the name."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        link = Path(d) / "harmless_name.json"
        link.symlink_to(REPO / "results" / "frozen_eval" / "untouched-fever.json")
        paths_guard._resolve.cache_clear()
        assert _refused(lambda: open(link)), "a symlink alias bypassed classification"


def t_claim_scopes_to_kind():
    """An allowlisted entry gets ONLY its kinds, and only its own file may claim it."""
    assert paths_guard.classify(REPO / "work" / "lotte" / "x.json") == "lotte"
    assert paths_guard.classify(REPO / "results" / "frozen_eval" / "untouched-fever.json") \
        == "untouched_labels"


def t_claim_from_wrong_file_refused():
    """Any file could otherwise grant itself the capability by naming someone else."""
    try:
        paths_guard.claim("m8src.final_run")
    except paths_guard.ProtectedPathRefusal as e:
        assert "may only claim itself" in str(e) or "not from" in str(e), str(e)
        return
    raise AssertionError("test_guards.py successfully claimed m8src.final_run")


def t_unknown_claim_refused():
    try:
        paths_guard.claim("m8src.some_probe")
    except paths_guard.ProtectedPathRefusal:
        return
    raise AssertionError("an off-allowlist entry was accepted as a claim")


def t_no_public_uninstall():
    assert not hasattr(paths_guard, "uninstall"), \
        "a public uninstall() is a hole: any code could turn the guard off"
    try:
        paths_guard._uninstall("wrong-token")
    except paths_guard.ProtectedPathRefusal:
        return
    raise AssertionError("the guard uninstalled on a wrong token")


def t_guard_installed_by_paths_import():
    """The capability must arrive with `import m8base`, which every m8src module does, so a module
    cannot opt out by forgetting."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import m8base, paths_guard; "
         "print(paths_guard.status()['installed'])" % str(M8SRC)],
        capture_output=True, text=True, cwd=str(REPO))
    assert out.stdout.strip() == "True", f"guard not installed on plain import: {out.stdout!r} "\
                                         f"{out.stderr[-300:]!r}"


def t_guard_in_a_fresh_process_refuses():
    """End to end, in a subprocess: the refusal is not an artifact of this test's state."""
    code = ("import sys; sys.path.insert(0, %r); import m8base, paths_guard\n"
            "try:\n"
            "    open(%r)\n"
            "except paths_guard.ProtectedPathRefusal:\n"
            "    print('REFUSED')\n" % (
                str(M8SRC), str(REPO / "results" / "frozen_eval" / "untouched-fever.json")))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO))
    assert out.stdout.strip() == "REFUSED", f"fresh process did not refuse: {out.stdout!r} " \
                                            f"{out.stderr[-300:]!r}"


# ---------------------------------------------------------------- G2 static -----------------

_PROTECTED_LITERALS = re.compile(r"untouched-|frozen_eval/untouched|work/lotte|work/m9reserve")


def t_static_no_unclaimed_mention():
    """No m8src file may mention a protected path unless it is on the allowlist (or is this test,
    or is the guard itself). This is the half that survives a refactor: a helper cannot inherit
    the capability by being imported from an allowlisted module, because the helper's own source
    would have to name the path."""
    exempt = {"paths_guard.py", "test_guards.py"}
    exempt |= {name.split(".", 1)[1] + ".py" for name in paths_guard.ALLOWLIST}
    offenders = []
    for f in sorted(M8SRC.glob("*.py")):
        if f.name in exempt:
            continue
        src = f.read_text()
        for i, line in enumerate(src.splitlines(), 1):
            if _PROTECTED_LITERALS.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{f.name}:{i}: {line.strip()[:90]}")
    assert not offenders, "unclaimed modules name a protected path:\n    " + \
                          "\n    ".join(offenders)


def t_static_every_allowlist_entry_is_named_in_the_ledger():
    """The allowlist and the ledger cannot drift: G2 says adding a name is an amendment."""
    ledger = (REPO / "m8" / "LEDGER.md").read_text()
    missing = [n for n in paths_guard.ALLOWLIST if n.split(".", 1)[1] not in ledger]
    assert not missing, f"allowlist entries with no mention in m8/LEDGER.md: {missing}"


# ---------------------------------------------------------------- G1 ------------------------

def t_probe_guard_refuses_unregistered():
    try:
        probe_guard.assert_registered("B99", strict_commit=False)
    except probe_guard.ProbeNotRegistered:
        return
    raise AssertionError("an unregistered probe id was accepted")


def t_probe_guard_refuses_tbd_bar():
    """A TBD bar must be a refusal, not a placeholder to run through (LEDGER 4.7 / G4)."""
    reg = probe_guard.registry()["probes"]
    tbd = [p for p, r in reg.items() if "tbd" in str(r.get("bar", "")).lower()]
    assert tbd, "no TBD bars at all -- if every bar is frozen, retire this check deliberately"
    for p in tbd:
        try:
            probe_guard.assert_registered(p, strict_commit=False)
        except probe_guard.ProbeNotRegistered as e:
            assert "TBD" in str(e), str(e)
            continue
        raise AssertionError(f"probe {p} has a TBD bar and was accepted")


def t_probe_guard_refuses_incomplete_row():
    """Every wave-2 stub must be refused: a stub is a placeholder for a registration, not one."""
    reg = probe_guard.registry()["probes"]
    # B16 is wave 2 but descriptive-and-exempt by registration, so it is legitimately runnable;
    # the stubs are the ones that will become bars.
    stubs = [p for p, r in reg.items()
             if r.get("wave") == 2 and r.get("adopts") != "nothing"]
    assert stubs, "no wave-2 stubs found"
    for p in stubs:
        try:
            probe_guard.assert_registered(p, strict_commit=False)
        except probe_guard.ProbeNotRegistered:
            continue
        raise AssertionError(f"wave-2 stub {p} was accepted as runnable")


def t_probe_guard_refuses_bar_pending():
    """A bar can READ complete and still be unfinished; bar_pending names what is missing.

    This test used to iterate the registry for rows carrying `bar_pending` -- and for most of the
    milestone there were NONE, so it passed by asserting nothing over an empty list. It now
    SYNTHESIZES the case against a row that is otherwise complete, so the code path is exercised
    whatever the registry happens to contain, and a row whose bar says TBD cannot stand in for it
    (a TBD bar refuses one check earlier, which is a different guard).
    """
    reg = copy.deepcopy(probe_guard.registry())
    donor = next(p for p, r in reg["probes"].items()
                 if "tbd" not in str(r.get("bar", "")).lower() and not r.get("bar_pending")
                 and all(str(r.get(f, "")).strip() for f in ("bar", "endpoint", "comparator",
                                                             "multiplicity", "no_survivor")))
    reg["probes"][donor]["bar_pending"] = "the floor term"
    orig = probe_guard.registry
    probe_guard.registry = lambda: reg
    try:
        try:
            probe_guard.assert_registered(donor, strict_commit=False)
        except probe_guard.ProbeNotRegistered as e:
            assert "bar_pending" in str(e), str(e)
        else:
            raise AssertionError(f"a complete row with bar_pending was accepted ({donor})")
    finally:
        probe_guard.registry = orig

    # and every row that actually declares it must refuse for SOME registered reason
    for p, r in probe_guard.registry()["probes"].items():
        if not r.get("bar_pending"):
            continue
        try:
            probe_guard.assert_registered(p, strict_commit=False)
        except probe_guard.ProbeNotRegistered:
            continue
        raise AssertionError(f"probe {p} with bar_pending was accepted")


def t_probe_guard_stamps_results(tmp=None):
    """A metric that reaches disk without a registry sha did not run under a registration."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.json"
        prov = probe_guard.write_result(out, {"x": 1}, "B2", strict_commit=False)
        body = json.loads(out.read_text())
        assert body["_registration"]["registry_sha256"] == prov["registry_sha256"]
        assert len(body["_registration"]["ledger_commit"]) == 40
        try:
            probe_guard.write_result(Path(d) / "n.json", {"x": 1}, "B9", strict_commit=False)
        except probe_guard.ProbeNotRegistered:
            return
    raise AssertionError("write_result wrote a result for a TBD-bar probe")


def t_probe_guard_classifies_changes():
    """LEDGER 5.4: an unknown config key must FAIL, not be argued into a category later."""
    assert probe_guard.classify_change("tokenizer_id") == "qualifying_table"
    assert probe_guard.classify_change("doc_side_head") == "qualifying_non_table"
    assert probe_guard.classify_change("steps_a") == "not_qualifying"
    assert probe_guard.classify_change("lr_tuning") == "not_qualifying"
    assert probe_guard.classify_change("some_new_knob") == "unknown"


def t_probe_guard_requires_committed_ledger():
    problems, head = probe_guard.check_committed()
    assert isinstance(problems, list) and len(head) == 40, "commit check did not run"


def t_probe_guard_parses_the_registry():
    reg = probe_guard.registry()["probes"]
    for pid in ("S0", "T1", "B2", "B3", "B7", "B6-pre", "B17"):
        assert pid in reg, f"{pid} missing from m8/registry.json"


def t_ledger_and_registry_agree():
    """The prose renders the registry; a probe id in one and not the other is drift."""
    ledger = (REPO / "m8" / "LEDGER.md").read_text()
    for pid in probe_guard.registry()["probes"]:
        assert pid in ledger, f"registry probe {pid} has no mention in m8/LEDGER.md"


def main():
    print("m8 guard suite (LEDGER G1 + G2)")
    groups = {
        "G2 runtime": [n for n in globals()
                       if n.startswith("t_") and not n.startswith(("t_static_",
                                                                   "t_probe_guard_"))],
        "G2 static": [n for n in globals() if n.startswith("t_static_")],
        "G1": [n for n in globals() if n.startswith("t_probe_guard_")],
    }
    for label, names in groups.items():
        print(f" {label}:")
        for name in names:
            check(name[2:], globals()[name])
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURES")
        return 1
    print("all guard checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
