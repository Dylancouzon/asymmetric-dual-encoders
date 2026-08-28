"""The fusion parameter must be provably fitted on the artifact being frozen.

Codex one-shot-path review 2026-08-28, MAJOR 1: `select_fusion` wrote no run id, table hash or
preprocessing fingerprint into the spec, and `freeze.write` accepted whatever spec its caller
handed it without consulting the selection file or the gate result. So a parameter fitted on
artifact A could be frozen with artifact B, undetected, and applied in the single one-shot run.
`released_system` was also a free string: anything other than the exact word "fusion" silently
meant dense, and an unknown fusion family silently meant convex.

Every check below is a refusal that must happen -- plus one check that a COMPLETE, consistent
freeze verifies, so the guards cannot pass by failing closed on everything.

KNOWN GAP, stated rather than papered over: this exercises `load_selected_fusion`,
`assert_gate_passed` and `load_and_verify` directly, not `freeze.write` end to end. `write` also
calls `ensure_release`, `assert_releasable` and `teacher_code.verify`, which need a real table, a
real run record and a real HF snapshot; a fixture faithful enough to be worth having is a larger
piece of work than the coverage it would add. The TOCTOU window `write` used to have is closed by
hashing once and re-checking at the end, and that re-check is not covered here.

Run after touching freeze.py, select_fusion.py or fusion.py.

    ../.venv/bin/python test_freeze_binding.py
"""
import json
import sys
import tempfile
from pathlib import Path

import freeze
import fusion

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def refuses(name, fn, must_mention=None):
    """`fn` must raise SystemExit, and its message must name the thing that is wrong."""
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        ok = must_mention is None or must_mention.lower() in msg.lower()
        check(name, ok, f"refused, but the message does not mention {must_mention!r}: {msg[:200]}")
        return
    check(name, False, "did NOT refuse")


RID = "fixture-run"


class Fixture:
    """A minimal on-disk repo: release table, metadata, fusion selection, gate result."""

    def __init__(self, td):
        self.repo = Path(td)
        self.work = self.repo / "work"
        (self.work / "runs").mkdir(parents=True)
        (self.repo / "results").mkdir()
        self.npz = self.work / "runs" / f"{RID}.release.npz"
        self.npz.write_bytes(b"not really a table, but it has a stable sha256")
        self.meta_p = self.npz.with_name(self.npz.stem + ".meta.json")
        self.meta = {"preproc": {"prefix": "", "pool_mode": "sqrt"},
                     "preproc_fingerprint": "adb24fb2e8cad66f", "weights_folded": True}
        self.meta_p.write_text(json.dumps(self.meta))
        for f in ("m7_dev_manifest.json", "eval_manifest.json", "perquery.json"):
            (self.repo / "results" / f).write_text('{"fixture": true}')
        self.write_spec()
        self.write_gate()

    def bm25_key(self, **over):
        k = {"format": fusion.CACHE_FORMAT, "n_docs": 10, "n_queries": 3,
             "depth": fusion.DEPTH, "doc_ids_sha256": "a" * 64, "doc_texts_sha256": "b" * 64,
             "q_ids_sha256": "c" * 64, "q_texts_sha256": "d" * 64,
             "config": fusion.BM25_CONFIG, "versions": fusion._pkg_versions()}
        k.update(over)
        return k

    def spec(self, **over):
        s = {"family": "convex", "param": 0.5, "dev_macro": 0.55,
             "grid": [{"family": "rrf", "param": 60, "macro": 0.51},
                      {"family": "convex", "param": 0.5, "macro": 0.55},
                      {"family": "convex", "param": 1.0, "macro": 0.49},
                      {"family": "convex0", "param": 0.6, "macro": 0.52}],
             "depth": fusion.DEPTH, "components": ["nq-250k"],
             "fitted_against": "int8 table (the released artifact)",
             "selected_on": {
                 "run_id": RID,
                 "table_relpath": f"work/runs/{self.npz.name}",
                 "table_sha256": freeze.sha256_file(self.npz),
                 "table_meta_sha256": freeze.sha256_file(self.meta_p),
                 "preproc": self.meta["preproc"],
                 "preproc_fingerprint": self.meta["preproc_fingerprint"],
                 "encoder_spec": freeze.encoder_fingerprint(),
                 "dev_manifest_sha256": freeze.sha256_file(
                     self.repo / "results" / "m7_dev_manifest.json"),
                 "bm25_run_keys": {"nq-250k": self.bm25_key()}},
             "released_system": "fusion"}
        s.update(over)
        return s

    def write_spec(self, spec=None, publish=None):
        s = spec if spec is not None else self.spec()
        (self.work / "runs" / f"{RID}.fusion.json").write_text(json.dumps(s))
        (self.repo / "results" / f"m7_fusion_{RID}.json").write_text(
            json.dumps(publish if publish is not None else s))

    def write_gate(self, **over):
        g = {"run_id": RID, "PASS": True,
             "conditions": {"G1": {"pass": True}, "G2": {"pass": True},
                            "G3": {"pass": True}, "G4": {"pass": True}},
             "artifact": {"release": self.npz.name,
                          "sha256": freeze.sha256_file(self.npz),
                          "meta_sha256": freeze.sha256_file(self.meta_p)}}
        g.update(over)
        (self.repo / "results" / f"m7_gate_{RID}.json").write_text(json.dumps(g))

    def load(self):
        return freeze.load_selected_fusion(RID, freeze.sha256_file(self.npz),
                                           freeze.sha256_file(self.meta_p), self.meta)

    def gate(self):
        return freeze.assert_gate_passed(RID, freeze.sha256_file(self.npz),
                                         freeze.sha256_file(self.meta_p))


def main():
    orig = (freeze.REPO, freeze.WORK, freeze.FREEZE)
    with tempfile.TemporaryDirectory() as td:
        fx = Fixture(td)
        freeze.REPO, freeze.WORK = fx.repo, fx.work
        try:
            print("fusion spec <-> artifact binding")
            check("a spec selected on this artifact loads", fx.load()["param"] == 0.5)

            sel = fx.spec()["selected_on"]
            fx.write_spec(fx.spec(selected_on={**sel, "table_sha256": "0" * 64}))
            refuses("a spec fitted on a DIFFERENT table is refused", fx.load, "table_sha256")

            fx.write_spec(fx.spec(selected_on={**sel, "run_id": "some-other-run"}))
            refuses("a spec naming a different run id is refused", fx.load, "run_id")

            fx.write_spec(fx.spec(selected_on={**sel, "preproc_fingerprint": "deadbeefdeadbeef"}))
            refuses("a spec with a different preproc fingerprint is refused", fx.load,
                    "preproc_fingerprint")

            fx.write_spec(fx.spec(selected_on={**sel, "table_meta_sha256": "0" * 64}))
            refuses("a spec fitted against different table metadata is refused", fx.load,
                    "table_meta_sha256")

            fx.write_spec(fx.spec(selected_on={**sel, "dev_manifest_sha256": "0" * 64}))
            refuses("a spec fitted against a different dev suite is refused", fx.load,
                    "dev_manifest_sha256")

            fx.write_spec(fx.spec(selected_on={
                **sel, "encoder_spec": dict(sel["encoder_spec"], repo="someone/other-encoder")}))
            refuses("a spec selected under a different encoder is refused", fx.load, "encoder")

            s = fx.spec()
            s.pop("selected_on")
            fx.write_spec(s)
            refuses("a spec with no provenance block is refused", fx.load, "selected_on")

            fx.write_spec(fx.spec(family="convexx"))
            refuses("an unknown fusion family is refused at freeze time", fx.load, "family")

            fx.write_spec(fx.spec(param="0.5"))
            refuses("a non-numeric fusion param is refused", fx.load, "param")

            fx.write_spec(fx.spec(depth=100))
            refuses("a spec selected at another depth is refused", fx.load, "depth")

            fx.write_spec(fx.spec(), publish=fx.spec(param=0.7))
            refuses("a hand-edited committed copy is refused", fx.load, "differs")

            # the frozen point must be the argmax of the grid the selection actually searched
            fx.write_spec(fx.spec(param=0.6, dev_macro=0.55))
            refuses("a point that is not the grid's argmax is refused", fx.load, "argmax")
            fx.write_spec(fx.spec(grid=[]))
            refuses("a spec with no grid is refused", fx.load, "grid")
            fx.write_spec(fx.spec(dev_macro=0.99))
            refuses("a dev_macro that is not the grid's best is refused", fx.load, "dev_macro")

            # the BM25 runs the parameter was fitted against must still be the same function
            sel2 = fx.spec()["selected_on"]
            fx.write_spec(fx.spec(selected_on={**sel2, "bm25_run_keys": {}}))
            refuses("an empty bm25_run_keys block is refused", fx.load, "bm25_run_keys")
            fx.write_spec(fx.spec(selected_on={
                **sel2, "bm25_run_keys": {"nq-250k": fx.bm25_key(versions={"bm25s": "0.0.0"})}}))
            refuses("a bm25s version change since selection is refused", fx.load,
                    "lexical function has changed")
            fx.write_spec(fx.spec(selected_on={
                **sel2, "bm25_run_keys": {"nq-250k": fx.bm25_key(config={"k1": 9.9})}}))
            refuses("changed BM25 parameters are refused", fx.load, "parameters")
            fx.write_spec(fx.spec(selected_on={
                **sel2, "bm25_run_keys": {"other-comp": fx.bm25_key()}}))
            refuses("bm25 keys for the wrong components are refused", fx.load, "cover")

            fx.write_spec()
            (fx.repo / "results" / f"m7_fusion_{RID}.json").unlink()
            refuses("a missing committed copy is refused", fx.load, "missing")
            fx.write_spec()

            print("\ngate binding")
            check("a PASSing gate on this artifact is accepted", fx.gate()["PASS"] is True)
            fx.write_gate(PASS=False, conditions={"G3": {"pass": False}})
            refuses("a NO-GO gate is refused", fx.gate, "NO-GO")
            # "false" is a non-empty string and used to be truthy
            fx.write_gate(PASS="false")
            refuses("a stringy PASS is refused", fx.gate, "NO-GO")
            # a summary flag cannot carry a failing condition through
            fx.write_gate(PASS=True, conditions={"G1": {"pass": True}, "G3": {"pass": False}})
            refuses("PASS=true with a failing condition is refused", fx.gate, "not passing")
            fx.write_gate(conditions={})
            refuses("a gate with no conditions is refused", fx.gate, "no conditions")
            fx.write_gate(artifact="not-a-dict")
            refuses("a malformed artifact block is refused, not crashed on", fx.gate, "artifact")
            fx.write_gate(artifact={"release": "x.npz", "sha256": "0" * 64,
                                    "meta_sha256": freeze.sha256_file(fx.meta_p)})
            refuses("a gate run on a different table is refused", fx.gate, "different table")
            fx.write_gate(run_id="another-run")
            refuses("a gate file naming another run id is refused", fx.gate, "run_id")
            (fx.repo / "results" / f"m7_gate_{RID}.json").unlink()
            refuses("a missing gate result is refused", fx.gate, "no gate result")
            fx.write_gate()

            print("\nreleased_system is derived, not asserted")
            check("convex w=1.0 is the dense-only endpoint",
                  fusion.is_dense_only({"family": "convex", "param": 1.0}))
            check("convex0 w=1.0 is the dense-only endpoint",
                  fusion.is_dense_only({"family": "convex0", "param": 1.0}))
            check("convex w=0.5 is not dense-only",
                  not fusion.is_dense_only({"family": "convex", "param": 0.5}))
            check("rrf is never dense-only",
                  not fusion.is_dense_only({"family": "rrf", "param": 10}))
            check("the enum has exactly the two members final_run dispatches on",
                  tuple(freeze.RELEASED_SYSTEMS) == ("dense", "fusion"))

            print("\nload_and_verify on a COMPLETE fixture, then one mutation at a time")
            # The earlier version of this block used a manifest with a missing table and bogus
            # hashes, so it only proved that one error message mentioned the enum -- not that a
            # valid freeze passes and that the mutation is the ONLY thing that breaks it.
            fz = fx.repo / "FREEZE-fixture.json"
            freeze.FREEZE = fz
            sp = freeze.encoder_fingerprint()
            fx.meta.update({"teacher": sp["repo"], "teacher_revision": sp["revision"],
                            "dim": sp["dim"]})
            fx.meta_p.write_text(json.dumps(fx.meta))
            fx.write_spec()
            good = {
                "run_id": RID,
                "table_relpath": f"work/runs/{fx.npz.name}",
                "table_sha256": freeze.sha256_file(fx.npz),
                "table_meta_sha256": freeze.sha256_file(fx.meta_p),
                "table_bytes": fx.npz.stat().st_size,
                "preproc": fx.meta["preproc"],
                "preproc_fingerprint": fx.meta["preproc_fingerprint"],
                "teacher": sp["repo"], "teacher_revision": sp["revision"],
                "encoder_spec": sp,
                "fusion": fx.spec(), "released_system": "fusion",
                "dev_manifest_sha256": freeze.sha256_file(
                    fx.repo / "results" / "m7_dev_manifest.json"),
                "eval_manifest_sha256": freeze.sha256_file(
                    fx.repo / "results" / "eval_manifest.json"),
                "perquery_sha256": freeze.sha256_file(fx.repo / "results" / "perquery.json"),
            }

            def verify_with(**over):
                fz.write_text(json.dumps({**good, **over}))
                return freeze.load_and_verify()

            try:
                verify_with()
                check("a complete, consistent freeze VERIFIES", True)
            except SystemExit as e:
                check("a complete, consistent freeze VERIFIES", False, str(e)[:400])

            refuses("released_system 'Fusion' is not silently read as dense",
                    lambda: verify_with(released_system="Fusion"), "'Fusion' is not one of")
            refuses("released_system must agree with the selection's grid point",
                    lambda: verify_with(released_system="dense"), "but the frozen selection")
            refuses("an edited fusion param is caught against the selection file",
                    lambda: verify_with(fusion=fx.spec(param=0.6)), "differs from")
            refuses("a changed preproc fingerprint is caught",
                    lambda: verify_with(preproc_fingerprint="0" * 16), "preprocessing fingerprint")
            refuses("a changed table hash is caught",
                    lambda: verify_with(table_sha256="0" * 64), "not the frozen artifact")
            refuses("a different frozen encoder is caught",
                    lambda: verify_with(encoder_spec=dict(sp, pooling="__sentinel__")),
                    "active encoder differs")
            refuses("a changed dev manifest is caught",
                    lambda: verify_with(dev_manifest_sha256="0" * 64), "changed after the freeze")
        finally:
            freeze.REPO, freeze.WORK, freeze.FREEZE = orig

    print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
