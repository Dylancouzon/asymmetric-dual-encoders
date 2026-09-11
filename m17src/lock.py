"""M17 step 6: the lock, committed in two dated halves (`m17/STATUS.md` step 6).

Why two halves. The on-clock protected screen (`prepare_data.py --protected-screen`) is part of
every stage marker's identity, so it rebuilds `teacher`, `bank`, `vocab`, `cache` and the
manifests. The vocabulary list, the extended tokenizer, the new rows and the candidate cache that
train are therefore produced ON the clock, after the screen; the unscreened full build only
prices them. `training.decision_protocol.lock` requires exactly that these hashes and the V0
export hash are committed "before the V0 read or any real development read" — not before the
clock — so:

  --phase pre       PRE-CLOCK. Binds the protocol block, the recipe identity, the seeds, the
                    fixed manifests (panel FINAL, alias test, support, exclusions, k8s source,
                    warm start, FREEZE, the builder's source) and the measured allocation from
                    the unscreened full build. Its identities are recorded as `pre_screen_build`
                    — a price, not the executed artifact. Status -> EXECUTABLE (the executor's
                    protected screen needs it: `require_executable` guards `protected10`).
  --phase executed  ON THE CLOCK, after the screened rebuild and before any read. Requires the
                    screen receipt, the same protocol/recipe identity as the pre half, records the
                    executed vocabulary/tokenizer/new-rows/cache/bank/teacher identities, exports
                    V0 (folded + int8 exactly like a release, no optimizer step) through the M17
                    gates and records its hash. Status -> LOCKED_EXECUTABLE. Only then may V0
                    be read.

Neither half overwrites an existing half: a later change is a dated amendment the owner writes
by hand, preserving the original in git. No development component, panel row or protected
surface is read here; the V0 export writes a table and runs the synthetic-fixture gates only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REGISTRY_PATH, RESULTS, admit_read, freeze, registry, sha_file,   # noqa: E402
                    sha_json, write_json)

PRE_STATUS, EXEC_STATUS = "EXECUTABLE", "LOCKED_EXECUTABLE"
PREPARE_PHASE = "coverage and verified alias pairs, teacher targets and candidate cache"
FIXED_FILES = {
    "panel_manifest": RESULTS / "m17_panel_manifest.json",
    "alias_test_manifest": RESULTS / "m17_alias_test_manifest.json",
    "support_manifest": RESULTS / "m17_support_manifest.json",
    "k8s_source_manifest": RESULTS / "m17_k8s_source_manifest.json",
    "judgments_ingest": RESULTS / "m17_judgments_ingest.json",
    "prepare_timing": RESULTS / "m17_prepare_timing.json",
}
# ext/prepared.json `hashes` fields that the screened rebuild may legitimately change (they are
# downstream of `protected` in `prepare_data.STAGES`) versus those that must not move at all.
EXECUTED_HASHES = ("vocabulary", "tokenizer", "new_rows", "student_ids", "cache_identity",
                   "cache_artifact", "bank_ids", "bank_vector_bytes", "teacher_q", "v1_q",
                   "query_pool", "doc_domain_join")
INVARIANT_HASHES = ("warm_start_npz", "freeze", "alias_pairs_jsonl", "exclusion_file",
                    "k8s_jsonl", "support_manifest", "prepare_data_py")


def _today():
    return _dt.date.today().isoformat()


def _rel(path: Path) -> str:
    root = REGISTRY_PATH.parents[1]
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _manifest(build: Path, form: str):
    return json.loads(admit_read(build / form / "prepared.json").read_text())


def protocol_identity(reg, fz=None):
    """What the lock binds besides artifacts: the decision protocol and the build recipe."""
    import prepare_data
    tr = reg["training"]
    return {
        "decision_protocol_sha256": sha_json(tr["decision_protocol"]),
        "arms_sha256": sha_json(tr["arms"]),
        "screen_contrasts_sha256": sha_json(tr["screen_contrasts"]),
        "recipe_sha256": prepare_data.recipe_identity(reg, fz or freeze())["sha256"],
        "seeds": {"prepared": 0, "screen": int(tr["seed_screen"]),
                  "replication": int(tr["seed_replication"])},
        "steps": {"screen": int(tr["screen_steps"]), "final": int(tr["final_steps_per_fresh_run"])},
        "batch": int(tr["batch"]),
        "snapshot_steps": list(reg["checkpoint_averaging"]["checkpoint_steps"]),
        # every operative setting the protocol refers to (tie band, tolerances, learning rates,
        # vocabulary caps, ...) — the whole registry minus the fields the lock itself writes
        # (Astra lock review P1-6)
        "registry_sha256_excluding_lock_fields": sha_json(
            {k: v for k, v in reg.items() if k not in ("status", "lock", "allocation_hours")}),
    }


def _check_full_build(ext, proto):
    """The locked build is the FULL pool at the locked seed: a `--size` build is complete too
    (all ten stages ran) but is not what trains (Astra lock review P1-5)."""
    if ext.get("size") is not None:
        raise SystemExit(f"M17 LOCK REFUSED: the build is a --size {ext['size']} subsample, not "
                         "the full pool.")
    if int(ext.get("seed", -1)) != int(proto["seeds"]["prepared"]):
        raise SystemExit(f"M17 LOCK REFUSED: the build's seed {ext.get('seed')} is not the locked "
                         f"prepared seed {proto['seeds']['prepared']}.")


def _fixed_sha(fixed_files):
    return {k: sha_file(admit_read(p)) for k, p in fixed_files.items()}


def measured_allocation(build: Path, prior=None):
    """The unscreened full build's wall clock, and the ceiling it implies for the on-clock
    screened rebuild (teacher/document caches warm, every stage after `protected` re-run).

    `prior` = `{"seconds": float, "stages": [...], "source": str}` prices the stages an earlier
    invocation of the SAME output directory built and this one resumed (the 2026-09-11 full
    build: attempt 1 built pool/domain/protected, died in teacher, attempt 2 resumed). It is
    accepted only when those are exactly the stages this record did not run, and it is recorded
    beside the measurement, never folded into `wall_clock_kind`."""
    import prepare_data
    rec = json.loads(admit_read(build / "build_record.json").read_text())
    ran = rec.get("stages_run_this_invocation") or []
    not_run = [st for st in prepare_data.STAGES if st not in ran]
    if rec.get("complete_build") and rec.get("wall_clock_kind") == "full build":
        seconds, summed = float(rec["wall_clock_seconds"]), None
    elif prior and sorted(prior.get("stages") or []) == sorted(not_run) and not_run:
        seconds = float(rec["wall_clock_seconds"]) + float(prior["seconds"])
        summed = {"this_invocation_seconds": rec["wall_clock_seconds"], "this_invocation_ran": ran,
                  "prior_seconds": float(prior["seconds"]), "prior_stages": not_run,
                  "prior_source": str(prior.get("source") or "")}
    else:
        raise SystemExit(f"M17 LOCK REFUSED: {build}/build_record.json is a "
                         f"{rec.get('wall_clock_kind')!r} that did not run {not_run}; only a "
                         "complete full build, or a resumed one plus --prior-seconds for exactly "
                         "those stages, prices the preparation phase.")
    hours = seconds / 3600
    return {"full_build_wall_clock_seconds": round(seconds, 3),
            "full_build_hours": round(hours, 3),
            "summed_across_invocations": summed,
            "rss_high_water_gib": rec.get("rss_high_water_gib"),
            "memory_gib": rec.get("memory_gib"), "gpu_peak_gib": rec.get("gpu_peak_gib"),
            "queries": rec.get("size") or _manifest(build, "ext")["queries"],
            # one screened rebuild is at most one more full build; the ceiling is two builds
            # rounded up, never below one hour
            "phase_ceiling_hours": max(1, math.ceil(2 * hours))}


def lock_pre(reg, build: Path, date=None, fixed_files=None, prior=None):
    fixed_files = fixed_files or FIXED_FILES
    if reg.get("status") != "DRAFT_NOT_EXECUTABLE":
        raise SystemExit(f"M17 LOCK REFUSED: status is {reg.get('status')!r}; the pre half "
                         "applies to a DRAFT_NOT_EXECUTABLE registry only.")
    if "lock" in reg:
        raise SystemExit("M17 LOCK REFUSED: `lock` already exists; amend by hand, dated.")
    ext, base = _manifest(build, "ext"), _manifest(build, "base")
    if ext["protected_screen"]["state"] != "deferred_to_clock":
        raise SystemExit("M17 LOCK REFUSED: the pre half prices the UNSCREENED build; this one "
                         f"records protected_screen.state={ext['protected_screen']['state']!r}.")
    panel = json.loads(admit_read(fixed_files["panel_manifest"]).read_text())
    # the sealed manifest reads "FINAL — sealed; cloud-software relevance is MODEL-JUDGED (A6)"
    if not str(panel.get("status") or "").startswith("FINAL"):
        raise SystemExit(f"M17 LOCK REFUSED: panel manifest status {panel.get('status')!r} != FINAL.")
    proto = protocol_identity(reg)
    _check_full_build(ext, proto)
    alloc = measured_allocation(build, prior)
    fixed = _fixed_sha(fixed_files)
    lock = {
        "_note": "step-6 lock, PRE-CLOCK half. Binds protocol, recipe, seeds, fixed manifests "
                 "and the measured allocation. `pre_screen_build` prices the preparation; the "
                 "executed vocabulary/tokenizer/cache identities are committed by the on-clock "
                 "half (`lock.executed`) before the V0 read. Amendments are dated and preserve "
                 "this block.",
        "date": date or _today(),
        "status_set": PRE_STATUS,
        "protocol": proto,
        "fixed_files_sha256": fixed,
        "panel_sha256": panel["panel_jsonl"]["sha256"],   # the sealed panel.jsonl bytes
        "invariant_build_inputs_sha256": {k: ext["hashes"][k] for k in INVARIANT_HASHES},
        "pre_screen_build": {
            "out": _rel(build),
            "queries": ext["queries"], "seed": ext["seed"],
            "ext_hashes": {k: ext["hashes"][k] for k in EXECUTED_HASHES},
            "base_tokenizer_sha256": base["hashes"]["tokenizer"],
            "vocabulary_terms": ext["stages"]["vocab"]["selected"],
            "unscreened_new_source_rows": ext["protected_screen"]["unscreened_new_source_rows"],
            "note": "unscreened; every hash here may change under the on-clock protected screen "
                    "and is superseded by `lock.executed`",
        },
        "measured_allocation": alloc,
        "clock": {"hard_total_hours": 72,
                  "start": "an owner ruling (which invocation starts the 72 h, and what pre-clock "
                           "work is exempt) is recorded in LEDGER.md before the first "
                           "--protected-screen invocation; `lock.executed.clock_started` "
                           "records the timestamp under that ruling"},
    }
    # the allocation row the measurement replaces (placeholder 16 h -> measured ceiling)
    rows = reg["allocation_hours"]
    hit = [r for r in rows if r["phase"] == PREPARE_PHASE]
    if len(hit) != 1:
        raise SystemExit(f"M17 LOCK REFUSED: allocation phase {PREPARE_PHASE!r} not found once.")
    row = hit[0]
    row["placeholder_hours_before_measurement"] = row["hours"]
    row["hours"] = min(int(row["hours"]), alloc["phase_ceiling_hours"])
    row["measured"] = {k: alloc[k] for k in ("full_build_wall_clock_seconds", "full_build_hours",
                                              "rss_high_water_gib", "gpu_peak_gib", "queries")}
    lock["allocation_hours_total"] = sum(int(r["hours"]) for r in rows)
    lock["allocation_unallocated_hours"] = 72 - lock["allocation_hours_total"]
    reg["lock"] = lock
    reg["status"] = PRE_STATUS
    return reg


def export_v0(build: Path, out: Path, reg=None, log=print):
    """V0: the warm start's effective rows plus the count-weighted new rows, folded and int8
    quantized exactly like a release, no optimizer step (`training.untrained_vocab_export_v0`)."""
    import export
    from tokenizers import Tokenizer
    reg = reg or registry()
    ext = _manifest(build, "ext")
    d = build / "ext"
    # The bytes V0 folds must be the bytes the manifest (and therefore the lock) names
    # (Astra lock review P1-1); `train._load_prepared` does the same for training.
    from common import sha_array
    new_rows = np.load(admit_read(d / ext["new_rows"])).astype(np.float32)
    # the builder's conventions: files by bytes, `new_rows` by `sha_array` (array bytes, dtype,
    # shape — `prepare_data.stage_vocab`), as `train._load_prepared` also compares them
    for key, got in (("tokenizer", sha_file(admit_read(d / ext["tokenizer"]))),
                     ("new_rows", sha_array(np.load(admit_read(d / ext["new_rows"])))),
                     ("warm_start_npz", sha_file(admit_read(d / ext["warm_start"])))):
        if got != ext["hashes"][key]:
            raise SystemExit(f"M17 V0 REFUSED: {d}/{key} hashes {got[:12]} but prepared.json "
                             f"records {str(ext['hashes'][key])[:12]}.")
    eff, diag = export.effective_rows(d / ext["warm_start"])
    if eff.shape[0] != int(reg["base_vocab"]):
        raise SystemExit(f"M17 V0 REFUSED: warm start has {eff.shape[0]} rows, not "
                         f"base_vocab {reg['base_vocab']}.")
    rows = np.concatenate([eff, new_rows], 0)
    tok = Tokenizer.from_file(str(admit_read(d / ext["tokenizer"])))
    cache = json.loads(admit_read(d / "cache.json").read_text())
    prov = {"m17_run_id": "V0", "arm": "V0", "trained": False,
            "definition": reg["training"]["untrained_vocab_export_v0"]["definition"],
            "prepared_dir": str(d), "warm_start_diag": diag,
            "tokenizer_sha256": ext["hashes"]["tokenizer"],
            "vocabulary_sha256": ext["hashes"]["new_rows"],
            "candidate_cache_sha256": cache["identity"]["sha256"],
            "vocabulary_terms": ext["stages"]["vocab"]["selected"],
            # The vocabulary was discovered over the whole pool, which carries Kubernetes
            # documentation rows whenever the k8s source file entered the build; CC BY 4.0
            # attribution therefore ships with V0 (over-attributing is harmless, the reverse is
            # a licence breach). Per-term sources are not recorded by the vocab stage.
            "vocabulary_sources": ([export.ATTRIBUTION_TRIGGER]
                                   if ext["hashes"].get("k8s_jsonl") else []),
            "registry_lock_date": (reg.get("lock") or {}).get("date")}
    bundle = export.build_bundle(out, rows, tok, prov, reg, form="endpoint")
    gates = export.run_gates(bundle, log=log)
    return bundle, gates


def lock_executed(reg, build: Path, v0_out: Path, clock_started: str, date=None, log=print,
                  export=None, fixed_files=None):
    if reg.get("status") != PRE_STATUS or "lock" not in reg:
        raise SystemExit(f"M17 LOCK REFUSED: status {reg.get('status')!r}; the executed half "
                         f"needs the pre half ({PRE_STATUS}).")
    if "executed" in reg["lock"]:
        raise SystemExit("M17 LOCK REFUSED: `lock.executed` already exists; amend by hand, dated.")
    ext, base = _manifest(build, "ext"), _manifest(build, "base")
    scr = ext["protected_screen"]
    if scr.get("state") != "complete" or not scr.get("receipt"):
        raise SystemExit("M17 LOCK REFUSED: the executed half needs the SCREENED build "
                         f"(protected_screen.state={scr.get('state')!r}, receipt={scr.get('receipt')!r}).")
    lock = reg["lock"]
    proto = protocol_identity(reg)
    _check_full_build(ext, proto)
    if proto != lock["protocol"]:
        raise SystemExit("M17 LOCK REFUSED: the protocol/recipe identity moved since the pre "
                         "half; that is a dated amendment, not an execution.")
    moved = {k: (lock["invariant_build_inputs_sha256"][k], ext["hashes"][k])
             for k in INVARIANT_HASHES if lock["invariant_build_inputs_sha256"][k] != ext["hashes"][k]}
    if moved:
        raise SystemExit(f"M17 LOCK REFUSED: invariant build inputs changed since the pre half: {moved}")
    if ext["registry_status_at_build"] != PRE_STATUS:
        raise SystemExit(f"M17 LOCK REFUSED: the screened build ran under status "
                         f"{ext['registry_status_at_build']!r}, not {PRE_STATUS}.")
    fixed_now = _fixed_sha(fixed_files or FIXED_FILES)
    moved = sorted(k for k in fixed_now if fixed_now[k] != lock["fixed_files_sha256"].get(k))
    if moved:
        raise SystemExit(f"M17 LOCK REFUSED: fixed manifests changed since the pre half: {moved}")
    # The builder records the receipt as {"path": <repo-relative>, "sha256": ...}
    # (`prepare_data.stage_protected`); the bytes must still hash to what the screen wrote.
    rec = scr["receipt"]
    rel = rec["path"] if isinstance(rec, dict) else rec
    receipt = Path(rel) if Path(rel).is_absolute() else REGISTRY_PATH.parents[1] / rel
    if not receipt.exists():
        receipt = build / Path(rel).name
    receipt_sha = sha_file(admit_read(receipt))
    if isinstance(rec, dict) and rec.get("sha256") != receipt_sha:
        raise SystemExit(f"M17 LOCK REFUSED: {receipt} hashes {receipt_sha[:12]} but the manifest "
                         f"records {str(rec.get('sha256'))[:12]}.")
    bundle, gates = (export or export_v0)(build, v0_out, reg, log=log)
    prov = json.loads((bundle / "provenance.json").read_text())   # a failed gate raised already
    lock["executed"] = {
        "_note": "step-6 lock, ON-CLOCK half, committed after the screened rebuild and BEFORE "
                 "the V0 read or any development read. These are the identities that train.",
        "date": date or _today(), "clock_started": clock_started,
        "status_set": EXEC_STATUS,
        "build": _rel(build),
        "ext_hashes": {k: ext["hashes"][k] for k in EXECUTED_HASHES},
        # the control arms (C, L) train the base form: bind it whole, not only its tokenizer
        # (Astra lock review P1-4); `train._check_executed_lock` compares per form
        "base_hashes": {k: base["hashes"].get(k) for k in EXECUTED_HASHES},
        "vocabulary_terms": ext["stages"]["vocab"]["selected"],
        "protected_screen": {k: scr.get(k) for k in ("state", "screened", "dropped", "dropped_by_text",
                                                      "dropped_by_document_receipt",
                                                      "documents_screened", "receipt")},
        "protected_receipt_sha256": receipt_sha,
        "v0_export": {"bundle": str(bundle), "model_npz_sha256": prov["model_npz_sha256"],
                      "tokenizer_sha256": prov["tokenizer_sha256"],
                      "config_sha256": prov["config_sha256"], "gates": gates,
                      "read": False},
        "changed_since_pre_screen": sorted(k for k in EXECUTED_HASHES
                                           if lock["pre_screen_build"]["ext_hashes"][k]
                                           != ext["hashes"][k]),
    }
    reg["status"] = EXEC_STATUS
    return reg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("pre", "executed"), required=True)
    ap.add_argument("--build", required=True, help="the full prepared directory")
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--v0-out", default=None, help="executed: where the V0 bundle is written")
    ap.add_argument("--clock-started", default=None,
                    help="executed: ISO timestamp of the first --protected-screen invocation")
    ap.add_argument("--prior-seconds", type=float, default=None,
                    help="pre: wall clock of the stages an earlier invocation built and this "
                         "one resumed (see measured_allocation)")
    ap.add_argument("--prior-stages", default=None, help="pre: comma-separated, e.g. pool,domain,protected")
    ap.add_argument("--prior-source", default=None, help="pre: where those seconds were read")
    ap.add_argument("--dry-run", action="store_true", help="print the block, write nothing")
    args = ap.parse_args(argv)
    path = Path(args.registry)
    reg = json.loads(admit_read(path).read_text())
    build = Path(args.build).resolve()
    if args.phase == "pre":
        prior = ({"seconds": args.prior_seconds, "stages": args.prior_stages.split(","),
                  "source": args.prior_source} if args.prior_seconds is not None else None)
        reg = lock_pre(reg, build, prior=prior)
        shown = reg["lock"]
    else:
        if args.dry_run:
            raise SystemExit("--dry-run is for the pre half only: the executed half exports V0, "
                             "which writes a bundle (Astra lock review P2-11).")
        if not (args.v0_out and args.clock_started):
            raise SystemExit("--v0-out and --clock-started are required for the executed half")
        reg = lock_executed(reg, build, Path(args.v0_out), args.clock_started)
        shown = reg["lock"]["executed"]
    print(json.dumps(shown, indent=1, sort_keys=True))
    if args.dry_run:
        print(f"[dry run] status would become {reg['status']!r}; nothing written")
        return 0
    write_json(path, reg)
    print(f"registry -> {path} status {reg['status']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
