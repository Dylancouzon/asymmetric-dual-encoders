"""One-shot M14 transaction around M13's triggered descriptive reserved batch.

The original six-set result remains the durable record of the gate.  This transaction spends the
separate inherited reserved access, resumes only at a missing atomic system output, and replaces
the earlier ``INCOMPLETE_RESERVED`` marker with a report derived from all three completed systems.
M13 only prepares and verifies this implementation; owner ruling R19 assigns execution to M14.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import access13 as A
import reserved_support as R

TAG = "m8-reserved-spent"
MANIFEST = A.REPO / "results" / "m13_reserved_manifest.json"
RESULT = A.REPO / "results" / "m13_reserved_run.json"
PREENCODE = A.REPO / "results" / "m13_reserved_preencode.json"
LOCK = A.REPO / "work" / "m13-reserved.lock"
ARCHIVE_ROOT = A.REPO / "work" / "m20-archive"
CODE_FILES = (
    "m13src/score13.py",
    "m13src/reserved_transaction.py",
    "m13src/reserved_support.py",
    "m20src/roster.py",
    "m20/beir15_registry.json",
    "m8src/pre_encode.py",
    "m8src/paths_guard.py",
    "m7src/evalkit.py",
    "m7src/fusion.py",
    "m12src/qfusion.py",
    "m10src/nano10.py",
    "m13/LOTTE_GATE_MANIFEST.json",
)


def _sha_bytes(blob):
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def registry_without_m20_amendment(path):
    """The registry bytes as they were before the dated M20 roster amendment.

    The amendment adds exactly two keys under `reserved` and rewrites `systems_included`.
    Reversing those three edits and re-serializing must reproduce the pre-amendment file
    byte-for-byte; anything else -- a moved weight, a changed seed, a different partition -- shows
    up as a hash mismatch here rather than being waved through as "the registry was amended".
    """
    import collections

    live = json.loads(Path(path).read_text(), object_pairs_hook=collections.OrderedDict)
    reserved = live["reserved"]
    original = reserved.pop("_systems_included_original", None)
    amendment = reserved.pop("_amended_2026_09_16", None)
    if original is None or amendment is None:
        raise ValueError("live registry carries no dated M20 roster amendment to reverse")
    reserved["systems_included"] = original
    return (json.dumps(live, indent=1) + "\n").encode()


def code_identity(repo=A.REPO):
    per_file = {name: A.sha256_file(Path(repo) / name) for name in CODE_FILES}
    return A.sha_json(per_file)


def _status_lines(cfg):
    return [line for line in A.sh_raw(
        cfg, "git", "status", "--porcelain=v1", "--untracked-files=all").splitlines()
            if line.strip()]


def _clean_pushed(cfg):
    problems = []
    dirty = _status_lines(cfg)
    # Stage A rewrites this receipt every run, including its timestamp, and the receipt is tracked
    # in git -- so at handoff it is a MODIFIED tracked file, not an untracked one, and allowing
    # only the `??` form refused the run it was written for (Astra review, 2026-09-18, P2). Both
    # forms are accepted: `_preencode(verify_shards=True)` validates the receipt's content before
    # `_begin` runs, `_begin` pins its sha256 into the manifest, and `_begin`'s commit includes it.
    allowed = {"?? results/m13_reserved_preencode.json",
               " M results/m13_reserved_preencode.json"}
    unexpected = [line for line in dirty if line not in allowed]
    if unexpected:
        problems.append(f"working tree has unexpected changes: {unexpected}")
    if PREENCODE.exists() and "?? results/m13_reserved_preencode.json" not in dirty \
            and not A.sh(cfg, "git", "ls-files", "--error-unmatch", str(PREENCODE)):
        problems.append("pre-encode receipt is neither the sole expected untracked file nor tracked")
    head = A.sh(cfg, "git", "rev-parse", "HEAD")
    upstream = A.sh(cfg, "git", "rev-parse", "@{u}")
    if head != upstream:
        problems.append("HEAD is not the exact pushed upstream tip")
    return problems


def _registration(cfg, conf):
    problems = []
    reserved = conf.get("reserved") or {}
    if reserved.get("systems_included") != list(R.SYSTEMS):
        problems.append("registered reserved systems differ from the executor")
    if [str(x).lower() for x in reserved.get("datasets", [])] != list(R.DATASETS):
        problems.append("registered reserved datasets differ from the executor")
    if reserved.get("alpha") != 0.0 or reserved.get("gate") is not None:
        problems.append("reserved batch is not registered as zero-alpha descriptive")
    if reserved.get("B") != 10_000 or reserved.get("seed") != 902:
        problems.append("reserved bootstrap constants changed")
    if float(reserved.get("min_free_gb", 0)) != 120:
        problems.append("reserved free-disk requirement changed")
    if shutil.disk_usage(cfg.repo).free / 1e9 < 120:
        problems.append("less than the registered 120 GB free")
    return problems


def _prior(cfg, conf):
    prior = json.loads(Path(cfg.result_path).read_text())
    if prior.get("end_status") not in ("INCOMPLETE_RESERVED", "COMPLETE"):
        raise ValueError(f"unexpected six-set end status {prior.get('end_status')!r}")
    decision = prior.get("decision_record") or {}
    if decision.get("reserved_batch_runs") is not True:
        raise ValueError("the frozen six-set decision did not trigger the reserved batch")
    if prior.get("registry_sha256") != A.sha256_file(cfg.registry_path):
        # The M20 roster amendment (R20/R22) deliberately changes the registry after the six-set
        # result pinned it. Demanding an unchanged hash would make a registered amendment
        # unexecutable; accepting any change would let a weight or threshold move unseen. So prove
        # mechanically that undoing the amendment reproduces the pinned bytes exactly.
        undone = registry_without_m20_amendment(cfg.registry_path)
        if _sha_bytes(undone) != prior.get("registry_sha256"):
            raise ValueError("six-set result and live registry differ by more than the registered "
                             "M20 roster amendment")
    if prior.get("freeze_sha256") != json.loads(cfg.state_path.read_text())["freeze_sha256"]:
        raise ValueError("six-set result and run manifest name different checkpoints")
    freeze = json.loads(Path(cfg.freeze_path).read_text())
    checkpoint = Path(cfg.repo) / freeze.get("checkpoint", "missing")
    if not checkpoint.is_file():
        raise ValueError("frozen Nano checkpoint is missing")
    checkpoint_sha = A.sha256_file(checkpoint)
    if (freeze.get("checkpoint_sha256") != checkpoint_sha or
            prior.get("freeze_sha256") != checkpoint_sha):
        raise ValueError("live frozen Nano checkpoint differs from the six-set result")
    return prior


def _preencode(cfg, verify_shards=True):
    record = json.loads(PREENCODE.read_text())
    if record.get("status") != "COMPLETE":
        raise ValueError("reserved document pre-encode is incomplete")
    # The pre-encode summary is keyed by DOCUMENT TOWER directory, not by system: three of the
    # eight systems share the Stella shards and two have no document vectors at all.
    for system in sorted({R.cache_dir_for(name) for name in R.DENSE_SYSTEMS}):
        for dataset in R.DATASETS:
            row = record.get("systems", {}).get(system, {}).get(dataset, {})
            path = Path(cfg.repo) / row.get("manifest", "missing")
            if row.get("status") != "COMPLETE" or not path.exists():
                raise ValueError(f"{system}/{dataset}: missing complete pre-encode manifest")
            if R.sha_file(path) != row.get("manifest_sha256"):
                raise ValueError(f"{system}/{dataset}: pre-encode manifest changed")
            vectors = R.ShardedVectors(path, verify=verify_shards)
            if vectors.n != row.get("n_docs") or len(vectors.paths) != row.get("n_shards"):
                raise ValueError(f"{system}/{dataset}: pre-encode summary differs from cache")
    return record


def preflight(cfg=None, verify_shards=True):
    cfg = cfg or A.production()
    conf = json.loads(cfg.registry_path.read_text())
    problems = _clean_pushed(cfg) + _registration(cfg, conf)
    try:
        prior = _prior(cfg, conf)
    except Exception as error:  # reported without opening protected data
        problems.append(f"six-set prerequisite: {type(error).__name__}: {error}")
        prior = None
    try:
        preencode = _preencode(cfg, verify_shards=verify_shards)
    except Exception as error:
        problems.append(f"pre-encode prerequisite: {type(error).__name__}: {error}")
        preencode = None
    reserved_cfg = replace(cfg, spent_tag=TAG, lock_path=LOCK)
    exists, where = A.spent_tag_exists(reserved_cfg, conf)
    if exists:
        problems.append(f"{TAG} already exists ({where}); fresh access is unavailable")
    return problems, prior, preencode


def _begin(cfg, conf, prior, preencode):
    implementation = A.sh(cfg, "git", "rev-parse", "HEAD")
    manifest = {
        "status": "READY",
        "implementation_commit": implementation,
        "code_sha256": code_identity(cfg.repo),
        "registry_sha256": A.sha256_file(cfg.registry_path),
        "freeze_sha256": prior["freeze_sha256"],
        "freeze_file_sha256": A.sha256_file(cfg.freeze_path),
        "six_result_sha256": A.sha256_file(cfg.result_path),
        "preencode_sha256": A.sha256_file(PREENCODE),
        "systems": list(R.SYSTEMS),
        "datasets": list(R.DATASETS),
        "spent_tag": TAG,
        "written": A.utcnow(),
        "scope": "descriptive reserved NDO-3 plus double-contaminated FEVER sensitivity; zero alpha",
    }
    A.write_atomic(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    A.ledger_append(cfg, f"\n- {A.utcnow()} — **RESERVED-RUN-BEGIN** implementation "
                    f"`{implementation[:12]}`; prior six result `{manifest['six_result_sha256'][:12]}`; "
                    f"systems {list(R.SYSTEMS)}; zero alpha.")
    ok, command, error = A.commit_and_push(
        cfg, [cfg.ledger_path, MANIFEST, PREENCODE],
        f"m13: RESERVED-RUN-BEGIN {manifest['six_result_sha256'][:12]}")
    if not ok:
        raise RuntimeError(f"reserved BEGIN not durable: {command}: {error}")
    begin = A.sh(cfg, "git", "rev-parse", "HEAD")
    ok, error = A.sh_ok(cfg, "git", "tag", "-a", TAG, "-m",
                       f"M13 reserved-four access spent {A.utcnow()} begin {begin[:12]}")
    if not ok:
        raise RuntimeError(f"could not create {TAG}: {error}")
    ok, error = A.sh_ok(cfg, "git", "push", "-q", "origin", f"refs/tags/{TAG}")
    if not ok:
        A.sh(cfg, "git", "tag", "-d", TAG)
        raise RuntimeError(f"could not push {TAG}; local tag removed, no protected read: {error}")
    print(f"[reserved13] access SPENT and pushed: {TAG} at {begin[:12]}", flush=True)
    return manifest, begin


def _continuation(cfg, conf):
    reserved_cfg = replace(cfg, spent_tag=TAG, lock_path=LOCK)
    exists, where = A.spent_tag_exists(reserved_cfg, conf)
    if not (exists and where == "origin") or not MANIFEST.exists():
        raise ValueError("reserved continuation requires the pushed tag and BEGIN manifest")
    manifest = json.loads(MANIFEST.read_text())
    begin = A.remote_tag_commit(reserved_cfg)
    problems = []
    if begin != A.sh(cfg, "git", "rev-parse", "HEAD"):
        problems.append("continuation HEAD is not the tagged RESERVED-RUN-BEGIN commit")
    if manifest.get("code_sha256") != code_identity(cfg.repo):
        problems.append("reserved executor code changed after the tag")
    if manifest.get("registry_sha256") != A.sha256_file(cfg.registry_path):
        problems.append("reserved registry changed after the tag")
    if manifest.get("preencode_sha256") != A.sha256_file(PREENCODE):
        problems.append("pre-encode receipt changed after the tag")
    if manifest.get("freeze_file_sha256") != A.sha256_file(cfg.freeze_path):
        problems.append("Nano freeze metadata changed after the tag")
    allowed = {
        f"?? {Path(cfg.scores_dir).relative_to(cfg.repo)}/reserved/{R.R20.slug(system)}.json"
        for system in R.SYSTEMS
    } | {
        f"?? {Path(cfg.scores_dir).relative_to(cfg.repo)}/reserved/{R.R20.slug(system)}.json.tmp"
        for system in R.SYSTEMS
    }
    unexpected = [line for line in _status_lines(cfg) if line not in allowed]
    if unexpected:
        problems.append(f"continuation checkout has unrelated changes: {unexpected}")
    if problems:
        raise ValueError("; ".join(problems))
    _preencode(cfg, verify_shards=True)
    return manifest, begin


def publish(cfg):
    """Re-commit and push an already-computed reserved result after a failed push.

    The transaction writes its result files and only then commits. A crash or a network failure in
    between leaves complete, durable-on-disk outputs that the ordinary entry point refuses to touch
    because `RESULT.exists()`. That refusal is right -- nothing may be re-scored -- but it left no
    way to publish what was already paid for. This path opens no payload, scores nothing, changes
    no number, and refuses unless the result is already present and already COMPLETE.

    It deliberately does NOT try to finish a finalization that crashed between writing RESULT and
    updating the six-set result. That window leaves the scores durable and the numbers intact; only
    the bookkeeping is unfinished, and completing it automatically would mean a second, weaker
    authentication path running after the access is spent. A human decides there instead
    (Sol re-review, 2026-09-18: the automatic version authenticated pointers, not content).
    """
    if not RESULT.exists():
        raise ValueError("there is no computed reserved result to publish")
    final = json.loads(Path(cfg.result_path).read_text())
    if final.get("end_status") != "COMPLETE" or "reserved" not in final:
        raise ValueError("the six-set result does not already carry a complete reserved report")
    if final["reserved"].get("manifest_sha256") != A.sha256_file(MANIFEST):
        raise ValueError("the reserved result does not match the BEGIN manifest on disk")
    ok, command, error = A.commit_and_push(
        cfg, [cfg.ledger_path, cfg.result_path, cfg.scores_dir / "reserved", RESULT],
        f"m13: RESERVED-RUN-END {A.sha256_file(RESULT)[:12]} (publish-only)")
    if not ok:
        raise RuntimeError(f"reserved result still not durable: {command}: {error}")
    print(json.dumps({"status": "PUBLISHED", "result_sha256": A.sha256_file(RESULT)}), flush=True)
    return 0


def run(cfg=None, preflight_only=False, publish_only=False):
    import score13 as S

    cfg = cfg or A.production()
    conf = json.loads(cfg.registry_path.read_text())
    reserved_cfg = replace(cfg, spent_tag=TAG, lock_path=LOCK)
    A.acquire_lock(reserved_cfg)
    if publish_only:
        return publish(cfg)
    exists, where = A.spent_tag_exists(reserved_cfg, conf)
    if preflight_only:
        problems, _prior_result, _preencode_result = preflight(cfg)
        print(json.dumps({"status": "REFUSED" if problems else "PASSED", "problems": problems}))
        return 2 if problems else 0
    if RESULT.exists():
        raise ValueError("completed reserved result already exists; preserve it")
    prior = _prior(cfg, conf)
    if exists:
        manifest, begin = _continuation(cfg, conf)
    else:
        problems = _clean_pushed(cfg) + _registration(cfg, conf)
        preencode = _preencode(cfg, verify_shards=True)
        if problems:
            raise ValueError("reserved preflight refused: " + "; ".join(problems))
        manifest, begin = _begin(cfg, conf, prior, preencode)

    cfg.extra.update(reserved_production=True, reserved_device="cuda", reserved_chunk=50_000,
                     reserved_identity={"begin_commit": begin,
                                        "code_sha256": manifest["code_sha256"],
                                        "registry_sha256": manifest["registry_sha256"],
                                        "spent_tag": TAG})
    summary = S.reserved_batch(cfg, conf, None)
    if summary.get("status") != "complete" or "contrasts" not in summary:
        raise RuntimeError("reserved batch did not return a complete descriptive report")
    # R22's archive needs the reserved queries and qrels. They are exported HERE, from the payload
    # objects this transaction already holds in memory, so the archiving pass never has to reopen
    # protected data. The exporter enforces that rather than assuming it: it reads only from the
    # process cache, reuses and re-hashes an archive an earlier attempt wrote when nothing is
    # cached, and refuses outright otherwise. Until 2026-09-18 this line claimed "no additional
    # protected read" while the exporter reopened all four payloads (Astra review, P1).
    archived = R.export_reserved_payload_archive(cfg, ARCHIVE_ROOT)
    record = {
        **summary,
        "reserved_payload_archive": {"root": str(ARCHIVE_ROOT.relative_to(cfg.repo)),
                                     "datasets": archived},
        "manifest_sha256": A.sha256_file(MANIFEST),
        "begin_commit": begin,
        "spent_tag": TAG,
        "prior_incomplete_result_sha256": manifest["six_result_sha256"],
        "preencode_sha256": manifest["preencode_sha256"],
        "written": A.utcnow(),
    }
    A.write_atomic(RESULT, json.dumps(record, indent=2) + "\n")
    updated = dict(prior)
    updated["original_end_status"] = prior["end_status"]
    updated["original_incomplete_result_sha256"] = manifest["six_result_sha256"]
    updated["reserved"] = record
    updated["end_status"] = "COMPLETE"
    updated["reserved_completed"] = A.utcnow()
    A.write_atomic(cfg.result_path, json.dumps(updated, indent=1) + "\n")
    A.ledger_append(cfg, f"- {A.utcnow()} — **RESERVED-RUN-END** result "
                    f"`{A.sha256_file(RESULT)[:16]}`; all three systems / four datasets complete; "
                    "descriptive only, zero alpha.")
    ok, command, error = A.commit_and_push(
        cfg, [cfg.ledger_path, cfg.result_path, cfg.scores_dir / "reserved", RESULT],
        f"m13: RESERVED-RUN-END {A.sha256_file(RESULT)[:12]}")
    if not ok:
        raise RuntimeError(f"reserved result not durable: {command}: {error}")
    print(json.dumps({"status": "COMPLETE", "contrasts": record["contrasts"]}), flush=True)
    return 0
