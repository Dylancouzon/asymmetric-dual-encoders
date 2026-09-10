"""M13 — the access machinery for M10's ONE six-set scoring transaction.

`m9src/final9.py`'s guard, ported to M10's paths and tag (`m10-six-spent`,
`results/m10_final_run.json`, `work/m10final.lock`) and parameterised through ONE `Config` object so
the synthetic rehearsal (`m13src/rehearse13.py`) can point every path, the origin URL and the tag at
fixtures while the production code below runs unmodified.

Three things changed on the way over, each deliberate:

1. **`seal_protected_paths` is FATAL on the executor path.** final9 printed a warning and continued,
   which made its own provenance claim unverifiable exactly when it mattered. Here the guard either
   installs or the run refuses. (`--recover` keeps the softer contract only insofar as it, too,
   refuses: there is no path that proceeds unsealed.)
2. **Preflight reads NO protected payload.** It checks the six's qids through
   `results/eval_manifest.json` against `results/perquery.json` — metadata and the comparator, never
   `results/frozen_eval/`. That is ruling R1's recommendation (`m13/STAGE1_DESIGN.md` §3); the
   payload-level checks live inside the transaction (`score13.payload_checks`), one switch away.
3. **`min_free_gb` is a Config field, defaulted to 120.** M10's registry has NO such field (M9/M7
   inherited the requirement in prose only — see `m13/EXECUTION.md`, "restore inherited 120 GB
   space requirement"). If R7 adds `min_free_gb` to the registry, `preflight` prefers it.

Everything else — flock over O_EXCL, the fail-closed spent-tag check with a pinned origin, the
BEGIN-commit-then-tag-then-push ordering, `write_atomic`'s fsync of the rename — is final9's, kept
because the failures it encodes were paid for once already.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "m7src", REPO / "m9src", REPO / "m10src", REPO / "m8src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BEGIN = "FINAL-RUN-BEGIN"
END = "FINAL-RUN-END"

# The six document caches, as `teacher.encode_cached` names them (`m7src/final_run.py:score_set`,
# `m12src/run_six_dbsf.py:66`): `final-six-<ds>-docs` at fp32.
DOC_CACHE_FMT = "final-six-{ds}-docs"


@dataclass
class Config:
    """Every path, name and injected component the transaction touches. Production defaults are the
    real ones; the rehearsal builds one of these pointing at `work/m13rehearse/`."""

    repo: Path = REPO
    registry_path: Path = REPO / "m10" / "final_run_registry.json"
    result_path: Path = REPO / "results" / "m10_final_run.json"
    scores_dir: Path = REPO / "results" / "m10_final_scores"
    ledger_path: Path = REPO / "m10" / "LEDGER.md"
    freeze_path: Path = REPO / "m10" / "FREEZE.json"
    perquery_path: Path = REPO / "results" / "perquery.json"
    manifest_path: Path = REPO / "results" / "eval_manifest.json"
    frozen_eval_dir: Path = REPO / "results" / "frozen_eval"
    enc_root: Path = REPO / "work" / "enc"
    lock_path: Path = REPO / "work" / "m10final.lock"
    spent_tag: str = "m10-six-spent"
    ledger_prefix: str = "m10"
    # None -> the registry's `origin_url`. The rehearsal pins its bare fixture remote here.
    origin_url: str | None = None
    min_free_gb: float = 120.0
    serving_parity_artifacts: tuple[str, ...] = ("results/m10_student_parity_box.json",)
    doc_cache_fmt: str = DOC_CACHE_FMT
    # "nano" -> m10src/final10; "m9" -> m9src/final_stats + m9src/final9.decide, untouched.
    decision_layer: str = "nano"
    # RULING R1 is pending. True = the payload-level checks (lengths, duplicates, qid-set equality
    # against the frozen payload) run INSIDE the transaction, first after the tag. Flip to False
    # only if the owner rules for M7's payload-reading preflight instead.
    payload_checks_inside: bool = True
    # Only these paths may be dirty when a post-tag continuation re-enters. Anything else is
    # undeclared output drift.
    allowed_drift: tuple[str, ...] = ()
    # --- injected components (None -> the production implementation in score13) ---------------
    corpus_reader: object = None        # (ds) -> (doc_ids, doc_texts); None -> final_run's HF read
    doc_vector_loader: object = None    # (cfg, ds, doc_texts) -> (n, d) array
    load_student: object = None         # (freeze_blob) -> obj with .encode(texts) -> ndarray
    load_anchor: object = None          # () -> obj with .encode(texts) -> ndarray
    reserved_encoder: object = None     # (cfg, system, ds) -> ...; None -> the refusing stub
    topk: int = 100
    chunk: int = 200_000
    extra: dict = field(default_factory=dict)

    @property
    def state_path(self) -> Path:
        return self.scores_dir / "run_state.json"

    def score_path(self, ds: str) -> Path:
        return self.scores_dir / f"{ds}.json"


def production() -> Config:
    return Config()


# ---------------------------------------------------------------------------- shell / hashing

def sh(cfg, *a):
    return subprocess.run(a, cwd=cfg.repo, capture_output=True, text=True).stdout.strip()


def sh_raw(cfg, *a):
    """`sh` without the strip. `git status --porcelain` puts a SPACE in column 1, and stripping the
    whole output eats the first line's status character — which silently shifted every parsed path
    by one (M7 kept `sh_raw` for the same reason)."""
    return subprocess.run(a, cwd=cfg.repo, capture_output=True, text=True).stdout


def sh_ok(cfg, *a):
    r = subprocess.run(a, cwd=cfg.repo, capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout).strip()


def branch_name(cfg):
    return sh(cfg, "git", "rev-parse", "--abbrev-ref", "HEAD")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(obj):
    """`m7src/hashing.sha` — sha256(json.dumps(obj, sort_keys=True)).

    This IS the manifest's convention: `scripts/freeze_eval_assets.py`:44 writes
    `"qids_sha256": sha(sorted(q_ids))` with `sha` defined at :22, and
    `scripts/verify_manifest.py`:34 re-derives it the same way. Re-implemented here rather than
    imported so `access13` does not drag `m7src`'s torch imports into a preflight.
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_atomic(path, text):
    """write_text truncates first, so a kill mid-write destroys the sole confirmatory result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(str(path.parent), os.O_RDONLY)          # the rename itself must be durable
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------- the boundary

def seal_protected_paths(fatal=True):
    """Install `m8src/paths_guard`. FATAL by default — final9 warned and continued.

    A boundary that degrades to a warning is a promise, and the executor's whole claim is that it
    cannot read a reserved payload by accident. `results/frozen_eval/<six>.json` is NOT protected
    (only `untouched-*` is), so the guard does not obstruct the six.
    """
    try:
        import paths_guard
        paths_guard.install()
        return True
    except Exception as e:                                       # noqa: BLE001 — reported verbatim
        if fatal:
            raise SystemExit(
                f"REFUSED: could not install m8src/paths_guard ({e!r}). The reserved-payload "
                f"boundary is not a promise this executor is allowed to make without it.")
        print(f"WARNING: paths_guard not installed ({e!r}); the boundary is NOT enforced.")
        return False


_LOCKS = {}


def acquire_lock(cfg):
    """flock, NOT O_EXCL-plus-staleness (final9 blocker 5): the kernel drops the lock when the
    holder dies, so there is no stale state to reclaim and no liveness to guess.

    Re-entrant WITHIN a process: flock is per open file description, so a second `open` of the same
    lock file in the same process conflicts with the first. One process holding it twice is not the
    hazard this guards (two processes scoring the six is), and refusing there would break a
    `--recover` that follows a run in the same process.
    """
    import fcntl
    key = str(Path(cfg.lock_path).resolve())
    if key in _LOCKS:
        return _LOCKS[key]
    Path(cfg.lock_path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(cfg.lock_path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"REFUSED: {cfg.lock_path} is flock-held by a live executor. Two "
                         f"concurrent runs could both score the six.")
    fh.truncate(0)
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    # Held for the process lifetime and never unlinked: unlinking while our descriptor is open
    # would let another process lock a fresh inode and score concurrently.
    _LOCKS[key] = fh
    return fh


def spent_tag_exists(cfg, conf):
    """-> (exists, where). FAILS CLOSED, and pins `origin` by URL (final9 blocker 2)."""
    want_url = cfg.origin_url or conf.get("origin_url")
    if not want_url:
        raise SystemExit("REFUSED: no `origin_url` (registry or config). The spend receipt's "
                         "witness must be a pinned remote.")
    got_url = sh(cfg, "git", "remote", "get-url", "origin")
    if got_url != want_url:
        raise SystemExit(f"REFUSED: origin is {got_url!r}, not the registered {want_url!r}. "
                         f"The spend receipt's witness must be the registered remote.")
    tag = cfg.spent_tag
    r = subprocess.run(["git", "ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
                       cwd=cfg.repo, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: cannot reach origin to check {tag} "
                         f"({(r.stderr or '').strip()[:200]}). Refusing to assume the access is "
                         f"unspent. Fix connectivity and retry.")
    if r.stdout.strip():
        return True, "origin"
    if sh(cfg, "git", "tag", "-l", tag):
        return True, "local-only"
    return False, ""


# ---------------------------------------------------------------------------- preflight

def _doc_cache_dirs(cfg, ds):
    """The `final-six-<ds>-docs-fp32-<12 hex>` directories, without deriving the corpus hash.

    Deriving the exact key needs the corpus text, i.e. a download; preflight stays metadata-only and
    checks that a plausible cache is present and shard-clean. `encode_cached(verify=True)` is what
    actually authenticates the bytes, inside the transaction.
    """
    stem = cfg.doc_cache_fmt.format(ds=ds)
    return sorted(cfg.enc_root.glob(f"{stem}-fp32-*"))


def _cache_problems(cfg, ds):
    dirs = _doc_cache_dirs(cfg, ds)
    if not dirs:
        return [f"no document cache `{cfg.doc_cache_fmt.format(ds=ds)}-fp32-*` under {cfg.enc_root}"]
    problems = []
    for d in dirs:
        man = d / "shards.json"
        if not man.exists():
            problems.append(f"{d.name}: no shards.json (its bytes were never hash-recorded)")
            continue
        try:
            shards = json.loads(man.read_text()).get("shards") or {}
        except json.JSONDecodeError:
            problems.append(f"{d.name}: shards.json is unreadable")
            continue
        if not shards:
            problems.append(f"{d.name}: shards.json records no shard")
        missing = [s for s in shards if not (d / f"shard_{s}.npy").exists()]
        if missing:
            problems.append(f"{d.name}: {len(missing)} recorded shard(s) missing on disk")
    # One cache per dataset is expected; several means an ambiguous corpus and is worth saying.
    if len(dirs) > 1 and not problems:
        problems.append(f"{ds}: {len(dirs)} document caches match "
                        f"{cfg.doc_cache_fmt.format(ds=ds)}-fp32-*; the corpus identity is ambiguous")
    return problems


def manifest_qid_check(cfg, conf, datasets):
    """The manifest-only qid check. Reads `eval_manifest.json` (committed metadata) and
    `perquery.json` (the comparator). NEVER `results/frozen_eval/`."""
    problems = []
    try:
        man = json.loads(cfg.manifest_path.read_text()).get("datasets") or {}
    except (OSError, json.JSONDecodeError) as e:
        return [f"{cfg.manifest_path.name} unreadable ({type(e).__name__})"]
    try:
        pq = json.loads(cfg.perquery_path.read_text()).get("datasets") or {}
    except (OSError, json.JSONDecodeError) as e:
        return [f"{cfg.perquery_path.name} unreadable ({type(e).__name__})"]
    for ds in datasets:
        if ds not in man:
            problems.append(f"eval_manifest.json has no `datasets.{ds}` entry")
            continue
        if ds not in pq:
            problems.append(f"perquery.json has no `{ds}` block")
            continue
        qids = pq[ds].get("qids") or []
        if not qids:
            problems.append(f"perquery.json `{ds}` has no qids")
            continue
        if any(not isinstance(q, str) for q in qids):
            problems.append(f"perquery.json `{ds}` has non-string qids; BEIR qids are strings and "
                            f"a JSON round-trip would re-sort them")
        if len(set(qids)) != len(qids):
            problems.append(f"perquery.json `{ds}` has {len(qids) - len(set(qids))} duplicate qid(s)")
        if qids != sorted(qids):
            # The comparator rows are aligned POSITIONALLY to this list and the file's own note
            # says "qids sorted lexicographically per dataset". A reorder leaves the qid-set hash
            # unchanged while silently re-pairing every score, which no other check can see.
            problems.append(f"perquery.json `{ds}` qids are not in sorted order; the system rows "
                            f"are aligned positionally and a reorder re-pairs every score")
        got = sha_json(sorted(qids))
        want = man[ds].get("qids_sha256")
        if want is None:
            problems.append(f"eval_manifest.json `{ds}` has no qids_sha256")
        elif got != want:
            problems.append(f"`{ds}`: perquery qid-set sha {got[:12]} != manifest {want[:12]} "
                            f"({len(qids)} qids here, manifest says {man[ds].get('n_queries')})")
        n = man[ds].get("n_queries")
        if isinstance(n, int) and n != len(qids):
            problems.append(f"`{ds}`: perquery has {len(qids)} qids, the manifest pins {n}")
        for sysname, row in (pq[ds].get("systems") or {}).items():
            if len(row) != len(qids):
                problems.append(f"perquery.json `{sysname}`/`{ds}` has {len(row)} values for "
                                f"{len(qids)} qids")
    return problems


def preflight(cfg, conf, infra_retry=False):
    """Everything static that must hold BEFORE the six are touched. Opens no protected payload.

    A failing preflight that read labels could be repeated indefinitely, reading protected data
    without ever consuming the access (M7 final-lock blocker 4). Paths, hashes, sizes and manifests.
    """
    problems = []
    tag = cfg.spent_tag

    exists, where = spent_tag_exists(cfg, conf)
    if exists and where == "origin":
        problems.append(f"{tag} exists on origin: the six-set access is already spent."
                        + (" --infra-retry is inadmissible: the failure did NOT precede the spend."
                           if infra_retry else ""))
    elif exists and where == "local-only":
        # origin absence is positively established here (ls-remote exited 0 and empty), so the
        # crash landed between tag creation and push — before the durable spend.
        if not infra_retry:
            problems.append(f"a LOCAL-ONLY {tag} exists while origin has none: a crash between tag "
                            f"creation and push. Rerun with --infra-retry, which clears the "
                            f"remnant; the access was never durably spent.")
        else:
            ok, err = sh_ok(cfg, "git", "tag", "-d", tag)
            print(f"[access13] cleared local-only {tag} remnant ({'ok' if ok else err})")
    if cfg.result_path.exists():
        problems.append(f"{cfg.result_path} already exists; a scored result must never be "
                        f"overwritten. Use --recover to recompute decisions from it.")
    if any(cfg.scores_dir.glob("*.json")) if cfg.scores_dir.exists() else False:
        problems.append(f"{cfg.scores_dir} already holds per-dataset scores; a fresh run must not "
                        f"start on top of them (post-tag continuation is a different mode).")

    if sh(cfg, "git", "status", "--porcelain"):
        problems.append("working tree is dirty; the freeze commit must be clean.")
    if sh(cfg, "git", "rev-parse", "HEAD") != sh(cfg, "git", "rev-parse", "@{u}"):
        problems.append("HEAD is not pushed to its upstream.")

    if not cfg.freeze_path.exists():
        problems.append(f"{cfg.freeze_path} is missing: the candidate is not frozen.")
    else:
        fz = json.loads(cfg.freeze_path.read_text())
        ckpt = cfg.repo / fz["checkpoint"]
        if not ckpt.exists():
            problems.append(f"frozen checkpoint {ckpt} is missing.")
        elif sha256_file(ckpt) != fz["checkpoint_sha256"]:
            problems.append(f"checkpoint sha256 does not match {cfg.freeze_path.name}.")

    # The comparator snapshot is the SOURCE of every conjunct's b-side. Opening it read-only does
    # not pin its contents, so the digest is verified explicitly (M9 final-lock blocker 6).
    want = conf["comparator_source"]["sha256"]
    if not cfg.perquery_path.exists():
        problems.append(f"{cfg.perquery_path} is missing: the frozen comparator vectors.")
    else:
        got = sha256_file(cfg.perquery_path)
        if got != want:
            problems.append(f"{cfg.perquery_path.name} sha256 {got[:12]} != registered {want[:12]}.")

    if conf.get("ratified_by_owner") is not True:
        problems.append("final_run_registry.ratified_by_owner is not true: the decision lock has "
                        "not been ratified by the owner.")

    min_gb = conf.get("min_free_gb", cfg.min_free_gb)     # R7 may add the field; config until then
    try:
        free_gb = shutil.disk_usage(cfg.repo).free / 1e9
        if free_gb < min_gb:
            problems.append(f"only {free_gb:.1f} GB free under {cfg.repo}; the inherited "
                            f"requirement is {min_gb:.0f} GB (document caches and the reserved "
                            f"batch need the room).")
    except OSError as e:
        problems.append(f"cannot stat free space under {cfg.repo} ({e})")

    for rel in cfg.serving_parity_artifacts:
        if not (cfg.repo / rel).exists():
            problems.append(f"serving-parity artifact {rel} is missing; the shipped path was never "
                            f"shown to reproduce the scored one.")

    datasets = conf["partitions"]["all6"]
    for ds in datasets:
        problems += _cache_problems(cfg, ds)
    problems += manifest_qid_check(cfg, conf, datasets)
    return problems


# ---------------------------------------------------------------------------- spend, durably

def ledger_append(cfg, line):
    cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.ledger_path, "a") as f:
        f.write(line + "\n")


def spend_access(cfg, freeze_sha):
    """Push the receipt BEFORE the first protected read. Push failure aborts before any access.

    -> the BEGIN commit sha, which a post-tag continuation must later match HEAD against.
    """
    stamp = datetime.now(timezone.utc).isoformat()
    tag = cfg.spent_tag
    ledger_append(cfg, f"\n- {stamp} — **{BEGIN}** freeze `{freeze_sha[:12]}` "
                       f"pid {os.getpid()} host `{os.uname().nodename}`")
    for cmd in (("git", "add", str(cfg.ledger_path)),
                ("git", "commit", "-q", "-m", f"{cfg.ledger_prefix}: {BEGIN} {freeze_sha[:12]}"),
                ("git", "push", "-q", "origin", "HEAD")):
        ok, err = sh_ok(cfg, *cmd)
        if not ok:
            raise SystemExit(f"ABORT before any protected read: {' '.join(cmd)} failed ({err}). "
                             f"No access consumed.")
    # Positively verify the BEGIN entry reached origin: pushing an UNCHANGED HEAD would otherwise
    # satisfy the check while the ledger entry existed nowhere durable (final9 blocker 1).
    head = sh(cfg, "git", "rev-parse", "HEAD")
    if head != sh(cfg, "git", "rev-parse", "origin/" + branch_name(cfg)):
        raise SystemExit("ABORT before any protected read: HEAD is not the pushed origin tip after "
                         "the BEGIN commit. No access consumed.")
    ok, err = sh_ok(cfg, "git", "tag", "-a", tag, "-m",
                    f"M10 six-set access spent {stamp} freeze {freeze_sha[:12]}")
    if not ok:
        raise SystemExit(f"ABORT before any protected read: could not create {tag} ({err}). "
                         f"No access consumed.")
    ok, err = sh_ok(cfg, "git", "push", "-q", "origin", f"refs/tags/{tag}")
    if not ok:
        sh(cfg, "git", "tag", "-d", tag)
        raise SystemExit(f"ABORT before any protected read: could not PUSH {tag} ({err}). Local tag "
                         f"removed; no access consumed. The receipt must be durable on origin "
                         f"before the six are opened.")
    print(f"[access13] access SPENT and pushed: {tag}. Everything from here is irreversible.")
    return head


def commit_and_push(cfg, paths, message):
    """-> (ok, failing command, stderr). The caller decides what a failure means; a decision that
    is on disk but not on origin must not be acted on."""
    for cmd in (("git", "add", *[str(p) for p in paths]),
                ("git", "commit", "-q", "-m", message),
                ("git", "push", "-q", "origin", "HEAD")):
        ok, err = sh_ok(cfg, *cmd)
        if not ok:
            return False, " ".join(cmd), err
    return True, "", ""


def utcnow():
    return datetime.now(timezone.utc).isoformat()
