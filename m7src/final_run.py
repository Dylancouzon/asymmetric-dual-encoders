"""THE final run. One shot, guarded.

Refuses to start unless:
  * the working tree is clean and HEAD equals the freeze commit PUSHED to origin (the remote
    timestamp is the external witness),
  * m7/LEDGER.md holds no prior final-run entry (unless --infra-retry, which is allowed once
    per crash with identical commit, config and inputs and is itself logged).

It is the sole reader of six-set and untouched-final qrels, reads them only from
results/frozen_eval/ after verifying eval_manifest.json corpus hashes against a fresh HF
download, scores the six first and the untouched-final sets after, and appends every access
to m7/LEDGER.md itself.

Confirmatory decisions are exactly three, one-sided, Holm step-down at family alpha = 0.025:
  C1  released int8 dense table   >  lr-dense-pertask (0.4583)   -- Tier 2, the release bar
  C2  released int8 dense table   >  bm25 (0.4174)               -- Tier 3
  C3  released system             >  opensearch-doc-v3-gte (0.4868) -- Tier 1, the aim
Everything else this script prints is exploratory and labeled as such.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

import boot
import fusion
from _paths import REPO
from core import DATASETS
from evalkit import per_query_ndcg, topk_ids_scores
import freeze
from hashing import sha
from table import Preproc, load_table
import teacher
from teacher import QUERY_PREFIX, encode_cached

LEDGER = REPO / "m7" / "LEDGER.md"
FROZEN = REPO / "results" / "frozen_eval"
MANIFEST = REPO / "results" / "eval_manifest.json"
UNTOUCHED = ["fever", "dbpedia-entity", "cqadup-android", "cqadup-english"]
CONFIRMATORY = {"C1_int8_table_gt_lr_dense_pertask": ("int8-table", "lr-dense-pertask"),
                "C2_int8_table_gt_bm25": ("int8-table", "bm25"),
                "C3_released_system_gt_opensearch": ("released-system", "opensearch-doc-v3-gte")}
ALPHA = 0.025


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout.strip()


def sh_raw(*a):
    """`sh` without the strip. Column-oriented git output loses its first field to `.strip()`."""
    return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout


def ledger(line):
    with open(LEDGER, "a") as f:
        f.write(line.rstrip() + "\n")


FREEZE_TAG = "m7-freeze"
# The number of FINAL-RUN-BEGIN entries the ledger may hold. The first is the run itself, the
# second is its one permitted infrastructure retry. `n_begin > MAX` let a THIRD through, because
# the second retry saw n_begin == 2 and `2 > 2` is false -- three readings where the docstring
# promised two (Codex review #4, BLOCKER 2).
MAX_BEGINS = 2
SIX = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]
OUT = REPO / "results" / "m7_final_run.json"
# The durable, external spent receipt. OUT is untracked and the ledger is an editable text file,
# so deleting the result and trimming the ledger's completion lines resurrected the one shot for
# --infra-retry (Codex pre-freeze review 2026-08-28, BLOCKER 2). An annotated tag pushed to origin
# survives both; the guard refuses a scoring run while it exists, locally or remotely.
SPENT_TAG = "m7-six-spent"
# One process at a time: the guard is read-only and the first durable marker used to come only
# after preflight, so two concurrent launches could both pass and both score (BLOCKER 3).
LOCK = REPO / "work" / ".final-run.lock"


def acquire_lock():
    LOCK.parent.mkdir(exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                pid = int(LOCK.read_text().split()[0])
            except (ValueError, IndexError, OSError):
                pid = None
            alive = False
            if pid is not None:
                try:
                    os.kill(pid, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
            if alive:
                raise SystemExit(f"FINAL RUN REFUSED: another final_run process (pid {pid}) holds "
                                 f"{LOCK}. Two concurrent runs could both score the six.")
            if attempt == 1:
                print(f"[final_run] removing stale lock {LOCK} (pid {pid} is gone)")
                LOCK.unlink(missing_ok=True)
    raise SystemExit(f"FINAL RUN REFUSED: could not acquire {LOCK}")


def spent_tag_exists():
    """-> (exists, where). Checks origin first (the external witness), then local tags."""
    remote = sh_raw("git", "ls-remote", "origin", f"refs/tags/{SPENT_TAG}",
                    f"refs/tags/{SPENT_TAG}^{{}}").strip()
    if remote:
        return True, "origin"
    if sh("git", "tag", "-l", SPENT_TAG):
        return True, "local"
    return False, ""


def write_atomic(path, text):
    """Write, fsync, rename. `write_text` truncates first, so a kill or a full disk mid-write
    destroys the sole confirmatory result and leaves a file no mode can parse (Codex review #4,
    BLOCKER 1)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    d = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(d)
    finally:
        os.close(d)


def six_already_scored(fz):
    """-> (spent, why). The one-shot access is SPENT the moment a result file holding a `six`
    block for this frozen table exists, whatever the ledger says. The ledger is a text file the
    operator can edit; this is the artifact itself, and `--infra-retry` must not get past it
    (Codex review #4, BLOCKER 3)."""
    if not OUT.exists():
        return False, ""
    try:
        blob = json.loads(OUT.read_text())
    except json.JSONDecodeError:
        return True, (f"{OUT.name} exists but is unparseable. A confirmatory run has already "
                      "written there; recover it from git or from the crash, do not rerun.")
    if not blob.get("six"):
        return False, ""
    same = (blob.get("freeze") or {}).get("table_sha256") == fz["table_sha256"]
    return True, (f"{OUT.name} already holds scored `six` results for "
                  + ("this frozen table" if same else "a DIFFERENT table"))


def guard(freeze_hash, infra_retry, branch, fz, untouched_only=False):
    """`untouched_only` resumes the non-confirmatory tail after the six are already scored and on
    disk. It is the ONLY way past the completed-run refusal, it refuses unless the six's results
    are present for this exact frozen table, and it never re-scores them."""
    problems = []
    # BENCH_DATASETS is an env var with an M2-era five-dataset default; a stale export in the
    # same shell would silently redefine "the six" and the macro computed over it.
    if list(DATASETS) != SIX:
        problems.append(f"DATASETS is {list(DATASETS)}, not the six (check BENCH_DATASETS)")
    # The scorer itself appends to the ledger and the access trail, so a crashed attempt leaves
    # those two files changed. An infra retry must tolerate exactly that and nothing else
    # (review #2 MAJOR 20: the old guard could never pass after any post-BEGIN crash).
    ALLOWED_DRIFT = {"m7/LEDGER.md", "m7/SIX_ACCESS.log"}
    if untouched_only:      # the six's results are already written and are necessarily dirty
        ALLOWED_DRIFT = ALLOWED_DRIFT | {"results/m7_final_run.json"}
    # `git status --porcelain` is "XY PATH" with XY exactly two columns, so the path starts at 3 --
    # but `sh` strips the whole output, which eats the leading space of the FIRST line when its
    # status is " M". That silently truncated the first dirty path ("m7/LEDGER.md" -> "7/LEDGER.md"),
    # so ALLOWED_DRIFT never matched it and a legitimate --infra-retry would have been refused for
    # the very file the retry is allowed to touch. Split without stripping the leading column.
    dirty = [l[3:].strip() for l in sh_raw("git", "status", "--porcelain").splitlines() if l.strip()]
    stray_dirty = [d for d in dirty if d not in ALLOWED_DRIFT]
    if stray_dirty:
        problems.append(f"working tree is not clean beyond the scorer's own files: {stray_dirty}")
    # A first run demands a spotless tree. A retry and a tail-resume are both expected to find the
    # scorer's own files modified, and `stray_dirty` above already refuses anything else.
    if dirty and not (infra_retry or untouched_only):
        problems.append("working tree is not clean")
    head = sh("git", "rev-parse", "HEAD")
    if head != freeze_hash and not infra_retry:
        problems.append(f"HEAD {head[:12]} != freeze commit {freeze_hash[:12]}")
    remote = sh("git", "ls-remote", "origin", f"refs/heads/{branch}").split("\t")[0]
    if remote != freeze_hash:
        problems.append(f"freeze commit is not pushed: origin/{branch} is {remote[:12]}")
    # The freeze commit is resolved from an immutable pushed TAG, not taken on the caller's word.
    # `--freeze-hash` was the only identification of "the reviewed freeze commit", so any clean
    # pushed HEAD could be declared one (Codex one-shot-path review, BLOCKER 3).
    #
    # PEEL THE TAG. For an ANNOTATED tag -- which is exactly what the documented procedure
    # `git tag -a m7-freeze ...` creates -- `refs/tags/X` names the tag OBJECT, not the commit,
    # so comparing it with a commit hash could never match and the final run could never have
    # started. `refs/tags/X^{}` is the peeled commit and exists only for annotated tags, so query
    # both and prefer the peeled one (Codex review #4, BLOCKER 6).
    tag_refs = {}
    for line in sh_raw("git", "ls-remote", "origin", f"refs/tags/{FREEZE_TAG}",
                       f"refs/tags/{FREEZE_TAG}^{{}}").splitlines():
        if "\t" in line:
            h, ref = line.split("\t", 1)
            tag_refs[ref.strip()] = h.strip()
    tagged = (tag_refs.get(f"refs/tags/{FREEZE_TAG}^{{}}")      # annotated: the peeled commit
              or tag_refs.get(f"refs/tags/{FREEZE_TAG}")        # lightweight: the commit itself
              or "")
    if not tagged:
        problems.append(f"no pushed tag `{FREEZE_TAG}`: the freeze commit must be marked by an "
                        f"immutable pushed tag, not asserted on the command line. "
                        f"`git tag -a {FREEZE_TAG} -m ... <commit> && git push origin {FREEZE_TAG}`")
    elif tagged != freeze_hash:
        problems.append(f"--freeze-hash {freeze_hash[:12]} != the pushed tag `{FREEZE_TAG}` "
                        f"({tagged[:12]}); the tag is authoritative")

    text = LEDGER.read_text()
    prior_hashes = re.findall(r"FINAL-RUN-BEGIN freeze=([0-9a-f]{40}) table=([0-9a-f]{64})", text)
    n_complete = len(re.findall(r"FINAL-RUN complete in", text))
    n_begin = len(prior_hashes)
    if n_complete and not untouched_only:
        # A COMPLETE run is the one shot. Retry existed for an aborted run; it did not check
        # whether the prior run had finished, so a completed result could be re-rolled forever.
        problems.append(f"m7/LEDGER.md records {n_complete} COMPLETED final run(s). There is "
                        "exactly one confirmatory access and it has been spent; a further run is "
                        "a NEW milestone with its own pre-registration, never a retry.")
    # THE ARTIFACT, NOT THE LEDGER, IS WHAT SAYS THE ACCESS IS SPENT. The ledger is a text file
    # the operator can edit; deleting one "FINAL-RUN complete in" line and passing --infra-retry
    # used to buy a second scoring of the six (Codex review #4, BLOCKER 3).
    spent, why = six_already_scored(fz)
    if spent and not untouched_only:
        problems.append(f"the confirmatory access is SPENT: {why}. A further run is a NEW milestone "
                        "with its own pre-registration, never a retry. To finish the "
                        "non-confirmatory tail, use --untouched-only.")
    # The receipt survives a deleted result file and an edited ledger (BLOCKER 2): if the spent
    # tag exists anywhere, the six have been scored, whatever the local files now say.
    tag_spent, where = spent_tag_exists()
    if tag_spent and not untouched_only:
        problems.append(f"the spent receipt `{SPENT_TAG}` exists ({where}): the confirmatory "
                        "access has been used. A further run is a NEW milestone with its own "
                        "pre-registration.")
    if untouched_only:
        # This mode may only ADD the non-confirmatory tail to a result that already exists. It
        # must never be a way to re-score the six -- and it keys on the RESULT FILE, not on the
        # ledger marker, so a crash between the result write and the marker append does not wedge
        # the run in a state no mode accepts (Codex review #4, BLOCKER 1).
        if infra_retry:
            problems.append("--untouched-only and --infra-retry are mutually exclusive")
        if not OUT.exists():
            problems.append("--untouched-only with no results/m7_final_run.json: the six have not "
                            "been scored, and this mode never scores them")
        else:
            try:
                prior = json.loads(OUT.read_text())
            except json.JSONDecodeError as e:
                prior = {}
                problems.append(f"--untouched-only: {OUT.name} is unparseable ({e}); recover the "
                                "confirmatory result before appending anything to it")
            if not prior.get("six"):
                problems.append("--untouched-only: results/m7_final_run.json holds no `six` block")
            if (prior.get("freeze") or {}).get("table_sha256") != fz["table_sha256"]:
                problems.append("--untouched-only: the existing result was produced with a "
                                "different table than the one now frozen")
            # The six's scores and the tier decisions in that file must be the ones the ledger
            # recorded when they were written. Without this, a caller could edit the confirmatory
            # numbers and have --untouched-only rewrite them as an apparently resumed result.
            digest = sha({"six": prior.get("six"), "confirmatory": prior.get("confirmatory"),
                          "holm": prior.get("holm")})
            recorded = re.findall(r"FINAL-RUN-SIX-SHA256 ([0-9a-f]{64})", text)
            if not recorded:
                problems.append("--untouched-only: the ledger records no FINAL-RUN-SIX-SHA256 for "
                                "the confirmatory block, so it cannot be shown unedited")
            elif digest != recorded[-1]:
                problems.append(f"--untouched-only: the confirmatory block in {OUT.name} does not "
                                f"match the digest the ledger recorded when it was written "
                                f"({digest[:12]} vs {recorded[-1][:12]}). It has been edited.")
    if infra_retry and n_begin >= MAX_BEGINS:
        problems.append(f"the ledger already holds {n_begin} FINAL-RUN-BEGIN entries and the cap is "
                        f"{MAX_BEGINS}. Repeated infrastructure failure is a reason to fix the "
                        "infrastructure, not to keep drawing from the six.")
    if prior_hashes and not (infra_retry or untouched_only):
        problems.append("m7/LEDGER.md already holds a final-run entry; a code fix requires a NEW "
                        "pushed freeze commit, and no later run may be relabeled as final")
    if infra_retry:
        if not prior_hashes:
            problems.append("--infra-retry with no prior FINAL-RUN entry in the ledger")
        else:
            pf, pt = prior_hashes[-1]
            if pt != fz["table_sha256"]:
                problems.append("--infra-retry with a different table than the aborted run")
            # The retry must be the SAME COMMIT, not merely a tree that differs only in allowed
            # paths -- an empty commit, or one touching only the ledger, used to pass (Codex
            # review #4, MAJOR 7).
            if pf != freeze_hash:
                problems.append(f"--infra-retry names freeze commit {freeze_hash[:12]} but the "
                                f"aborted run began at {pf[:12]}; a retry is the same commit")
            if pf != head:
                problems.append(f"--infra-retry: HEAD {head[:12]} is not the aborted run's freeze "
                                f"commit {pf[:12]}")
            # only the ledger may have changed since the aborted run's freeze commit
            changed = [l for l in sh("git", "diff", "--name-only", f"{pf}..HEAD").splitlines() if l]
            stray = [c for c in changed if c not in ALLOWED_DRIFT]
            if stray:
                problems.append("--infra-retry is for infrastructure only, but these files changed "
                                f"since the aborted run's freeze commit: {stray}. A code change "
                                "requires a new pushed freeze commit with the diff and its "
                                "classification.")
    if problems:
        print("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
        sys.exit(2)
    return head


def verify_and_load(ds, kind):
    """kind: 'six' | 'untouched'. Verifies corpus hashes, then reads labels from frozen_eval only."""
    key = {"six": "datasets", "untouched": "m7_untouched_final"}[kind]
    man = json.loads(MANIFEST.read_text())[key][ds]
    if ds.startswith("cqadup-"):
        import devsuite
        doc_ids, doc_texts, *_ = devsuite.load(ds)
    else:
        # Corpus ONLY. This used `load_beir` for the six, which also downloads and parses fresh
        # test qrels -- discarded, but it made "labels are read from frozen_eval only" false
        # (Codex pre-freeze review 2026-08-28, MINOR). The trail entry is kept.
        if kind == "six":
            from core import _m7_access_trail
            _m7_access_trail(ds)
        from datasets import load_dataset
        from core import doc_text
        corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
        doc_ids, doc_texts = [str(x) for x in corpus["_id"]], [doc_text(r) for r in corpus]
    from hashing import sha_stream_list
    for field, got in (("corpus_ids_sha256", sha_stream_list(doc_ids)),
                       ("corpus_text_sha256", sha_stream_list(doc_texts)),
                       ("n_docs", len(doc_ids))):
        if man[field] != got:
            print(f"FINAL RUN ABORTED: {ds}.{field} mismatch vs the frozen manifest")
            sys.exit(3)
    name = f"{ds}.json" if kind == "six" else f"untouched-{ds}.json"
    froz = json.loads((FROZEN / name).read_text())
    q_ids = sorted(froz["queries"])
    ledger(f"- {datetime.now(timezone.utc).isoformat()} — **FINAL-RUN access** ({kind}) `{ds}`: "
           f"{len(doc_ids):,} docs / {len(q_ids):,} queries, corpus hashes verified against a fresh "
           f"HF download, labels read from `results/frozen_eval/{name}`.")
    return doc_ids, doc_texts, q_ids, [froz["queries"][q] for q in q_ids], froz["qrels"]


def score_set(ds, kind, table_path, pre, fusion_spec, encode_dtype=torch.float32):
    doc_ids, doc_texts, q_ids, q_texts, qrels = verify_and_load(ds, kind)
    # verify=True: the document vectors behind a confirmatory number must come from bytes this
    # process wrote or re-hashed. The cache lives under a gitignored, mutable work/ tree, and
    # `encode_cached` used to skip any shard that merely existed while `combined.f16` was accepted
    # on byte size alone (Codex one-shot-path review 2026-08-28, MAJOR 4). A cache that cannot be
    # authenticated now aborts and names the directory to delete, rather than being reused.
    dv = encode_cached(f"final-{kind}-{ds}-docs", doc_texts, prefix="", dtype=encode_dtype,
                       verify=True)
    tag = "six" if kind == "six" else "unt"
    tqv = np.asarray(encode_cached(f"final-{tag}-{ds}-q-pfx", q_texts, prefix=QUERY_PREFIX,
                                   dtype=encode_dtype, verify=True), dtype=np.float32)
    chunk = 200_000
    out, runs = {}, {}
    for variant in ("fp16", "int8"):
        m = load_table(table_path, variant=variant)
        runs[f"{variant}-table"] = topk_ids_scores(m.encode(q_texts, pre), dv, doc_ids,
                                                  k=fusion.DEPTH, chunk=chunk, qids=q_ids)
        del m
        torch.cuda.empty_cache()
    runs["bm25"] = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts)  # the shared builder, B5
    runs["teacher-symmetric"] = topk_ids_scores(tqv, dv, doc_ids, k=fusion.DEPTH, chunk=chunk, qids=q_ids)
    if fusion_spec:
        runs["fusion"] = fusion.apply_frozen(fusion_spec, runs["int8-table"], runs["bm25"])
    for k, r in runs.items():
        out[k] = per_query_ndcg(r, qrels)
    return out


def preflight(kinds=("six", "untouched")):
    """Everything static that must hold, checked BEFORE the six are touched.

    Exists because the failure it prevents is the worst one available: `UNTOUCHED` named four
    datasets while the manifest and `frozen_eval/` held two, so this script would have scored the
    six -- consuming the ONE confirmatory access -- computed the tier decisions, scored two
    untouched sets, then died on a KeyError and written NOTHING. The access would have been spent
    and no result produced. Found by the pre-freeze Codex review, 2026-08-28.

    So: every manifest key, every frozen payload, every required field, for all ten datasets,
    before `FINAL-RUN-BEGIN`. A missing asset must cost a second, not the deliverable.

    `kinds`: a --untouched-only resume passes ("untouched",) and this function then never opens a
    six-set payload -- it used to open and hash all six on every resume, making the advertised
    "never goes anywhere near the six" false (Codex pre-freeze review 2026-08-28, BLOCKER 4).
    Reading a frozen payload is not a scoring access, but it is a read of confirmatory labels and
    is therefore logged to m7/SIX_ACCESS.log below rather than left unrecorded (BLOCKER 5).
    """
    man = json.loads(MANIFEST.read_text())
    need = {"six": ("datasets", DATASETS, lambda d: f"{d}.json"),
            "untouched": ("m7_untouched_final", UNTOUCHED, lambda d: f"untouched-{d}.json")}
    need = {k: v for k, v in need.items() if k in kinds}
    if "six" in kinds:
        with open(REPO / "m7" / "SIX_ACCESS.log", "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} final_run.preflight read+hashed "
                    "the six frozen query/qrels payloads (verification only, nothing scored)\n")
    fields = ("n_docs", "n_queries", "corpus_ids_sha256", "corpus_text_sha256",
              "qids_sha256", "qrels_sha256")
    problems = []
    for kind, (key, names, fname) in need.items():
        sect = man.get(key) or {}
        for ds in names:
            if ds not in sect:
                problems.append(f"{kind} `{ds}`: no `{key}` entry in eval_manifest.json")
                continue
            missing = [f for f in fields if f not in sect[ds]]
            if missing:
                problems.append(f"{kind} `{ds}`: manifest entry lacks {missing}")
            fp = FROZEN / fname(ds)
            if not fp.exists():
                problems.append(f"{kind} `{ds}`: {fp.relative_to(REPO)} missing")
                continue
            try:
                froz = json.loads(fp.read_text())
            except Exception as e:
                problems.append(f"{kind} `{ds}`: {fp.name} unreadable ({type(e).__name__})")
                continue
            if not isinstance(froz.get("queries"), dict) or not froz["queries"]:
                problems.append(f"{kind} `{ds}`: {fp.name} has no `queries` mapping")
            if not isinstance(froz.get("qrels"), dict) or not froz["qrels"]:
                problems.append(f"{kind} `{ds}`: {fp.name} has no `qrels` mapping")
            # the payload the scorer actually reads must match the hashes the manifest pins.
            # verify_and_load checks the CORPUS only; queries and qrels were never verified, so
            # changed query text with unchanged qids would have scored silently (Codex MAJOR 3).
            if isinstance(froz.get("queries"), dict) and isinstance(froz.get("qrels"), dict):
                # qtexts_sha256 binds the QUERY TEXT. Without it the previous comment here was
                # false: changing a value in `queries` while leaving its key and the qrels alone
                # passed every check and was then encoded and scored (Codex review #4, MAJOR 1).
                for field, got in (("qids_sha256", sha(sorted(froz["queries"]))),
                                   ("qtexts_sha256",
                                    sha([froz["queries"][q] for q in sorted(froz["queries"])])),
                                   ("qrels_sha256", sha(froz["qrels"]))):
                    if field not in sect[ds]:
                        problems.append(f"{kind} `{ds}`: the manifest has no `{field}`; regenerate "
                                        "it with scripts/freeze_eval_assets.py before freezing")
                    elif sect[ds][field] != got:
                        problems.append(f"{kind} `{ds}`: {fp.name} {field} mismatch vs the "
                                        f"frozen manifest")
    pq = REPO / "results" / "perquery.json"
    if "six" not in kinds:
        pass    # the tail makes no confirmatory comparison, so the comparator rows are not needed
    elif not pq.exists():
        problems.append("results/perquery.json missing: the frozen comparator vectors")
    else:
        blob = json.loads(pq.read_text())
        for _, (a_name, b_name) in CONFIRMATORY.items():
            for sysname in (a_name, b_name):
                if sysname in ("int8-table", "released-system"):
                    continue      # produced by this run, not read from the frozen file
                for ds in DATASETS:
                    d = blob["datasets"].get(ds, {})
                    row = d.get("systems", {}).get(sysname)
                    if row is None:
                        problems.append(f"perquery.json: comparator `{sysname}` has no row for "
                                        f"`{ds}`")
                        continue
                    # Presence was all that was checked, so a row one qid short passed preflight
                    # and then aborted `strict=True` AFTER the six had been scored -- spending the
                    # access for nothing (Codex review #4, MAJOR 6).
                    qids = d.get("qids") or []
                    if len(row) != len(qids):
                        problems.append(f"perquery.json: `{sysname}`/`{ds}` has {len(row)} values "
                                        f"for {len(qids)} qids")
                    if len(set(qids)) != len(qids):
                        problems.append(f"perquery.json: `{ds}` has duplicate qids")
                    froz_p = FROZEN / f"{ds}.json"
                    if froz_p.exists():
                        want = set(json.loads(froz_p.read_text()).get("queries") or {})
                        if want and set(qids) != want:
                            problems.append(
                                f"perquery.json: `{ds}` qids do not match the frozen payload "
                                f"({len(set(qids) - want)} extra, {len(want - set(qids))} missing) "
                                "-- strict alignment would abort after the six were scored")
    if problems:
        # NOT "not touched": the frozen payloads WERE read and hashed above (that read is logged
        # to SIX_ACCESS.log); what has not happened is any scoring or any new-model number.
        print("FINAL RUN REFUSED by preflight (nothing was scored; frozen payloads were read for "
              "hash verification only):\n  - " + "\n  - ".join(problems))
        sys.exit(4)
    print(f"preflight OK ({'+'.join(kinds)}): manifest entries, frozen payloads and comparator "
          f"rows all present and hash-matched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-hash", required=True)
    ap.add_argument("--branch", default="m7-query-encoder")
    ap.add_argument("--infra-retry", action="store_true")
    ap.add_argument("--untouched-only", action="store_true",
                    help="score ONLY the non-confirmatory untouched-final tail, appending to an "
                         "existing results/m7_final_run.json. For resuming after a crash in that "
                         "stage; it never re-scores the six.")
    # The untouched-final four are RESERVED for M8 (registered 2026-08-28 before any six-set
    # number, instructions-m8.md): the default run scores the six ONLY and never opens an
    # untouched payload. Scoring the tail is Dylan's explicit override and burns the sets for M8.
    ap.add_argument("--run-untouched-tail", action="store_true",
                    help="OVERRIDE the M8 reservation and score the untouched-final tail after "
                         "the six (10.1M docs, tens of hours). Once scored they are "
                         "development-visible and no longer fresh for M8.")
    a = ap.parse_args()

    # One process at a time, from the first moment (BLOCKER 3): the guard is read-only, so two
    # concurrent launches could both pass it and both score. Stale locks (dead pid) self-clear.
    acquire_lock()
    import atexit
    atexit.register(lambda: LOCK.unlink(missing_ok=True))

    # everything decisive comes from the committed freeze manifest, not the command line
    fz = freeze.load_and_verify()
    spec = fz["fusion"]
    pre = Preproc(**fz["preproc"])
    if pre.fingerprint() != fz["preproc_fingerprint"]:
        raise SystemExit("FINAL RUN REFUSED: preproc fingerprint does not match its own fields")
    table_path = REPO / fz["table_relpath"]

    # GUARD FIRST, then preflight. `preflight` opens and parses all six frozen query/qrels
    # payloads, so with the old ordering every refused invocation -- including every
    # --untouched-only resume -- read the confirmatory labels before the one-shot check ran. It is
    # not a second scoring pass, but if "access" means reading confirmatory labels then it was one
    # (Codex review #4, BLOCKER 5). Guard is git commands and one JSON parse: cheap. preflight
    # still runs before FINAL-RUN-BEGIN, so a missing asset still costs a second, not the run.
    head = guard(a.freeze_hash, a.infra_retry, a.branch, fz, untouched_only=a.untouched_only)
    # The teacher's remote code was verified when the freeze was WRITTEN; the snapshot is mutable
    # and lives outside git, so re-verify it here -- and BEFORE preflight opens any frozen payload,
    # so a pin mismatch or snapshot failure costs nothing that needs recording
    # (Codex review #4 MAJOR 9; pre-freeze review 2026-08-28 BLOCKER 5).
    import teacher_code
    code_ok, code_problems = teacher_code.verify()
    if not code_ok:
        raise SystemExit("FINAL RUN REFUSED: the teacher's pinned files no longer match "
                         "results/m7_teacher_code_pin.json:\n  - " + "\n  - ".join(code_problems))
    t0 = time.time()
    out = OUT
    # IMMUTABLE SNAPSHOT OF THE TABLE. `load_and_verify` hashes it once, then `score_set` reopens
    # it by pathname for every variant and every dataset. Another process calling `ensure_release`
    # in between could substitute a different table -- and because work/ is gitignored the
    # clean-tree guard sees nothing, so the result would record the frozen hash beside another
    # table's scores, or even mix two tables across datasets (Codex review #4, BLOCKER 4).
    import shutil
    snap = REPO / "work" / "runs" / f".final-run-snapshot-{fz['table_sha256'][:16]}.npz"
    shutil.copy2(table_path, snap)
    got = freeze.sha256_file(snap)
    if got != fz["table_sha256"]:
        raise SystemExit(f"FINAL RUN REFUSED: the table changed between verification and snapshot "
                         f"({got[:12]} vs frozen {fz['table_sha256'][:12]})")
    shutil.copy2(table_path.parent / (table_path.stem + ".meta.json"),
                 snap.parent / (snap.stem + ".meta.json"))
    table_path = snap
    # Preflight LAST among the checks, and only for the kinds this invocation will score: it is
    # the one check that opens frozen payloads, so everything that can refuse for free runs first
    # (BLOCKER 5), and a --untouched-only resume never opens a six-set payload (BLOCKER 4).
    if a.untouched_only:
        kinds = ("untouched",)
    elif a.run_untouched_tail:
        kinds = ("six", "untouched")
    else:
        kinds = ("six",)    # the tail is reserved for M8; its payloads are not even opened
    preflight(kinds=kinds)
    if a.untouched_only:
        ledger(f"\n- {datetime.now(timezone.utc).isoformat()} — untouched-final RESUME "
               f"(--untouched-only) on freeze={head}; the six are not re-scored.")
        # If the crash landed between the local tag and its push, the external receipt is still
        # missing -- retry it here so a resume heals the receipt rather than leaving it local.
        if sh("git", "tag", "-l", SPENT_TAG) and not sh_raw(
                "git", "ls-remote", "origin", f"refs/tags/{SPENT_TAG}").strip():
            p = subprocess.run(["git", "push", "origin", SPENT_TAG],
                               cwd=REPO, capture_output=True, text=True)
            print(f"[final_run] spent receipt `{SPENT_TAG}` was local-only; push "
                  f"{'succeeded' if p.returncode == 0 else 'FAILED: ' + (p.stderr or '').strip()}")
        prior = json.loads(out.read_text())
        # Registered resume path for a crash between the confirmatory write and the clean-4
        # append (pre-freeze review 2026-08-28, MAJOR 7): the clean-4 block is a pure function of
        # the stored confirmatory per-query values and the frozen comparator vectors, so it is
        # recomputed here WITHOUT touching any six-set payload.
        if prior.get("clean4_robustness") is None:
            pq = json.load(open(REPO / "results" / "perquery.json"))
            by_sys = dict(prior["six"])
            # fz, not prior["freeze"]: the ledger digest covers six/confirmatory/holm only, so the
            # stored freeze block is editable -- the committed FREEZE.json was just re-verified.
            rs = fz["released_system"]
            by_sys["released-system"] = by_sys["fusion" if rs == "fusion" else "int8-table"]
            prior["clean4_robustness"] = clean4_block(by_sys, pq)
            write_atomic(out, json.dumps(prior, indent=1))
            ledger("- clean-4 robustness recomputed during --untouched-only resume from the "
                   "stored confirmatory per-query values and the frozen comparator vectors; no "
                   "six-set payload was read.")
        return untouched_stage(prior, out, table_path, pre, spec, t0,
                               tiers_from=prior.get("holm", {}))
    ledger(f"\n### FINAL-RUN {datetime.now(timezone.utc).isoformat()}\n"
           f"- FINAL-RUN-BEGIN freeze={head} table={fz['table_sha256']}\n"
           f"- pushed to origin/{a.branch}; table `{fz['table_relpath']}` "
           f"({fz['table_bytes']} bytes), preproc `{fz['preproc']}` "
           f"(fingerprint {fz['preproc_fingerprint']}), fusion `{json.dumps(spec)}`, "
           f"released system `{fz['released_system']}`"
           f"{', INFRA-RETRY' if a.infra_retry else ''}.")

    six = {ds: score_set(ds, "six", table_path, pre, spec) for ds in DATASETS}
    print("\n=== the six (KNOWN-TEST, development-informed) ===")
    systems = sorted({k for v in six.values() for k in v})
    by_sys = {s: {ds: six[ds][s] for ds in DATASETS if s in six[ds]} for s in systems}
    for s in systems:
        means = {ds: float(np.mean(list(v.values()))) for ds, v in by_sys[s].items()}
        print(f"  {s:20s} avg-6 {np.mean(list(means.values())):.4f}  " +
              " ".join(f"{d}={means[d]:.4f}" for d in DATASETS))

    pq = json.load(open(REPO / "results" / "perquery.json"))
    # Explicit dispatch, not `"fusion" if x == "fusion" else dense`: that spelling made every
    # value other than the exact string "fusion" mean DENSE, so a typo in the freeze would have
    # silently judged C3 on the wrong system. `load_and_verify` validates the enum; this refuses
    # again rather than trusting a neighbouring check on the one irreversible path.
    rs = fz["released_system"]
    if rs == "fusion":
        by_sys["released-system"] = by_sys["fusion"]
    elif rs == "dense":
        by_sys["released-system"] = by_sys["int8-table"]
    else:
        raise SystemExit(f"FINAL RUN ABORTED: released_system {rs!r} is not one of "
                         f"{freeze.RELEASED_SYSTEMS}")
    conf, pvals = {}, {}
    for name, (a_name, b_name) in CONFIRMATORY.items():
        A = by_sys[a_name]
        # ALWAYS the frozen per-query comparator vectors, never a locally recomputed row: the
        # mandate says "never re-run a comparator system". A fresh BM25 run exists in by_sys for
        # the fusion input and as an exploratory row, and must not be used here.
        B = boot.from_perquery_json(pq, b_name, set(DATASETS))
        if not B:
            raise SystemExit(f"FINAL RUN ABORTED: no frozen per-query vectors for {b_name}")
        # strict=True: the registered rule is strict alignment on EVERY confirmatory path.
        # The sign-flip call below is strict and would abort first, so nothing could have
        # shrunk silently -- but relying on a neighbouring call for a guarantee is how the
        # guarantee gets lost when the calls are reordered.
        r = boot.paired(A, B, alternative="greater", strict=True)   # intervals only (B3)
        t = boot.signflip(A, B, alternative="greater", strict=True)  # THE p-value (B3)
        r["signflip"] = t
        conf[name] = r
        pvals[name] = t["p"]
    decisions = boot.holm(pvals, alpha=ALPHA)
    # Tier rule (pre-registered, review #2 BLOCKER 5 + MAJOR 8): a tier win requires BOTH the
    # Holm-corrected sign-flip rejection AND the paired-bootstrap CI resolved above zero. The
    # mandate's tier text is written in CIs; the sign-flip carries the multiplicity control; the
    # conjunction satisfies both and is conservative under either's failure mode.
    n_fam = len(CONFIRMATORY)
    for name in CONFIRMATORY:
        h = decisions[name]
        h["reject_holm_signflip"] = h["reject"]
        # ci95_raw, NEVER ci95: the display field is rounded to four decimals, so a true
        # lower endpoint of +4e-5 reads as 0.0000 and a resolved tier becomes "unresolved"
        # on the ONE-SHOT run. boot.py and m7/LEDGER.md both state the raw-endpoint rule;
        # this line broke it, in the single place where it is irreversible.
        h["ci_resolved"] = bool(conf[name]["ci95_raw"][0] > 0)
        # SIMULTANEOUS bound, added 2026-08-28 before any confirmatory number existed. Three
        # separate one-sided 2.5% intervals are not a family-wise 2.5%, and the sign-flip leg is
        # exact only under the SHARP null -- the repo's own weak-null simulation rejects at 0.038
        # for a nominal 0.025 on the worst pair. Bonferroni over the family puts each bound at
        # 0.025/3 = 0.8333%, where that same simulation measures 0.013 and 0.008. Strictly harder
        # than the previous rule, and fixed before the numbers, which is the only time a bar may
        # move.
        lb = (conf[name].get("one_sided_lower_raw") or {}).get("0.8333")
        h["bonferroni_level"] = round(ALPHA / n_fam, 6)
        h["one_sided_lower_bonferroni"] = lb
        h["ci_resolved_simultaneous"] = bool(lb is not None and lb > 0)
        h["reject"] = bool(h["reject_holm_signflip"] and h["ci_resolved"]
                           and h["ci_resolved_simultaneous"])

    print(f"\n=== confirmatory (one-sided; tier = Holm(sign-flip) AND CI>0 AND simultaneous "
          f"Bonferroni lower bound at {ALPHA}/{n_fam} > 0) ===")
    for name in CONFIRMATORY:
        d, h = conf[name], decisions[name]
        print(f"  {'REJECT H0' if h['reject'] else 'not resolved'}  {name}: d={d['delta']:+.4f} "
              f"CI={d['ci95']} p={d['signflip']['p_str']} (sign-flip) "
              f"thr={h['threshold']:.4f} ci_resolved={h['ci_resolved']}")

    blob = {"freeze_commit": head, "freeze": fz, "infra_retry": bool(a.infra_retry),
            # RAW floats, never round(x, 6): the in-memory tier decision uses full precision, and
            # a rounded artifact could not reproduce a close decision exactly (pre-freeze review
            # 2026-08-28, MAJOR 8) -- and the clean-4 resume path recomputes FROM these values.
            "six": {s: {ds: {q: float(x) for q, x in v.items()} for ds, v in by_sys[s].items()}
                    for s in systems},
            "untouched_final": None,
            "confirmatory": conf, "holm": decisions, "alpha": ALPHA,
            # Which cache bytes produced these vectors: shard hashes for every encode consumed,
            # so the document side of a one-shot number is auditable after the fact.
            "encode_provenance": dict(teacher.PROVENANCE),
            "clean4_robustness": None,
            "seconds": round(time.time() - t0, 1)}
    # PERSIST THE CONFIRMATORY RESULT BEFORE ANYTHING ELSE CAN FAIL. The tail encodes 10.1M
    # documents, and even the clean-4 robustness block below is more work after all six have been
    # scored and every tier decision made. An OOM or a bad comparator row there would have spent
    # the access and left no result (Codex review #4, BLOCKER 1 and MAJOR 6). Atomic write: a
    # truncating `write_text` on the sole confirmatory file is not recoverable.
    write_atomic(out, json.dumps(blob, indent=1))
    tiers = [k for k, v in decisions.items() if v["reject"]]
    # The digest binds the six and the tier decisions, so a later --untouched-only resume can prove
    # the confirmatory block it is appending to has not been edited.
    six_digest = sha({"six": blob["six"], "confirmatory": conf, "holm": decisions})
    tail_note = ("the non-confirmatory untouched-final tail follows." if a.run_untouched_tail
                 else "the untouched-final tail is reserved for M8 and is not run.")
    ledger(f"- FINAL-RUN complete in {blob['seconds']:.0f}s (the six and all three confirmatory "
           f"decisions). Confirmatory rejections: {tiers or 'none'}. Results in "
           f"`results/m7_final_run.json`; {tail_note}\n"
           f"- FINAL-RUN-SIX-SHA256 {six_digest}")
    # THE DURABLE SPENT RECEIPT (pre-freeze review 2026-08-28, BLOCKER 2): the result file is
    # untracked and the ledger is editable, so an annotated tag pushed to origin is the record
    # that survives both. Best-effort AFTER the result is safe on disk: a network failure must
    # not cost the run, only print loudly.
    subprocess.run(["git", "tag", "-a", SPENT_TAG, "-m",
                    f"confirmatory six-set access spent; six_digest={six_digest}", head],
                   cwd=REPO, capture_output=True, text=True)
    push = subprocess.run(["git", "push", "origin", SPENT_TAG],
                          cwd=REPO, capture_output=True, text=True)
    if push.returncode != 0:
        print(f"[final_run] WARNING: could not push the spent receipt `{SPENT_TAG}` to origin "
              f"({(push.stderr or '').strip()}). Push it manually; until then only the local tag, "
              "the result file and the ledger mark the access as spent.")
    print("\nwrote results/m7_final_run.json (the six and all three confirmatory decisions)")

    # Pre-registered clean-4 robustness (teacher-exposure restriction). Runs AFTER the persist,
    # so a failure here costs a labelled robustness block, not the result -- and a crash between
    # the two writes is recoverable: --untouched-only recomputes this block from the stored
    # per-query values (pre-freeze review 2026-08-28, MAJOR 7).
    blob["clean4_robustness"] = clean4_block(by_sys, pq)
    write_atomic(out, json.dumps(blob, indent=1))
    if a.run_untouched_tail:
        return untouched_stage(blob, out, table_path, pre, spec, t0, tiers_from=decisions)
    ledger("- untouched-final tail SKIPPED: the four sets stay un-scored, reserved as M8's "
           "confirmatory evaluation (instructions-m8.md, registered 2026-08-28 pre-run). "
           "`--run-untouched-tail` or `--untouched-only` would burn them; neither was used.")
    print("\nuntouched-final tail skipped (reserved for M8). The run is complete.")


CLEAN4 = {"scifact", "nfcorpus", "scidocs", "trec-covid"}


def clean4_block(by_sys, pq):
    """The pre-registered exposure-restricted robustness read: same three comparisons on the four
    datasets with no disclosed teacher benchmark overlap. Labeled, never a tier decision. A pure
    function of per-query values already in memory or on disk -- it reads no six-set payload."""
    robustness = {}
    for name, (a_name, b_name) in CONFIRMATORY.items():
        A4 = {ds: v for ds, v in by_sys[a_name].items() if ds in CLEAN4}
        B4 = boot.from_perquery_json(pq, b_name, CLEAN4)
        rr = boot.paired(A4, B4, alternative="greater", strict=True)
        rr["signflip"] = boot.signflip(A4, B4, alternative="greater", strict=True)
        robustness[name] = rr
        print(f"  [clean-4 robustness] {name}: d={rr['delta']:+.4f} CI={rr['ci95']} "
              f"p={rr['signflip']['p_str']}")
    return {"_note": "pre-registered exposure-restricted robustness "
                     "(no disclosed teacher benchmark overlap); NOT a tier decision",
            **robustness}


def untouched_stage(blob, out, table_path, pre, spec, t0, tiers_from=None):
    """The non-confirmatory tail. Separated so it can be resumed with --untouched-only after a
    crash without going anywhere near the six.

    It encodes 10.1M documents -- 37x the six -- so it is hours, not minutes, and it used to run
    BEFORE anything was written to disk. A crash here (disk, OOM, a Windows Update reboot) would
    have spent the one confirmatory access and left no result at all, the same failure `preflight`
    exists to prevent, arriving from the other end of the script.
    """
    man = json.loads(MANIFEST.read_text())["m7_untouched_final"]
    todo = [d for d in UNTOUCHED if d not in (blob.get("untouched_final") or {})]
    print("\n=== untouched-final (scored after the six; no recipe change after this point) ===")
    print(f"  {sum(man[d]['n_docs'] for d in todo):,} documents to encode across {len(todo)} set(s); "
          "hours, not minutes. The confirmatory result is already on disk and does not depend on "
          "this stage completing.")
    unt = dict(blob.get("untouched_final") or {})
    for ds in todo:
        scored = score_set(ds, "untouched", table_path, pre, spec)
        print(f"  {ds}: " + " ".join(f"{s}={np.mean(list(v.values())):.4f}"
                                     for s, v in scored.items()))
        unt[ds] = {s: {q: float(x) for q, x in v.items()} for s, v in scored.items()}
        # rewrite after each set, so a crash costs one dataset rather than the whole tail
        blob["untouched_final"] = unt
        # MERGE, never replace. In a resumed `--untouched-only` process `teacher.PROVENANCE` starts
        # empty and holds only tail encodes, so assigning it would erase every six-set document and
        # query cache hash -- exactly the audit trail this field exists for (Codex review #4,
        # MAJOR 8).
        blob["encode_provenance"] = {**(blob.get("encode_provenance") or {}),
                                     **dict(teacher.PROVENANCE)}
        blob["seconds"] = round(time.time() - t0, 1)
        write_atomic(out, json.dumps(blob, indent=1))
    tiers = [k for k, v in (tiers_from or {}).items() if v.get("reject")]
    ledger(f"- untouched-final done ({len(unt)}/{len(UNTOUCHED)} sets). Confirmatory rejections "
           f"stand at {tiers or 'none'}. `results/m7_final_run.json` updated.")
    print("\nwrote results/m7_final_run.json")


if __name__ == "__main__":
    main()
