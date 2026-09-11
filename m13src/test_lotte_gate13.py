"""What LoTTE read #1's executor must refuse, and the arithmetic it must get right — on a synthetic
git tree with a bare origin. No test opens `work/lotte`; the fixture writes seven tiny "remediated"
slices under a tmp path, pins them with `freeze_lotte`'s hash scheme, writes the two E arm records
with real checkpoint bytes, commits and pushes the manifest and pin, injects a deterministic document
encoder in place of stella and a constant dependency identity, and hands the executor stub students
whose quality is a parameter — so both branches, both veto outcomes, the crash-and-recover path and
every refusal run on a CPU in seconds. The one real-path check (the guard claim) runs in a
subprocess and only CLASSIFIES a protected path; it opens nothing.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "m7src", REPO / "m8src", REPO / "m10src", REPO / "m13src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build13 as BD                                                      # noqa: E402
import lotte_gate13 as G                                                  # noqa: E402

DIM = 32
REAL_REG = json.loads((REPO / "m13" / "LOTTE_GATE_REGISTRATION.json").read_text())
SLICES = list(REAL_REG["surface"]["slices"])
DEP = {"repo": "fixture/bge-small", "sha256": "d" * 64}


# --------------------------------------------------------------------------------- fixture ----

def _unit(text, dim=DIM):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def doc_vec(doc_text):
    return _unit("doc|" + doc_text)


class DocEncoder:
    """Stands in for `teacher.encode_cached`: deterministic unit vectors, stored fp16 like the
    real caches, plus a cache record. Counts calls and can be told to crash on a given slice."""

    def __init__(self):
        self.calls = []
        self.crash_on = None

    def __call__(self, name, texts):
        self.calls.append(name)
        if self.crash_on and self.crash_on in name:
            raise RuntimeError("injected crash during the document encode")
        vecs = np.stack([doc_vec(t) for t in texts]).astype(np.float16)
        return vecs, {"fixture": True, "cache_key": name,
                      "combined_sha256": hashlib.sha256(vecs.tobytes()).hexdigest()}


class Student:
    """The `encode_queries` contract. `quality` is the fraction of queries answered with the
    normalised sum of their positives' vectors (a perfect answer); the rest get an unrelated
    vector. Deterministic per query text, so two students of equal quality score identically."""

    def __init__(self, quality):
        self.quality = quality
        self.calls = 0

    def encode_queries(self, texts, batch_size=256, prefix=""):
        self.calls += 1
        out = []
        for t in texts:
            positives = t.split(" about ", 1)[1].split(" || ")
            r = int(hashlib.sha256(("pick|" + t).encode()).hexdigest()[:8], 16) / 2 ** 32
            if r < self.quality:
                v = np.sum([doc_vec(p) for p in positives], axis=0)
                out.append(v / np.linalg.norm(v))
            else:
                out.append(_unit("noise|" + t))
        return np.stack(out).astype(np.float32)


def write_slice(d, key, n_docs, n_queries):
    """Docs `0..n_docs-1`, queries `0..n_queries-1` — the SAME integer ids on both sides, as in
    LoTTE, so a scorer that keeps BEIR's self-hit rule drops every query's first positive. Query j
    is relevant to doc j, and every even j also to doc j + n_queries. -> (counts, pin entry)."""
    d.mkdir(parents=True, exist_ok=True)
    docs = {str(i): f"passage {key} {i} {'lorem' if i % 3 else 'ipsum'}" for i in range(n_docs)}
    with open(d / "collection.tsv", "w") as fh:
        for pid, text in docs.items():
            fh.write(f"{pid}\t{text}\n")
    q_ids, q_texts, qrels, pairs = [], [], {}, 0
    with open(d / "questions.forum.tsv", "w") as fh, open(d / "qas.forum.jsonl", "w") as qa:
        for j in range(n_queries):
            pos = [str(j)] + ([str(j + n_queries)] if j % 2 == 0 and j + n_queries < n_docs else [])
            text = f"question {key} {j} about " + " || ".join(docs[p] for p in pos)
            fh.write(f"{j}\t{text}\n")
            qa.write(json.dumps({"qid": j, "answer_pids": [int(p) for p in pos]}) + "\n")
            q_ids.append(str(j)); q_texts.append(text); qrels[str(j)] = sorted(pos); pairs += len(pos)
    counts = {"queries_after_remedy": n_queries, "docs_after_remedy": n_docs,
              "qrels_pairs_after_remedy": pairs}
    pin = {"n_docs": n_docs, "n_queries": n_queries, "n_qrels_pairs": pairs,
           "hashes": G.slice_hashes(list(docs), list(docs.values()), q_ids, q_texts, qrels),
           "read_relpath": str(d)}
    return counts, pin


def _git(repo, *a):
    r = subprocess.run(("git", "-c", "user.email=t@t", "-c", "user.name=t") + a, cwd=repo,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(a)}: {r.stderr}"
    return r.stdout.strip()


def commit_and_push(repo, msg="fixture"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    _git(repo, "push", "-q", "origin", "HEAD")


def build_world(root, *, e1_batch, quality=None, n_queries=12, count_error=None,
                write_manifest=True, publish=True):
    """-> a Config over a synthetic GIT tree with a bare origin: the REAL registration's structure
    with the fixture's counts, a copy of the real screen registry and build config, an E1 verdict
    bound to the registry, two complete E arm records carrying the registered recipe and real
    checkpoint bytes, seven tiny slices of different sizes, their pin, and (by default) the
    manifest written by the executor itself, all committed and pushed."""
    quality = quality or {"E-bs32": 1.0, "E-bs128": 1.0}
    repo, origin = Path(root) / "repo", Path(root) / "origin.git"
    for d in ("m13", "m10", "results", "work/lotte/remediated", "work/m10arms"):
        (repo / d).mkdir(parents=True, exist_ok=True)
    reg = json.loads(json.dumps(REAL_REG))
    pin = {"_note": "fixture pin, freeze_lotte's scheme", "slices": {}}
    for i, key in enumerate(SLICES):
        topic, split = key.split("/")
        counts, p = write_slice(repo / "work" / "lotte" / "remediated" / topic / split, key,
                                n_docs=40 + 7 * i, n_queries=n_queries)
        if count_error == key:
            counts["docs_after_remedy"] += 1
        reg["surface"]["slices"][key] = counts
        pin["slices"][key] = p
    reg["surface"]["total_queries"] = sum(s["queries_after_remedy"]
                                          for s in reg["surface"]["slices"].values())
    (repo / "m13" / "LOTTE_GATE_REGISTRATION.json").write_text(json.dumps(reg, indent=1))
    shutil.copy(REPO / "m13" / "build_config.json", repo / "m13" / "build_config.json")
    (repo / "results" / "m8_lotte_pin.json").write_text(json.dumps(pin, indent=1))
    shutil.copy(REPO / "m10" / "screen_registry.json", repo / "m10" / "screen_registry.json")
    reg_sha = G.sha256_file(repo / "m10" / "screen_registry.json")
    (repo / "results" / "m10_screen_verdicts.json").write_text(json.dumps(
        {"selected": {"batch": e1_batch}, "registry_sha256": reg_sha}))
    for arm, batch in (("E-bs32", 32), ("E-bs128", 128)):
        d = repo / "work" / "m10arms" / arm
        d.mkdir(parents=True, exist_ok=True)
        ck = d / "cycle3.pt"
        ck.write_bytes(hashlib.sha256(arm.encode()).digest() * 64)
        sha = G.sha256_file(ck)
        (repo / "results" / f"m10_arm_{arm}.json").write_text(json.dumps(
            {"arm": arm, "status": "complete", "complete": True, "terminal": True, "smoke": False,
             "seed": 0, "registry_sha256": reg_sha,
             "recipe": {"student": "bge-small", "n_layers": 3, "head": "linear",
                        "objective": "squared_l2", "pattern": "75/25",
                        "dose_examples": 5_000_000, "batch": batch},
             "params": 33_000_000, "device": "cuda", "git_head": "f" * 40,
             "checkpoints": {"cycle3": {"path": f"work/m10arms/{arm}/cycle3.pt", "sha256": sha}},
             "final_checkpoint": f"work/m10arms/{arm}/cycle3.pt", "final_checkpoint_sha256": sha}))
    students = {arm: Student(q) for arm, q in quality.items()}
    enc = DocEncoder()
    cfg = G.Config(
        repo=repo,
        registration_path=repo / "m13" / "LOTTE_GATE_REGISTRATION.json",
        build_config_path=repo / "m13" / "build_config.json",
        record_path=repo / "m13" / "LOTTE_GATE.json",
        manifest_path=repo / "m13" / "LOTTE_GATE_MANIFEST.json",
        pin_path=repo / "results" / "m8_lotte_pin.json",
        verdicts_path=repo / "results" / "m10_screen_verdicts.json",
        registry_path=repo / "m10" / "screen_registry.json",
        arm_records_dir=repo / "results",
        remediated_dir=repo / "work" / "lotte" / "remediated",
        enc_root=repo / "work" / "lotte" / "enc",
        gate_work_dir=repo / "work" / "lotte" / "gate13",
        device="cpu", allow_cpu=True, topk=100, chunk=1000,
        doc_encoder=enc,
        load_student=lambda ar, data, device: students[ar["arm"]],
        dependency_identity=lambda key: dict(DEP),
        claim_guard=False)
    cfg.students, cfg.enc = students, enc
    (repo / ".gitignore").write_text("work/\n")
    _git(repo, "init", "-q")
    _git(origin.parent, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    if write_manifest and str(e1_batch).upper() != "PENDING":
        G.write_manifest(cfg, verbose=False)
    if publish:
        commit_and_push(repo)
    return cfg


def _rewrite(path, **over):
    d = json.loads(Path(path).read_text())
    d.update(over)
    Path(path).write_text(json.dumps(d))


def _crash_then_recover_world(root, quality=None):
    cfg = build_world(root, e1_batch=128, quality=quality or {"E-bs32": 1.0, "E-bs128": 0.5})
    cfg.enc.crash_on = "science-dev"                       # the fifth slice by size
    with pytest.raises(RuntimeError, match="injected crash"):
        G.run(cfg, verbose=False)
    cfg.enc.crash_on = None
    return cfg


# ------------------------------------------------------------------------------ registration ----

def test_the_real_registration_validates_without_opening_lotte():
    reg, sha = G.registration(G.Config())
    assert len(reg["surface"]["slices"]) == 7 and reg["executed"] is False
    assert reg["veto"]["margin"] == 0.004 and reg["veto"]["bootstrap"]["seed"] == 903
    assert reg["veto"]["bootstrap"]["quantile_method"] == "inverted_cdf", "ruling R17"
    assert sha == G.sha256_file(REPO / "m13" / "LOTTE_GATE_REGISTRATION.json")
    assert G.registered_recipe(G.Config()) == {"student": "bge-small", "n_layers": 3,
                                               "head": "linear", "objective": "squared_l2",
                                               "pattern": "75/25"}


def test_the_registration_is_held_to_the_locked_constants(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    reg["veto"]["margin"] = 0.005
    cfg.registration_path.write_text(json.dumps(reg))
    with pytest.raises(SystemExit, match="veto.margin"):
        G.registration(cfg)
    reg["veto"]["margin"] = 0.004
    for method in (None, "higher", "linear"):               # R17 admits inverted_cdf and nothing else
        reg["veto"]["bootstrap"]["quantile_method"] = method
        cfg.registration_path.write_text(json.dumps(reg))
        with pytest.raises(SystemExit, match="R17"):
            G.registration(cfg)
    reg["veto"]["bootstrap"]["quantile_method"] = "inverted_cdf"
    reg["executed"] = True
    cfg.registration_path.write_text(json.dumps(reg))
    with pytest.raises(SystemExit, match="never flips"):
        G.registration(cfg)


# --------------------------------------------------------------------------------- preflight ----

def test_preflight_opens_no_lotte_path_and_orders_slices_smallest_first(tmp_path):
    cfg = build_world(tmp_path, e1_batch=128)
    cfg.remediated_dir = tmp_path / "does-not-exist"          # a preflight that read it would fail
    plan = G.run(cfg, preflight_only=True, verbose=False)
    assert plan["branch"] == "bs128" and plan["veto_runs"] is True
    assert plan["candidate"]["arm"] == "E-bs128" and plan["comparator"]["arm"] == "E-bs32"
    sizes = [json.loads(cfg.registration_path.read_text())["surface"]["slices"][k]["docs_after_remedy"]
             for k in plan["slice_order"]]
    assert sizes == sorted(sizes) and len(sizes) == 7
    assert plan["manifest_commit"] and plan["pin_commit"] and plan["code_identity"] == G.code_identity()
    assert not cfg.record_path.exists() and not cfg.gate_work_dir.exists()
    assert cfg.enc.calls == [] and all(s.calls == 0 for s in cfg.students.values())


def test_a_pending_e1_verdict_refuses(tmp_path):
    cfg = build_world(tmp_path, e1_batch="PENDING")
    with pytest.raises(SystemExit, match="PENDING"):
        G.run(cfg, preflight_only=True, verbose=False)
    with pytest.raises(SystemExit, match="PENDING"):
        G.write_manifest(cfg, verbose=False)


def test_a_verdict_bound_to_another_registry_refuses(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    _rewrite(cfg.verdicts_path, registry_sha256="0" * 64)
    with pytest.raises(SystemExit, match="bound to registry sha"):
        G.run(cfg, preflight_only=True, verbose=False)


@pytest.mark.parametrize("over, match", [
    ({"status": "failed", "complete": False}, "failed or unfinished"),
    ({"smoke": True}, "SMOKE record"),
    ({"arm": "E-bs32"}, "not 'E-bs128'"),
    ({"seed": 1}, "seed 1 is not the registered 0"),
    ({"registry_sha256": "0" * 64}, "cannot feed a gate under another"),
    ({"final_checkpoint_sha256": "1" * 64}, "do not name one cycle-3 checkpoint"),
    ({"recipe": {"student": "bge-small"}}, "recipe lacks"),
])
def test_an_arm_record_that_is_not_this_complete_registered_arm_refuses(tmp_path, over, match):
    cfg = build_world(tmp_path, e1_batch=32)
    _rewrite(cfg.arm_records_dir / "m10_arm_E-bs128.json", **over)   # the OTHER arm in bs32, too
    with pytest.raises(SystemExit, match=match):
        G.run(cfg, preflight_only=True, verbose=False)


@pytest.mark.parametrize("field, value", [("student", "MiniLM-L6-v2"), ("n_layers", 4),
                                          ("head", "mlp"), ("objective", "leaf_norm_e2"),
                                          ("pattern", "50/50")])
def test_a_record_whose_recipe_is_not_the_registered_one_refuses(tmp_path, field, value):
    """Sol 2026-09-10, finding 3: a wrong-recipe checkpoint labelled with the right arm, seed and
    registry sha must not be frozen into a manifest or read."""
    cfg = build_world(tmp_path, e1_batch=128, write_manifest=False)
    p = cfg.arm_records_dir / "m10_arm_E-bs128.json"
    rec = json.loads(p.read_text())
    rec["recipe"][field] = value
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="differs from the registered knobs"):
        G.write_manifest(cfg, verbose=False)
    with pytest.raises(SystemExit, match="differs from the registered knobs"):
        G.run(cfg, preflight_only=True, verbose=False)


def test_both_e_records_are_required_even_in_the_bs32_branch(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    (cfg.arm_records_dir / "m10_arm_E-bs128.json").unlink()
    with pytest.raises(SystemExit, match="after BOTH 5M E arms"):
        G.run(cfg, preflight_only=True, verbose=False)


def test_a_checkpoint_whose_bytes_moved_refuses(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    (cfg.repo / "work" / "m10arms" / "E-bs32" / "cycle3.pt").write_bytes(b"not the published bytes")
    with pytest.raises(SystemExit, match="refusing to read a checkpoint that is not the published"):
        G.run(cfg, preflight_only=True, verbose=False)


def test_the_student_is_loaded_from_the_bytes_that_were_hashed(tmp_path):
    """Hash-then-load must not reopen the file. The loader receives the exact bytes the hash was
    computed on; swapping the file after preflight is caught at load time."""
    cfg = build_world(tmp_path, e1_batch=32)
    seen = {}
    cfg.load_student = lambda ar, data, device: seen.setdefault(ar["arm"], data) and cfg.students[ar["arm"]]
    rec = G.run(cfg, verbose=False)
    assert hashlib.sha256(seen["E-bs32"]).hexdigest() == rec["candidate_sha256"]
    cfg2 = build_world(tmp_path / "b", e1_batch=32)
    real_preflight = G.preflight

    def swap_after_preflight(*a, **k):
        plan = real_preflight(*a, **k)
        (cfg2.repo / "work" / "m10arms" / "E-bs32" / "cycle3.pt").write_bytes(b"swapped")
        return plan
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "preflight", swap_after_preflight)
        with pytest.raises(SystemExit, match="not the published one"):
            G.run(cfg2, verbose=False)
    assert not cfg2.record_path.exists()


def test_a_changed_dependency_refuses_at_load(tmp_path):
    """Sol 2026-09-10, finding 4: `Nano10` loads its tokenizer and backbone configuration by
    repository name. The manifest records their identity; the read re-derives and compares it."""
    cfg = build_world(tmp_path, e1_batch=32)
    assert json.loads(cfg.manifest_path.read_text())["candidate"]["dependencies"] == DEP
    cfg.dependency_identity = lambda key: {"repo": DEP["repo"], "sha256": "e" * 64}
    with pytest.raises(SystemExit, match="not the one the manifest recorded"):
        G.run(cfg, verbose=False)
    assert not cfg.record_path.exists()


def test_the_two_dependency_identity_derivations_agree_for_the_real_student():
    """Astra closing re-check, P3: the manifest hashes the tokenizer and config loaded by name, the
    read hashes the constructed student's own objects. They must be the same bytes for bge-small,
    or the real read would refuse falsely. Offline, from the cached weights (HARNESS.md)."""
    import nano10 as N
    cfg = G.Config()
    by_name = G.dependency_identity(cfg, "bge-small")
    from_model = G.dependency_identity(cfg, "bge-small",
                                       model=N.Nano10("bge-small", n_layers=3, head="linear"))
    assert by_name == from_model and by_name["repo"] == "BAAI/bge-small-en-v1.5"


def test_an_existing_gate_record_refuses_a_second_execution(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    G.run(cfg, verbose=False)
    assert cfg.record_path.exists()
    for kw in ({}, {"preflight_only": True}, {"recover": True}):
        with pytest.raises(SystemExit, match="ONE access"):
            G.run(cfg, verbose=False, **kw)


# ------------------------------------------------------------------------- manifest and pin ----

def test_the_read_refuses_without_the_committed_and_pushed_manifest(tmp_path):
    cfg = build_world(tmp_path, e1_batch=128, write_manifest=False)
    with pytest.raises(SystemExit, match="second manifest commit"):
        G.run(cfg, preflight_only=True, verbose=False)
    G.write_manifest(cfg, verbose=False)                      # written but NOT committed
    with pytest.raises(SystemExit, match="not tracked by git"):
        G.run(cfg, preflight_only=True, verbose=False)
    _git(cfg.repo, "add", "-A")                               # staged is not committed
    with pytest.raises(SystemExit, match="uncommitted changes"):
        G.run(cfg, preflight_only=True, verbose=False)
    _git(cfg.repo, "commit", "-q", "-m", "manifest")          # committed is not pushed
    with pytest.raises(SystemExit, match="not on any remote branch"):
        G.run(cfg, preflight_only=True, verbose=False)
    _git(cfg.repo, "push", "-q", "origin", "HEAD")
    plan = G.run(cfg, preflight_only=True, verbose=False)
    assert plan["manifest_commit"] == _git(cfg.repo, "rev-parse", "HEAD")
    _rewrite(cfg.manifest_path, written_at="later")           # edited after its commit
    with pytest.raises(SystemExit, match="uncommitted changes"):
        G.run(cfg, preflight_only=True, verbose=False)


def test_a_manifest_that_disagrees_with_the_live_records_refuses(tmp_path):
    """A swapped record and checkpoint (same dose, same batch, same recipe) must not pass merely
    because they are self-consistent; the committed manifest is the anchor."""
    cfg = build_world(tmp_path, e1_batch=128)
    ck = cfg.repo / "work" / "m10arms" / "E-bs128" / "cycle3.pt"
    ck.write_bytes(b"a different but complete run's checkpoint")
    sha = G.sha256_file(ck)
    _rewrite(cfg.arm_records_dir / "m10_arm_E-bs128.json", final_checkpoint_sha256=sha,
             checkpoints={"cycle3": {"path": "work/m10arms/E-bs128/cycle3.pt", "sha256": sha}})
    with pytest.raises(SystemExit, match="does not describe the live inputs.*candidate.sha256"):
        G.run(cfg, preflight_only=True, verbose=False)
    with pytest.raises(SystemExit, match="already exists"):
        G.write_manifest(cfg, verbose=False)


def test_write_manifest_records_both_arms_for_bs128_and_one_for_bs32(tmp_path):
    cfg = build_world(tmp_path, e1_batch=128)
    m = json.loads(cfg.manifest_path.read_text())
    assert m["branch"] == "bs128" and m["candidate"]["arm"] == "E-bs128"
    assert m["comparator"]["arm"] == "E-bs32" and m["e1_verdict_sha256"] == G.sha256_file(cfg.verdicts_path)
    assert set(m["candidate"]) >= {"arm", "checkpoint", "sha256", "record_sha256", "recipe", "seed",
                                   "dependencies"}
    assert set(m["candidate"]["recipe"]) == set(G.RECIPE_FIELDS)
    cfg32 = build_world(tmp_path / "b", e1_batch=32)
    m32 = json.loads(cfg32.manifest_path.read_text())
    assert m32["branch"] == "bs32" and m32["candidate"]["arm"] == "E-bs32" and m32["comparator"] is None


def test_the_read_refuses_without_a_committed_pin_and_on_a_pin_mismatch(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    pin = json.loads(cfg.pin_path.read_text())
    cfg.pin_path.unlink()
    with pytest.raises(SystemExit, match="R18"):
        G.run(cfg, preflight_only=True, verbose=False)
    cfg.pin_path.write_text(json.dumps(pin, indent=1))        # restored byte-identically: clean
    G.run(cfg, preflight_only=True, verbose=False)
    cfg.pin_path.write_text(json.dumps(pin))                  # same content, different bytes
    with pytest.raises(SystemExit, match="LoTTE pin .* uncommitted changes"):
        G.run(cfg, preflight_only=True, verbose=False)
    cfg.pin_path.write_text(json.dumps(pin, indent=1))
    # the slice's text changes but its counts do not: only the pin can see it
    d = cfg.remediated_dir / "science" / "dev"
    lines = (d / "collection.tsv").read_text().splitlines()
    lines[3] = lines[3].split("\t")[0] + "\tan altered passage"
    (d / "collection.tsv").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="science/dev: the slice on disk does not match .*doc_texts_sha256"):
        G.run(cfg, verbose=False)
    assert not cfg.record_path.exists()


def test_a_slice_whose_counts_differ_from_the_registration_refuses_before_scoring(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32, count_error="science/dev")
    with pytest.raises(SystemExit, match="science/dev .* is not the registered slice"):
        G.run(cfg, verbose=False)
    assert not cfg.record_path.exists()
    assert cfg.students["E-bs32"].calls < 7, "scoring must stop at the bad slice"


def test_duplicate_qrel_rows_or_positives_refuse(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    d = cfg.remediated_dir / "lifestyle" / "test"
    rows = (d / "qas.forum.jsonl").read_text().splitlines()
    (d / "qas.forum.jsonl").write_text("\n".join(rows + [rows[0]]) + "\n")
    reg = json.loads(cfg.registration_path.read_text())
    pin = json.loads(cfg.pin_path.read_text())
    with pytest.raises(SystemExit, match="duplicate qrel rows"):
        G.read_slice(cfg, "lifestyle/test", reg["surface"]["slices"]["lifestyle/test"],
                     pin["slices"]["lifestyle/test"])
    r0 = json.loads(rows[0])
    r0["answer_pids"] = r0["answer_pids"] + [r0["answer_pids"][0]]
    (d / "qas.forum.jsonl").write_text("\n".join([json.dumps(r0)] + rows[1:]) + "\n")
    with pytest.raises(SystemExit, match="duplicate positives"):
        G.read_slice(cfg, "lifestyle/test", reg["surface"]["slices"]["lifestyle/test"],
                     pin["slices"]["lifestyle/test"])


def test_read_slice_hashes_match_the_freeze_lotte_scheme(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    pin = json.loads(cfg.pin_path.read_text())
    sl = G.read_slice(cfg, "writing/dev", reg["surface"]["slices"]["writing/dev"],
                      pin["slices"]["writing/dev"])
    assert sl["n_queries"] == 12 and set(sl["hashes"]) == set(G.HASH_FIELDS)
    # sorted-key JSON of the doc id list, exactly as m8src/freeze_lotte.sha hashes it
    assert sl["hashes"]["doc_ids_sha256"] == hashlib.sha256(
        json.dumps(sl["doc_ids"], sort_keys=True).encode()).hexdigest()
    assert sl["hashes"] == pin["slices"]["writing/dev"]["hashes"]
    assert all(isinstance(v, dict) and set(v.values()) == {1} for v in sl["qrels"].values())


# ---------------------------------------------------------------------------------- scoring ----

def test_queries_whose_ids_equal_their_positives_are_still_scored(tmp_path):
    """LoTTE's qids and pids share the integer space; BEIR's self-hit rule in
    `evalkit.run_from_arrays` would drop query j's positive j. The `q:` namespace keeps it."""
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    pin = json.loads(cfg.pin_path.read_text())
    sl = G.read_slice(cfg, "lifestyle/dev", reg["surface"]["slices"]["lifestyle/dev"],
                      pin["slices"]["lifestyle/dev"])
    dv, _ = DocEncoder()("x", sl["doc_texts"])
    s = G.score_slice(cfg, Student(1.0), sl, dv)
    assert set(s["ndcg10"]) == set(sl["q_ids"]) and set(s["success5"]) == set(sl["q_ids"])
    assert all(v == 1.0 for v in s["ndcg10"].values()), s["ndcg10"]
    assert all(v == 1.0 for v in s["success5"].values())
    bad = G.score_slice(cfg, Student(0.0), sl, dv)
    assert np.mean(list(bad["ndcg10"].values())) < 0.5


def test_a_missing_query_refusal_prints_a_count_not_identifiers(tmp_path, monkeypatch):
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    pin = json.loads(cfg.pin_path.read_text())
    sl = G.read_slice(cfg, "lifestyle/dev", reg["surface"]["slices"]["lifestyle/dev"],
                      pin["slices"]["lifestyle/dev"])
    dv, _ = DocEncoder()("x", sl["doc_texts"])
    import evalkit
    real = evalkit.per_query_ndcg
    monkeypatch.setattr(evalkit, "per_query_ndcg",
                        lambda run, qrels, cut=10: {k: v for k, v in real(run, qrels, cut).items()
                                                    if not k.endswith(":3")})
    with pytest.raises(SystemExit) as e:
        G.score_slice(cfg, Student(1.0), sl, dv)
    msg = str(e.value)
    assert "1 of 12 queries were not scored" in msg and "q:" not in msg


def test_success_at_k_reads_the_top_k_by_score():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4}
    assert G.success_at_k(scores, {"e": 1}, 5) == 1.0
    assert G.success_at_k(scores, {"f": 1}, 5) == 0.0
    assert G.success_at_k(scores, {"a": 1}, 1) == 1.0


def test_the_paired_bootstrap_is_paired_deterministic_and_slice_macro():
    rng = np.random.default_rng(0)
    cand, comp = {}, {}
    for k, n in (("s1", 50), ("s2", 30), ("s3", 80)):          # unequal sizes: slice weighting
        q = [f"{i}" for i in range(n)]
        x = rng.uniform(size=n)
        cand[k] = dict(zip(q, x))
        comp[k] = dict(zip(q, x - (0.01 if k != "s3" else 0.04)))
    a = G.paired_bootstrap(cand, comp, B=2000, seed=903)
    b = G.paired_bootstrap(cand, comp, B=2000, seed=903)
    assert a == b and a["plan_sha256"] == b["plan_sha256"] and a["draws_sha256"] == b["draws_sha256"]
    # the estimand is the MACRO of slice means (0.01, 0.01, 0.04 -> 0.02), not the pooled mean
    assert abs(a["delta_macro_raw"] - 0.02) < 1e-12
    pooled = (50 * 0.01 + 30 * 0.01 + 80 * 0.04) / 160
    assert abs(a["delta_macro_raw"] - pooled) > 1e-3
    assert abs(a["upper_q975_raw"] - 0.02) < 1e-9, "constant paired deltas have no spread"
    assert a["quantile_method"] == "inverted_cdf"
    c = G.paired_bootstrap(cand, comp, B=2000, seed=904)
    assert c["plan_sha256"] != a["plan_sha256"] and c["delta_macro_raw"] == a["delta_macro_raw"]
    shuffled = {k: dict(reversed(list(v.items()))) for k, v in comp.items()}
    assert G.paired_bootstrap(cand, shuffled, B=2000, seed=903)["draws_sha256"] == a["draws_sha256"]
    with pytest.raises(ValueError, match="different qids"):
        comp2 = {k: dict(v) for k, v in comp.items()}
        comp2["s1"].pop("0")
        G.paired_bootstrap(cand, comp2, B=10, seed=903)
    with pytest.raises(ValueError, match="slices differ"):
        G.paired_bootstrap(cand, {"s1": comp["s1"]}, B=10, seed=903)


def test_the_upper_bound_method_matters_at_the_boundary_which_is_why_r17_pins_it():
    rng = np.random.default_rng(3)
    q = [f"{i}" for i in range(40)]
    x = rng.uniform(size=40)
    cand = {"s": dict(zip(q, x))}
    comp = {"s": dict(zip(q, x + rng.normal(scale=0.05, size=40)))}
    a = G.paired_bootstrap(cand, comp, B=10000, seed=903)
    b = G.paired_bootstrap(cand, comp, B=10000, seed=903, method="higher")
    assert b["upper_q975_raw"] >= a["upper_q975_raw"]
    assert a["draws_sha256"] == b["draws_sha256"], "the draws are identical; only the read-out differs"


def test_the_veto_rule_needs_both_the_point_and_the_bound():
    plan = {"comparator": {"arm": "E-bs32"}}
    assert G.decide(plan, {"delta_macro_raw": -0.01, "upper_q975_raw": -0.005}, 0.004) == ("veto", True)
    assert G.decide(plan, {"delta_macro_raw": -0.01, "upper_q975_raw": -0.003}, 0.004) == ("no_veto", False)
    assert G.decide(plan, {"delta_macro_raw": -0.003, "upper_q975_raw": -0.005}, 0.004) == ("no_veto", False)
    assert G.decide({"comparator": None}, None, 0.004) == ("skipped", None)


# ------------------------------------------------------------------------------ the two branches

def test_the_bs32_branch_skips_the_veto_and_still_reads_the_observational_row(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32, quality={"E-bs32": 1.0, "E-bs128": 0.0})
    rec = G.run(cfg, verbose=False)
    assert rec["executed"] is True and rec["branch"] == "bs32" and rec["decision"] == "skipped"
    assert rec["comparator_sha256"] is None and rec["comparator"] is None
    assert rec["candidate"]["arm"] == "E-bs32"
    assert rec["delta_candidate_minus_comparator"] is None and rec["bootstrap_upper_bound"] is None
    assert set(rec["macro_ndcg10"]) == {"candidate"} and set(rec["success_at_5"]) == {"candidate"}
    assert rec["macro_ndcg10"]["candidate"]["macro"] == 1.0
    assert set(rec["macro_ndcg10"]["candidate"]["per_slice"]) == set(SLICES)
    assert rec["surface"]["n_slices"] == 7 and rec["surface"]["total_queries"] == 7 * 12
    assert cfg.students["E-bs128"].calls == 0, "the bs32 branch loads one student"
    assert rec["manifest_sha256"] == G.sha256_file(cfg.manifest_path)
    assert rec["pin_sha256"] == G.sha256_file(cfg.pin_path) and rec["pin_commit"]
    assert rec["dependencies"] == {"candidate": DEP}
    g = BD.check_gate(cfg.record_path, verdicts_path=cfg.verdicts_path, e1_batch=32,
                      arm_records_dir=cfg.arm_records_dir)
    assert g["decision"] == "skipped" and g["arm_records_checked"] is True
    assert BD.check_gate_manifest(g, manifest_path=cfg.manifest_path, smoke=True)["sha256"] == rec["manifest_sha256"]
    with pytest.raises(SystemExit, match="E1 selected bs128"):
        BD.check_gate(cfg.record_path, verdicts_path=cfg.verdicts_path, e1_batch=128,
                      arm_records_dir=cfg.arm_records_dir)


def test_the_bs128_branch_vetoes_a_clearly_worse_candidate(tmp_path):
    cfg = build_world(tmp_path, e1_batch=128, quality={"E-bs32": 1.0, "E-bs128": 0.2})
    rec = G.run(cfg, verbose=False)
    assert rec["branch"] == "bs128" and rec["decision"] == "veto" and rec["veto_fired"] is True
    assert rec["candidate"]["arm"] == "E-bs128" and rec["comparator"]["arm"] == "E-bs32"
    boot = rec["delta_candidate_minus_comparator"]
    assert boot["B"] == 10000 and boot["seed"] == 903 and boot["paired_within_slice"] is True
    assert boot["quantile_method"] == "inverted_cdf"
    assert boot["delta_macro_raw"] < -0.004 and boot["upper_q975_raw"] < -0.004
    assert rec["bootstrap_upper_bound"] == boot["upper_q975_raw"]
    assert set(boot["n_by_slice"]) == set(SLICES) and all(n == 12 for n in boot["n_by_slice"].values())
    assert "bs32" in rec["veto"]["consequence"]
    g = BD.check_gate(cfg.record_path, verdicts_path=cfg.verdicts_path, e1_batch=128,
                      arm_records_dir=cfg.arm_records_dir)
    assert g["decision"] == "veto" and g["comparator_sha256"] == rec["comparator_sha256"]
    assert BD.check_gate_manifest(g, manifest_path=cfg.manifest_path, smoke=True)["branch"] == "bs128"


def test_the_bs128_branch_does_not_veto_an_equal_candidate(tmp_path):
    cfg = build_world(tmp_path, e1_batch=128, quality={"E-bs32": 0.7, "E-bs128": 0.7})
    rec = G.run(cfg, verbose=False)
    assert rec["decision"] == "no_veto" and rec["veto_fired"] is False
    boot = rec["delta_candidate_minus_comparator"]
    assert boot["delta_macro_raw"] == 0.0 and boot["upper_q975_raw"] == 0.0
    assert rec["macro_ndcg10"]["candidate"] == rec["macro_ndcg10"]["comparator"]
    g = BD.check_gate(cfg.record_path, verdicts_path=cfg.verdicts_path, e1_batch=128,
                      arm_records_dir=cfg.arm_records_dir)
    assert g["decision"] == "no_veto"


# ------------------------------------------------------------------- lock, receipt and recovery

def test_a_crashed_read_leaves_a_receipt_and_refuses_a_plain_rerun(tmp_path):
    cfg = _crash_then_recover_world(tmp_path)
    receipt = cfg.gate_work_dir / "receipt.json"
    assert receipt.exists() and not cfg.record_path.exists()
    persisted = sorted(p.name for p in cfg.gate_work_dir.glob("slice-*.json"))
    assert len(persisted) == 4, persisted
    journal = (cfg.gate_work_dir / "slices.jsonl").read_text().splitlines()
    assert len(journal) == 4
    with pytest.raises(SystemExit, match="a read has already STARTED"):
        G.run(cfg, verbose=False)
    with pytest.raises(SystemExit, match="a read has already STARTED"):
        G.run(cfg, preflight_only=True, verbose=False)


def test_recover_completes_the_same_read_without_re_reading_finished_slices(tmp_path):
    cfg = _crash_then_recover_world(tmp_path)
    calls_before = list(cfg.enc.calls)
    assert len(calls_before) == 5                                # four scored, the fifth crashed
    rec = G.run(cfg, recover=True, verbose=False)
    new_calls = cfg.enc.calls[len(calls_before):]
    assert len(new_calls) == 3 and not any("lifestyle-test" in c for c in new_calls), \
        "only the three unfinished slices are read"
    assert rec["receipt"]["recovered"] is True and len(rec["receipt"]["recovered_slices"]) == 4
    assert rec["receipt"]["attempts_including_this"] == 2
    assert set(rec["macro_ndcg10"]["candidate"]["per_slice"]) == set(SLICES)
    # the recovered record is what an uninterrupted read would have produced
    cfg2 = build_world(tmp_path / "clean", e1_batch=128, quality={"E-bs32": 1.0, "E-bs128": 0.5})
    rec2 = G.run(cfg2, verbose=False)
    for f in ("decision", "macro_ndcg10", "success_at_5", "bootstrap_upper_bound"):
        assert rec[f] == rec2[f], f
    with pytest.raises(SystemExit, match="ONE access"):
        G.run(cfg, recover=True, verbose=False)


def test_recover_refuses_a_different_identity(tmp_path):
    cfg = _crash_then_recover_world(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "code_identity", lambda: "9" * 64)
        with pytest.raises(SystemExit, match="identity differs .*code_identity"):
            G.run(cfg, recover=True, verbose=False)
    cfg2 = build_world(tmp_path / "fresh", e1_batch=128)
    with pytest.raises(SystemExit, match="no receipt"):
        G.run(cfg2, recover=True, verbose=False)


def test_recover_refuses_an_edited_or_unjournaled_slice_file(tmp_path):
    """Sol 2026-09-10, finding 8: a persisted slice is evidence only if its digest was journaled
    when it was written and its content is well-formed."""
    cfg = _crash_then_recover_world(tmp_path)
    p = G._slice_path(cfg, "lifestyle/test")
    s = json.loads(p.read_text())
    s["per_role"]["candidate"]["ndcg10"][next(iter(s["per_role"]["candidate"]["ndcg10"]))] = 0.0
    p.write_text(json.dumps(s))                                # valid JSON, unjournaled bytes
    with pytest.raises(SystemExit, match="not the file this read journaled"):
        G.run(cfg, recover=True, verbose=False)
    # a journaled but malformed file (a score out of range) also refuses
    cfg2 = _crash_then_recover_world(tmp_path / "b")
    p2 = G._slice_path(cfg2, "lifestyle/test")
    s2 = json.loads(p2.read_text())
    s2["per_role"]["candidate"]["ndcg10"][next(iter(s2["per_role"]["candidate"]["ndcg10"]))] = 1.5
    G.write_atomic(p2, s2)
    G._journal(cfg2, "lifestyle/test", G.sha256_file(p2))
    with pytest.raises(SystemExit, match="rows are malformed"):
        G.run(cfg2, recover=True, verbose=False)


def test_recover_refuses_when_a_journaled_slice_file_is_missing(tmp_path):
    """Astra 2026-09-10 re-check, P1: a completed slice whose file was lost is never re-read."""
    cfg = _crash_then_recover_world(tmp_path)
    G._slice_path(cfg, "lifestyle/test").unlink()
    calls = len(cfg.enc.calls)
    with pytest.raises(SystemExit, match="journaled it as completed; a completed slice is never re-read"):
        G.run(cfg, recover=True, verbose=False)
    assert len(cfg.enc.calls) == calls, "nothing was re-read"


def test_a_second_process_cannot_read_or_recover_while_the_first_holds_the_lock(tmp_path):
    """Sol 2026-09-10, finding 1: O_EXCL protects creation only; the flock protects the attempt."""
    cfg = _crash_then_recover_world(tmp_path)
    fd = os.open(cfg.gate_work_dir / "lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)             # "the first process is still reading"
    try:
        with pytest.raises(SystemExit, match="another gate process holds"):
            G.run(cfg, recover=True, verbose=False)
    finally:
        os.close(fd)
    rec = G.run(cfg, recover=True, verbose=False)              # released: the recovery proceeds
    assert rec["receipt"]["recovered"] is True
    cfg2 = build_world(tmp_path / "b", e1_batch=32)
    plan = G.preflight(cfg2, verbose=False)
    G._create_receipt(cfg2, plan)
    with pytest.raises(FileExistsError):
        G._create_receipt(cfg2, plan)


def test_a_code_change_during_the_read_writes_no_record(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    ids = iter([G.code_identity(), "8" * 64, "8" * 64])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "code_identity", lambda: next(ids))
        with pytest.raises(SystemExit, match="code identity changed during the read"):
            G.run(cfg, verbose=False)
    assert not cfg.record_path.exists()


# ----------------------------------------------------------------------------------- record ----

def test_the_record_carries_no_query_text_and_the_per_query_rows_stay_in_the_tree(tmp_path):
    cfg = build_world(tmp_path, e1_batch=128, quality={"E-bs32": 1.0, "E-bs128": 0.5})
    rec = G.run(cfg, verbose=False)
    blob = json.dumps(rec)
    assert "question " not in blob and "passage " not in blob and "q_texts" not in blob
    assert "answer_pids" not in blob
    pq = Path(rec["perquery"]["path"])
    assert pq.parent == cfg.gate_work_dir and pq.exists()
    assert rec["perquery"]["sha256"] == G.sha256_file(pq)
    per = json.loads(pq.read_text())["per_role"]
    assert set(per) == {"candidate", "comparator"}
    assert set(per["candidate"]["ndcg10"]) == set(SLICES)
    for key in SLICES:
        assert set(rec["surface"]["slices"][key]) == {"n_docs", "n_queries", "n_qrels_pairs",
                                                      "hashes", "read_relpath"}
        assert rec["surface"]["doc_caches"][key]["n_rows"] == rec["surface"]["slices"][key]["n_docs"]
        assert G._slice_path(cfg, key).exists()
    for f in ("e1_verdict_sha256", "candidate_sha256", "comparator_sha256", "read_at",
              "code_identity", "registration_sha256", "manifest_sha256", "manifest_commit",
              "pin_sha256", "pin_commit", "git_head", "environment"):
        assert rec[f], f
    assert rec["e1_verdict_sha256"] == G.sha256_file(cfg.verdicts_path)
    assert rec["registration_sha256"] == G.sha256_file(cfg.registration_path)
    assert rec["code_identity"] == G.code_identity()
    assert rec["receipt"]["attempts_including_this"] == 1 and rec["receipt"]["recovered"] is False
    lines = (cfg.gate_work_dir / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["decision"] == rec["decision"]


def test_the_written_record_is_byte_for_byte_what_run_returned(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    rec = G.run(cfg, verbose=False)
    assert json.loads(cfg.record_path.read_text()) == json.loads(json.dumps(rec, default=str))


def test_the_code_identity_covers_the_four_files():
    h = hashlib.sha256()
    for name in ("m13src/lotte_gate13.py", "m10src/nano10.py", "m7src/evalkit.py", "m7src/teacher.py"):
        h.update((REPO / name).read_bytes())
    assert G.code_identity() == h.hexdigest()


# ------------------------------------------------------------------------------------ guard ----

def test_the_guard_entry_is_this_modules_own_and_covers_lotte_only():
    """In a fresh interpreter: before the claim, classifying a `work/lotte` path refuses; after
    `claim_lotte()` it is permitted and `m9reserve` still is not. Nothing is opened."""
    code = f"""
import sys
sys.path[:0] = {[str(REPO / p) for p in ("m13src", "m8src", "m7src", "m10src")]!r}
import lotte_gate13 as G, paths_guard
p = G.REPO / "work" / "lotte" / "remediated" / "x" / "y" / "collection.tsv"
paths_guard.install()
try:
    paths_guard.check(p); print("UNGUARDED")
except paths_guard.ProtectedPathRefusal:
    print("refused-before-claim")
c = G.claim_lotte()
print(c["entry"], c["kinds"])
print(paths_guard.check(p))
try:
    paths_guard.check(G.REPO / "work" / "m9reserve" / "x.json"); print("UNGUARDED")
except paths_guard.ProtectedPathRefusal:
    print("m9reserve-refused")
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       cwd=REPO, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout.strip().splitlines()
    assert out == ["refused-before-claim", "m13src.lotte_gate13 ['lotte']", "lotte",
                   "m9reserve-refused"], r.stdout


def test_the_allowlist_entry_is_named_in_the_ledger():
    import paths_guard
    assert G.GUARD_ENTRY in paths_guard.ALLOWLIST
    assert paths_guard.ALLOWLIST[G.GUARD_ENTRY]["kinds"] == {"lotte"}
    assert "lotte_gate13" in (REPO / "m8" / "LEDGER.md").read_text()


def test_the_cli_defaults_to_cuda_and_offers_the_three_modes():
    ap = G.build_argparser()
    a = ap.parse_args([])
    assert a.device == "cuda" and not a.preflight_only and not a.recover and not a.write_manifest
    assert ap.parse_args(["--preflight-only"]).preflight_only is True
    assert ap.parse_args(["--recover"]).recover is True
    assert ap.parse_args(["--write-manifest"]).write_manifest is True
