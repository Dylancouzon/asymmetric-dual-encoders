"""What LoTTE read #1's executor must refuse, and the arithmetic it must get right — on a synthetic
tree. No test opens `work/lotte`; the fixture writes seven tiny "remediated" slices under a tmp
path, injects a deterministic document encoder in place of stella, and hands the executor stub
students whose quality is a parameter, so both branches, both veto outcomes and every refusal run
on a CPU in seconds. The one real-path check (the guard claim) runs in a subprocess and only
CLASSIFIES a protected path; it opens nothing.
"""
from __future__ import annotations

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


# --------------------------------------------------------------------------------- fixture ----

def _unit(text, dim=DIM):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def doc_vec(doc_text):
    return _unit("doc|" + doc_text)


def fixture_doc_encoder(name, texts):
    """Stands in for `teacher.encode_cached`: deterministic unit vectors, stored fp16 like the
    real caches, plus a cache record."""
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
    is relevant to doc j, and every even j also to doc j + n_queries."""
    d.mkdir(parents=True, exist_ok=True)
    docs = {str(i): f"passage {key} {i} {'lorem' if i % 3 else 'ipsum'}" for i in range(n_docs)}
    with open(d / "collection.tsv", "w") as fh:
        for pid, text in docs.items():
            fh.write(f"{pid}\t{text}\n")
    qas, pairs = [], 0
    with open(d / "questions.forum.tsv", "w") as fh:
        for j in range(n_queries):
            pos = [str(j)] + ([str(j + n_queries)] if j % 2 == 0 and j + n_queries < n_docs else [])
            fh.write(f"{j}\tquestion {key} {j} about " + " || ".join(docs[p] for p in pos) + "\n")
            qas.append({"qid": j, "answer_pids": [int(p) for p in pos]})
            pairs += len(pos)
    with open(d / "qas.forum.jsonl", "w") as fh:
        for row in qas:
            fh.write(json.dumps(row) + "\n")
    return {"queries_after_remedy": n_queries, "docs_after_remedy": n_docs,
            "qrels_pairs_after_remedy": pairs}


def build_world(root, *, e1_batch, quality=None, n_queries=12, count_error=None):
    """-> a Config over a synthetic tree: the REAL registration's structure with the fixture's
    counts, a copy of the real screen registry, an E1 verdict bound to it, two complete E arm
    records with real checkpoint bytes, and seven tiny slices of different sizes."""
    quality = quality or {"E-bs32": 1.0, "E-bs128": 1.0}
    repo = Path(root) / "repo"
    for d in ("m13", "m10", "results", "work/lotte/remediated", "work/m10arms"):
        (repo / d).mkdir(parents=True, exist_ok=True)
    reg = json.loads(json.dumps(REAL_REG))
    for i, key in enumerate(SLICES):
        topic, split = key.split("/")
        counts = write_slice(repo / "work" / "lotte" / "remediated" / topic / split, key,
                             n_docs=40 + 7 * i, n_queries=n_queries)
        if count_error == key:
            counts["docs_after_remedy"] += 1
        reg["surface"]["slices"][key] = counts
    reg["surface"]["total_queries"] = sum(s["queries_after_remedy"]
                                          for s in reg["surface"]["slices"].values())
    (repo / "m13" / "LOTTE_GATE_REGISTRATION.json").write_text(json.dumps(reg, indent=1))
    shutil.copy(REPO / "m10" / "screen_registry.json", repo / "m10" / "screen_registry.json")
    (repo / "results" / "m10_screen_verdicts.json").write_text(json.dumps(
        {"selected": {"batch": e1_batch},
         "registry_sha256": G.sha256_file(repo / "m10" / "screen_registry.json")}))
    for arm, batch in (("E-bs32", 32), ("E-bs128", 128)):
        d = repo / "work" / "m10arms" / arm
        d.mkdir(parents=True, exist_ok=True)
        ck = d / "cycle3.pt"
        ck.write_bytes(hashlib.sha256(arm.encode()).digest() * 64)
        sha = G.sha256_file(ck)
        (repo / "results" / f"m10_arm_{arm}.json").write_text(json.dumps(
            {"arm": arm, "status": "complete", "complete": True, "terminal": True, "smoke": False,
             "recipe": {"student": "bge-small", "n_layers": 3, "head": "linear",
                        "dose_examples": 5_000_000, "batch": batch},
             "params": 33_000_000, "device": "cuda", "git_head": "f" * 40,
             "checkpoints": {"cycle3": {"path": f"work/m10arms/{arm}/cycle3.pt", "sha256": sha}},
             "final_checkpoint": f"work/m10arms/{arm}/cycle3.pt", "final_checkpoint_sha256": sha}))
    students = {arm: Student(q) for arm, q in quality.items()}
    cfg = G.Config(
        repo=repo,
        registration_path=repo / "m13" / "LOTTE_GATE_REGISTRATION.json",
        record_path=repo / "m13" / "LOTTE_GATE.json",
        verdicts_path=repo / "results" / "m10_screen_verdicts.json",
        registry_path=repo / "m10" / "screen_registry.json",
        arm_records_dir=repo / "results",
        remediated_dir=repo / "work" / "lotte" / "remediated",
        enc_root=repo / "work" / "lotte" / "enc",
        gate_work_dir=repo / "work" / "lotte" / "gate13",
        device="cpu", allow_cpu=True, topk=100, chunk=1000,
        doc_encoder=fixture_doc_encoder,
        load_student=lambda ar, device: students[ar["arm"]],
        claim_guard=False)
    cfg.students = students
    return cfg


def _rewrite(path, **over):
    d = json.loads(Path(path).read_text())
    d.update(over)
    Path(path).write_text(json.dumps(d))


# ------------------------------------------------------------------------------ registration ----

def test_the_real_registration_validates_without_opening_lotte():
    reg, sha = G.registration(G.Config())
    assert len(reg["surface"]["slices"]) == 7 and reg["executed"] is False
    assert reg["veto"]["margin"] == 0.004 and reg["veto"]["bootstrap"]["seed"] == 903
    assert sha == G.sha256_file(REPO / "m13" / "LOTTE_GATE_REGISTRATION.json")


def test_the_registration_is_held_to_the_locked_constants(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    reg["veto"]["margin"] = 0.005
    cfg.registration_path.write_text(json.dumps(reg))
    with pytest.raises(SystemExit, match="veto.margin"):
        G.registration(cfg)
    reg["veto"]["margin"] = 0.004
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
    assert not cfg.record_path.exists() and not cfg.gate_work_dir.exists()


def test_a_pending_e1_verdict_refuses(tmp_path):
    cfg = build_world(tmp_path, e1_batch="PENDING")
    with pytest.raises(SystemExit, match="PENDING"):
        G.run(cfg, preflight_only=True, verbose=False)


def test_a_verdict_bound_to_another_registry_refuses(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    _rewrite(cfg.verdicts_path, registry_sha256="0" * 64)
    with pytest.raises(SystemExit, match="bound to registry sha"):
        G.run(cfg, preflight_only=True, verbose=False)


@pytest.mark.parametrize("over, match", [
    ({"status": "failed", "complete": False}, "failed or unfinished"),
    ({"smoke": True}, "SMOKE record"),
    ({"final_checkpoint_sha256": "1" * 64}, "do not name one cycle-3 checkpoint"),
    ({"recipe": {"student": "bge-small"}}, "recipe lacks"),
])
def test_an_arm_record_that_is_not_a_complete_registered_arm_refuses(tmp_path, over, match):
    cfg = build_world(tmp_path, e1_batch=32)
    _rewrite(cfg.arm_records_dir / "m10_arm_E-bs128.json", **over)   # the OTHER arm in bs32, too
    with pytest.raises(SystemExit, match=match):
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


def test_an_existing_gate_record_refuses_a_second_execution(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    G.run(cfg, verbose=False)
    assert cfg.record_path.exists()
    with pytest.raises(SystemExit, match="ONE access"):
        G.run(cfg, verbose=False)
    with pytest.raises(SystemExit, match="ONE access"):
        G.run(cfg, preflight_only=True, verbose=False)


# ----------------------------------------------------------------------------------- slices ----

def test_a_slice_whose_counts_differ_from_the_registration_refuses_before_scoring(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32, count_error="science/dev")
    with pytest.raises(SystemExit, match="science/dev .* is not the registered slice"):
        G.run(cfg, verbose=False)
    assert not cfg.record_path.exists()
    assert all(s.calls == 0 for s in cfg.students.values()) or \
        cfg.students["E-bs32"].calls < 7, "scoring must stop at the bad slice"


def test_read_slice_hashes_match_the_freeze_lotte_scheme(tmp_path):
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    sl = G.read_slice(cfg, "writing/dev", reg["surface"]["slices"]["writing/dev"])
    assert sl["n_queries"] == 12 and set(sl["hashes"]) == {
        "doc_ids_sha256", "doc_texts_sha256", "query_ids_sha256", "query_texts_sha256",
        "qrels_sha256"}
    # sorted-key JSON of the doc id list, exactly as m8src/freeze_lotte.sha hashes it
    assert sl["hashes"]["doc_ids_sha256"] == hashlib.sha256(
        json.dumps(sl["doc_ids"], sort_keys=True).encode()).hexdigest()
    assert all(isinstance(v, dict) and set(v.values()) == {1} for v in sl["qrels"].values())


# ---------------------------------------------------------------------------------- scoring ----

def test_queries_whose_ids_equal_their_positives_are_still_scored(tmp_path):
    """LoTTE's qids and pids share the integer space; BEIR's self-hit rule in
    `evalkit.run_from_arrays` would drop query j's positive j. The `q:` namespace keeps it."""
    cfg = build_world(tmp_path, e1_batch=32)
    reg = json.loads(cfg.registration_path.read_text())
    sl = G.read_slice(cfg, "lifestyle/dev", reg["surface"]["slices"]["lifestyle/dev"])
    dv, _ = fixture_doc_encoder("x", sl["doc_texts"])
    s = G.score_slice(cfg, Student(1.0), sl, dv)
    assert set(s["ndcg10"]) == set(sl["q_ids"]) and set(s["success5"]) == set(sl["q_ids"])
    assert all(v == 1.0 for v in s["ndcg10"].values()), s["ndcg10"]
    assert all(v == 1.0 for v in s["success5"].values())
    bad = G.score_slice(cfg, Student(0.0), sl, dv)
    assert np.mean(list(bad["ndcg10"].values())) < 0.5


def test_success_at_k_reads_the_top_k_by_score():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4}
    assert G.success_at_k(scores, {"e": 1}, 5) == 1.0
    assert G.success_at_k(scores, {"f": 1}, 5) == 0.0
    assert G.success_at_k(scores, {"a": 1}, 1) == 1.0


def test_the_paired_bootstrap_is_paired_deterministic_and_slice_macro():
    rng = np.random.default_rng(0)
    cand, comp = {}, {}
    for k in ("s1", "s2", "s3"):
        q = [f"{i}" for i in range(50)]
        x = rng.uniform(size=50)
        cand[k] = dict(zip(q, x))
        comp[k] = dict(zip(q, x - 0.01))                  # every query exactly +0.01
    a = G.paired_bootstrap(cand, comp, B=2000, seed=903)
    b = G.paired_bootstrap(cand, comp, B=2000, seed=903)
    assert a == b and a["plan_sha256"] == b["plan_sha256"] and a["draws_sha256"] == b["draws_sha256"]
    assert abs(a["delta_macro_raw"] - 0.01) < 1e-12
    assert abs(a["upper_q975_raw"] - 0.01) < 1e-9, "a constant paired delta has no spread"
    # a different seed is a different plan; the point estimate does not move
    c = G.paired_bootstrap(cand, comp, B=2000, seed=904)
    assert c["plan_sha256"] != a["plan_sha256"] and c["delta_macro_raw"] == a["delta_macro_raw"]
    # dict order does not matter: pairing is by qid, slices enter sorted
    shuffled = {k: dict(reversed(list(v.items()))) for k, v in comp.items()}
    assert G.paired_bootstrap(cand, shuffled, B=2000, seed=903)["draws_sha256"] == a["draws_sha256"]
    with pytest.raises(ValueError, match="different qids"):
        comp2 = {k: dict(v) for k, v in comp.items()}
        comp2["s1"].pop("0")
        G.paired_bootstrap(cand, comp2, B=10, seed=903)
    with pytest.raises(ValueError, match="slices differ"):
        G.paired_bootstrap(cand, {"s1": comp["s1"]}, B=10, seed=903)


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
    # the record is what build13 reads, and it passes check_gate for the branch E1 selected
    g = BD.check_gate(cfg.record_path, verdicts_path=cfg.verdicts_path, e1_batch=32,
                      arm_records_dir=cfg.arm_records_dir)
    assert g["decision"] == "skipped" and g["arm_records_checked"] is True
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
    assert boot["delta_macro_raw"] < -0.004 and boot["upper_q975_raw"] < -0.004
    assert rec["bootstrap_upper_bound"] == boot["upper_q975_raw"]
    assert set(boot["n_by_slice"]) == set(SLICES) and all(n == 12 for n in boot["n_by_slice"].values())
    assert "bs32" in rec["veto"]["consequence"]
    g = BD.check_gate(cfg.record_path, verdicts_path=cfg.verdicts_path, e1_batch=128,
                      arm_records_dir=cfg.arm_records_dir)
    assert g["decision"] == "veto" and g["comparator_sha256"] == rec["comparator_sha256"]


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


def test_a_no_veto_that_is_worse_but_not_resolved_does_not_fire(tmp_path):
    """Worse by less than the margin, or worse with a bound above -margin: the E1 selection
    stands. The margin and the bound are both required (m10/LOTTE_LOCK.md)."""
    cfg = build_world(tmp_path, e1_batch=128, quality={"E-bs32": 1.0, "E-bs128": 0.98})
    rec = G.run(cfg, verbose=False)
    boot = rec["delta_candidate_minus_comparator"]
    assert boot["delta_macro_raw"] <= 0.0
    assert rec["decision"] == ("veto" if (boot["delta_macro_raw"] < -0.004 and
                                          boot["upper_q975_raw"] < -0.004) else "no_veto")


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
    for f in ("e1_verdict_sha256", "candidate_sha256", "comparator_sha256", "read_at",
              "code_identity", "registration_sha256", "git_head", "environment"):
        assert rec[f], f
    assert rec["e1_verdict_sha256"] == G.sha256_file(cfg.verdicts_path)
    assert rec["registration_sha256"] == G.sha256_file(cfg.registration_path)
    assert rec["code_identity"] == G.code_identity()
    assert rec["attempts_before_this_record"] == 1
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


def test_the_cli_defaults_to_cuda_and_offers_preflight_only():
    ap = G.build_argparser()
    a = ap.parse_args([])
    assert a.device == "cuda" and a.preflight_only is False
    assert ap.parse_args(["--preflight-only"]).preflight_only is True
