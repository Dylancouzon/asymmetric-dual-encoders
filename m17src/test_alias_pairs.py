"""Alias training-pair mining: the admission rules, the ambiguity refusals and the spot-check.

Synthetic fixtures in `tmp_path` throughout; no real store, glossary clone, result file or
protected surface is read.
"""
from __future__ import annotations

import json

import alias_pairs as A
import support_manifest as S


# --------------------------------------------------------------------------- definition rule

def test_initial_matched_definition_is_mined():
    assert ("Container Storage Interface", "CSI") in A.find_definitions(
        "The Container Storage Interface (CSI) defines a standard.")


def test_initials_that_do_not_match_are_rejected():
    assert A.find_definitions("The Quick Brown Fox (XYZ) jumped.") == []


def test_function_words_may_be_skipped_when_matching_initials():
    got = A.find_definitions("The Bank of England (BOE) raised rates.")
    assert any(a == "BOE" for _, a in got)


def test_ordinary_english_words_are_never_treated_as_abbreviations():
    assert A.find_definitions("Information Technology (IT) departments exist.") == []
    assert "it" in A.ABBR_DENY


def test_overlong_and_overshort_abbreviations_are_rejected():
    assert A.find_definitions("Alpha (A) beta.") == []
    assert A.find_definitions("A B C D E F G H I (ABCDEFGHI) x.") == []


# --------------------------------------------------------------------------- pair construction

def _pairs(text, entries, rule="R2_abbreviation_expansion"):
    return A.make_pairs(text, "g0", "squad-train", entries, rule, {})


def test_a_pair_is_two_query_views_not_two_bare_terms():
    text = ("The Container Storage Interface (CSI) lets storage vendors write plugins. "
            "Every Container Storage Interface driver runs in its own pod today.")
    out = _pairs(text, [("Container Storage Interface", "CSI")])
    assert out, "expected at least one carrier"
    p = out[0]
    assert "Container Storage Interface" in p["view_a"] and "CSI" in p["view_b"]
    assert len(p["view_a"].split()) >= A.MIN_CARRIER_WORDS


def test_the_parenthetical_definition_is_stripped_before_substituting():
    text = "The Container Storage Interface (CSI) defines a standard interface for storage."
    p = _pairs(text, [("Container Storage Interface", "CSI")])[0]
    assert "(CSI)" not in p["view_a"] and "CSI (CSI)" not in p["view_b"]


def test_a_carrier_with_two_occurrences_is_skipped():
    text = ("Container Storage Interface plugins use the Container Storage Interface protocol "
            "for every attach and detach call made by the kubelet.")
    assert _pairs(text, [("Container Storage Interface", "CSI")]) == []


def test_overlong_carriers_are_skipped():
    text = "Container Storage Interface " + " ".join(f"w{i}" for i in range(60)) + "."
    assert _pairs(text, [("Container Storage Interface", "CSI")]) == []


def test_substitution_must_change_the_view():
    c = {}
    assert A.make_pairs("k8s ingress routes traffic into the cluster today.",
                        "g0", "k8s-docs-en", [("k8s", "k8s")], "R1_k8s_glossary_aka", c) == []
    assert c["substitution_noop"] == 1


def test_family_id_uses_the_step_one_scheme_rooted_at_the_carrier_document():
    p = _pairs("The Pod Disruption Budget (PDB) limits voluntary disruption for an application.",
               [("Pod Disruption Budget", "PDB")])[0]
    assert p["family_id"] == S.group_id("docgroup:g0")
    assert len(p["family_id"]) == 16


def test_pair_id_is_a_function_of_the_normalized_views():
    p = _pairs("The Pod Disruption Budget (PDB) limits voluntary disruption for an application.",
               [("Pod Disruption Budget", "PDB")])[0]
    assert p["pair_id"] == S.group_id(S.normalize(p["view_a"]) + "\x00" + S.normalize(p["view_b"]))


# --------------------------------------------------------------------------- glossary rule

def test_glossary_aka_forms_are_read_from_the_pinned_front_matter(tmp_path, monkeypatch):
    g = tmp_path / "glossary"
    g.mkdir()
    (g / "hpa.md").write_text(
        "---\ntitle: Horizontal Pod Autoscaler\nid: hpa\naka: \n- HPA\ntags:\n- x\n---\nbody\n")
    (g / "flow.md").write_text('---\ntitle: Group Version Resource\naka: ["GVR"]\n---\nbody\n')
    (g / "empty.md").write_text("---\ntitle: Cluster\naka: \ntags:\n- y\n---\nbody\n")
    monkeypatch.setattr(A, "GLOSSARY", g)
    got = dict(A.glossary_aliases())
    assert got["Horizontal Pod Autoscaler"] == "HPA"
    assert got["Group Version Resource"] == "GVR"
    assert "Cluster" not in got, "an empty aka field is not an alias"


def test_missing_glossary_directory_yields_no_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "GLOSSARY", tmp_path / "absent")
    assert A.glossary_aliases() == []


# --------------------------------------------------------------------------- end to end

def _world(tmp_path, monkeypatch, docs, glossary=None, k8s_docs=()):
    train = tmp_path / "train"
    (train / "stores").mkdir(parents=True)
    for src, store in A.MINED_STORES.items():
        rows = docs.get(src, [])
        (train / "stores" / f"{store}.json").write_text(
            json.dumps({"ids": [f"d{i}" for i in range(len(rows))], "texts": list(rows)}))
    k8s = tmp_path / "k8s.jsonl"
    with open(k8s, "w", encoding="utf-8") as f:
        for i, t in enumerate(k8s_docs):
            f.write(json.dumps({"path": f"content/en/docs/{i}.md", "title": "t", "text": t,
                                "sha256": "x"}) + "\n")
    step2a = tmp_path / "step2a.json"
    step2a.write_text(json.dumps({"decontamination": {"result": {"flagged_paths": []}}}))
    monkeypatch.setattr(S, "TRAIN", train)
    monkeypatch.setattr(S, "K8S_JSONL", k8s)
    monkeypatch.setattr(S, "K8S_STEP2A", step2a)
    monkeypatch.setattr(A, "MANIFEST_DIR", tmp_path / "manifest")
    monkeypatch.setattr(A, "POOL", tmp_path / "manifest" / "alias_pairs.jsonl")
    monkeypatch.setattr(A, "SUMMARY", tmp_path / "manifest" / "summary.json")
    monkeypatch.setattr(A, "SPOTCHECK", tmp_path / "spotcheck.jsonl")
    monkeypatch.setattr(A, "EXCLUDE_DEFAULT", tmp_path / "manifest" / "alias_test_families.json")
    monkeypatch.setattr(A, "GLOSSARY", glossary or (tmp_path / "absent"))
    return tmp_path


def test_ambiguous_abbreviation_is_never_expanded_globally(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, docs={"squad-train": [
        "The Container Storage Interface (CSI) is one standard used across many clusters.",
        "The Crime Scene Investigation (CSI) team arrived at the building before dawn.",
    ]})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    assert s["counters"]["ambiguous_abbreviations_dropped"] == 1
    assert s["pairs"] == 0


def test_unambiguous_definition_produces_pairs_and_a_summary(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, docs={"squad-train": [
        "The Pod Disruption Budget (PDB) limits how many pods go down at once.",
        "Every Pod Disruption Budget is evaluated by the eviction API before a drain.",
    ]})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    assert s["pairs"] >= 1
    assert s["pairs_by_rule"]["R2_abbreviation_expansion"] == s["pairs"]
    assert s["protected_payloads_opened"] is False and s["gpu_used"] is False


def test_registered_rules_without_material_are_reported_as_gaps(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, docs={})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    assert {g["rule"] for g in s["rule_gaps"]} == {"R3_dataset_paraphrases",
                                                   "R4_wikipedia_redirects"}


def test_absent_exclusion_file_proceeds_with_a_note(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, docs={"squad-train": [
        "The Pod Disruption Budget (PDB) limits how many pods go down at once.",
        "Every Pod Disruption Budget is evaluated by the eviction API before a drain."]})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    assert s["excluded_families"]["applied"] is False
    assert "MUST be re-filtered" in s["excluded_families"]["note"]


def test_excluded_families_are_removed_when_the_file_exists(tmp_path, monkeypatch):
    root = _world(tmp_path, monkeypatch, docs={"squad-train": [
        "The Pod Disruption Budget (PDB) limits how many pods go down at once.",
        "Every Pod Disruption Budget is evaluated by the eviction API before a drain."]})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    fams = sorted({json.loads(l)["family_id"] for l in open(A.POOL, encoding="utf-8")})
    ex = root / "exclude.json"
    ex.write_text(json.dumps({"families": fams}))
    s2 = A.build(seed=0, sample_fraction=0.02, exclude_path=str(ex))
    assert s["pairs"] > 0 and s2["pairs"] == 0
    assert s2["counters"]["pairs_removed_by_excluded_families"] == s["pairs"]


def test_spotcheck_sample_is_seeded_and_reproducible(tmp_path, monkeypatch):
    docs = ["The Pod Disruption Budget (PDB) limits how many pods go down at once."]
    docs += [f"Every Pod Disruption Budget is checked by component number {i} before a drain."
             for i in range(40)]
    _world(tmp_path, monkeypatch, docs={"squad-train": docs})
    a = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    first = open(A.SPOTCHECK, encoding="utf-8").read()
    b = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    assert first == open(A.SPOTCHECK, encoding="utf-8").read()
    assert a["spotcheck"]["n"] == b["spotcheck"]["n"] >= 1
    c = A.build(seed=1, sample_fraction=0.02, exclude_path=None)
    assert c["spotcheck"]["n"] == a["spotcheck"]["n"]


def test_duplicate_views_are_deduplicated_across_documents(tmp_path, monkeypatch):
    doc = ("The Pod Disruption Budget (PDB) limits how many pods go down at once. "
           "Every Pod Disruption Budget is evaluated by the eviction API before a drain.")
    _world(tmp_path, monkeypatch, docs={"squad-train": [doc, doc]})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    assert s["counters"]["duplicate_pairs_dropped"] >= 1


def test_esci_is_not_mined():
    assert "esci-us" not in A.MINED_STORES


def test_heldout_text_shas_remove_pairs_even_when_family_ids_differ(tmp_path, monkeypatch):
    root = _world(tmp_path, monkeypatch, docs={"squad-train": [
        "The Pod Disruption Budget (PDB) limits how many pods go down at once.",
        "Every Pod Disruption Budget is evaluated by the eviction API before a drain."]})
    s = A.build(seed=0, sample_fraction=0.02, exclude_path=None)
    shas = [S.group_id(S.normalize(json.loads(l)["view_a"]))
            for l in open(A.POOL, encoding="utf-8")]
    ex = root / "exclude.json"
    ex.write_text(json.dumps({"excluded_family_ids": ["fam:unrelated"],
                              "excluded_text_shas": shas}))
    s2 = A.build(seed=0, sample_fraction=0.02, exclude_path=str(ex))
    assert s["pairs"] > 0 and s2["pairs"] == 0
    assert s2["excluded_families"]["text_shas"] == len(shas)


def test_space_padded_parentheses_are_still_definitions():
    # FEVER's store is pre-tokenized with spaces inside parentheses, so the definition pattern
    # has to tolerate them or that whole source mines zero pairs.
    got = A.find_definitions("the Container Storage Interface ( CSI ) is a standard")
    assert ("Container Storage Interface", "CSI") in got
    assert A.find_definitions("the Quick Brown Fox ( XYZ ) jumped") == []
