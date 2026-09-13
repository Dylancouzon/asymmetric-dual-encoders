import json

import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from m19src import term_inventory


def _tokenizer(tmp_path):
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "k": 3, "##8": 4, "##s": 5,
             "s": 6, "##3": 7, "whole": 8}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def _catalog():
    return [
        {"term": "k8s", "meaning": "Kubernetes", "demand": "deployment",
         "intent_axes": ["a", "b", "c"]},
        {"term": "s3", "meaning": "object storage", "demand": "snapshots",
         "intent_axes": ["a", "b", "c"]},
    ]


def test_inventory_uses_serving_match_deduplicates_and_excludes_nonindexable(tmp_path, monkeypatch):
    tokenizer_path = _tokenizer(tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"artifact_id": "a", "doc_id": "a:1", "kind": "issue_opening", "title": "K8s setup",
         "text": "k8s k8s", "normalized_text_sha256": "one"},
        {"artifact_id": "a", "doc_id": "a:2", "kind": "issue_comment", "title": "",
         "text": "K8S", "normalized_text_sha256": "two"},
        {"artifact_id": "b", "doc_id": "b:1", "kind": "issue_opening", "title": "",
         "text": "s3 setup", "normalized_text_sha256": "three"},
        {"artifact_id": "c", "doc_id": "c:1", "kind": "issue_event", "title": "k8s",
         "text": "k8s", "normalized_text_sha256": "four"},
        {"artifact_id": "d", "doc_id": "d:1", "kind": "issue_opening", "title": "copy",
         "text": "s3 copy", "normalized_text_sha256": "copy"},
        {"artifact_id": "e", "doc_id": "e:1", "kind": "issue_opening", "title": "copy",
         "text": "s3 copy", "normalized_text_sha256": "copy"},
        {"artifact_id": "f", "doc_id": "f:1", "kind": "issue_opening", "title": "",
         "text": "s3 hidden", "normalized_text_sha256": "hidden", "indexable": False},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    corpus.write_bytes(payload)
    monkeypatch.setattr(term_inventory, "admit_read", lambda path: path)
    report, packets = term_inventory.inventory(corpus, tokenizer_path, _catalog())
    got = {row["term"]: row for row in report["catalog"]}
    assert got["k8s"]["artifact_support"] == 1
    assert got["k8s"]["raw_serving_match_occurrences"] == 4
    assert got["s3"]["artifact_support"] == 2  # b plus one representative of copied d/e
    assert {item["artifact_id"] for item in packets["s3"]} == {"b", "d"}
    assert report["indexable_documents_scanned"] == 5
    assert report["consumed_corpus_sha256"] == term_inventory.sha_bytes(payload)


def test_real_added_token_boundary_normalization_fixtures(tmp_path, monkeypatch):
    tokenizer_path = _tokenizer(tmp_path)
    monkeypatch.setattr(term_inventory, "admit_read", lambda path: path)
    payload, base = term_inventory._load_tokenizer_bytes(tokenizer_path)
    matcher, audit = term_inventory._extend_tokenizer(payload, ["s3"])
    token_id = audit["term_ids"]["s3"]
    for text in ("s3", "S3", "(s3)", "s3's", "s3-compatible", "s\u200b3"):
        assert token_id in matcher.encode(text, add_special_tokens=False).ids, text
    for text in ("xs3", "s3x", "_s3", "s3_bucket", "s3s"):
        assert token_id not in matcher.encode(text, add_special_tokens=False).ids, text
    assert audit["inherited_ids_unchanged"] is True
    assert base.get_vocab(with_added_tokens=True)["s"] == 6


def test_selection_enforces_semantics_distinct_axes_and_fixed_order():
    report = {"catalog": [
        {"term": "first", "fragment_count": 2, "artifact_support": 5, "meaning": "m",
         "demand": "d", "intent_axes": ["a", "b", "c"]},
        {"term": "bad", "fragment_count": 2, "artifact_support": 100, "meaning": "",
         "demand": "", "intent_axes": ["a", "a", "a"]},
        {"term": "second", "fragment_count": 3, "artifact_support": 6, "meaning": "m",
         "demand": "d", "intent_axes": ["a", "b", "c"]},
    ]}
    selected, rejected = term_inventory.select_roster(report, minimum=2, maximum=2)
    assert [row["term"] for row in selected] == ["first", "second"]
    assert rejected[0]["term"] == "bad"
    assert set(rejected[0]["reasons"]) == {
        "fewer_than_three_distinct_credible_intent_axes", "missing_stable_meaning",
        "missing_demand_basis",
    }


def test_source_qualification_requires_five_non_equivalent_witnesses():
    selected = [{"term": "s3", "intent_axes": ["a", "b", "c"]}]
    packets = {"s3": [
        {"artifact_id": f"a{i}", "witness_digest": f"d{i}"} for i in range(5)
    ]}
    valid = {"s3": {
        "witness_artifact_ids": [f"a{i}" for i in range(5)],
        "intent_evidence": {"a": ["a0"], "b": ["a1"], "c": ["a2"]},
    }}
    assert term_inventory._qualify(selected, packets, valid)["s3"]["witness_digests"]
    invalid = json.loads(json.dumps(valid))
    invalid["s3"]["witness_artifact_ids"][-1] = "a3"
    with pytest.raises(SystemExit, match="five distinct"):
        term_inventory._qualify(selected, packets, invalid)


def test_consumed_input_mismatch_stops_mutation_during_scan():
    inheritance = {"inherited_data": {
        "corpus_jsonl": {"sha256": "corpus-a"},
        "zero_v1_tokenizer": {"sha256": "tokenizer-a"},
    }}
    with pytest.raises(SystemExit, match="consumed_corpus"):
        term_inventory._bind_consumed({
            "consumed_corpus_sha256": "corpus-b", "consumed_tokenizer_sha256": "tokenizer-a",
        }, inheritance)


def test_pair_publication_resumes_identical_first_output(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    one, two = {"id": 1}, {"id": 2}
    first.write_text(json.dumps(one))
    monkeypatch.setattr(term_inventory, "load_json", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(
        term_inventory, "create_json",
        lambda path, obj: path.write_text(json.dumps(obj)),
    )
    assert term_inventory._publish_or_verify(first, one) == "verified"
    assert term_inventory._publish_or_verify(second, two) == "created"
    assert json.loads(first.read_text()) == one
    assert json.loads(second.read_text()) == two


def test_competing_publisher_cannot_replace(monkeypatch, tmp_path):
    path = tmp_path / "lock.json"
    monkeypatch.setattr(term_inventory, "load_json", lambda target: json.loads(target.read_text()))

    def compete(target, obj):
        target.write_text(json.dumps({"other": True}))
        raise FileExistsError(target)

    monkeypatch.setattr(term_inventory, "create_json", compete)
    with pytest.raises(SystemExit, match="competing immutable output differs"):
        term_inventory._publish_or_verify(path, {"ours": True})
    assert json.loads(path.read_text()) == {"other": True}
