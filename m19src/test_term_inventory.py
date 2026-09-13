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


def test_inventory_counts_distinct_artifacts_and_excludes_events(tmp_path, monkeypatch):
    tokenizer_path = _tokenizer(tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"artifact_id": "a", "kind": "issue_opening", "title": "K8s setup", "text": "k8s k8s"},
        {"artifact_id": "a", "kind": "issue_comment", "title": "", "text": "K8S"},
        {"artifact_id": "b", "kind": "issue_opening", "title": "", "text": "s3 setup"},
        {"artifact_id": "c", "kind": "issue_event", "title": "k8s", "text": "k8s"},
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(term_inventory, "admit_read", lambda path: path)
    catalog = [
        {"term": "k8s", "meaning": "x", "demand": "x", "intent_axes": ["a", "b", "c"]},
        {"term": "s3", "meaning": "x", "demand": "x", "intent_axes": ["a", "b", "c"]},
    ]
    report = term_inventory.inventory(corpus, tokenizer_path, catalog)
    got = {row["term"]: row for row in report["catalog"]}
    assert got["k8s"]["artifact_support"] == 1
    assert got["k8s"]["occurrences"] == 4
    assert got["s3"]["artifact_support"] == 1
    assert report["indexable_documents_scanned"] == 3


def test_selection_obeys_fixed_order_and_stops_below_minimum():
    report = {"catalog": [
        {"term": "first", "fragment_count": 2, "artifact_support": 5,
         "intent_axes": ["a", "b", "c"]},
        {"term": "whole", "fragment_count": 1, "artifact_support": 100,
         "intent_axes": ["a", "b", "c"]},
        {"term": "second", "fragment_count": 3, "artifact_support": 6,
         "intent_axes": ["a", "b", "c"]},
    ]}
    selected, rejected = term_inventory.select_roster(report, minimum=2, maximum=2)
    assert [row["term"] for row in selected] == ["first", "second"]
    assert rejected == [{"term": "whole", "reasons": ["released_tokenizer_not_fragmented"]}]
    with pytest.raises(SystemExit, match="only 2"):
        term_inventory.select_roster(report, minimum=3, maximum=3)


def test_added_token_policy_preserves_old_ids(tmp_path, monkeypatch):
    tokenizer_path = _tokenizer(tmp_path)
    monkeypatch.setattr(term_inventory, "admit_read", lambda path: path)
    audit = term_inventory._added_token_audit(tokenizer_path, ["k8s", "s3"])
    assert audit["base_vocab"] == 9
    assert audit["term_ids"] == {"k8s": 9, "s3": 10}
    assert audit["policy"]["single_word"] is True
