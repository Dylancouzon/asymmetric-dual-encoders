from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from m19src import queries
from m19src.term_inventory import _extend_tokenizer


def _fixture():
    base = Tokenizer(models.WordPiece({"[UNK]": 0, "k": 1, "##8": 2, "##s": 3,
                                       "s": 4, "##3": 5}, unk_token="[UNK]"))
    base.normalizer = normalizers.BertNormalizer(lowercase=True)
    base.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    payload = base.to_str().encode()
    tokenizer, audit = _extend_tokenizer(payload, ["k8s", "s3"])
    roster = {"terms": [{"term": "k8s"}, {"term": "s3"}],
              "selected_added_token_audit": audit}
    registry = {
        "query_counts_per_term": {"development_bare": 1, "development_short": 1,
                                  "development_longer": 1, "confirmation_short": 1,
                                  "confirmation_longer_per_two_terms": 1},
        "query_protocol": {"development_numeric_version_controls_minimum": 2,
                           "confirmation_numeric_version_controls_minimum": 1,
                           "numeric_version_terms_minimum": 1},
    }

    def row(qid, text, term, primary, split, tags=()):
        return {"query_id": qid, "text": text, "term": term, "primary_class": primary,
                "intent_family": qid, "source_type": "deterministic", "author_id": "author",
                "prospective_useful_artifact_ids": ["useful-" + qid], "answerable": True,
                "spent_overlap": False, "tags": list(tags), "source_artifact_id": None,
                "source_family": split + "-" + qid}

    development = [
        row("d-k-b", "k8s", "k8s", "bare_adaptation", "d"),
        row("d-k-s", "k8s probes", "k8s", "short_context", "d"),
        row("d-k-l", "k8s probes changed after version 1 restart", "k8s", "longer_control", "d", ["version"]),
        row("d-s-b", "s3", "s3", "bare_adaptation", "d"),
        row("d-s-s", "s3 credentials", "s3", "short_context", "d"),
        row("d-s-l", "s3 endpoint changed after version 2 upgrade", "s3", "longer_control", "d", ["numeric"]),
    ]
    confirmation = [
        row("c-k-s", "k8s networking", "k8s", "short_context", "c"),
        row("c-s-s", "s3 compatibility", "s3", "short_context", "c"),
        row("c-k-l", "k8s readiness changed in version 3 cluster", "k8s", "longer_control", "c", ["version"]),
    ]
    return development, confirmation, roster, tokenizer, registry


def test_validates_shapes_counts_safety_and_no_overlap():
    report = queries.validate_splits(*_fixture())
    assert report["development_queries"] == 6
    assert report["confirmation_queries"] == 3
    assert report["intent_family_overlap"] == 0


def test_refuses_bare_confirmation_and_cross_split_intent():
    development, confirmation, roster, tokenizer, registry = _fixture()
    confirmation[0]["primary_class"] = "bare_adaptation"
    confirmation[0]["text"] = "k8s"
    with pytest.raises(ValueError, match="development-only"):
        queries.validate_splits(development, confirmation, roster, tokenizer, registry)
    development, confirmation, roster, tokenizer, registry = _fixture()
    confirmation[0]["intent_family"] = development[0]["intent_family"]
    with pytest.raises(ValueError, match="overlaps"):
        queries.validate_splits(development, confirmation, roster, tokenizer, registry)


def test_refuses_multiple_roster_terms_and_empty_safety():
    development, confirmation, roster, tokenizer, registry = _fixture()
    development[1]["text"] = "k8s s3"
    with pytest.raises(ValueError, match="exactly one"):
        queries.validate_splits(development, confirmation, roster, tokenizer, registry)
    development, confirmation, roster, tokenizer, registry = _fixture()
    for row in development:
        row["tags"] = []
    with pytest.raises(ValueError, match="safety slice"):
        queries.validate_splits(development, confirmation, roster, tokenizer, registry)


def test_split_publication_is_no_clobber_and_never_reads_existing_confirmation(tmp_path,
                                                                               monkeypatch):
    development, confirmation, *_ = _fixture()
    monkeypatch.setattr(queries, "atomic_create_bytes", lambda path, payload: Path(path).write_bytes(payload)
                        if not Path(path).exists() else (_ for _ in ()).throw(FileExistsError(path)))
    dev_path, conf_path = tmp_path / "dev.jsonl", tmp_path / "confirmation.jsonl"
    first = queries.seal_splits(dev_path, conf_path, development, confirmation)
    with pytest.raises(SystemExit, match="already exists"):
        queries.seal_splits(dev_path, conf_path, development, confirmation)
    assert first["confirmation_sha256"] == queries.sha_bytes(conf_path.read_bytes())
