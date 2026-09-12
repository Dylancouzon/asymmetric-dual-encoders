"""judgments_ingest: report parsing, sheet alignment refusals, spot-check arithmetic, receipt
rebinding. Synthetic fixtures only; nothing under results/ or work/ is touched."""
import json

import pytest

import judgments_ingest as J


def _sheet():
    rows = []
    for q in ("k8s-docs-en:k8s-0000", "k8s-docs-en:k8s-0001"):
        for c in ("a.md", "b.md"):
            rows.append({"kind": "panel-candidate", "query_id": q, "candidate_doc_id": c,
                         "relevant_yes_no": None, "judge": None, "notes": ""})
    rows.append({"kind": "alias-pair", "pair_id": "alias-0001", "candidate_doc_id": "x.md",
                 "relevant_yes_no": None, "judge": None, "notes": ""})
    return rows


def test_parse_report_takes_the_last_jsonl_block(tmp_path):
    p = tmp_path / "r.log"
    p.write_text("chatter\n```jsonl\n{\"line\": 9}\n```\nmore\n```jsonl\n{\"line\": 1, \"a\": 1}\n\n"
                 "{\"line\": 2, \"a\": 2}\n```\ntail\n")
    assert [r["line"] for r in J.parse_report(p)] == [1, 2]
    (tmp_path / "none.log").write_text("no block here")
    with pytest.raises(SystemExit):
        J.parse_report(tmp_path / "none.log")


def test_align_requires_every_row_once_with_matching_ids():
    sheet = _sheet()
    good = [{"line": i + 1, "id": r["query_id"], "candidate": r["candidate_doc_id"],
             "answer": "yes" if i % 2 == 0 else "no"} for i, r in enumerate(sheet[:4])]
    ans = J.align_sheet(sheet, good, "panel-candidate", "query_id")
    assert [ans[i][0] for i in (1, 2, 3, 4)] == ["yes", "no", "yes", "no"]
    with pytest.raises(SystemExit):                       # one row missing
        J.align_sheet(sheet, good[:3], "panel-candidate", "query_id")
    bad = [dict(g) for g in good]
    bad[0]["candidate"] = "z.md"                           # misaligned candidate
    with pytest.raises(SystemExit):
        J.align_sheet(sheet, bad, "panel-candidate", "query_id")
    dup = good + [good[0]]
    with pytest.raises(SystemExit):
        J.align_sheet(sheet, dup, "panel-candidate", "query_id")
    wrong_kind = [{"line": 5, "id": "alias-0001", "candidate": "x.md", "answer": "yes"}]
    with pytest.raises(SystemExit):                       # alias row offered as a panel row
        J.align_sheet(sheet, wrong_kind, "panel-candidate", "query_id")
    with pytest.raises(SystemExit):                       # non yes/no
        J.align_sheet(sheet, [{**good[0], "answer": "maybe"}] + good[1:], "panel-candidate",
                      "query_id")


def test_apply_sheet_never_overwrites_an_existing_judgment():
    sheet = _sheet()
    sheet[0]["relevant_yes_no"], sheet[0]["judge"] = "no", "dylan"
    with pytest.raises(SystemExit):
        J.apply_sheet(sheet, {1: ("yes", "")}, "model")
    J.apply_sheet(sheet, {2: ("yes", "clear")}, "model")
    assert sheet[1]["relevant_yes_no"] == "yes" and sheet[1]["judge"] == "model"
    assert "[model] clear" in sheet[1]["notes"]


def test_apply_panel_and_alias_statuses():
    sheet = _sheet()
    for i, r in enumerate(sheet[:4]):
        r["relevant_yes_no"], r["judge"] = ("yes" if i == 0 else "no"), "m"
    sheet[4]["relevant_yes_no"], sheet[4]["judge"] = "no", "m"
    records = [{"query_id": "k8s-docs-en:k8s-0000", "judgment_status": "PENDING_HUMAN",
                "qrels": {}, "candidates": ["a.md", "b.md"]},
               {"query_id": "k8s-docs-en:k8s-0001", "judgment_status": "PENDING_HUMAN",
                "qrels": {}, "candidates": ["a.md", "b.md"]},
               {"query_id": "squad-train:1", "judgment_status": "DATASET_QRELS", "qrels": {"d": 1}}]
    st = J.apply_panel(records, sheet, "m")
    assert st == {"queries": 2, "relevant_documents": 1, "queries_without_relevant": 1}
    assert records[0]["qrels"] == {"a.md": 1} and records[0]["judgment_status"] == "MODEL_JUDGED"
    assert records[1]["qrels"] == {} and records[1]["judgment_status"] == "MODEL_JUDGED"
    assert records[2]["judgment_status"] == "DATASET_QRELS"
    alias = [{"pair_id": "alias-0001", "judgment_status": "PENDING_HUMAN"},
             {"pair_id": "alias-0002", "judgment_status": "VERIFIED_BY_SOURCE"}]
    assert J.apply_alias(alias, sheet, "m") == {"REJECTED_BY_MODEL": 1}
    assert alias[1]["judgment_status"] == "VERIFIED_BY_SOURCE"


def test_spotcheck_threshold_is_strictly_more_than_five_percent():
    sample = [{"pair_id": f"p{i}", "rule": "R2", "source": "s", "form_a": "a", "form_b": "b"}
              for i in range(100)]
    rows = [{"pair_id": f"p{i}", "wrong": i < 5, "note": "x" if i < 5 else ""} for i in range(100)]
    res, _ = J.apply_spotcheck(sample, rows)
    assert res["wrong"] == 5 and res["exceeds_threshold"] is False
    rows[5]["wrong"] = True
    res, _ = J.apply_spotcheck(sample, rows)
    assert res["wrong"] == 6 and res["exceeds_threshold"] is True
    with pytest.raises(SystemExit):                       # missing one pair
        J.apply_spotcheck(sample, rows[:-1])
    with pytest.raises(SystemExit):                       # non-boolean
        J.apply_spotcheck(sample, [{**rows[0], "wrong": "no"}] + rows[1:])


def test_rebind_refuses_non_judgment_field_changes():
    before = [{"query_id": "q", "query": "t", "qrels": {}, "judgment_status": "PENDING_HUMAN"}]
    after = [{"query_id": "q", "query": "CHANGED", "qrels": {"a": 1},
              "judgment_status": "MODEL_JUDGED", "judge": "m"}]
    with pytest.raises(SystemExit):
        J.rebind_screen(before, after)
