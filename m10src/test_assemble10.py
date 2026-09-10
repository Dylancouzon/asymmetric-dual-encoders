"""Step 8's assembly: the hold-out by DOCUMENT, A8's action, the quota cut, the screen order and
the row schema `corpus_loader.SOURCES["generated"]` requires. All synthetic; no GPU, no index."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import pytest

import assemble10 as A
import corpus10
import corpus_loader as CL
import qfilter


def row(text, form="finance", seed_id="wikipedia-body:1#0", doc=None):
    return {"text": text, "form": form, "seed_id": seed_id,
            "doc": doc if doc is not None else seed_id.split("#", 1)[0],
            "prompt_sha256": "0" * 64}


def manifest(held_docs=("wikipedia-body:999",), quota=143_000, span_removed=3):
    return {"form": "finance", "complete": True, "quota": quota,
            "forms12_holdout_docs": sorted(held_docs),
            "screens_applied": {"out_of_rubric_range": 1, "copied_span": span_removed,
                                "kept": 10, "n_input": 14}}


def q(nwords=12):
    """A finance query inside the frozen rubric's range (8-30 words)."""
    return " ".join(f"w{i}" for i in range(nwords))


# ---- the hold-out, by DOCUMENT ----------------------------------------------------------------

def test_holdout_by_document_drops_a_novel_query_from_a_held_document():
    """The mandate holds out 500 seed DOCUMENTS; the driver held by seed PASSAGE. A query from
    chunk #7 of a held document is novel TEXT and still must not be trained on."""
    rows = [row(q() + " alpha", seed_id="wikipedia-body:999#7"),      # held doc, other chunk
            row(q() + " beta", seed_id="wikipedia-body:123#0")]
    kept, rep = A.local_screens("finance", rows, manifest(held_docs=["wikipedia-body:999"]))
    assert [r["text"] for r in kept] == [q() + " beta"]
    assert rep["forms12_holdout_doc"] == 1
    assert rep["exact_dup"] == 0, "the held row is novel text -- exact dedup cannot catch it"


def test_a_manifest_without_holdout_documents_is_refused():
    with pytest.raises(SystemExit):
        A.local_screens("finance", [row(q())], dict(manifest(), forms12_holdout_docs=[]))


# ---- screen order and counts ------------------------------------------------------------------

def test_every_screen_is_counted_separately_and_in_the_registered_order():
    rows = [row("too short"),                                   # under the 8-word floor
            row(q()), row(q()),                                 # the second is an exact dup
            row(q() + " gamma", seed_id="wikipedia-body:999#3")]  # held document
    kept, rep = A.local_screens("finance", rows, manifest(held_docs=["wikipedia-body:999"]))
    assert rep["out_of_rubric_range"] == 1
    assert rep["exact_dup"] == 1
    assert rep["forms12_holdout_doc"] == 1
    assert rep["n_input"] == 4 and rep["kept_after_local"] == 1 == len(kept)
    assert list(A.SCREEN_ORDER)[:3] == ["out_of_rubric_range", "exact_dup", "forms12_holdout_doc"]
    assert rep["copied_span_already_applied_by_driver"] is True
    assert rep["copied_span"] == 0, "not re-run when the driver already applied it"


def test_the_rubric_filter_is_idempotent_on_driver_output():
    rows = [row(q(n)) for n in (8, 12, 30)]                     # the frozen range is 8-30
    kept, rep = A.local_screens("finance", rows, manifest())
    assert rep["out_of_rubric_range"] == 0 and len(kept) == 3
    assert qfilter.RANGES["finance"] == (8, 30)


def test_the_span_fallback_refuses_rather_than_silently_skipping():
    man = dict(manifest(), screens_applied={"out_of_rubric_range": 0})   # no copied_span key
    with pytest.raises(SystemExit):
        A.local_screens("finance", [row(q())], man)
    # ... and runs when the seed text IS supplied
    src = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    rows = [row(src, seed_id="s#0"), row(q(), seed_id="s#0")]
    kept, rep = A.local_screens("finance", rows, man, seed_text={"s#0": src})
    assert rep["copied_span"] == 1 and [r["text"] for r in kept] == [q()]


# ---- A8 gate 1: the action this module applies -------------------------------------------------

def test_a8_above_the_cut_keeps_representatives_only():
    """One template with a swapped slot: the W10 short rule sees it, and above 25% the form is
    cut to representatives -- a real cut, never topped up."""
    tmpl = ("what is the return on investment for a {} fund over the last ten years please")
    texts = [tmpl.format(f"x{i}") for i in range(200)]
    kept, a8 = corpus10.a8_action("finance", texts)
    assert a8["near_dup_rate_raw"] > corpus10.A8_MAX_NEAR_DUP_RATE
    assert a8["action"].startswith("cut to representatives")
    assert len(kept) == 0 or len(kept) == a8["retained"]
    # under 50,000 retained the form is DROPPED from the build and reported
    assert a8["dropped_from_build"] is True and kept == []


def test_a8_drops_a_form_that_falls_under_fifty_thousand():
    texts = [f"unique finance question number {i} about {i} bonds and {i} yields" for i in range(60)]
    kept, a8 = corpus10.a8_action("finance", texts)
    assert a8["near_dup_rate_raw"] <= corpus10.A8_MAX_NEAR_DUP_RATE
    assert a8["retained"] == 60 and a8["dropped_from_build"] is True and kept == []
    assert corpus10.A8_MIN_RETAINED == 50_000


def test_keep_by_text_maps_a8_strings_back_onto_rows_in_order():
    rows = [row("a " + q()), row("b " + q()), row("c " + q())]
    out = A._keep_by_text(rows, ["c " + q(), "a " + q()])
    assert [r["text"] for r in out] == ["a " + q(), "c " + q()], "row order is preserved"


# ---- the quota cut -----------------------------------------------------------------------------

def test_quota_cut_is_a_no_op_below_quota():
    rows = [row(q() + f" {i}") for i in range(100)]
    out, rep = A.quota_cut(rows, 143_000)
    assert out == rows and rep["cut"] is False and rep["n_after"] == 100


def test_quota_cut_is_uniform_seed_zero_and_deterministic():
    rows = [row(q() + f" {i}") for i in range(1000)]
    out, rep = A.quota_cut(rows, 200)
    out2, _ = A.quota_cut(rows, 200)
    assert len(out) == 200 and rep["cut"] is True and rep["removed"] == 800
    assert [r["text"] for r in out] == [r["text"] for r in out2], "seed 0, deterministic"
    pos = [rows.index(r) for r in out]
    assert pos == sorted(pos), "file order is preserved; the DRAW is what is uniform"
    assert max(pos) > 800 and sum(1 for p in pos if p < 200) < 70, "never a positional prefix"


# ---- the row schema the loader requires --------------------------------------------------------

def test_the_output_row_schema_is_what_corpus_loader_requires(tmp_path):
    spec = CL.SOURCES["generated"]
    assert spec["require_id"] == "seed_id" and Path(spec["path"]).name == A.OUT_NAME
    p = tmp_path / A.OUT_NAME
    p.write_text("\n".join(json.dumps({k: row(q())[k] for k in A.ROW_KEYS})
                           for _ in range(3)) + "\n")
    texts, forms, ids = CL._rows_from_jsonl(p, default_form=spec.get("form"), with_ids=True,
                                            require_id=spec["require_id"])
    assert len(texts) == 3 and set(forms) == {"finance"} and all(i for i in ids)


def test_a_row_without_a_seed_id_is_refused_by_the_loader(tmp_path):
    p = tmp_path / A.OUT_NAME
    p.write_text(json.dumps({"text": q(), "form": "finance"}) + "\n")
    with pytest.raises(SystemExit):
        CL._rows_from_jsonl(p, with_ids=True, require_id="seed_id")


# ---- the smoke guard ---------------------------------------------------------------------------

def test_a_partial_run_refuses_to_write_the_real_artifact():
    with pytest.raises(SystemExit, match="refusing to write the real"):
        A.assemble(forms=["finance"], limit=10, allow_incomplete=True, out_dir=A.OUT)


def test_data_cut_reads_the_registered_rule_and_registers_nothing():
    reg = json.loads((REPO / "m10" / "screen_registry.json").read_text())["data_cut"]
    assert reg["rule"].startswith("min of the three")
    assert set(A.data_cut.__defaults__[1]) == {"A2", "A3", "A4"}


# ---- end to end, with the index screens stubbed -------------------------------------------------

def _fake_gen_dir(tmp_path, texts, held_docs=("wikipedia-body:999",)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "generated_finance.jsonl").write_text(
        "\n".join(json.dumps(r) for r in texts) + "\n")
    (tmp_path / "manifest_finance.json").write_text(json.dumps(manifest(held_docs=held_docs,
                                                                       quota=1)))
    return tmp_path


def test_end_to_end_applies_a8_then_the_quota_and_writes_the_loader_schema(tmp_path, monkeypatch):
    tmpl = "what is the return on investment for a {} fund over the last ten years please"
    rows = [row(tmpl.format(f"x{i}"), seed_id=f"wikipedia-body:{i}#0") for i in range(60)]
    rows += [row(f"how do {i} municipal bonds differ from corporate debt instruments in practice",
                 seed_id=f"wikipedia-body:9{i}#0") for i in range(40)]
    gen_dir = _fake_gen_dir(tmp_path / "gen", rows)
    monkeypatch.setattr(A, "index_screens",
                        lambda texts, **k: (set(), set(), {"stubbed": True,
                                                           "n_candidates": len(texts)}))
    rep = A.assemble(forms=["finance"], gen_dir=gen_dir, out_dir=tmp_path / "out",
                     smoke_ignore_a8_min_retained=True, verbose=False)
    a8 = rep["per_form"]["finance"]["a8_gate1"]
    assert a8["near_dup_rate_raw"] > 0.25 and a8["action"].startswith("cut to representatives")
    assert a8["_smoke_min_retained_not_applied"] is True
    # the cut is applied to the ROWS, then the quota (10) cuts what is left
    assert rep["per_form"]["finance"]["quota_cut"]["n_before"] == a8["retained"]
    assert rep["per_form"]["finance"]["quota_cut"]["cut"] is True
    assert rep["total_rows"] == 1
    out = [json.loads(l) for l in (tmp_path / "out" / A.OUT_NAME).read_text().splitlines()]
    assert len(out) == 1
    assert all(set(r) == set(A.ROW_KEYS) for r in out)
    texts, forms, ids = CL._rows_from_jsonl(tmp_path / "out" / A.OUT_NAME, with_ids=True,
                                            require_id="seed_id")
    assert len(texts) == 1 and set(forms) == {"finance"}


def test_a_smoke_report_never_lands_on_the_real_results_path(tmp_path, monkeypatch):
    """Pitfall 9. `--smoke-ignore-a8-min-retained` alone makes a run partial, and a partial run's
    report belongs beside its own output -- this exact case wrote the real report once."""
    gen_dir = _fake_gen_dir(tmp_path / "gen",
                            [row(q() + f" {i}", seed_id=f"wikipedia-body:{i}#0")
                             for i in range(20)])
    monkeypatch.setattr(A, "index_screens", lambda texts, **k: (set(), set(), {"stubbed": True}))
    rep = A.assemble(forms=["finance"], gen_dir=gen_dir, out_dir=tmp_path / "out",
                     smoke_ignore_a8_min_retained=True, verbose=False)
    assert rep["_partial"] is True
    assert Path(rep["report_path"]) == tmp_path / "out" / A.REPORT_NAME
    assert not (A.RESULTS / A.REPORT_NAME).exists() or \
        json.loads((A.RESULTS / A.REPORT_NAME).read_text())["_partial"] is False
