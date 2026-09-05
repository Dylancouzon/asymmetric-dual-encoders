"""Step 8's gating logic. The two constants that must not be conflated are asserted first."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import pytest

import corpus10 as C
import decontam


def test_the_two_sketch_thresholds_are_different_and_are_the_registered_ones():
    """A8's diversity gate is 16/32; the decontamination screens are 8/32. Conflating them would
    either gut the corpus or wave through template collapse."""
    assert C.A8_NEAR_DUP_SHARE == 16, "instructions-m10.md:462"
    assert decontam.DUP_SHARE == 8, "instructions-m10.md:444"
    assert C.A8_NEAR_DUP_SHARE != decontam.DUP_SHARE
    assert decontam.SKETCH == 32
    assert C.A8_MAX_NEAR_DUP_RATE == 0.25 and C.A8_MIN_RETAINED == 50_000
    assert C.HOLDOUT_PER_FORM == 500 and C.SPAN_K == 5


# ---- FORMS-12 hold-out ----------------------------------------------------------------------

def test_holdout_is_uniform_not_a_prefix():
    ids = [f"doc{i:05d}" for i in range(5000)]
    held, kept = C.holdout_seed_ids(ids, n=500, seed=0)
    assert len(held) == 500 and len(kept) == 4500
    assert not set(held) & set(kept), "a held seed must never also be trained on"
    assert sorted(held + kept) == sorted(ids), "every seed is either held or kept"
    # a prefix hold-out would put every held id in the first 500 of the sorted order
    pos = [ids.index(h) for h in held]
    assert max(pos) > 4000, "the hold-out must reach the whole store, not its prefix"
    assert sum(1 for p in pos if p < 500) < 120, "prefix not over-represented"


def test_holdout_is_deterministic_in_the_seed():
    ids = [f"d{i}" for i in range(3000)]
    assert C.holdout_seed_ids(ids, n=100, seed=0)[0] == C.holdout_seed_ids(ids, n=100, seed=0)[0]
    assert C.holdout_seed_ids(ids, n=100, seed=1)[0] != C.holdout_seed_ids(ids, n=100, seed=0)[0]


def test_holdout_refuses_a_store_too_small_to_hold_out_from():
    with pytest.raises(SystemExit, match="hold-out"):
        C.holdout_seed_ids([f"d{i}" for i in range(400)], n=500)


# ---- the own-seed-passage span screen -------------------------------------------------------

SRC = ("The Bethlem myopathy is a slowly progressive muscle disease that was first described in "
       "nineteen seventy six by Bethlem and van Wijngaarden in a Dutch family cohort study.")


def test_a_copied_span_is_rejected():
    copied = "a slowly progressive muscle disease that was first described"
    assert C.copied_span(copied, SRC), "a five-word span lifted from the seed is not a query"


def test_a_genuine_question_about_the_passage_survives():
    q = "what causes muscle weakness in children with an inherited collagen disorder"
    assert not C.copied_span(q, SRC)


def test_the_harvested_span_itself_is_excluded_or_every_harvest_self_rejects():
    """A harvested title IS a span of its document, so without the exclusion this screen would
    reject 100% of harvested text by construction."""
    span = "a slowly progressive muscle disease that was first described"
    assert C.copied_span(span, SRC), "without the exclusion it self-matches"
    assert not C.copied_span(span, SRC, exclude_span=span), "with it, it must survive"


def test_excluding_the_span_does_not_splice_new_kgrams_from_its_neighbours():
    """The exclusion removes the span's k-grams, not its characters. Deleting the substring would
    join 'myopathy is' to 'in nineteen seventy' and create k-grams the source never had."""
    span = "slowly progressive muscle disease that"
    # the splice: removing the span's CHARACTERS would join "...myopathy is a" to "was first
    # described...", creating 5-grams the source never contained. Removing its K-GRAMS does not.
    spliced = "myopathy is a was first described"
    assert not C.copied_span(spliced, SRC, exclude_span=span)
    # and a genuine 5-gram from elsewhere in the source is still caught -- the exclusion is
    # surgical, not a blanket pass. (My first fixture here failed because it happened to contain
    # "in nineteen seventy six by", which really is a span of the source.)
    assert C.copied_span("published in nineteen seventy six by Bethlem", SRC, exclude_span=span)


def test_a_query_shorter_than_five_words_cannot_copy_a_span():
    assert not C.copied_span("muscle disease", SRC)


# ---- A8 gate 1 ------------------------------------------------------------------------------

def _vary(base, n):
    return [f"{base} number {i} with its own distinct trailing words here" for i in range(n)]


def test_near_dup_gate_keeps_the_FIRST_occurrence_as_representative():
    q = "how do I configure the network interface on a headless server machine at boot"
    qs = [q + " today", q + " today", q + " now"]
    reps, deduped, rep = C.near_dup_gate(qs)
    assert rep["exact_dups_removed"] == 1
    assert rep["post_exact_dedup"] == 2
    assert reps[0] == qs[0], "the earlier query is the representative"


def test_near_dup_gate_reports_a_rate_over_the_post_exact_dedup_count():
    qs = _vary("how do I reset the thing", 40)
    reps, deduped, rep = C.near_dup_gate(qs)
    assert rep["n_input"] == 40 and rep["post_exact_dedup"] == 40
    assert rep["near_duplicates"] + rep["representatives"] == rep["post_exact_dedup"]
    assert 0.0 <= rep["near_dup_rate"] <= 1.0


def test_a8_cuts_to_representatives_only_above_25_percent():
    """Template collapse: one stem repeated with a trailing counter shares its 8-grams."""
    stem = ("please tell me in detail how one would go about resetting the main configuration "
            "value on this particular device without losing any of the existing saved settings")
    qs = [f"{stem} {i}" for i in range(60)]
    kept, rep = C.a8_action("howto", qs)
    assert rep["near_dup_rate"] > 0.25, f"rate {rep['near_dup_rate']} -- fixture is not collapsed"
    assert "cut to representatives" in rep["action"]
    assert len(kept) == rep["representatives"] or rep["dropped_from_build"]


def test_a8_does_not_cut_a_diverse_form_but_still_removes_exact_duplicates():
    qs = [f"an entirely different question about topic {i} phrased in its own particular way"
          for i in range(40)]
    qs.append(qs[0])                                   # one exact duplicate
    kept, rep = C.a8_action("comparison", qs)
    assert rep["near_dup_rate"] <= 0.25
    assert "no cut" in rep["action"]
    assert rep["exact_dups_removed"] == 1
    assert rep["retained"] == 40, "the 40 distinct queries are retained; the duplicate is not"
    # `kept` is [] because the registered 50,000 retention floor drops any form this small. That
    # is the rule, not a bug -- every small fixture hits it, so assert on `retained` instead.
    assert rep["dropped_from_build"] is True and kept == []


def test_a8_drops_a_form_that_falls_under_the_retention_floor():
    qs = [f"a distinct query number {i} about its own separate subject matter here"
          for i in range(30)]
    kept, rep = C.a8_action("yesno", qs)
    assert rep["retained"] < C.A8_MIN_RETAINED
    assert rep["dropped_from_build"] is True and kept == []
    assert "FORM DROPPED" in rep["action"]


def test_manifest_sha256_is_order_independent_over_keys():
    a = C.manifest_sha256({"x": 1, "y": [2, 3]})
    b = C.manifest_sha256({"y": [2, 3], "x": 1})
    assert a == b and len(a) == 64


# ---- the step-8 driver, against a STUB OpenAI-compatible server -----------------------------
#
# Generation is a ~10-GPU-hour spend and the card is busy for hours at a time, so the glue around
# it is tested against a fake endpoint. What this cannot test is output QUALITY -- that is the
# decision-15 smoke's job.

import json as _json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Stub(BaseHTTPRequestHandler):
    REPLY = None          # set per test: a callable(form_prompt, n) -> list[str]

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = _json.dumps({"data": [{"id": "Qwen/Qwen3-8B-AWQ"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = _json.loads(self.rfile.read(n) or b"{}")
        qs = type(self).REPLY(req)
        body = _json.dumps({"choices": [{"message": {"content": _json.dumps(qs)},
                                         "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _seed_rows(n, text="A passage of ordinary prose about some subject matter and its context."):
    return [(f"p{i:05d}", f"{text} Instance {i}.") for i in range(n)]


def test_partition_removes_gate_seeds_BEFORE_drawing_the_holdout():
    rows = _seed_rows(3000)
    gate = {r[0] for r in rows[:400]}
    build, held, rep = C.partition_seeds(rows, gate_ids=gate, holdout_n=500)
    assert rep["gate_seeds_excluded"] == 400 and rep["forms12_holdout"] == 500
    assert len(build) == 3000 - 400 - 500
    held_ids = {r[0] for r in held}
    assert not held_ids & gate, "a gate seed must never be reported as held out"
    assert not {r[0] for r in build} & gate
    assert not {r[0] for r in build} & held_ids


def test_screen_form_reports_each_screen_separately_and_in_order():
    import qfilter
    lo, hi = qfilter.RANGES["howto"]
    seed_text = {"p1": "the quick brown fox jumps over the lazy dog again and again today"}
    short = "what does the fox do in the well known typing exercise"
    copies = ("the quick brown fox jumps over the lazy dog and then keeps going onward for a "
              "while longer until it finally stops somewhere well beyond the far end of the field")
    prot = ("how do I train a dog to stop chasing foxes in the garden without using any kind of "
            "punishment based method at all and without a professional trainer present")
    # assert the fixture is in range FIRST -- an accidentally out-of-range row would be counted by
    # the first screen and prove nothing about the later ones
    assert len(short.split()) < lo
    for q in (copies, prot):
        assert lo <= len(q.split()) <= hi, f"{len(q.split())} words, need {lo}-{hi}"
    rows = [{"query": q, "form": "howto", "seed_id": "p1"} for q in (short, copies, prot)]
    kept, rep = C.screen_form(rows, seed_text,
                              protected_hit=lambda q: "punishment" in q, doc_hit=None)
    assert rep["out_of_rubric_range"] == 1, "only the short row"
    assert rep["copied_span"] == 1, "the second row copies a 5-gram from its own seed passage"
    assert rep["protected_index"] == 1
    assert rep["kept"] == 0 and kept == []


def test_build_form_end_to_end_against_the_stub(stub):
    srv, base = stub
    # 30-word on-form `howto` strings, distinct per request so A8 does not collapse them
    def reply(req):
        s = req.get("seed", 0)
        return [f"how do I configure setting number {s}{j} on the appliance without losing any "
                f"of the saved values that were already present before I began the process"
                for j in range(req.get("n", 5) if "n" in req else 5)]
    _Stub.REPLY = lambda req: reply(req)
    rows = _seed_rows(1200)
    gate = {r[0] for r in rows[:400]}
    qs, rep = C.build_form("howto", rows, quota=50, n_per_seed=5, gate_ids=gate, base=base,
                           protected_hit=None, doc_hit=None, workers=8, margin=1.0,
                           verbose=False)
    assert rep["seeds"]["gate_seeds_excluded"] == 400
    assert rep["seeds"]["forms12_holdout"] == 500
    assert rep["generation"]["contract_rate"] == 1.0, rep["generation"]["first_failures"]
    assert rep["screens"]["n_input"] > 0
    assert len(rep["holdout_seed_ids"]) == 500
    # every returned row carries its provenance, which the manifest needs
    assert all(set(r) == {"query", "form", "seed_id"} for r in qs)
    assert all(r["form"] == "howto" for r in qs)


def test_build_form_drops_a_form_whose_output_is_template_collapsed(stub):
    """The A8 diversity gate is the guard against 4-bit repetition (decision 14's rationale).
    An identical reply for every seed must not survive as a corpus."""
    srv, base = stub
    _Stub.REPLY = lambda req: [
        "how do I reset the configuration on this device without losing the saved settings "
        "that were already stored on it before I started"] * 5
    rows = _seed_rows(1200)
    qs, rep = C.build_form("howto", rows, quota=50, n_per_seed=5, base=base, workers=8,
                           margin=1.0, verbose=False)
    # exact duplicates collapse to one, so the form cannot reach the retention floor
    assert rep["a8"]["retained"] < C.A8_MIN_RETAINED
    assert rep["dropped_from_build"] is True and qs == []
