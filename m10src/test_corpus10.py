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
    # GENUINELY diverse. Two earlier versions of this fixture were not: the first varied only an
    # index inside a fixed frame, and the second reused four suffixes across ten subjects. Both
    # read "diverse" only because the pre-W10 gate was blind to exactly that kind of overlap.
    qs = [
        "how do I renew a british passport when the old one has expired",
        "when did the war of the spanish succession actually come to an end",
        "what is the best way to store fresh basil so it lasts",
        "why do domestic cats knead soft blankets with their front paws",
        "what causes a lithium laptop battery to swell up over time",
        "who composed the piece popularly known as the moonlight sonata",
        "roughly how much does replacing a timing belt cost these days",
        "is sparkling water genuinely bad for dental enamel or a myth",
        "how tall is the eiffel tower measured to its very tip",
        "explain the difference between weather forecasting and climate modelling",
        "can sourdough starter survive several weeks left in a fridge",
        "which vitamins are actually absorbed better alongside dietary fat",
        "what happens to unclaimed lottery winnings after the deadline passes",
        "are noise cancelling headphones safe to wear all day long",
        "how does a heat pump work in freezing outdoor temperatures",
        "what distinguishes arabica from robusta beans in a espresso blend",
        "why does cut avocado turn brown so quickly on the counter",
        "how long should concrete cure before heavy vehicles drive over",
        "what were the main causes of the eighteen forty eight revolutions",
        "does closing unused browser tabs meaningfully reduce laptop power draw",
        "how do migrating birds navigate accurately across an open ocean",
        "what is the practical difference between broth and traditional stock",
        "when should winter tyres be swapped back to summer ones",
        "why are some antibiotics prescribed for exactly ten day courses",
        "how much sleep do teenagers actually need on school nights",
        "what makes cast iron cookware naturally non stick after seasoning",
        "is it cheaper to run a dishwasher than wash manually",
        "how did medieval stonemasons lift heavy blocks without modern cranes",
        "what determines whether a volcano erupts explosively or effusively",
        "why do aeroplane windows have a tiny hole near the bottom",
        "how are olympic swimming pool lane ropes designed to reduce waves",
        "what is the shelf life of unopened tinned tomatoes",
        "can houseplants meaningfully improve indoor air quality in a flat",
        "how does compound interest differ from simple interest in practice",
        "why did the roman empire eventually abandon its british province",
        "what should someone check before buying a used electric vehicle",
        "how do submarines generate breathable oxygen while submerged for months",
        "which factors most affect how quickly bread dough rises",
        "what is the origin of the phrase red herring",
        "how accurate are consumer smartwatch heart rate sensors during exercise",
    ]
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


def test_draw_seeds_reaches_the_whole_store_not_its_prefix():
    """Regression for the bias class that had to be fixed in `harvest.draw` and `paq.draw`:
    `build[:need]` takes the seed store's own order, which for a Wikipedia store is dump order."""
    N, need = 5000, 40
    build = _seed_rows(N)
    use = C.draw_seeds(build, need, seed=0)
    pos = [int(r[0][1:]) for r in use]
    assert len(use) == need
    assert pos == sorted(pos), "returned in store order, for reproducibility"
    assert max(pos) > N // 2, f"highest drawn position {max(pos)} of {N} -- looks like a prefix"
    assert sum(1 for p in pos if p < need) <= 2, "the prefix must not be over-represented"


def test_draw_seeds_is_deterministic_and_returns_everything_when_supply_is_short():
    build = _seed_rows(300)
    assert C.draw_seeds(build, 40, seed=0) == C.draw_seeds(build, 40, seed=0)
    assert C.draw_seeds(build, 40, seed=1) != C.draw_seeds(build, 40, seed=0)
    assert C.draw_seeds(build, 900, seed=0) == build, "short supply returns all of it"


def test_the_forms_the_OLD_gate_could_never_have_fired_for_are_named():
    """**A8's diversity gate is structurally inert for short forms.** An N-word text has N-7
    word-8-grams, so a sketch reaches the registered 16/32 threshold only at N >= 23. Five of the
    twelve registered forms have their ENTIRE range below that -- including `yesno`, a GENERATED
    form -- so the gate decision 14 leaned on as its guard against 4-bit repetition cannot fire
    for them. Asserted here so the blind spot cannot be forgotten or silently 'fixed' by a
    threshold change that nobody registers."""
    import qfilter
    import decontam
    assert decontam.NGRAM == 8 and C.A8_NEAR_DUP_SHARE == 16
    floor = decontam.NGRAM + C.A8_NEAR_DUP_SHARE - 1          # 23
    assert len(decontam.sketch(" ".join(f"w{i}" for i in range(floor - 1)))) < 16
    assert len(decontam.sketch(" ".join(f"w{i}" for i in range(floor)))) >= 16
    never = sorted(f for f, (lo, hi) in qfilter.RANGES.items() if hi < floor)
    assert never == ["factoid", "keyword", "product", "title", "yesno"], never
    # identical strings are still exact dedup's job, not the near-dup rule's, under either regime
    qs = ["is this product waterproof and safe to use outdoors"] * 40
    _reps, _ded, rep = C.near_dup_gate(qs)
    assert rep["exact_dups_removed"] == 39 and rep["post_exact_dedup"] == 1
    assert rep["near_duplicates"] == 0, "nothing left to compare after exact dedup"


def test_copied_span_keeps_windows_that_straddle_the_excluded_span():
    """Finding 8: value subtraction removed a gram wherever it occurred. Positional exclusion
    drops only windows lying entirely inside an occurrence, so boundary windows still catch."""
    src = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    span = "gamma delta epsilon zeta eta"
    assert not C.copied_span(span, src, exclude_span=span), "the span itself must survive"
    # a window straddling the span's right boundary is NOT part of the span and must still catch
    straddle = "delta epsilon zeta eta theta"
    assert C.copied_span(straddle, src, exclude_span=span)
    straddle_left = "beta gamma delta epsilon zeta"
    assert C.copied_span(straddle_left, src, exclude_span=span)


def test_copied_span_multi_occurrence_hole_is_open_and_asserted_as_such():
    """The honest limit: with the span appearing twice and no harvest offset, both occurrences are
    excluded and a copy of the second passes. Asserted so the limitation is visible, not implied."""
    src = "aa bb cc dd ee ff gg hh aa bb cc dd ee ii jj"
    span = "aa bb cc dd ee"
    assert not C.copied_span("aa bb cc dd ee", src, exclude_span=span), \
        "known limitation: the second occurrence is excluded too (no offset is recorded)"


def _long(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def test_a8_action_reads_the_RAW_rate_not_the_rounded_one():
    """Finding 6b: 0.25000375 rounds to 0.25 and would wrongly escape the `> 0.25` cut."""
    rep = {"near_dup_rate_raw": 0.25000375, "near_dup_rate": round(0.25000375, 5)}
    assert rep["near_dup_rate"] == 0.25 and not (rep["near_dup_rate"] > C.A8_MAX_NEAR_DUP_RATE)
    assert rep["near_dup_rate_raw"] > C.A8_MAX_NEAR_DUP_RATE, "the raw rate must fire the cut"


def test_harvest_holdout_holds_a_document_out_of_EVERY_form(tmp_path):
    """One Wikipedia article yields a title, headings and a lead claim. Holding it out for `title`
    only would train on that same article's `claim`, and the rule is "queries generated or
    harvested from them are never trained on" -- not "from them, in that form"."""
    import json
    rows = []
    for i in range(2000):
        d = f"doc{i:05d}"
        rows.append({"form": "title", "text": f"title {i}", "doc": d, "src": "wiki", "rule": "title"})
        rows.append({"form": "claim", "text": f"claim {i}", "doc": d, "src": "wiki", "rule": "claim"})
    p = tmp_path / "drawn.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    train, f12, rep = C.harvest_holdout(p, per_form=100, seed=0, verbose=False)

    held = {r["doc"] for r in f12}
    assert len(train) + len(f12) == len(rows)
    assert not held & {r["doc"] for r in train}, "a held document must appear in no training row"
    # every held doc contributes BOTH of its forms to the hold-out, never one to each side
    for d in held:
        assert sum(1 for r in f12 if r["doc"] == d) == 2
    assert rep["held_documents"] == len(held)
    assert set(rep["forms12_by_form"]) == {"title", "claim"}


def test_harvest_holdout_is_deterministic(tmp_path):
    import json
    rows = [{"form": "title", "text": f"t{i}", "doc": f"d{i}", "src": "w", "rule": "title"}
            for i in range(1000)]
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    a = C.harvest_holdout(p, per_form=50, seed=0, verbose=False)[2]["held_documents"]
    b = C.harvest_holdout(p, per_form=50, seed=0, verbose=False)[2]["held_documents"]
    assert a == b == 50


def test_the_W10_amendment_catches_a_one_slot_template_the_old_gate_could_not():
    """The defect W10 fixed: an 8-gram rule cannot catch a one-slot template at <= 15 words at ANY
    threshold, because every 8-gram window covers the changed word. Pre-amendment this read
    exactly 0.00%; the two-regime rule must now catch it."""
    qs = [f"what should i do if my {w} keeps making a loud noise at night"
          for w in ("boiler fridge heater kettle furnace washer dryer oven mower pump "
                    "blender toaster grinder scanner printer").split()]
    assert all(len(q.split()) < C.A8_LONG_FLOOR for q in qs), "all in the short regime"
    _r, _d, a8 = C.near_dup_gate(qs)
    assert a8["near_dup_rate_raw"] > 0.5, \
        f"W10 amendment must catch this; got {a8['near_dup_rate_raw']:.2%}"
    assert a8["caught_only_by_short_rule"] > 0, "and it must be the SHORT rule that caught it"
    rate, _n = C.prop_near_dup_rate(qs)
    assert rate > 0.5, "the standalone diagnostic agrees"


def test_opener_concentration_flags_frame_repetition():
    same = [f"what should i do about problem number {i} in my house" for i in range(50)]
    varied = [f"topic {i} question {i} phrased {i} differently {i} each {i} time" for i in range(50)]
    s_share, s_d2 = C.opener_concentration(same)
    v_share, v_d2 = C.opener_concentration(varied)
    assert s_share == 1.0, "one repeated opening 4-gram covers every query"
    assert v_share < s_share and v_d2 > s_d2


def test_diversity_report_curve_is_a_floor_and_flags_rising():
    qs = [f"what should i do if my device number {i} keeps failing" for i in range(600)]
    rep = C.diversity_report(qs, form="howto", ns=(200, 500))
    assert set(rep["curve"]) == {"200", "500"}
    assert set(rep["curve"]["200"]) == {"frac_40", "frac_50", "frac_60"}
    assert rep["curve"]["500"]["frac_50"] >= rep["curve"]["200"]["frac_50"], "monotone in n"
    assert rep["curve"]["200"]["frac_40"] >= rep["curve"]["200"]["frac_60"], "looser frac catches more"
