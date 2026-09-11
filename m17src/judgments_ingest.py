"""M17 ruling A6: ingest the model judgments of the three descriptive sheets and re-seal.

Codex gpt-6-astra judged, read-only and from the row text alone (briefs
`research/m17-astra-judge-{panel,senses,spotcheck}-brief-2026-09-11.md`):

  * the 360 Kubernetes (query, candidate) panel rows      -> panel qrels, `MODEL_JUDGED`
  * the 80 pending alias senses of the held-out alias test -> `VERIFIED_BY_MODEL` / `REJECTED_BY_MODEL`
  * the 308-pair 2 % spot check of the alias TRAINING pool -> wrong count against the 5 % rule

This script parses each report's single ```jsonl block, refuses an incomplete or misaligned
report, writes the answers into the review sheet with `judge = codex-gpt-6-astra` (the sheet
refuses to be rebuilt once answered), updates `work/m17/panel/panel.jsonl` and
`work/m17/alias/alias_test.jsonl`, rebinds the ancestry-screen receipt to the new panel bytes
after checking that ONLY the judgment fields changed, re-seals the panel through
`panel_build.seal()`, and writes the seeded human double-check slice (20 rows, 10 senses,
20 pairs, model answers hidden) for Dylan. The panel and the alias test never route selection.

Nothing here reads a development component, a protected surface or any vector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, RESULTS, admit_read, sha_file, write_json  # noqa: E402
import panel_build as P                                              # noqa: E402

JUDGE = "codex-gpt-6-astra"
PENDING = RESULTS / "m17_panel_pending_judgments.jsonl"
SPOT_SAMPLE = RESULTS / "m17_alias_spotcheck_sample.jsonl"
SPOT_RESULT = RESULTS / "m17_alias_spotcheck_result.json"
ALIAS_JSONL = REPO / "work" / "m17" / "alias" / "alias_test.jsonl"
ALIAS_MANIFEST = RESULTS / "m17_alias_test_manifest.json"
INGEST_RECORD = RESULTS / "m17_judgments_ingest.json"
DOUBLECHECK = RESULTS / "m17_human_doublecheck_slice.jsonl"
SPOT_THRESHOLD = 0.05          # registry data.alias_spotcheck_rule: MORE than 5 % wrong
JUDGMENT_FIELDS = {"qrels", "judgment_status", "judge"}
DOUBLECHECK_N = {"panel-candidate": 20, "alias-pair": 10, "spotcheck": 20}


# ------------------------------------------------------------------ report parsing

def parse_report(path):
    """The LAST ```jsonl fenced block of a Codex log, one dict per line."""
    text = admit_read(Path(path)).read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"```jsonl\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        raise SystemExit(f"M17 INGEST REFUSED: no ```jsonl block in {path}")
    rows = []
    for ln in blocks[-1].splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise SystemExit(f"M17 INGEST REFUSED: bad JSON line in {path}: {ln[:80]!r} ({e})")
    return rows


def _yes_no(v):
    v = str(v).strip().lower()
    if v not in ("yes", "no"):
        raise SystemExit(f"M17 INGEST REFUSED: answer {v!r} is not yes/no")
    return v


def align_sheet(sheet, rows, kind, id_key):
    """Match report rows to sheet rows of `kind` by 1-based line, then verify id + candidate.

    Every sheet row of that kind must receive exactly one answer; a report that skips, repeats
    or mislabels a row is refused rather than partially applied.
    """
    want = {i + 1: r for i, r in enumerate(sheet) if r["kind"] == kind}
    got = {}
    for j in rows:
        line = int(j["line"])
        if line not in want:
            raise SystemExit(f"M17 INGEST REFUSED: report line {line} is not a {kind} row")
        r = want[line]
        if str(j.get("id")) != str(r[id_key]) or str(j.get("candidate")) != str(r["candidate_doc_id"]):
            raise SystemExit(f"M17 INGEST REFUSED: line {line} id/candidate mismatch: report "
                             f"{j.get('id')!r}/{j.get('candidate')!r} vs sheet "
                             f"{r[id_key]!r}/{r['candidate_doc_id']!r}")
        if line in got:
            raise SystemExit(f"M17 INGEST REFUSED: line {line} answered twice")
        got[line] = (_yes_no(j["answer"]), str(j.get("note", "") or ""))
    missing = sorted(set(want) - set(got))
    if missing:
        raise SystemExit(f"M17 INGEST REFUSED: {len(missing)} {kind} rows unanswered, e.g. "
                         f"lines {missing[:5]}")
    return got


# ------------------------------------------------------------------ the three sheets

def apply_sheet(sheet, answers, judge):
    for line, (ans, note) in answers.items():
        r = sheet[line - 1]
        if r.get("relevant_yes_no") not in (None, "") or r.get("judge") not in (None, ""):
            raise SystemExit(f"M17 INGEST REFUSED: sheet line {line} is already judged by "
                             f"{r.get('judge')!r}; the sheet is never overwritten")
        r["relevant_yes_no"], r["judge"] = ans, judge
        if note:
            r["notes"] = (r.get("notes") or "") + f"[{judge}] {note}"


def apply_panel(records, sheet, judge):
    yes = {}
    for r in sheet:
        if r["kind"] != "panel-candidate":
            continue
        yes.setdefault(r["query_id"], {})
        if r["relevant_yes_no"] == "yes":
            yes[r["query_id"]][r["candidate_doc_id"]] = 1
    n_q = n_rel = 0
    for rec in records:
        if rec.get("judgment_status") != "PENDING_HUMAN":
            continue
        if rec["query_id"] not in yes:
            raise SystemExit(f"M17 INGEST REFUSED: panel query {rec['query_id']} has no judged rows")
        rels = yes[rec["query_id"]]
        bad = set(rels) - set(rec.get("candidates", []))
        if bad:
            raise SystemExit(f"M17 INGEST REFUSED: {rec['query_id']} judged a non-candidate {bad}")
        rec["qrels"] = dict(sorted(rels.items()))
        rec["judgment_status"] = "MODEL_JUDGED"
        rec["judge"] = judge
        n_q += 1
        n_rel += len(rels)
    return {"queries": n_q, "relevant_documents": n_rel,
            "queries_without_relevant": sum(1 for q, v in yes.items() if not v)}


def apply_alias(records, sheet, judge):
    ans = {r["pair_id"]: r["relevant_yes_no"] for r in sheet if r["kind"] == "alias-pair"}
    n = Counter()
    for rec in records:
        if rec.get("judgment_status") != "PENDING_HUMAN":
            continue
        if rec["pair_id"] not in ans:
            raise SystemExit(f"M17 INGEST REFUSED: alias pair {rec['pair_id']} has no judged row")
        rec["judgment_status"] = ("VERIFIED_BY_MODEL" if ans[rec["pair_id"]] == "yes"
                                  else "REJECTED_BY_MODEL")
        rec["judge"] = judge
        n[rec["judgment_status"]] += 1
    return dict(n)


def apply_spotcheck(sample, rows):
    by_id = {r["pair_id"]: r for r in sample}
    seen = {}
    for j in rows:
        pid = str(j["pair_id"])
        if pid not in by_id:
            raise SystemExit(f"M17 INGEST REFUSED: spot-check pair {pid} is not in the sample")
        if pid in seen:
            raise SystemExit(f"M17 INGEST REFUSED: spot-check pair {pid} judged twice")
        if not isinstance(j["wrong"], bool):
            raise SystemExit(f"M17 INGEST REFUSED: spot-check pair {pid} `wrong` is not boolean")
        seen[pid] = (j["wrong"], str(j.get("note", "") or ""))
    missing = sorted(set(by_id) - set(seen))
    if missing:
        raise SystemExit(f"M17 INGEST REFUSED: {len(missing)} spot-check pairs unanswered")
    wrong = [pid for pid, (w, _) in seen.items() if w]
    share = len(wrong) / len(sample)
    return {
        "judge": JUDGE, "n_pairs": len(sample), "wrong": len(wrong), "wrong_share": round(share, 4),
        "threshold_share": SPOT_THRESHOLD,
        "exceeds_threshold": share > SPOT_THRESHOLD,
        "consequence": ("tighten the abbreviation filter and rebuild the alias pool before lock"
                        if share > SPOT_THRESHOLD else "none: the pool stands"),
        "wrong_by_rule": dict(Counter(by_id[p]["rule"] for p in wrong)),
        "wrong_by_source": dict(Counter(by_id[p]["source"] for p in wrong)),
        "wrong_pairs": [{"pair_id": p, "form_a": by_id[p]["form_a"], "form_b": by_id[p]["form_b"],
                         "rule": by_id[p]["rule"], "note": seen[p][1]} for p in wrong],
    }, seen


# ------------------------------------------------------------------ receipts

def rebind_screen(before, after):
    """The ancestry screen was run on the panel's query text and gold groups; ingesting judgments
    changes only `qrels`/`judgment_status`/`judge`. Verify that field by field, then bind the
    receipt to the new bytes instead of re-streaming 2.9 GB of ancestor manifests."""
    if len(before) != len(after):
        raise SystemExit("M17 INGEST REFUSED: panel record count changed during ingest")
    for b, a in zip(before, after):
        keys = {k for k in set(b) | set(a) if b.get(k) != a.get(k)}
        if keys - JUDGMENT_FIELDS:
            raise SystemExit(f"M17 INGEST REFUSED: {b['query_id']} changed non-judgment fields "
                             f"{sorted(keys - JUDGMENT_FIELDS)}")
    scr = json.loads(admit_read(P.SCREEN_JSON).read_text())
    if not P.PANEL_JSONL.exists():
        raise SystemExit("panel.jsonl missing")
    old_ok = scr.get("panel_sha256_sealed") or scr.get("panel_sha256")
    scr.setdefault("rebound_after_judgment_ingest", []).append(
        {"from_panel_sha256": old_ok, "to_panel_sha256": sha_file(P.PANEL_JSONL),
         "fields_allowed_to_change": sorted(JUDGMENT_FIELDS), "judge": JUDGE})
    scr["panel_sha256_sealed"] = sha_file(P.PANEL_JSONL)
    write_json(P.SCREEN_JSON, scr)


def doublecheck_slice(sheet, sample, seed):
    """Seeded slice for Dylan, model answers HIDDEN so the human judgment is independent."""
    import random
    rng = random.Random(seed)
    out = []
    for kind in ("panel-candidate", "alias-pair"):
        rows = [r for r in sheet if r["kind"] == kind]
        for r in rng.sample(rows, min(DOUBLECHECK_N[kind], len(rows))):
            row = {k: v for k, v in r.items() if k not in ("relevant_yes_no", "judge", "notes")}
            row.update({"kind": kind, "human_relevant_yes_no": None, "human_judge": None})
            out.append(row)
    for r in rng.sample(sample, min(DOUBLECHECK_N["spotcheck"], len(sample))):
        out.append({"kind": "spotcheck", **r, "human_wrong": None, "human_judge": None})
    return out


# ------------------------------------------------------------------ main

def ingest(panel_log, senses_log, spot_log, judge=JUDGE, seed=17, verbose=True):
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    sheet = P.read_jsonl(PENDING)
    if any(r.get("judge") for r in sheet):
        raise SystemExit("M17 INGEST REFUSED: the review sheet already carries judgments")
    panel_rows = parse_report(panel_log)
    sense_rows = parse_report(senses_log)
    spot_rows = parse_report(spot_log)

    panel_ans = align_sheet(sheet, panel_rows, "panel-candidate", "query_id")
    sense_ans = align_sheet(sheet, sense_rows, "alias-pair", "pair_id")
    sample = P.read_jsonl(SPOT_SAMPLE)
    spot_result, _ = apply_spotcheck(sample, spot_rows)

    apply_sheet(sheet, panel_ans, judge)
    apply_sheet(sheet, sense_ans, judge)

    records = P.read_jsonl(P.PANEL_JSONL)
    before = [dict(r) for r in records]
    panel_stats = apply_panel(records, sheet, judge)
    alias_records = P.read_jsonl(ALIAS_JSONL)
    alias_stats = apply_alias(alias_records, sheet, judge)

    # write everything only after every check passed
    P._write_jsonl(PENDING, sheet)
    P._write_jsonl(P.PANEL_JSONL, records)
    P._write_jsonl(P.SELECTION_JSONL, [r for r in records if r["partition"] == "selection"])
    P._write_jsonl(P.AUDIT_JSONL, [r for r in records if r["partition"] == "audit"])
    rebind_screen(before, records)
    P._write_jsonl(ALIAS_JSONL, alias_records)
    P._write_jsonl(DOUBLECHECK, doublecheck_slice(sheet, sample, seed))
    write_json(SPOT_RESULT, {**spot_result, "sample": str(SPOT_SAMPLE.relative_to(REPO)),
                             "sample_sha256": sha_file(SPOT_SAMPLE),
                             "report_log_sha256": sha_file(Path(spot_log))})

    man = json.loads(admit_read(ALIAS_MANIFEST).read_text())
    man["by_judgment_status"] = dict(Counter(r["judgment_status"] for r in alias_records))
    man["status"] = ("FINAL — pending senses model-judged (A6); "
                     f"{man['by_judgment_status'].get('REJECTED_BY_MODEL', 0)} pairs rejected")
    man["files"]["pairs_sha256"] = sha_file(ALIAS_JSONL)
    man["model_judging_a6"] = {"judge": judge, "senses_judged": sum(alias_stats.values()),
                               "by_status": alias_stats,
                               "report_log_sha256": sha_file(Path(senses_log))}
    write_json(ALIAS_MANIFEST, man)

    log(f"[ingest] panel: {panel_stats}; alias senses: {alias_stats}; "
        f"spot check: {spot_result['wrong']}/{spot_result['n_pairs']} wrong "
        f"({spot_result['wrong_share']:.3%}, exceeds 5 %: {spot_result['exceeds_threshold']})")
    P.seal(verbose=verbose)
    rec = {
        "_schema": "m17-judgments-ingest-v1", "date": "2026-09-11", "ruling": "A6",
        "judge": judge, "script_sha256": sha_file(Path(__file__).resolve()),
        "reports": {k: {"path": str(v), "sha256": sha_file(Path(v))}
                    for k, v in (("panel", panel_log), ("senses", senses_log), ("spotcheck", spot_log))},
        "panel": panel_stats, "alias_senses": alias_stats,
        "spotcheck": {k: spot_result[k] for k in ("n_pairs", "wrong", "wrong_share",
                                                    "exceeds_threshold", "consequence")},
        "sheet_sha256": sha_file(PENDING), "panel_sha256": sha_file(P.PANEL_JSONL),
        "alias_test_sha256": sha_file(ALIAS_JSONL),
        "human_doublecheck_slice": {"path": str(DOUBLECHECK.relative_to(REPO)), "seed": seed,
                                    "sizes": DOUBLECHECK_N, "model_answers_hidden": True},
        "panel_manifest_sha256": sha_file(P.MANIFEST_OUT),
    }
    write_json(INGEST_RECORD, rec)
    log(f"[ingest] record -> {INGEST_RECORD}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--senses", required=True)
    ap.add_argument("--spotcheck", required=True)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    ingest(a.panel, a.senses, a.spotcheck, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
