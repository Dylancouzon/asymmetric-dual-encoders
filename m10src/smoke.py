"""M10 step 1 -- the generation smoke (decision 15).

Per generated form: 40 seed passages x 5 queries = 200 queries. Two gates:
  * contract >= 90%  -- computed here, from `forms.parse` on the raw reply;
  * on-form  >= 80%  -- a 50-query sample judged by an INDEPENDENT Fable subagent against the
    form's registered description in `forms.FORMS`; this script only produces the sample.

Outputs `work/m10gen/smoke/<form>.json` (everything) and `m10/SMOKE.md` (the pushable record:
gates, rates, the 50-query judge samples and the prompt hash each form's approval binds to).
"""
import hashlib, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))
OUT = REPO / "work" / "m10gen" / "smoke"

import forms, gen, seeds as seedmod

GENERATED = ["howto", "argument", "finance", "comparison", "yesno", "conversational", "health"]
CONTRACT_GATE, ONFORM_GATE, JUDGE_N = 0.90, 0.80, 50
N_SEEDS, N_PER_SEED = 40, 5


def prompt_hash(form):
    """Approval binds to this: system turn + the form's instruction wording."""
    return hashlib.blake2b((forms.SYSTEM + "\x00" + forms.FORMS[form]).encode(),
                           digest_size=8).hexdigest()


def judge_sample(queries, k):
    """A deterministic spread across the seeds, not the first k.

    `queries` arrives grouped by seed and the seeds are score-sorted, so `queries[:50]` is the
    ~10 strongest seeds' output and would overstate the on-form rate (Codex + Fable 2026-09-04).
    Round-robin over seeds instead: one query from each seed before a second from any.
    """
    by_seed = {}
    for q in queries:
        by_seed.setdefault(q["seed_id"], []).append(q["query"])
    out, r = [], 0
    while len(out) < k and any(len(v) > r for v in by_seed.values()):
        for sid in by_seed:                       # dict order == first-seen order, deterministic
            if len(by_seed[sid]) > r and len(out) < k:
                out.append(by_seed[sid][r])
        r += 1
    return out


def onform_rate(verdicts, need):
    """-> (rate, reason). A form can only PASS on `need` explicit boolean verdicts.

    `write_md` used to print PASS from a caller-supplied float with no verdicts behind it, so
    `{"onform": 0.8, "verdicts": []}` auto-approved a form without decision 15's evidence.
    """
    if not isinstance(verdicts, list) or len(verdicts) < need:
        return None, f"only {len(verdicts) if isinstance(verdicts, list) else 0}/{need} verdicts"
    vals = [v.get("on_form") if isinstance(v, dict) else v for v in verdicts[:need]]
    if any(not isinstance(v, bool) for v in vals):
        return None, "a verdict is not a boolean"
    return sum(vals) / need, None


def run(which=None, base=None, judge_n=JUDGE_N):
    which = which or GENERATED
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {"base": base} if base else {}
    print(gen.health(**({"base": base} if base else {})), flush=True)
    seen = set()                       # corpus-wide exact dedup across every form
    seed_map, seed_meta = seedmod.cached(which, per_form=N_SEEDS)
    rows = {}
    for form in which:
        s = seed_map[form]
        if len(s) < N_SEEDS:      # a short draw is recorded, not a crash that loses every form
            print(f"  WARNING {form}: only {len(s)} seeds of {N_SEEDS}", flush=True)
        r = gen.generate(form, s, n=N_PER_SEED, label=form, seen=seen, **kw)
        r["prompt_hash"] = prompt_hash(form)
        r["contract_pass"] = r["contract_rate"] >= CONTRACT_GATE
        r["judge_sample"] = judge_sample(r["queries"], judge_n)
        (OUT / f"{form}.json").write_text(json.dumps(r, indent=1))
        rows[form] = r
    (OUT / "seed_meta.json").write_text(json.dumps(seed_meta, indent=1))
    return rows, seed_meta


def write_md(rows, seed_meta, path=REPO / "m10" / "SMOKE.md", judged=None):
    """`judged`: {form: {"onform": float, "verdicts": [...], "note": str}} once Fable has read."""
    judged = judged or {}
    # Recompute every on-form rate from the verdicts themselves.
    rates, reasons = {}, {}
    for f in rows:
        want = min(JUDGE_N, len(rows[f]["judge_sample"]))
        rates[f], reasons[f] = onform_rate(judged.get(f, {}).get("verdicts", []), want)
    L = []
    A = L.append
    A("# M10 generation smoke — decision 15 (conditional pre-approval)\n")
    A(f"Generator `{gen.MODEL}` rev `{gen.REVISION[:12]}…`, vLLM on the box, thinking disabled, "
      f"temperature {gen.TEMPERATURE} / top-p {gen.TOP_P}, per-request "
      f"`seed = blake2b-64(seed_passage_id)`. {N_SEEDS} seed passages × {N_PER_SEED} queries = "
      f"200 per form. Seeds: `{seed_meta['store']}` (Wikipedia), length "
      f"{seedmod.MIN_WORDS}–{seedmod.MAX_WORDS} words, topical forms scored not first-fit; "
      f"{seed_meta['dropped_protected']} of {seed_meta['screened']} dropped on the protected "
      f"query index. Generated {time.strftime('%Y-%m-%d %H:%M %Z')}.\n")
    A("**Disclosed bias in this sample.** For the three topical forms (`health`, `finance`, "
      "`howto`) the seeds are the highest keyword-scoring passages of 400,000 Wikipedia "
      "candidates, so they carry their topic more strongly than a build-scale draw would. That "
      "is deliberate — this gate asks whether a PROMPT is well worded, and a health prompt on a "
      "sports biography fails for a reason unrelated to wording — but a PASS here does not "
      "predict the build's on-form rate. The build's own on-form rate is a separate reported "
      "diagnostic with no action attached (Fable consultation, `m10/LEDGER.md` §3).\n")
    A(f"**Gates.** contract ≥ {CONTRACT_GATE:.0%} (computed here) **and** on-form ≥ "
      f"{ONFORM_GATE:.0%} on {JUDGE_N} queries judged by an independent Fable subagent against the "
      "form's registered description. A form passing both is **auto-approved 6 h after this file "
      "is pushed** unless `Dylancouzon` comments `redraft: <form>: <note>` on the GitHub issue "
      "\"M10 smoke approval\" or replies via Remote Control; `approved: <form>` ends the window "
      "early. **Approval binds to the prompt hash below** — a revised prompt needs a new window "
      "(≤ 2 revisions, then the form is dropped from the build and its quota is not "
      "redistributed).\n")
    A("| form | contract | on-form | queries | out tok/s | prompt hash | verdict |")
    A("|---|---|---|---|---|---|---|")
    for f, r in rows.items():
        of = rates[f]
        ofs = "—" if of is None else f"{of:.0%}"
        v = ("**PASS**" if r["contract_pass"] and of is not None and of >= ONFORM_GATE
             else "**FAIL**" if of is not None or not r["contract_pass"]
             else f"pending judge ({reasons[f]})")
        A(f"| `{f}` | {r['contract_rate']:.0%} | {ofs} | {r['n_queries']} | "
          f"{r['out_tok_per_s']:.0f} | `{r['prompt_hash']}` | {v} |")
    A("")
    for f, r in rows.items():
        A(f"\n## `{f}`\n")
        A(f"*Registered description:* {forms.FORMS[f].format(n='N')}\n")
        if judged.get(f, {}).get("note"):
            A(f"*Judge:* {judged[f]['note']}\n")
        if r.get("transport_failures"):
            A(f"**{r['transport_failures']} transport failures** — not prompt evidence.\n")
        if r.get("truncated"):
            A(f"{r['truncated']} replies hit the token budget (`finish_reason=length`); "
              "truncation is a budget question, not a wording question.\n")
        A(f"Contract {r['contract_rate']:.0%} ({r['contract_ok']}/{r['n_seeds']} seeds, "
          f"{r['retried']} retried), {r['n_queries']} unique queries, "
          f"{r['exact_dupes']} exact duplicates removed.\n")
        if r["first_failures"]:
            A(f"First contract failure, truncated: `{r['first_failures'][0][:160]}`\n")
        vs = judged.get(f, {}).get("verdicts") or []
        A(f"Judge sample ({len(r['judge_sample'])} of {r['n_queries']}, spread across all "
          f"{r['n_seeds']} seeds){' — ✓/✗ is the judge verdict' if vs else ''}:\n")
        for i, q in enumerate(r["judge_sample"]):
            mark = ""
            if i < len(vs):
                v = vs[i].get("on_form") if isinstance(vs[i], dict) else vs[i]
                mark = "✓ " if v else "✗ "
            body = q if len(q) < 400 else q[:400] + "…"
            A(f"- {mark}{body}")
        A("")
    path.write_text("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    rows, meta = run(sys.argv[1:] or None)
    print("wrote", write_md(rows, meta))
