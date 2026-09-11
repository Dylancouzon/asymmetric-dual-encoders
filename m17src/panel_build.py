"""M17 step 2c: build and seal the judged evaluation panel.

Six declared domains (`registry.data.panel_domains`), ~600 queries, split 50/50 by QUERY FAMILY
into a selection partition and a sealed audit partition. Families are the split unit because two
questions on the same source document, or two near-duplicate phrasings, move together.

What the panel is, precisely:

* **Non-k8s queries** carry the *dataset's own* relevance judgments (`judgment_status:
  DATASET_QRELS`). Those are human judgments made by the dataset's annotators on the dataset's
  corpus, not independent judgments of this panel's mixed corpus: a distractor drawn from another
  source may also be relevant and is simply unjudged. That bias is recorded in the manifest.
* **Cloud-software queries** are built from the Kubernetes documentation slice and carry
  `judgment_status: PENDING_HUMAN` with candidate documents only. Nothing here fabricates a
  judgment, and no teacher ranking is used as a label anywhere: the candidates come from a
  lexical tf-idf neighbourhood over titles and lead text. `results/m17_panel_pending_judgments.jsonl`
  is the review sheet Dylan (or whoever is nominated) fills in.

Sources. Held-out-only draws from the admitted labelled sources whose licences are affirmative in
`research/m7-data-licensing.md`: SQuAD (CC BY-SA 4.0), HotpotQA (CC BY-SA 4.0), Mr. TyDi English
(Apache 2.0), plus the step-2a Kubernetes slice (CC BY 4.0). Two admitted sources are deliberately
NOT drawn from, and the reasons are recorded in the manifest:

* `fever-train` — FEVER is one of the reserved four AND a disclosed stella exposure (CLAUDE.md).
  Its training claims sit on the same Wikipedia surface as the reserved evaluation. Licensing
  permits it (CC BY-SA 3.0, approved for training); protocol caution does not. Reversible by an
  owner ruling, which would also require the executor's protected screen over those families.
* `esci-us` — Apache 2.0 and clean, but e-commerce product relevance maps to none of the six
  declared panel domains, and its store is 808k products.

"Held-out only" means `m7src/trainmix.heldout` (sha256(f"{source}:{qid}") % 50 == 0), the
registered M7 rule: such a query never entered M7 training. It does NOT mean unseen — see
`--stage screen` and `exposure_labels` in the manifest.

Stages (repository root, pre-clock, no GPU, `.venv/bin/python`):

    m17src/panel_build.py --stage draft     # queries, domains, families, corpus, partitions
    m17src/panel_build.py --stage screen    # ancestor-manifest exposure screen (slow, streams)
    m17src/panel_build.py --stage seal      # SE estimate + results/m17_panel_manifest.json

Nothing here reads `results/frozen_eval/untouched-*`, a reserved qrels cache, `work/m9reserve`,
a six-set/LoTTE payload or any development-suite component, and nothing here scores anything.
Protected-surface screening is the executor's (`results/m17_k8s_source_manifest.json` blocker).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "m17src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (RESULTS, WORK, admit_read, registry, sha_file, sha_json,  # noqa: E402
                    write_json)

TRAIN = REPO / "work" / "train"
MANIFEST_DIR = WORK / "manifest"
PANEL_DIR = WORK / "panel"
K8S_JSONL = WORK / "sources" / "k8s_docs_en.jsonl"
K8S_STEP2A = RESULTS / "m17_k8s_source_manifest.json"

PANEL_JSONL = PANEL_DIR / "panel.jsonl"
SELECTION_JSONL = PANEL_DIR / "selection.jsonl"
AUDIT_JSONL = PANEL_DIR / "audit.jsonl"
CORPUS_JSONL = PANEL_DIR / "corpus.jsonl"
DRAFT_JSON = PANEL_DIR / "draft.json"
CANDIDATES_CACHE = PANEL_DIR / "candidates_cache.jsonl"
DISTRACTOR_CACHE = PANEL_DIR / "distractor_pool_cache.jsonl"
SCREEN_JSON = PANEL_DIR / "ancestry_screen.json"
MANIFEST_OUT = RESULTS / "m17_panel_manifest.json"
SELECTION_MIRROR = RESULTS / "m17_panel_selection.jsonl"
PENDING_OUT = RESULTS / "m17_panel_pending_judgments.jsonl"
MIRROR_MAX_BYTES = 2_000_000

# Refused by name, the same list `evaluate.py` refuses. Nothing in this module opens a path,
# but a future edit that tries gets stopped here.
FORBIDDEN = ("frozen_eval/untouched-", "m9reserve", "reserved_qrels", "lotte", "devsuite",
             "work/dev/")

PANEL_SOURCES = ["squad-train", "hotpotqa-train", "mrtydi-en"]
STORE_OF = {"squad-train": "squad-ctx", "hotpotqa-train": "hotpotqa-corpus",
            "mrtydi-en": "mrtydi-docs"}
EXCLUDED_SOURCES = {
    "fever-train": ("reserved-adjacent: FEVER is one of the reserved four and a disclosed stella "
                    "exposure; its train claims share the reserved evaluation's Wikipedia "
                    "surface. Licensing (CC BY-SA 3.0) permits it; protocol caution does not."),
    "esci-us": ("Apache 2.0 and clean, but e-commerce product relevance maps to none of the six "
                "declared panel domains."),
}
HELDOUT_MOD = 50
SEED = 17
TARGET_PER_DOMAIN = 100
K8S_CANDIDATE_QUERIES = 120
K8S_CANDIDATES_PER_QUERY = 3
DISTRACTORS = {"squad-train": 3000, "hotpotqa-train": 6000, "mrtydi-en": 3000}
HOTPOT_STRATA_POOL = 60000     # seeded subsample used as the hotpot stratification pool
HUB_FANOUT = 50                # m17src/support_manifest.py's cutoff, same purpose


def _check_path(p):
    s = str(p).replace("\\", "/").lower()
    for bad in FORBIDDEN:
        if bad in s:
            raise SystemExit(f"M17 PANEL REFUSED: {p} names a protected or development surface "
                             f"({bad!r}).")
    return p


# --------------------------------------------------------------------- shared conventions

def normalize(text):
    """`m17src/support_manifest.normalize`, restated so this script imports no concurrently
    edited module: NFKC, casefold, whitespace runs collapsed, stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_id(normalized):
    """`m17src/support_manifest.group_id`: sha256(normalized text)[:16]. The interoperable key —
    the alias-test exclusion file is written in these terms, not in this script's family names."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def heldout(source, qid):
    """`m7src/trainmix.heldout`, restated (importing it pulls in `datasets` and M7's env)."""
    return int(hashlib.sha256(f"{source}:{qid}".encode()).hexdigest(), 16) % HELDOUT_MOD == 0


class Union:
    def __init__(self):
        self.parent = {}

    def add(self, k):
        self.parent.setdefault(k, k)

    def find(self, k):
        p = self.parent
        root = k
        while p[root] != root:
            root = p[root]
        while p[k] != root:
            p[k], k = root, p[k]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def family_id(root_uid):
    return "fam:" + hashlib.sha256(root_uid.encode()).hexdigest()[:16]


# --------------------------------------------------------------------- domain classifier

# A small keyword classifier, deliberately transparent. Domain labels from it are HEURISTIC and
# every record and the manifest say so. Scored over the query plus its gold document text; the
# highest-scoring domain wins if it clears MIN_SCORE, otherwise the source's map domain ('general')
# stands. Ties fall back to 'general' rather than to an arbitrary first key.
STRONG = 2
WEAK = 1
KEYWORDS = {
    "medicine": {
        STRONG: ["patient", "disease", "symptom", "diagnos", "therapy", "therapeutic", "cancer",
                 "tumour", "tumor", "vaccine", "infection", "clinical", "surgery", "surgical",
                 "antibiotic", "physician", "syndrome", "pathogen", "epidemic", "pandemic",
                 "medicine", "medical", "pharmaceutic", "dosage", "chemotherapy", "carcinoma"],
        WEAK: ["health", "hospital", "blood", "immune", "virus", "bacteria", "neuro", "cardiac",
               "drug", "nurse", "injury", "chronic", "inflammation", "mortality"],
    },
    "science-engineering": {
        STRONG: ["quantum", "thermodynamic", "semiconductor", "theorem", "astronomy",
                 "astrophysic", "chemical reaction", "electromagnetic", "algorithm", "isotope",
                 "catalyst", "polymer", "geolog", "calculus", "photosynthes", "engineering",
                 "spacecraft", "molecular", "particle physics", "aerodynamic"],
        WEAK: ["physics", "chemistry", "biology", "molecule", "electron", "equation", "velocity",
               "orbit", "voltage", "circuit", "enzyme", "protein", "species", "evolution",
               "mathematic", "gravity", "energy", "experiment", "laborator", "temperature",
               "engine", "mechanical", "software", "computer"],
    },
    "finance": {
        STRONG: ["inflation", "monetary polic", "fiscal", "interest rate", "stock market",
                 "shareholder", "dividend", "gross domestic product", "recession", "mortgage",
                 "bankrupt", "securities", "hedge fund", "central bank", "taxation", "subsidy",
                 "revenue", "investor"],
        WEAK: ["bank", "economy", "economic", "currency", "invest", "market", "tax", "debt",
               "loan", "profit", "trading", "financial", "finance", "insurance", "price",
               "commerce", "capital"],
    },
    "legal": {
        STRONG: ["supreme court", "plaintiff", "defendant", "statute", "legislation", "lawsuit",
                 "jurisdiction", "constitutional", "prosecutor", "verdict", "indict",
                 "legal system", "treaty", "copyright", "patent law", "civil rights act",
                 "criminal law", "appeal court", "court of appeal", "attorney"],
        WEAK: ["court", "law", "legal", "judge", "judicial", "constitution", "amendment",
               "criminal", "trial", "contract", "rights", "regulation", "parliament act",
               "sentence", "illegal", "convict"],
    },
}
MIN_SCORE = 4
DOMAIN_METHOD = (
    "HEURISTIC. Source-to-domain map (m17src/support_manifest.SOURCE_DOMAIN) gives 'general' for "
    "every admitted labelled source and 'cloud-software' for the Kubernetes slice. Within the "
    "Wikipedia-derived sources a keyword classifier over the query text plus its gold document "
    "text re-labels science-engineering / medicine / finance / legal when the winning domain "
    f"scores at least the recorded classifier_min_score (strong term {STRONG} points, weak term "
    f"{WEAK}, counted once per distinct term; module default {MIN_SCORE}) and strictly beats the "
    "runner-up. These are topical hints, not "
    "adjudicated domain labels, and per-domain numbers inherit that imprecision.")


def domain_scores(text):
    low = normalize(text)
    return {dom: sum(w * sum(1 for t in terms if t in low) for w, terms in tiers.items())
            for dom, tiers in KEYWORDS.items()}


def domain_from_scores(scores, source_domain="general", min_score=MIN_SCORE):
    best = max(scores, key=lambda d: (scores[d], d))
    ranked = sorted(scores.values(), reverse=True)
    runner_up = ranked[1] if len(ranked) > 1 else 0
    if scores[best] >= min_score and scores[best] > runner_up:
        return best
    return source_domain


def classify(text, source_domain="general", min_score=MIN_SCORE):
    """-> (domain, {domain: score}). `text` is query + gold document text, normalized inside."""
    scores = domain_scores(text)
    return domain_from_scores(scores, source_domain, min_score), scores


# --------------------------------------------------------------------- inputs

def load_source(name):
    return json.loads(admit_read(_check_path(TRAIN / "sources" / f"{name}.json")).read_text())


def load_store(name):
    b = json.loads(admit_read(_check_path(TRAIN / "stores" / f"{name}.json")).read_text())
    return b["ids"], b["texts"]


def load_doc_groups(source):
    p = MANIFEST_DIR / "doc_groups" / f"{source}.tsv.gz"
    if not p.exists():
        return {}
    out = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            did, gid = line.rstrip("\n").split("\t")
            out[did] = gid
    return out


def load_training_query_shas():
    """The normalized-text shas and family ids of every *training* query, from step 2b's
    `work/m17/manifest/query_families.tsv.gz`. Used to prove a panel query is not a training
    query and to find sibling exposure. Missing file is a recorded gap, not a silent pass."""
    p = MANIFEST_DIR / "query_families.tsv.gz"
    if not p.exists():
        return None
    shas, fams, by_source = set(), set(), Counter()
    with gzip.open(p, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_src, i_fam, i_sha = (header.index("source"), header.index("family_id"),
                               header.index("normalized_text_sha"))
        for line in f:
            r = line.rstrip("\n").split("\t")
            shas.add(r[i_sha])
            fams.add(r[i_fam])
            by_source[r[i_src]] += 1
    return {"shas": shas, "families": fams, "by_source": dict(by_source),
            "sha256": sha_file(p), "n": sum(by_source.values())}


def k8s_docs():
    """[{path, title, text, sha256}], with the step-2a near-duplicate paths excluded."""
    step2a = json.loads(K8S_STEP2A.read_text())
    flagged = set(step2a["decontamination"]["result"]["flagged_paths"])
    out = []
    with open(_check_path(K8S_JSONL)) as f:
        for line in f:
            r = json.loads(line)
            if r["path"] in flagged:
                continue
            out.append(r)
    return out, sorted(flagged), step2a


# --------------------------------------------------------------------- k8s candidate queries

HEADING_STOP = {"before you begin", "what's next", "whats next", "overview", "examples",
                "introduction", "summary", "feedback", "prerequisites", "objectives",
                "cleaning up", "clean up", "see also", "notes", "usage", "syntax", "options"}
TITLE_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)


def k8s_queries(docs, n_target=K8S_CANDIDATE_QUERIES, seed=SEED):
    """Candidate cloud-software queries from headings, glossary entries and task titles.

    One query per source document at most, spread over the documentation sections. No judgment
    is produced here: every item is PENDING_HUMAN and carries candidate documents only.
    """
    by_kind = defaultdict(list)
    for d in docs:
        path, title, text = d["path"], (d.get("title") or "").strip(), d["text"]
        if "/glossary/" in path and title:
            by_kind["glossary"].append((d, title, "glossary-term"))
        elif path.startswith("content/en/docs/tasks/") and title:
            by_kind["task"].append((d, title, "task-title"))
        elif title:
            heads = [h.strip() for h in TITLE_RE.findall(text)]
            heads = [h for h in heads
                     if normalize(h) not in HEADING_STOP and 2 <= len(h.split()) <= 9
                     and not h.startswith("{{")]
            if heads:
                by_kind["heading"].append((d, heads[0], "section-heading"))
            else:
                by_kind["heading"].append((d, title, "page-title"))
    quota = {"glossary": 40, "task": 40, "heading": 40}
    rng = random.Random(seed)
    picked = []
    for kind, want in quota.items():
        pool = sorted(by_kind.get(kind, []), key=lambda t: t[0]["path"])
        rng.shuffle(pool)
        picked += pool[:want]
    # top up from whatever is left if a bucket was short, keeping the order deterministic
    if len(picked) < n_target:
        chosen = {d["path"] for d, _, _ in picked}
        rest = sorted((t for v in by_kind.values() for t in v if t[0]["path"] not in chosen),
                      key=lambda t: t[0]["path"])
        rng.shuffle(rest)
        picked += rest[:n_target - len(picked)]
    picked = picked[:n_target]
    out = []
    for d, phrase, form in sorted(picked, key=lambda t: t[0]["path"]):
        q = phrase if form != "glossary-term" else f"what is {phrase}"
        out.append({"doc": d, "query": " ".join(q.split()), "form": form})
    return out


def tfidf_neighbors(docs, k=K8S_CANDIDATES_PER_QUERY - 1):
    """Lexical tf-idf cosine neighbours over title + lead text. NOT a teacher ranking: candidate
    documents for human judging must not come from any model's scores (m17/PLANNING.md)."""
    import numpy as np
    texts = [normalize((d.get("title") or "") + " " + d["text"][:600]) for d in docs]
    vocab, df = {}, Counter()
    rows = []
    for t in texts:
        c = Counter(w for w in t.split() if len(w) > 2)
        rows.append(c)
        for w in c:
            df[w] += 1
            vocab.setdefault(w, len(vocab))
    n = len(docs)
    mat = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, c in enumerate(rows):
        for w, f in c.items():
            mat[i, vocab[w]] = (1.0 + math.log(f)) * math.log(n / (1 + df[w]))
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= np.maximum(norms, 1e-9)
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)
    order = np.argsort(-sims, axis=1)[:, :k]
    return {docs[i]["path"]: [docs[int(j)]["path"] for j in order[i]] for i in range(n)}


# --------------------------------------------------------------------- draft stage

def _sample_strata(items, budget, rng):
    """Stratified seeded sample: equal budget per stratum, remainder by the largest strata."""
    by = defaultdict(list)
    for key, strat in items:
        by[strat].append(key)
    strata = sorted(by)
    if not strata:
        return []
    per = budget // len(strata)
    out, leftovers = [], []
    for s in strata:
        pool = sorted(by[s])
        rng.shuffle(pool)
        out += pool[:per]
        leftovers += pool[per:]
    rng.shuffle(leftovers)
    return out + leftovers[:max(0, budget - len(out))]


def assign_family_domains(items):
    """Give every member of a family the family's majority domain (ties to 'general').

    Domains are the bucket the 50/50 split happens inside, so a family whose members carried
    different heuristic labels could otherwise land partly in one domain's selection half and
    partly in another's audit half — the exact leak the family split exists to prevent.
    """
    fam_domain = defaultdict(Counter)
    for c in items:
        fam_domain[c["family_id"]][c["domain"]] += 1
    for c in items:
        counts = fam_domain[c["family_id"]]
        top = max(counts.values())
        winners = sorted(d for d, n in counts.items() if n == top)
        c["domain"] = winners[0] if len(winners) == 1 else "general"
        if "slice" in c:
            c["slice"] = f"{c['domain']}:{c['slice'].split(':', 1)[1]}"
    return items


def split_families(fams, order, want, member_cap=3):
    """Assign WHOLE families to selection/audit alternately until `want` queries are taken.

    The split unit is the family (`registry.data.panel_split_unit`): every member of a family
    lands in the same partition, so a near-duplicate phrasing or a second question on the same
    source document can never sit on both sides. `member_cap` stops one prolific source document
    from dominating a domain.
    """
    taken, sel_fams, aud_fams = [], [], []
    for fam in order:
        if len(taken) >= want:
            break
        members = sorted(fams[fam], key=lambda c: c["qid"])[:member_cap]
        part = "selection" if len(sel_fams) <= len(aud_fams) else "audit"
        (sel_fams if part == "selection" else aud_fams).append(fam)
        for m in members:
            m["partition"] = part
        taken += members
    return taken, sel_fams, aud_fams


def draft(seed=SEED, verbose=True, min_score=MIN_SCORE, reuse_cache=False):
    reg = registry()
    domains = list(reg["data"]["panel_domains"])
    target_total = int(reg["data"]["m17_panel_target_queries"])
    sel_frac = float(reg["data"]["panel_selection_fraction"])
    rng = random.Random(seed)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    train_q = load_training_query_shas()
    log(f"[panel] training-query manifest: "
        f"{train_q['n']:,} rows" if train_q else "[panel] training-query manifest MISSING")

    # ---- held-out candidates from the drawn sources
    candidates, gold_needed, doc_groups, trained_doc_groups = [], defaultdict(set), {}, set()
    for src in PANEL_SOURCES:
        blob = load_source(src)
        doc_groups[src] = load_doc_groups(src)
        n_ho = 0
        for p in blob["pairs"]:
            gids = {doc_groups[src].get(d, "unknown:" + d) for d in p["pos"]}
            if not heldout(src, p["qid"]):
                trained_doc_groups |= gids
                continue
            n_ho += 1
            if not p["pos"]:
                continue
            candidates.append({"source": src, "qid": p["qid"], "query": p["query"],
                               "pos": list(p["pos"]), "gold_groups": sorted(gids)})
            gold_needed[src] |= set(p["pos"])
        log(f"[panel] {src}: {len(blob['pairs']):,} pairs, {n_ho:,} held out")

    # a panel query may not be a training query under any spelling
    if train_q:
        before = len(candidates)
        candidates = [c for c in candidates
                      if group_id(normalize(c["query"])) not in train_q["shas"]]
        log(f"[panel] dropped {before - len(candidates)} held-out queries whose normalized text "
            "also appears as a training query")

    # ---- document text for golds, and the stratification pool for distractors
    gold_text, distractor_pool = {}, defaultdict(list)
    cached = reuse_cache and CANDIDATES_CACHE.exists() and DISTRACTOR_CACHE.exists()
    for src in ([] if cached else PANEL_SOURCES):
        store = STORE_OF[src]
        ids, texts = load_store(store)
        need = gold_needed[src]
        idx = {}
        for i, did in enumerate(ids):
            if did in need:
                idx[did] = i
        for did, i in idx.items():
            gold_text[(src, did)] = texts[i]
        pool_ix = list(range(len(ids)))
        if len(pool_ix) > HOTPOT_STRATA_POOL:
            pool_ix = rng.sample(pool_ix, HOTPOT_STRATA_POOL)
        for i in pool_ix:
            if ids[i] in need:
                continue
            distractor_pool[src].append((ids[i], domain_scores(texts[i][:1200])))
        log(f"[panel] store {store}: {len(ids):,} docs, {len(idx):,} golds resolved, "
            f"{len(distractor_pool[src]):,} in the stratification pool")
        del ids, texts

    # ---- domain label per candidate
    if cached:
        # The classifier inputs are cached so a threshold can be measured (and the labels
        # reproduced) without re-reading 2.9 GB of document stores. Candidate identity and the
        # distractor stratification pool are both fixed by the cache; only the labels move.
        candidates = read_jsonl(CANDIDATES_CACHE)
        for row in read_jsonl(DISTRACTOR_CACHE):
            distractor_pool[row["source"]].append((row["doc_id"], row["scores"]))
        log(f"[panel] reused caches: {len(candidates):,} candidates, "
            f"{sum(len(v) for v in distractor_pool.values()):,} stratification-pool documents")
    for c in candidates:
        if "classify_text" not in c:
            gold = " ".join(gold_text.get((c["source"], d), "")[:1200] for d in c["pos"][:2])
            c["classify_text"] = " ".join((c["query"] + " " + gold).split())[:2600]
        c["domain"] = domain_from_scores(domain_scores(c["classify_text"]), min_score=min_score)
        c["slice"] = f"{c['domain']}:{'factoid'}"
    if not cached:
        _write_jsonl(CANDIDATES_CACHE, [{k: c[k] for k in
                                         ("source", "qid", "query", "pos", "gold_groups",
                                          "classify_text", "domain")} for c in candidates])
        _write_jsonl(DISTRACTOR_CACHE, [{"source": s, "doc_id": d, "scores": sc}
                                        for s, rows in sorted(distractor_pool.items())
                                        for d, sc in rows])

    # ---- k8s candidates
    kdocs, flagged, step2a = k8s_docs()
    kq = k8s_queries(kdocs, seed=seed)
    neighbors = tfidf_neighbors(kdocs)
    ktext = {d["path"]: d for d in kdocs}
    k8s_items = []
    for i, item in enumerate(kq):
        path = item["doc"]["path"]
        cands = [path] + [p for p in neighbors.get(path, []) if p != path]
        k8s_items.append({
            "source": "k8s-docs-en", "qid": f"k8s-{i:04d}", "query": item["query"],
            "pos": [], "candidates": cands[:K8S_CANDIDATES_PER_QUERY],
            "gold_groups": [group_id(normalize(item["doc"]["text"]))],
            "domain": "cloud-software", "slice": f"cloud-software:{item['form']}",
            "form": item["form"]})
    log(f"[panel] kubernetes: {len(k8s_items)} candidate queries, "
        f"{sum(len(k['candidates']) for k in k8s_items)} pending judgments")

    # ---- families: union-find over shared gold document group and identical normalized text
    uf = Union()
    fanout = Counter()
    for c in candidates + k8s_items:
        for g in c["gold_groups"]:
            fanout[g] += 1
    for c in candidates + k8s_items:
        uid = f"{c['source']}:{c['qid']}"
        uf.add(uid)
        key_text = "txt:" + group_id(normalize(c["query"]))
        uf.add(key_text)
        uf.union(uid, key_text)
        for g in c["gold_groups"]:
            if fanout[g] > HUB_FANOUT:      # a hub document joins nothing (step 2b's rule)
                continue
            uf.add("doc:" + g)
            uf.union(uid, "doc:" + g)
    for c in candidates + k8s_items:
        c["family_id"] = family_id(uf.find(f"{c['source']}:{c['qid']}"))
    assign_family_domains(candidates + k8s_items)

    # ---- select ~TARGET_PER_DOMAIN per domain, whole families, sources interleaved
    by_domain = defaultdict(list)
    for c in candidates + k8s_items:
        by_domain[c["domain"]].append(c)
    chosen, per_domain_report = [], {}
    for dom in domains:
        pool = by_domain.get(dom, [])
        fams = defaultdict(list)
        for c in pool:
            fams[c["family_id"]].append(c)
        order = sorted(fams)
        rng.shuffle(order)
        want = TARGET_PER_DOMAIN if dom != "cloud-software" else K8S_CANDIDATE_QUERIES
        taken, sel_fams, aud_fams = split_families(fams, order, want)
        chosen += taken
        per_domain_report[dom] = {
            "queries": len(taken), "families": len(sel_fams) + len(aud_fams),
            "families_available": len(order),
            "selection_queries": sum(1 for c in taken if c["partition"] == "selection"),
            "audit_queries": sum(1 for c in taken if c["partition"] == "audit"),
            "selection_families": len(sel_fams), "audit_families": len(aud_fams),
            "short_of_target": max(0, want - len(taken))}
        log(f"[panel] {dom}: {len(taken)} queries in {len(sel_fams) + len(aud_fams)} families "
            f"({per_domain_report[dom]['selection_queries']} selection / "
            f"{per_domain_report[dom]['audit_queries']} audit)")

    # ---- corpus: every gold plus a seeded stratified distractor sample per source
    corpus = []
    gold_ids = defaultdict(set)
    for c in chosen:
        if c["source"] == "k8s-docs-en":
            continue
        gold_ids[c["source"]] |= set(c["pos"])
    for src in PANEL_SOURCES:
        for did in sorted(gold_ids[src]):
            corpus.append({"doc_id": did, "source": src, "store": STORE_OF[src], "role": "gold"})
        picked = _sample_strata([(d, domain_from_scores(s, min_score=min_score))
                                 for d, s in distractor_pool[src]], DISTRACTORS[src], rng)
        for did in sorted(picked):
            corpus.append({"doc_id": did, "source": src, "store": STORE_OF[src],
                           "role": "distractor"})
    for d in kdocs:
        corpus.append({"doc_id": d["path"], "source": "k8s-docs-en",
                       "store": "work/m17/sources/k8s_docs_en.jsonl",
                       "role": "candidate-or-distractor", "sha256": d["sha256"]})
    log(f"[panel] corpus: {len(corpus):,} documents")

    # ---- panel records
    records, pending = [], []
    for c in sorted(chosen, key=lambda c: (c["domain"], c["source"], c["qid"])):
        qid = f"{c['source']}:{c['qid']}"
        rec = {"query_id": qid, "query": c["query"], "source": c["source"],
               "domain": c["domain"], "domain_label": "heuristic", "slice": c["slice"],
               "family_id": c["family_id"], "partition": c["partition"],
               "normalized_text_sha": group_id(normalize(c["query"])),
               "gold_document_groups": c["gold_groups"]}
        if c["source"] == "k8s-docs-en":
            rec["judgment_status"] = "PENDING_HUMAN"
            rec["qrels"] = {}
            rec["candidates"] = c["candidates"]
            rec["candidate_source"] = "lexical tf-idf neighbourhood over title + lead text"
            for cand in c["candidates"]:
                d = ktext[cand]
                pending.append({
                    "kind": "panel-candidate",
                    "query_id": qid, "query": c["query"], "domain": c["domain"],
                    "partition": c["partition"], "candidate_doc_id": cand,
                    "candidate_title": (d.get("title") or Path(cand).stem),
                    "candidate_first_300_chars": " ".join(d["text"].split())[:300],
                    "relevant_yes_no": None, "judge": None, "notes": ""})
        else:
            rec["judgment_status"] = "DATASET_QRELS"
            rec["qrels"] = {d: 1 for d in c["pos"]}
        records.append(rec)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PANEL_JSONL, records)
    _write_jsonl(SELECTION_JSONL, [r for r in records if r["partition"] == "selection"])
    _write_jsonl(AUDIT_JSONL, [r for r in records if r["partition"] == "audit"])
    _write_jsonl(CORPUS_JSONL, corpus)
    assert_pending_unjudged(PENDING_OUT)
    _write_jsonl(PENDING_OUT, pending)
    draft_rec = {
        "seed": seed, "panel_sources": PANEL_SOURCES,
        "classifier_min_score": min_score,
        "classifier_threshold_yield": {
            str(ms): dict(Counter(domain_from_scores(domain_scores(c["classify_text"]),
                                                     min_score=ms) for c in candidates))
            for ms in (2, 3, 4, 5, 6)},
        "domain_availability": {d: sum(1 for c in candidates + k8s_items if c["domain"] == d)
                                for d in domains},
        "excluded_sources": EXCLUDED_SOURCES,
        "per_domain": per_domain_report,
        "n_queries": len(records), "n_families": len({r["family_id"] for r in records}),
        "n_pending_judgments": len(pending),
        "corpus": {"documents": len(corpus),
                   "per_source": dict(Counter(d["source"] for d in corpus)),
                   "per_role": dict(Counter(d["role"] for d in corpus)),
                   "distractor_budget": DISTRACTORS,
                   "stratification": "seeded stratified by the keyword classifier's label over a "
                                     f"seeded pool of at most {HOTPOT_STRATA_POOL:,} documents "
                                     "per store"},
        "training_query_manifest": ({"path": "work/m17/manifest/query_families.tsv.gz",
                                     "sha256": train_q["sha256"], "rows": train_q["n"]}
                                    if train_q else None),
        "k8s_flagged_paths_excluded": flagged,
        "k8s_jsonl_sha256": step2a["extraction"]["jsonl_sha256"],
        "target_total": target_total, "selection_fraction": sel_frac,
        "trained_doc_groups": len(trained_doc_groups),
    }
    # sibling exposure: held-out query, but a TRAINING query shares its gold document group
    for r in records:
        sib = [g for g in r["gold_document_groups"] if g in trained_doc_groups]
        r["sibling_trained_document_groups"] = len(sib)
    _write_jsonl(PANEL_JSONL, records)
    _write_jsonl(SELECTION_JSONL, [r for r in records if r["partition"] == "selection"])
    _write_jsonl(AUDIT_JSONL, [r for r in records if r["partition"] == "audit"])
    write_json(DRAFT_JSON, draft_rec)
    log(f"[panel] draft written: {len(records)} queries, {len(pending)} pending judgments")
    return draft_rec


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def read_jsonl(path):
    with open(admit_read(_check_path(path))) as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_pending_unjudged(path=None):
    """Refuse to rewrite the human review sheet once anyone has answered a row.

    Rebuilding the panel or the alias test rewrites `relevant_yes_no`/`judge` as empty, which
    destroys judging work that cannot be recovered from any other artifact. There is no merge
    flag: move or archive the answered sheet deliberately, then rebuild.
    """
    path = Path(path or PENDING_OUT)
    if not path.exists():
        return 0
    rows = read_jsonl(path)
    judged = [r for r in rows
              if r.get("relevant_yes_no") not in (None, "") or r.get("judge") not in (None, "")]
    if judged:
        raise SystemExit(
            f"M17 PANEL REFUSED: {path} already carries {len(judged)} answered review rows "
            "(relevant_yes_no or judge filled in). Rebuilding would erase them. Move the sheet "
            "aside deliberately if the panel really must be rebuilt.")
    return len(rows)


# --------------------------------------------------------------------- ancestry screen

def ancestor_streams():
    """The available ancestor training manifests of the warm start, as (name, iterator) pairs.

    M7 TRAIN query text is the superset the M9 pool and M10's `m9-pool` segment were derived from
    (`m9src/data.labelled_query_pool` = M8's filtered TRAIN query texts), so streaming it covers
    those two. `pseudoq-2000000-0.json` is the "2m" pseudo-query pool named in the warm start's
    own run id. M10's harvest and generated pools are streamed as the generated-query lineage.
    Any stream whose file is absent is reported as a gap and turns families exposure-unknown.
    """
    W = REPO / "work"
    out = []

    def _json_list(p, source=None):
        """A plain list of query strings. `source` applies the mod-50 held-out rule by INDEX,
        exactly as `m7src/mix.query_texts` does for the query-text-only sources: the held-out
        slice was never a training input, so streaming it would make every panel query match
        itself and report contamination that does not exist."""
        def it():
            rows = json.loads(Path(p).read_text())
            for i, r in enumerate(rows):
                if source and heldout(source, str(i)):
                    continue
                yield r if isinstance(r, str) else (r.get("query") or r.get("text") or "")
        return it

    def _jsonl(p, fields=("query", "text")):
        def it():
            with open(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    for k in fields:
                        if isinstance(r.get(k), str):
                            yield r[k]
                            break
        return it

    def _pairs(p, source):
        """TRAIN pairs only: the held-out slice is excluded by the same registered rule the panel
        draws with, so a panel query can never match itself here."""
        def it():
            blob = json.loads(Path(p).read_text())
            for pair in blob["pairs"]:
                if heldout(source, pair["qid"]):
                    continue
                yield pair["query"]
        return it

    for src in ("squad-train", "hotpotqa-train", "mrtydi-en", "fever-train", "esci-us"):
        f = TRAIN / "sources" / f"{src}.json"
        if f.exists():
            out.append((f"m7-train-queries:{src}", _pairs(f, src), str(f)))
    for src in ("nqopen", "triviaqa"):
        f = TRAIN / "querytext" / f"{src}.json"
        if f.exists():
            out.append((f"m7-querytext:{src}", _json_list(f, src), str(f)))
    pq = W / "pseudoq" / "pseudoq-2000000-0.json"
    if pq.exists():
        out.append(("m7-pseudoq-2m", _json_list(pq), str(pq)))
    hv = W / "m10harvest" / "harvest_train.jsonl"
    if hv.exists():
        out.append(("m10-harvest", _jsonl(hv), str(hv)))
    gq = W / "m10gen" / "generated_queries.jsonl"
    if gq.exists():
        out.append(("m10-generated", _jsonl(gq), str(gq)))
    return out


def screen(items, min_share=4, limit_per_stream=None, verbose=True):
    """Screen panel/alias query text against every available ancestor manifest.

    `items`: [{"key", "text", "family_id"}]. Candidate-side inverted index, ancestor text streamed
    against it — the direction and interface of `m10src/cov_screen.py::screen`, as step 2a used.
    The threshold is 4 shared word-grams rather than step 2a's 8 because the candidate side here
    is QUERY text: `decontam.query_grams` is the query path (8-grams plus short-text 4-grams), and
    an 8-gram vote is unreachable for a six-word question. Exact normalized match is reported
    separately and is the dominant signal at this length.
    """
    sys.path.insert(0, str(REPO / "m7src")) if str(REPO / "m7src") not in sys.path else None
    import decontam
    import numpy as np

    texts = [it["text"] for it in items]
    grams = [decontam.query_grams(t) for t in texts]
    exact = [decontam.exact_u64(t) for t in texts]
    idx = decontam.Inverted(grams, exact)
    hits_exact = defaultdict(set)
    hits_near = defaultdict(set)
    per_stream = {}
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    for name, factory, path in ancestor_streams():
        n = 0
        ex_n = nr_n = 0
        for text in factory():
            if not text:
                continue
            n += 1
            if limit_per_stream and n > limit_per_stream:
                break
            e, nr = idx.match(text, min_share=min_share, want_sketch=False)
            for i in np.asarray(e).tolist():
                hits_exact[items[i]["key"]].add(name)
                ex_n += 1
            for i in np.asarray(nr).tolist():
                hits_near[items[i]["key"]].add(name)
                nr_n += 1
        per_stream[name] = {"path": path, "n_streamed": n, "hits_exact": ex_n, "hits_near": nr_n}
        log(f"[screen] {name}: {n:,} texts, {ex_n} exact, {nr_n} near")
    return {"min_share": min_share, "n_candidates": len(items), "per_stream": per_stream,
            "hits_exact": {k: sorted(v) for k, v in hits_exact.items()},
            "hits_near": {k: sorted(v) for k, v in hits_near.items()},
            "interface": ("m7src/decontam.query_grams + Inverted(...).match(text, min_share, "
                          "want_sketch=False); candidate-side index, ancestor text streamed"),
            "streams_expected": ["m7-train-queries", "m7-querytext", "m7-pseudoq-2m",
                                 "m10-harvest", "m10-generated"]}


def exposure_label(rec, screen_hits_exact, screen_hits_near, all_streams_present):
    """-> (label, reason). Label is exposure-known-exposed / -clean / exposure-unknown.

    The reason matters as much as the label: 'this query's text was found in a training manifest'
    and 'this query's gold documents were in the warm start's document pool' are both exposure,
    but they are not the same finding, and a reader of the audit should not have to guess which
    one produced the label.
    """
    key = rec["query_id"]
    if key in screen_hits_exact:
        return "exposure-known-exposed", "ancestor-query-text-matched-exactly"
    if key in screen_hits_near:
        return "exposure-known-exposed", "ancestor-query-text-near-matched"
    if rec.get("sibling_trained_document_groups"):
        return "exposure-known-exposed", "training-query-shares-this-gold-document"
    if rec["source"] != "k8s-docs-en":
        # every gold document of an admitted labelled source sat in M7's document pool
        return "exposure-known-exposed", "gold-documents-were-in-the-ancestor-document-pool"
    if not all_streams_present:
        return "exposure-unknown", "an-ancestor-manifest-could-not-be-streamed"
    return "exposure-known-clean", "screened-against-every-available-ancestor-manifest-no-hit"


def run_screen(verbose=True):
    records = read_jsonl(PANEL_JSONL)
    items = [{"key": r["query_id"], "text": r["query"], "family_id": r["family_id"]}
             for r in records]
    alias = WORK / "alias" / "alias_test.jsonl"
    if alias.exists():
        for a in read_jsonl(alias):
            for view in ("view_a", "view_b"):
                items.append({"key": f"{a['pair_id']}:{view}", "text": a[view],
                              "family_id": a["family_id"]})
    out = screen(items, verbose=verbose)
    out["n_panel_queries"] = len(records)
    out["n_alias_views"] = len(items) - len(records)
    # Bind the receipt to the exact files it screened: a rebuilt panel whose new queries were
    # never streamed must not inherit this screen's "clean" labels (`seal` checks these).
    out["panel_sha256"] = sha_file(PANEL_JSONL)
    out["alias_sha256"] = sha_file(alias) if alias.exists() else None
    write_json(SCREEN_JSON, out)
    if verbose:
        print(f"[screen] written {SCREEN_JSON}")
    return out


# --------------------------------------------------------------------- SE and sealing

def expected_se(n_families, sd_delta=(0.15, 0.25)):
    """Expected standard error of the per-domain paired nDCG@10 difference.

    ASSUMPTION, stated because no observation exists: the per-query paired difference
    (candidate - control) has a standard deviation between 0.15 and 0.25 on a binary-relevance
    panel of this kind, and families are the independent unit, so n is the FAMILY count, not the
    query count. SE = sd / sqrt(n_families). A single system's own nDCG@10 has a larger spread
    (sd ~0.3-0.4) and therefore a larger SE; the paired difference is what the screen would read.
    """
    if not n_families:
        return {"n_families": 0, "se": None}
    return {"n_families": n_families,
            "se_paired_ndcg10": [round(sd / math.sqrt(n_families), 4) for sd in sd_delta],
            "sd_assumed": list(sd_delta),
            "se_single_system_ndcg10": [round(sd / math.sqrt(n_families), 4)
                                        for sd in (0.30, 0.40)]}


def _assert_screen_matches(scr):
    """The screen must be the one that ran on THESE files.

    `panel_sha256_sealed` is the hash sealing itself leaves behind after it writes the exposure
    labels into `panel.jsonl`, so re-sealing an unchanged panel is idempotent while a redrafted
    panel is refused.
    """
    alias = WORK / "alias" / "alias_test.jsonl"
    now = sha_file(PANEL_JSONL)
    if now not in (scr.get("panel_sha256"), scr.get("panel_sha256_sealed")):
        raise SystemExit(
            f"M17 PANEL REFUSED: {SCREEN_JSON} was written for panel sha256 "
            f"{str(scr.get('panel_sha256'))[:16]}, but {PANEL_JSONL} is now {now[:16]}. Re-run "
            "--stage screen: a stale receipt would certify unscreened queries as clean.")
    alias_now = sha_file(alias) if alias.exists() else None
    if alias_now != scr.get("alias_sha256"):
        raise SystemExit(
            f"M17 PANEL REFUSED: {SCREEN_JSON} was written for alias sha256 "
            f"{str(scr.get('alias_sha256'))[:16]}, but the alias test is now "
            f"{str(alias_now)[:16]}. Re-run --stage screen.")


def seal(verbose=True):
    reg = registry()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    records = read_jsonl(PANEL_JSONL)
    corpus = read_jsonl(CORPUS_JSONL)
    draft_rec = json.loads(DRAFT_JSON.read_text())
    if len(corpus) != draft_rec["corpus"]["documents"]:
        raise SystemExit(f"M17 PANEL REFUSED: {CORPUS_JSONL} holds {len(corpus)} documents, the "
                         f"draft recorded {draft_rec['corpus']['documents']}. Re-run --stage "
                         "draft rather than sealing a manifest against a different corpus.")
    straddle = {f for f in {r["family_id"] for r in records}
                if len({r["partition"] for r in records if r["family_id"] == f}) > 1}
    if straddle:
        raise SystemExit(f"M17 PANEL REFUSED: {len(straddle)} families straddle the selection and "
                         f"audit partitions, e.g. {sorted(straddle)[:3]}. The family is the split "
                         "unit (registry.data.panel_split_unit).")
    gold_in_corpus = {d["doc_id"] for d in corpus}
    missing = sorted({d for r in records for d in r["qrels"]} - gold_in_corpus)
    if missing:
        raise SystemExit(f"M17 PANEL REFUSED: {len(missing)} judged documents are not in the "
                         f"declared corpus, e.g. {missing[:3]}.")
    scr = json.loads(admit_read(SCREEN_JSON).read_text()) if SCREEN_JSON.exists() else None
    if scr:
        _assert_screen_matches(scr)
    all_streams = bool(scr) and all(
        any(name.startswith(exp) for name in scr["per_stream"])
        for exp in scr["streams_expected"])
    he = set(scr["hits_exact"]) if scr else set()
    hn = set(scr["hits_near"]) if scr else set()
    for r in records:
        r["exposure"], r["exposure_reason"] = (
            exposure_label(r, he, hn, all_streams) if scr
            else ("exposure-unknown", "no-ancestry-screen-was-run"))
    _write_jsonl(PANEL_JSONL, records)
    _write_jsonl(SELECTION_JSONL, [r for r in records if r["partition"] == "selection"])
    _write_jsonl(AUDIT_JSONL, [r for r in records if r["partition"] == "audit"])
    if scr:      # sealing rewrote panel.jsonl; keep the receipt bound to the sealed bytes
        scr["panel_sha256_sealed"] = sha_file(PANEL_JSONL)
        write_json(SCREEN_JSON, scr)

    per_domain = {}
    for dom in reg["data"]["panel_domains"]:
        rs = [r for r in records if r["domain"] == dom]
        sel = [r for r in rs if r["partition"] == "selection"]
        aud = [r for r in rs if r["partition"] == "audit"]
        per_domain[dom] = {
            "queries": len(rs),
            "selection": {"queries": len(sel), "families": len({r["family_id"] for r in sel})},
            "audit": {"queries": len(aud), "families": len({r["family_id"] for r in aud})},
            "relevant_documents": sum(len(r["qrels"]) for r in rs),
            "queries_with_judgments": sum(1 for r in rs if r["qrels"]),
            "pending_human_queries": sum(1 for r in rs
                                         if r["judgment_status"] == "PENDING_HUMAN"),
            "exposure": dict(Counter(r["exposure"] for r in rs)),
            "exposure_reasons": dict(Counter(r["exposure_reason"] for r in rs)),
            "families_total": len({r["family_id"] for r in rs}),
            "expected_se": expected_se(len({r["family_id"] for r in sel})),
        }
        log(f"[seal] {dom}: {len(sel)}/{len(aud)} sel/audit, "
            f"SE {per_domain[dom]['expected_se'].get('se_paired_ndcg10')}")

    sel_bytes = SELECTION_JSONL.stat().st_size
    mirrored = sel_bytes <= MIRROR_MAX_BYTES
    if mirrored:
        SELECTION_MIRROR.write_bytes(SELECTION_JSONL.read_bytes())
    elif SELECTION_MIRROR.exists():
        SELECTION_MIRROR.unlink()

    pending = read_jsonl(PENDING_OUT)
    # the sheet holds this script's panel candidates and, once alias_test_build.py has run, the
    # ambiguous alias pairs; both kinds are counted, neither is overwritten
    panel_pending = [p for p in pending if p.get("kind", "panel-candidate") == "panel-candidate"]
    alias_pending = [p for p in pending if p.get("kind") == "alias-pair"]
    manifest = {
        "milestone": "M17", "step": "2c", "date": "2026-09-11",
        "script": "m17src/panel_build.py",
        "status": "PROVISIONAL — pending human judgments unresolved",
        "provisional_reason": (
            f"{len(panel_pending)} cloud-software candidate judgments and {len(alias_pending)} "
            "ambiguous alias pairs are PENDING_HUMAN. Until they are "
            "resolved the cloud-software domain has no relevance judgments and the panel may not "
            "be scored as a six-domain panel. Resolving them re-seals this manifest with a new "
            "panel hash; the 72-hour clock starts from the sealed hash "
            "(registry.data.panel_timing)."),
        "gpu_used": False, "protected_payloads_opened": False, "scored_anything": False,
        "seed": draft_rec["seed"],
        "registry_targets": {"m17_panel_target_queries": reg["data"]["m17_panel_target_queries"],
                             "panel_selection_fraction": reg["data"]["panel_selection_fraction"],
                             "panel_split_unit": reg["data"]["panel_split_unit"]},
        "sources_drawn": PANEL_SOURCES + ["k8s-docs-en"],
        "source_licences": {
            "squad-train": "CC BY-SA 4.0 (research/m7-data-licensing.md), training-approved",
            "hotpotqa-train": "CC BY-SA 4.0, training-approved",
            "mrtydi-en": "Apache 2.0, training-approved",
            "k8s-docs-en": "CC BY 4.0, attribution required (research/m17-k8s-attribution.md)"},
        "sources_excluded": EXCLUDED_SOURCES,
        "draw_rule": ("held-out only: m7src/trainmix.heldout (sha256(source:qid) %% 50 == 0), the "
                      "registered M7 rule, plus removal of any held-out query whose normalized "
                      "text also occurs as a training query in step 2b's family manifest"),
        "domain_label_method": DOMAIN_METHOD,
        "classifier_min_score": draft_rec["classifier_min_score"],
        "classifier_threshold_yield": draft_rec["classifier_threshold_yield"],
        "classifier_threshold_choice": (
            "Measured before choosing, over the 3,576 held-out candidates: thresholds 2/3/4/5/6 "
            "label {science, medicine, finance, legal} as {142,133,149,215} / {75,75,85,101} / "
            "{34,56,38,61} / {17,35,24,40} / {8,24,12,25}. 3 was chosen as the loosest threshold "
            "that still needs a strong term plus corroboration; 2 would have reached ~100 per "
            "domain on a single weak term pair and was rejected as a looser label, not as an "
            "unavailable one. The four small domains are therefore SHORT of the ~100 target and "
            "that shortfall is a recorded gap, not a reason to draw from an unapproved source or "
            "from the training slice."),
        "domain_availability_at_chosen_threshold": draft_rec["domain_availability"],
        "judgment_semantics": {
            "DATASET_QRELS": ("the dataset's own annotator judgments, carried over. They judge the "
                              "dataset's corpus, not this panel's mixed corpus: an unjudged "
                              "distractor from another source may also be relevant, which biases "
                              "measured nDCG@10 downward by an unknown amount, equally for every "
                              "system scored on the same corpus."),
            "PENDING_HUMAN": ("candidate documents only, from a lexical tf-idf neighbourhood. No "
                              "judgment exists yet. Teacher rankings and seed-document identity "
                              "are not judgments (m17/PLANNING.md).")},
        "n_queries": len(records),
        "n_families": len({r["family_id"] for r in records}),
        "per_domain": per_domain,
        "partitions": {
            "selection": {"queries": sum(1 for r in records if r["partition"] == "selection"),
                          "families": len({r["family_id"] for r in records
                                           if r["partition"] == "selection"}),
                          "file": str(SELECTION_JSONL.relative_to(REPO)),
                          "mirrored_to_results": mirrored,
                          "mirror": (str(SELECTION_MIRROR.relative_to(REPO)) if mirrored
                                     else None),
                          "bytes": sel_bytes,
                          "sha256": sha_file(SELECTION_JSONL)},
            "audit": {"queries": sum(1 for r in records if r["partition"] == "audit"),
                      "families": len({r["family_id"] for r in records
                                       if r["partition"] == "audit"}),
                      "file": str(AUDIT_JSONL.relative_to(REPO)),
                      "text_stays_in_work": True,
                      "sha256": sha_file(AUDIT_JSONL)}},
        "corpus": {**draft_rec["corpus"],
                   "file": str(CORPUS_JSONL.relative_to(REPO)),
                   "sha256": sha_file(CORPUS_JSONL),
                   "note": ("document ids only; text is resolved at evaluation time from "
                            "work/train/stores/* and work/m17/sources/k8s_docs_en.jsonl. Exact "
                            "dense retrieval runs over this full declared corpus.")},
        "pending_judgments": {
            "file": str(PENDING_OUT.relative_to(REPO)),
            "items": len(pending),
            "panel_candidate_rows": len(panel_pending),
            "alias_pair_rows": len(alias_pending),
            "queries": len({p["query_id"] for p in panel_pending}),
            "candidates_per_query": K8S_CANDIDATES_PER_QUERY,
            "format": ("one JSONL row per (query, candidate) for kind=panel-candidate and per "
                       "ambiguous pair for kind=alias-pair; fill relevant_yes_no and judge"),
            "sha256": sha_file(PENDING_OUT)},
        "exposure_labels": {
            "counts": dict(Counter(r["exposure"] for r in records)),
            "reasons": dict(Counter(r["exposure_reason"] for r in records)),
            "definitions": {
                "exposure-known-exposed": ("the query text matched an ancestor training manifest, "
                                           "or a training query shares its gold document group, "
                                           "or its gold documents sat in M7's document pool"),
                "exposure-known-clean": ("screened against every available ancestor manifest with "
                                         "no exact or near hit, and its documents are in no "
                                         "ancestor store"),
                "exposure-unknown": ("an ancestor manifest could not be streamed, so ancestry "
                                     "cannot be established (registry.data."
                                     "ancestor_decontamination)")},
            "screen": scr and {k: scr[k] for k in ("min_share", "n_candidates", "per_stream",
                                                   "interface")},
            "all_expected_streams_present": all_streams,
            "not_screened": ("the six, the reserved four and LoTTE. Deferred to the M17 executor "
                             "exactly as results/m17_k8s_source_manifest.json records; every "
                             "interface reaching them materializes protected payloads."),
            "teacher_pretraining": ("stella's own pretraining exposure is not screenable from this "
                                    "repository and is not claimed either way.")},
        "expected_standard_error": {
            "per_domain_selection": {d: per_domain[d]["expected_se"] for d in per_domain},
            "assumption": expected_se.__doc__.strip(),
            "registry_registered_range": reg["data"]["panel_expected_se_registered"]},
        "panel_jsonl": {"file": str(PANEL_JSONL.relative_to(REPO)),
                        "sha256": sha_file(PANEL_JSONL),
                        "bytes": PANEL_JSONL.stat().st_size},
        "reader_contract": (
            "each row: query_id, query, source, domain, slice, family_id, partition, qrels "
            "{doc_id: grade}, judgment_status, exposure. m17src/evaluate.py takes qrels, domains "
            "and families keyed by the same query key, which is query_id here."),
        "limitations": [
            "Domain labels are heuristic keyword labels, not adjudicated ones.",
            "Sealed is not unseen: exposure labels above say what was established.",
            "Dataset qrels are incomplete on a mixed corpus; unjudged relevant distractors bias "
            "every system's nDCG@10 downward together.",
            "At these family counts the panel cannot resolve the registry's screening bands; it "
            "is descriptive and audit evidence only (registry.data.panel_role).",
        ],
    }
    write_json(MANIFEST_OUT, manifest)
    log(f"[seal] {MANIFEST_OUT} written; panel sha256 {manifest['panel_jsonl']['sha256'][:16]}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("draft", "screen", "seal", "all"), default="all")
    ap.add_argument("--min-score", type=int, default=MIN_SCORE,
                    help="keyword-classifier threshold; the chosen value is recorded")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="re-label from work/m17/panel/*_cache.jsonl instead of re-reading stores")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    v = not args.quiet
    if args.stage in ("draft", "all"):
        draft(verbose=v, min_score=args.min_score, reuse_cache=args.reuse_cache)
    if args.stage in ("screen", "all"):
        run_screen(verbose=v)
    if args.stage in ("seal", "all"):
        seal(verbose=v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
