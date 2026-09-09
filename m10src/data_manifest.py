"""`results/m10_data_manifest.json` — the provenance the recipe lock points at (§M10.1).

The mandate owes one file carrying "hashes and the provenance table", plus the decontamination
removals, the FORMS-12 hold-out, the teacher-target cache keys and the A8 gates
(`instructions-m10.md`:342, `m10/LEDGER.md` §0b). Every one of those facts is already MEASURED and
committed in its own artifact; what did not exist is a single file that names them, hashes the
DATA ITSELF, and can be cited by one line of the lock.

So this aggregates rather than re-measures. It does two things nothing else does:

1. **Hashes the corpus files on disk**, not the summaries about them. A provenance table that
   hashes only its own summary proves nothing about the bytes the arms trained on — the same hole
   the contrast step had until `check_against_arm_record` compared the COV file to the arm record.
2. **Refuses to emit a manifest with a hole in it.** A missing input is recorded as a refusal, not
   as an absent key: a provenance file whose gaps are invisible is worse than none, because the
   lock cites it as if it were complete.

Large files are hashed once and cached in the manifest by (path, size, mtime); `--rehash` forces.
The corpus is ~4.3 GB, so a full hash is minutes, not hours, and it is the point of the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "m10_data_manifest.json"

# The data the arms actually train on, by role. Paths are repo-relative and gitignored (they are
# tens of GB); the manifest is what makes them auditable from the repo.
CORPUS = {
    "harvest_train": "work/m10harvest/harvest_train.jsonl",
    "harvest_forms12_holdout": "work/m10harvest/harvest_forms12.jsonl",
    "harvest_drawn": "work/m10harvest/harvest_drawn.jsonl",
    # the RAW pools the draw came from. 21,087,043 rows harvested -> 1,250,000 drawn, so without
    # these the manifest can prove what was USED but not what it was drawn FROM, which is half of
    # what provenance means. ~3.3 GB, hashed once and cached.
    "harvest_pool_wikipedia": "work/m10harvest/wiki_harvest.jsonl",
    "harvest_pool_arxiv": "work/m10harvest/arxiv_harvest.jsonl",
    "harvest_pool_licensed": "work/m10harvest/pool_harvest.jsonl",
    "generated_queries": "work/m10gen/generated_queries.jsonl",
    "generated_seeds_wikibody": "work/m10gen/wikibody_seeds.jsonl",
}
GENERATED_PER_FORM = "work/m10gen/generated_{}.jsonl"
GEN_FORMS = ("argument", "comparison", "conversational", "finance", "health", "howto", "yesno")

# The measurements the manifest CITES. Each is already committed; the manifest hashes it so the
# lock inherits a single sha256 instead of eleven.
CITED = {
    "corpus_manifest": "m10_corpus_manifest.json",
    "assembly": "m10_assemble10.json",
    "data_cut": "m10_data_cut.json",
    "teacher_targets": "m10_targets10.json",
    "decontamination_rescreen": "m10_rescreen10.json",
    "a8_gate1_diversity": "m10_diversity_pilot.json",
    "a8_gate2_msmarco_overlap": "m10_a8_gate2.json",
    "a8_blindspot": "m10_a8_blindspot.json",
    "qfilter_effect": "m10_qfilter_effect.json",
    "exposure_table": "m10_exposure_table.json",
    "seed_supply": "m10_seed_supply.json",
    "route_precision": "m10_route_precision.json",
}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _entry(rel, cache, rehash=False):
    p = REPO / rel
    if not p.exists():
        return {"path": rel, "MISSING": True}
    st = p.stat()
    key = f"{rel}:{st.st_size}:{int(st.st_mtime)}"
    if not rehash and cache.get(key):
        return {"path": rel, "bytes": st.st_size, "sha256": cache[key], "_hash": "cached"}
    return {"path": rel, "bytes": st.st_size, "sha256": sha256_file(p), "_hash": "computed",
            "_key": key}


def build(rehash=False, verbose=True):
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    cache = {v.get("_key"): v.get("sha256") for sect in ("corpus", "generated_by_form")
             for v in (prev.get(sect) or {}).values() if isinstance(v, dict) and v.get("_key")}

    man = {
        "_what": "M10 data provenance (§M10.1). Hashes the corpus the arms trained on, and cites "
                 "every already-committed measurement about it. Built by `m10src/data_manifest.py`; "
                 "the recipe lock cites THIS file's sha256 rather than a dozen others.",
        "_not_a_measurement": "Nothing here is measured for the first time. If a number appears "
                              "both here and in a cited artifact, the artifact is authoritative.",
    }
    man["corpus"] = {k: _entry(v, cache, rehash) for k, v in CORPUS.items()}
    man["generated_by_form"] = {f: _entry(GENERATED_PER_FORM.format(f), cache, rehash)
                                for f in GEN_FORMS}
    man["cited_measurements"] = {}
    for name, fn in CITED.items():
        p = RESULTS / fn
        man["cited_measurements"][name] = ({"file": f"results/{fn}", "sha256": sha256_file(p)}
                                           if p.exists() else {"file": f"results/{fn}",
                                                               "MISSING": True})

    # -- the provenance table, read out of the cited artifacts rather than restated by hand
    cut = json.loads((RESULTS / "m10_data_cut.json").read_text())
    asm = json.loads((RESULTS / "m10_assemble10.json").read_text())
    resc = json.loads((RESULTS / "m10_rescreen10.json").read_text())
    reg = json.loads((REPO / "m10" / "screen_registry.json").read_text())
    man["provenance"] = {
        "sources": {
            "m9-pool": "the M9 training pool, re-screened against the M10 protected index",
            "paq-a2": "PAQ, 4,037,000 rows (screen arm A2, the volume control)",
            "paq-build": "PAQ, 1,000,000 rows, NESTED INSIDE the A2 sample",
            "harvest": "real titles / headings / claim sentences mined from Wikipedia, arXiv and "
                       "the licensed pool",
            "generated": "the seven forms no corpus contains, generated under the §Data contract",
        },
        "licences": "CC BY-SA sources (NQ, SQuAD, HotpotQA, FEVER, MIRACL, Mr.TyDi) carry model-card "
                    "attribution; PAQ is CC BY-SA 3.0 shipped in its tarball. MS MARCO is EXCLUDED "
                    "from training in every role and appears only as a validation surface "
                    "(`research/m7-data-licensing.md` §Rule change 2026-09-04). FineWeb is out of "
                    "M10 in every role.",
        "data_cut": cut.get("data_cut"),
        "unique_text_count": reg["data_cut"]["unique_text_count"],
        "generated_rows_realized": asm.get("total_rows"),
        "generated_by_form": asm.get("by_form"),
        "forms12_holdout": "by DOCUMENT across all forms: 1,500 documents held out -> 1,614 eval "
                           "rows and 1,248,386 train rows (`work/m10harvest/harvest_{train,"
                           "forms12}.jsonl`). Descriptive only; `evaluation.FORMS-12`.",
        "decontamination": {
            "_what": "every training source screened against the M10 protected index, which "
                     "includes the six's documents and the COV components",
            "artifact": "results/m10_rescreen10.json",
            "m9_pool_rescreen": {k: resc[k] for k in resc if k not in ("_what",)},
        },
        "teacher_targets": "stella_en_400M_v5, frozen; cache identity in results/m10_targets10.json",
    }

    # -- the two disclosures the lock owes about what was NOT done
    man["disclosures"] = {
        "own_source_5gram_screen": "NOT RUN. W11's second half: a word-5-gram screen of the harvest "
            "corpus against ITS OWN sources. It is near-vacuous by construction -- the harvest is "
            "titles, headings and lead claim sentences COPIED from those documents, so it would "
            "report near-total overlap with the corpus the text came from, which is what harvesting "
            "means. The screen that matters is against the PROTECTED index (the six's documents and "
            "the COV components), and that one RAN: `results/m10_rescreen10.json`. Disclosed rather "
            "than run, per the registered option.",
        "cure_v1": "Decision 12 (CUREv1 as a validation-only diagnostic) was ADOPTED 2026-09-04 and "
            "NEVER EXECUTED. Its own precondition is unmet: the harvest, PAQ and seed draws were "
            "screened against an index that did not contain it. If it is ever read, either "
            "re-screen first or disclose the gap beside the number. Never selection-bearing.",
        "generated_count": "the plan said ~1.0M generated queries; 834,463 were realized and used "
            "(`results/m10_assemble10.json`). Every registry sentence has been corrected to the "
            "realized figure.",
    }

    holes = [f"corpus/{k}" for k, v in man["corpus"].items() if v.get("MISSING")]
    holes += [f"generated_by_form/{k}" for k, v in man["generated_by_form"].items()
              if v.get("MISSING")]
    holes += [f"cited/{k}" for k, v in man["cited_measurements"].items() if v.get("MISSING")]
    man["complete"] = not holes
    man["holes"] = holes
    OUT.write_text(json.dumps(man, indent=1) + "\n")
    if verbose:
        n = len(man["corpus"]) + len(man["generated_by_form"])
        gb = sum(v.get("bytes", 0) for v in list(man["corpus"].values())
                 + list(man["generated_by_form"].values())) / 1e9
        print(f"{OUT.relative_to(REPO)}: {n} data files ({gb:.2f} GB), "
              f"{len(man['cited_measurements'])} cited measurements")
        print("  COMPLETE" if man["complete"] else f"  HOLES: {holes}")
    if holes:
        raise SystemExit(f"REFUSED to emit a complete-looking manifest: {holes}. A provenance file "
                         f"whose gaps are invisible is worse than none, because the lock cites it "
                         f"as if it were whole.")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehash", action="store_true", help="ignore the cached hashes")
    build(rehash=ap.parse_args().rehash)
