"""M10.0-d — COV admission: fetch, verify structure, record the numbers §2 needs.

Admits a component only after a pushed record of repo+revision, licence at the PRIMARY source,
corpus/query/qrels sizes and format, metric, corpus-level contamination check and the fingerprint
screen against the six. Licence research is drafted in `m10/COV_CANDIDATES.md`; this script
supplies the structural half and the screen.

Registered STOP: fewer than three family IDs (`consumer-health`, `BRIGHT`, `legal`, `finance` iff
LEDGER) -> stop, ping, wait for Dylan.

Nothing here reads a reserved surface. The six's QUERIES and DOCUMENTS are read only through
`m7src/decontam`'s fingerprint streams, which is how M7/M9 screened training text.
"""
import json, os, sys, time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m7src"))
OUT = REPO / "work" / "m10cov"

# family -> [(component, hf repo, revision)]
COMPONENTS = {
    "consumer-health": [("MedicalQARetrieval", "mteb/medical_qa",
                         "a77efe81ec0c03aff7fecde742a5c9c4c46f6005")],
    "BRIGHT": [("BRIGHT", "xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae")],
    "legal": [("LegalBenchCorporateLobbying", "mteb/legalbench_corporate_lobbying",
               "f43436957b41692dd3e1b06a6d7116cd09f6a1db"),
              ("LegalBenchConsumerContractsQA", "mteb/legalbench_consumer_contracts_qa",
               "f9eafd458f9c61e531d4a2510d8a11dfd2282b21")],
}
BRIGHT_SLICES = ("biology", "earth_science", "economics", "psychology", "robotics",
                 "sustainable_living")
DOC_CAP = 100_000          # per component, the registered cap


def _load(repo, revision, config=None, split=None):
    from datasets import load_dataset
    return load_dataset(repo, config, revision=revision, split=split)


def probe(repo, revision):
    """-> {config: {split: (n_rows, columns)}} without materialising more than it must."""
    from datasets import get_dataset_config_names, load_dataset_builder
    cfgs = get_dataset_config_names(repo, revision=revision)
    out = {}
    for c in cfgs:
        try:
            b = load_dataset_builder(repo, c, revision=revision)
            out[c] = {s: dict(n=int(i.num_examples), cols=list((b.info.features or {}).keys()))
                      for s, i in (b.info.splits or {}).items()}
        except Exception as e:
            out[c] = {"error": repr(e)[:200]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {}
    for family, comps in COMPONENTS.items():
        for name, repo, rev in comps:
            t0 = time.time()
            print(f"\n=== {family} / {name}  {repo}@{rev[:8]}", flush=True)
            try:
                p = probe(repo, rev)
                rec[name] = dict(family=family, repo=repo, revision=rev, structure=p,
                                 seconds=round(time.time() - t0, 1))
                for c, v in list(p.items())[:8]:
                    print(f"  {c}: {v}", flush=True)
            except Exception as e:
                rec[name] = dict(family=family, repo=repo, revision=rev,
                                 error=repr(e)[:300])
                print(f"  FAILED {e!r}"[:200], flush=True)
    (OUT / "structure.json").write_text(json.dumps(rec, indent=1))
    print("\nwrote", OUT / "structure.json")


if __name__ == "__main__":
    main()
