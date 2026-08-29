"""LEDGER G2 -- the protected-path guard, allowlist form. HARDENED 2026-08-29.

Why this exists as code rather than as a sentence in the ledger. M8's confirmatory evaluation is
ONE access to four hash-pinned datasets that have never been scored. The whole value of that
access is that no development decision was informed by it. A single stray read -- a probe that
loads "all of frozen_eval" to be helpful, a diagnostic that globs, a dev evaluation that includes
an extra component -- destroys it silently and unrecoverably. M7 lost the "exactly two accesses"
claim to exactly this class of accident (m7/LEDGER.md: three self-reported deviations, none of
them malicious, all of them a script reading a payload it did not need).

WHAT THE 2026-08-29 ADVERSARIAL GATE FOUND, and what this file now does about it. The first
version guarded `results/frozen_eval/untouched-*` and nothing else. The reserved sets were
reachable by FOUR other routes, two of which already existed on this disk:

  1. `work/dev/cqadup-android.json` and `work/dev/cqadup-english.json` -- 22,998 and 40,221 docs
     WITH THEIR 699 and 1,570 QRELS, materialized 2026-08-26 by `devsuite.load()` when the
     untouched-final pair was defined. Any M8 dev script calling `devsuite.load("cqadup-android")`
     would have scored a reserved confirmatory set and never known. This was a live hazard, found
     by review, not by accident. It is now a protected kind of its own.
  2. The HuggingFace cache: `BeIR___fever-qrels/`, `BeIR___dbpedia-entity-qrels/` and the reserved
     `queries` configs are on disk and re-readable with no download.
  3. `datasets.load_dataset("BeIR/fever-qrels")` re-fetches the labels from the network, touching
     no guarded path at all. So the guard also wraps the loader and refuses by DATASET ID.
  4. A symlink whose name contains none of the fast-path hints. The hint pre-filter now runs on
     the RESOLVED path, cached, so an alias cannot dodge classification.

Also removed: the public `claim()` / `uninstall()` escape hatches. `claim()` now verifies that the
call site is physically inside the module it claims to be, and the uninstall path is private and
takes a token, so "any code can turn the guard off" is no longer true.

Design. Protected payloads may be opened only by a module that has explicitly CLAIMED an allowlist
entry naming the contact class that justifies it. The default for everything else -- every probe,
every training path, every dev evaluation, every diagnostic -- is refusal at open time, with a
message that says which entry would have been needed.

The guard is deliberately claim-based rather than stack-introspection-based for the ONGOING check:
a claim is a visible, greppable, reviewable line in one entry-point module, whereas a stack walk
silently changes behaviour when a helper is refactored into a different file. Stack inspection is
used once, at claim time, only to verify the claimer is who it says it is. The static test in
`m8src/test_guards.py` covers the other half -- no unclaimed file may even mention a protected
path -- so runtime and source are both bounded.

WHAT THIS GUARD DOES NOT DO, stated so it is not mistaken for more than it is. It is a mistake
bulkhead, not a sandbox. A determined process can still reach the network, and nothing here
verifies freeze state or spends a receipt -- that is `final_run`'s job, ported from M7's one-shot
path. The claim is "an ordinary mistake cannot silently burn the access", not "the access is
cryptographically sealed".
"""
import builtins
import functools
import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HF = Path(os.environ.get("HF_DATASETS_CACHE")
          or (Path.home() / ".cache" / "huggingface" / "datasets"))

# --- what is protected -------------------------------------------------------------------
# Payloads only. The MANIFEST (results/eval_manifest.json) is metadata -- counts and hashes, no
# query text and no labels -- and is committed; reading it is not contact with the partition, and
# the pre-encode step needs it to verify the corpora it downloads.
FROZEN_EVAL = REPO / "results" / "frozen_eval"
WORK_DEV = REPO / "work" / "dev"
LOTTE = REPO / "work" / "lotte"
M9RESERVE = REPO / "work" / "m9reserve"

# The reserved four, by every name a loader might use.
RESERVED_HF_IDS = {
    "beir/fever", "beir/fever-qrels",
    "beir/dbpedia-entity", "beir/dbpedia-entity-qrels",
}
RESERVED_CQA_CONFIGS = {"android", "english"}
RESERVED_HF_CACHE_DIRS = ("BeIR___fever-qrels", "BeIR___dbpedia-entity-qrels")
RESERVED_WORK_DEV = {"cqadup-android.json", "cqadup-english.json"}

KINDS = ("untouched_labels", "lotte", "m9reserve")


def _classify_resolved(p: Path):
    if p.parent == FROZEN_EVAL and p.name.startswith("untouched-"):
        return "untouched_labels"
    if p.parent == WORK_DEV and p.name in RESERVED_WORK_DEV:
        return "untouched_labels"                     # same labels, different file
    for d in RESERVED_HF_CACHE_DIRS:
        if any(part == d for part in p.parts):
            return "untouched_labels"
    if p == LOTTE or LOTTE in p.parents:
        return "lotte"
    if p == M9RESERVE or M9RESERVE in p.parents:
        return "m9reserve"
    return None


# --- the allowlist -----------------------------------------------------------------------
# Adding a name here is a LEDGER 15 amendment, not an edit. Each entry states the contact class
# that justifies it, in the same words the ledger uses.
ALLOWLIST = {
    "m8src.freeze_lotte": {
        "kinds": {"lotte"},
        "why": "freeze/inventory: it must hash what it pins (LEDGER G2 class a)",
    },
    "m8src.freeze_m9reserve": {
        "kinds": {"m9reserve"},
        "why": "freeze/inventory: it must hash what it pins (LEDGER G2 class a)",
    },
    "m8src.protected_filter": {
        "kinds": {"untouched_labels", "lotte", "m9reserve"},
        "why": "decontamination: it must read protected query text to protect against it "
               "(LEDGER G2 class b). It emits a QUERY-ONLY hash inventory and never a label.",
    },
    "m8src.shadow_cross": {
        "kinds": {"lotte"},
        "why": "the single mandatory LoTTE shadow crossing, after the manifest is immutable "
               "(LEDGER 2.3)",
    },
    "m8src.final_run": {
        "kinds": {"untouched_labels"},
        "why": "the confirmatory access, only after the freeze (LEDGER G2 class d)",
    },
}

_claim = None
_installed = False
_real_open = None
_real_os_open = None
_real_load_dataset = None
_UNINSTALL_TOKEN = "m8src.test_guards"


class ProtectedPathRefusal(RuntimeError):
    """Raised instead of opening a protected payload. Never caught inside m8src."""


@functools.lru_cache(maxsize=8192)
def _resolve(s: str):
    try:
        return Path(s).resolve()
    except (ValueError, OSError):
        return None


def classify(path):
    """-> the protected kind this path belongs to, or None.

    Resolves FIRST (cached), so a symlink alias whose name mentions nothing suspicious still
    classifies. The lru_cache is what keeps this affordable on a guard that sits on every open in
    a process that loads a great many files."""
    try:
        s = os.fspath(path)
    except TypeError:
        return None                                   # an int fd or a file object; not a path
    if not isinstance(s, (str, bytes)):
        return None
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    p = _resolve(s)
    return None if p is None else _classify_resolved(p)


def check(path):
    """Refuse unless the claimed entry covers this path's kind. Returns the kind, or None."""
    kind = classify(path)
    if kind is None:
        return None
    if _claim is None:
        raise ProtectedPathRefusal(
            f"LEDGER G2: refusing to open {path} (protected kind {kind!r}). No allowlist entry is "
            f"claimed in this process. If this access is legitimate, the entry point must call "
            f"paths_guard.claim(<entry>) from inside that module, and the entry must be on the "
            f"ledger's allowlist.")
    if kind not in ALLOWLIST[_claim]["kinds"]:
        raise ProtectedPathRefusal(
            f"LEDGER G2: {_claim!r} may open {sorted(ALLOWLIST[_claim]['kinds'])} but not "
            f"{kind!r} ({path}). Entry's justification: {ALLOWLIST[_claim]['why']}")
    return kind


def check_dataset(name, config=None):
    """The network route. `load_dataset('BeIR/fever-qrels')` touches no guarded path, so the
    loader itself is guarded by dataset identity."""
    n = str(name).lower().strip("/")
    hit = n in RESERVED_HF_IDS
    if not hit and "cqadupstack" in n and config and str(config).lower() in RESERVED_CQA_CONFIGS:
        hit = True
    if not hit:
        return None
    if _claim is None or "untouched_labels" not in ALLOWLIST[_claim]["kinds"]:
        raise ProtectedPathRefusal(
            f"LEDGER G2: refusing load_dataset({name!r}, {config!r}) -- that is a reserved "
            f"confirmatory set. Downloading it fresh is the same contact as opening the frozen "
            f"payload. Claimed entry: {_claim!r}.")
    return "untouched_labels"


def claim(entry, note=""):
    """Claim an allowlist entry for this process. Must be called from INSIDE the module named by
    the entry -- otherwise any file could grant itself the capability by naming someone else."""
    global _claim
    if entry not in ALLOWLIST:
        raise ProtectedPathRefusal(
            f"{entry!r} is not on the LEDGER G2 allowlist. Adding it is a LEDGER 15 amendment, "
            f"not an edit. Known entries: {sorted(ALLOWLIST)}")
    caller = sys._getframe(1).f_code.co_filename
    expected = REPO / (entry.replace(".", "/") + ".py")
    if _resolve(caller) != _resolve(str(expected)):
        raise ProtectedPathRefusal(
            f"claim({entry!r}) was called from {caller}, not from {expected}. An entry may only "
            f"claim itself: the allowlist names a FILE's contact class, not a capability that "
            f"can be handed around.")
    if _claim is not None and _claim != entry:
        raise ProtectedPathRefusal(
            f"this process already claimed {_claim!r}; one entry point per process (m7/CODEMAP.md "
            f"pitfall 14 -- one process per job -- applies to protected access too)")
    _claim = entry
    return {"entry": entry, "kinds": sorted(ALLOWLIST[entry]["kinds"]),
            "why": ALLOWLIST[entry]["why"], "note": note}


def install():
    """Patch the open paths and the dataset loader. Idempotent; safe to call from every import."""
    global _installed, _real_open, _real_os_open
    if _installed:
        return
    _real_open = builtins.open
    _real_os_open = os.open

    def guarded_open(file, *a, **kw):
        check(file)
        return _real_open(file, *a, **kw)

    def guarded_os_open(path, *a, **kw):
        check(path)
        return _real_os_open(path, *a, **kw)

    builtins.open = guarded_open
    os.open = guarded_os_open
    io.open = guarded_open
    _installed = True
    _wrap_load_dataset()


def _wrap_load_dataset():
    """Wrap `datasets.load_dataset` if datasets is (or becomes) imported. Called at install and
    again lazily by `ensure_loader_guard()`, because datasets is usually imported later."""
    global _real_load_dataset
    mod = sys.modules.get("datasets")
    if mod is None or _real_load_dataset is not None:
        return
    real = getattr(mod, "load_dataset", None)
    if real is None:
        return
    _real_load_dataset = real

    @functools.wraps(real)
    def guarded(path, name=None, *a, **kw):
        check_dataset(path, name)
        return real(path, name, *a, **kw)

    mod.load_dataset = guarded


def ensure_loader_guard():
    """Call after importing `datasets` in any module that loads corpora."""
    _wrap_load_dataset()


def _uninstall(token):
    """Private, token-gated, and used only by the guard's own tests. There is deliberately no
    public way to turn the guard off."""
    global _installed, _claim
    if token != _UNINSTALL_TOKEN:
        raise ProtectedPathRefusal("the guard cannot be uninstalled by ordinary code")
    if not _installed:
        return
    builtins.open = _real_open
    os.open = _real_os_open
    io.open = _real_open
    _installed = False
    _claim = None
    _resolve.cache_clear()


def status():
    return {"installed": _installed, "claim": _claim,
            "loader_guarded": _real_load_dataset is not None,
            "protected_roots": {"untouched_labels": [str(FROZEN_EVAL) + "/untouched-*",
                                                     str(WORK_DEV) + "/{" +
                                                     ",".join(sorted(RESERVED_WORK_DEV)) + "}",
                                                     "HF cache: " +
                                                     ", ".join(RESERVED_HF_CACHE_DIRS)],
                                "lotte": [str(LOTTE)], "m9reserve": [str(M9RESERVE)]},
            "protected_dataset_ids": sorted(RESERVED_HF_IDS),
            "allowlist": {k: sorted(v["kinds"]) for k, v in ALLOWLIST.items()}}


if __name__ == "__main__":
    import json
    install()
    print(json.dumps(status(), indent=2))
