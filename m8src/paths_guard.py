"""LEDGER G2 -- the protected-path guard, allowlist form.

Why this exists as code rather than as a sentence in the ledger. M8's confirmatory evaluation is
ONE access to four hash-pinned datasets that have never been scored. The whole value of that
access is that no development decision was informed by it. A single stray read -- a probe that
loads "all of frozen_eval" to be helpful, a diagnostic that globs, a dev evaluation that includes
an extra component -- destroys it silently and unrecoverably. M7 lost the "exactly two accesses"
claim to exactly this class of accident (m7/LEDGER.md: three self-reported deviations, none of
them malicious, all of them a script reading a payload it did not need).

Design. Protected payloads may be opened only by a module that has explicitly CLAIMED an
allowlist entry naming the contact class that justifies it. The default for everything else --
every probe, every training path, every dev evaluation, every diagnostic -- is refusal at open
time, with a message that says which entry would have been needed.

The guard is deliberately claim-based rather than stack-introspection-based: a claim is a visible,
greppable, reviewable line in one entry-point module, whereas a stack walk silently changes
behaviour when a helper is refactored into a different file. The static test in
`m8src/test_guards.py` covers the other half -- no unclaimed file may even mention a protected
path -- so the two together bound both the runtime and the source.

Kinds, so an entry gets only what its contact class actually justifies:
  untouched_labels  the four reserved sets' query+qrel payloads. Opening one is the access.
  lotte             the shadow set. One crossing, ever (LEDGER 2.3).
  m9reserve         EUR-Lex / USPTO. Never scored in M8 (E4).
"""
import builtins
import io
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- what is protected -------------------------------------------------------------------
# Payloads only. The MANIFEST (results/eval_manifest.json) is metadata -- counts and hashes, no
# query text and no labels -- and is committed; reading it is not contact with the partition, and
# the pre-encode step needs it to verify the corpora it downloads.
PROTECTED = {
    "untouched_labels": (REPO / "results" / "frozen_eval", "untouched-*"),
    "lotte": (REPO / "work" / "lotte", "*"),
    "m9reserve": (REPO / "work" / "m9reserve", "*"),
}

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
               "(LEDGER G2 class b)",
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


class ProtectedPathRefusal(RuntimeError):
    """Raised instead of opening a protected payload. Never caught inside m8src."""


def claim(entry, note=""):
    """Claim an allowlist entry for this process. Call it once, at the top of the entry point,
    where a reviewer will see it."""
    global _claim
    if entry not in ALLOWLIST:
        raise ProtectedPathRefusal(
            f"{entry!r} is not on the LEDGER G2 allowlist. Adding it is a LEDGER 15 amendment, "
            f"not an edit. Known entries: {sorted(ALLOWLIST)}")
    if _claim is not None and _claim != entry:
        raise ProtectedPathRefusal(
            f"this process already claimed {_claim!r}; one entry point per process (m7/CODEMAP.md "
            f"pitfall 14 -- one process per job -- applies to protected access too)")
    _claim = entry
    return {"entry": entry, "kinds": sorted(ALLOWLIST[entry]["kinds"]),
            "why": ALLOWLIST[entry]["why"], "note": note}


# Cheap pre-filter: the guard sits on EVERY open in the process (HF datasets opens a great many
# files), so the common case must not cost a resolve() syscall. A path can only be protected if
# one of these substrings appears in it.
_HINTS = ("frozen_eval", "lotte", "m9reserve")


def classify(path):
    """-> the protected kind this path belongs to, or None."""
    try:
        s = os.fspath(path)
    except TypeError:
        return None                      # an int fd or a file object; not a path
    if not isinstance(s, (str, bytes)):
        return None
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    if not any(h in s for h in _HINTS):
        return None
    try:
        p = Path(s).resolve()
    except (ValueError, OSError):
        return None
    if p.parent == PROTECTED["untouched_labels"][0] and p.name.startswith("untouched-"):
        return "untouched_labels"
    for kind in ("lotte", "m9reserve"):
        root = PROTECTED[kind][0]
        if p == root or root in p.parents:
            return kind
    return None


def check(path):
    """Refuse unless the claimed entry covers this path's kind. Returns the kind, or None."""
    kind = classify(path)
    if kind is None:
        return None
    if _claim is None:
        raise ProtectedPathRefusal(
            f"LEDGER G2: refusing to open {path} (protected kind {kind!r}). No allowlist entry is "
            f"claimed in this process. If this access is legitimate, the entry point must call "
            f"paths_guard.claim(<entry>) and that entry must be on the ledger's allowlist.")
    if kind not in ALLOWLIST[_claim]["kinds"]:
        raise ProtectedPathRefusal(
            f"LEDGER G2: {_claim!r} may open {sorted(ALLOWLIST[_claim]['kinds'])} but not "
            f"{kind!r} ({path}). Entry's justification: {ALLOWLIST[_claim]['why']}")
    return kind


def install():
    """Patch the open paths. Idempotent; safe to call from every module's import."""
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


def uninstall():
    """Only for the guard's own tests."""
    global _installed, _claim
    if not _installed:
        return
    builtins.open = _real_open
    os.open = _real_os_open
    io.open = _real_open
    _installed = False
    _claim = None


def status():
    return {"installed": _installed, "claim": _claim,
            "protected": {k: str(v[0]) for k, v in PROTECTED.items()},
            "allowlist": {k: sorted(v["kinds"]) for k, v in ALLOWLIST.items()}}


if __name__ == "__main__":
    import json
    install()
    print(json.dumps(status(), indent=2))
