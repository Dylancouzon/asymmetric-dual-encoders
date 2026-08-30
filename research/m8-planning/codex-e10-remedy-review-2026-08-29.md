# Codex adversarial review — E10 remediation, 2026-08-29

gpt-5.6-sol, read-only, effort high. Verdict: **STOP / reopen E10** — the seven-slice
artifact is not decontaminated and its zero re-screen is tautological. Disposition in
`m8/LEDGER.md` §15, entry dated 2026-08-29 (E10 REOPENED).

I found two remediated LoTTE queries whose titles are verbatim prefixes of protected `cqadup-physics` corpus documents:

- `science/dev` qid `1147`, “How to design a house to be cooled passively?” remains in the [remediated questions](/home/dylan/asymetric-dual-encoders/work/lotte/remediated/science/dev/questions.forum.tsv:1139) and matches protected physics document `111653`.
- `recreation/test` qid `355`, “What is the terminal velocity of a sheep?” remains in the [remediated questions](/home/dylan/asymetric-dual-encoders/work/lotte/remediated/recreation/test/questions.forum.tsv:356) and matches protected physics document `129267`. The protected document explicitly says it was inspired by the Gaming.SE question—almost exactly the quoted-across-sites case you feared.

A targeted audit found **36 retained shadow queries sharing an 8-word run with protected CQADupStack corpus documents**. None matched the protected CQADupStack test-query list. Under your registered query rule, those are hits; the code simply never makes that comparison.

## Findings

### BLOCKER — The screen compares roles, not protected content

[`_cqa_index()`](/home/dylan/asymetric-dual-encoders/m8src/protected_filter.py:135) contains only the four CQADupStack corpora. LoTTE documents are checked against it, while LoTTE queries are checked only against the protected **query** inventory at [`_screen()`](/home/dylan/asymetric-dual-encoders/m8src/protected_filter.py:402).

That misses:

- LoTTE query ↔ protected corpus document, including the two confirmed cases above.
- LoTTE document ↔ protected query.
- LoTTE documents ↔ FEVER, DBpedia, the six corpora, NQ and HotpotQA. Despite the stated policy, the document index contains only 133,711 CQADupStack documents, not the full protected document universe.

Smallest correct repair: screen each shadow query against the union of protected queries and documents, and each shadow document against the union of protected documents and queries. Include every protected family named by policy. Reopen E10; the existing counts and artifact cannot be grandfathered.

### BLOCKER — The re-screen cannot fail on ordinary input

The first screen creates `dup_idx` and `leaked_qid_kind`; remediation then constructs their exact complement at [lines 456–474](/home/dylan/asymetric-dual-encoders/m8src/protected_filter.py:456). The second pass applies the same deterministic `_screen()` to those unchanged in-memory survivors at [lines 490–495](/home/dylan/asymetric-dual-encoders/m8src/protected_filter.py:490).

There is no ordinary data input that can produce a residual hit. An exact protected canary produces:

1. A first-pass hit.
2. Removal.
3. A guaranteed zero on the same detector.

Worse, files are written first, but the re-screen examines the in-memory lists, not the serialized files. A malformed, altered, or recontaminated output file would not be checked.

Smallest correct repair:

- Reopen and parse the written files before acceptance.
- Add a canary test that deliberately leaves or reinserts a known leak and proves acceptance fails.
- Use an independent acceptance detector: StackExchange IDs/URLs and migration/cross-site links, character shingles/edit similarity, and manually adjudicated semantic candidates. The current detector can remain the removal mechanism, but cannot certify itself.

This is exactly CODEMAP pitfalls 17 and 19.

### MAJOR — The 100× asymmetry is both real and diagnostic of a weaker document screen

The rates are not comparable:

- Query detection fires on **one** shared 8-gram, or one 4-gram for short queries, at [`query_hits()`](/home/dylan/asymetric-dual-encoders/m7src/decontam.py:278).
- Document detection requires **eight** shared entries from bottom-32 sketches: [`DUP_SHARE = 8`](/home/dylan/asymetric-dual-encoders/m7src/decontam.py:38), [`sketch()`](/home/dylan/asymetric-dual-encoders/m7src/decontam.py:157), and [`match()`](/home/dylan/asymetric-dual-encoders/m7src/decontam.py:187).
- A non-identical document under 15 normalized words has fewer than eight 8-grams and therefore cannot ever be a near hit.
- I reproduced a one-word edit of a 16-word document and a 14-word verbatim quotation; both return no document hit.
- LoTTE documents are answer passages, while CQADupStack documents are questions. Queries are question-versus-question. That creates a genuine modality mechanism for lower document overlap.
- The query inventory is much broader than the document inventory.

Normalization is only lowercase plus Unicode `isalnum()` splitting at [`norm_words()`](/home/dylan/asymetric-dual-encoders/m7src/decontam.py:55): no Unicode normalization, stemming, typo tolerance, synonymy, or reordering. Re-asked or edited titles readily evade it. The short-query logic is also directional: a protected short query embedded in a longer candidate is detected, but a short candidate extracted from a longer protected question is not.

Therefore `recreation/test = 0` is evidence about this detector, not evidence of no document overlap.

Smallest repair: length-adaptive character/word shingles with a containment score as well as Jaccard, plus exact source-ID/link checks. Freeze thresholds on adversarial fixtures before rerunning.

### MAJOR — Community equality is complete for these files, but community equivalence is not

The actual metadata is structurally sound: every survivor metadata row has a dataset field, and all 2,715,290 collection document IDs map through metadata. The `programmers` → `softwareengineering` rename is covered.

But exact hostname equality does not catch migration, quotation, or re-asking:

- The two exact ELL queries matching protected English queries show that `ell.stackexchange.com` and `english.stackexchange.com` are not independent content populations.
- The retained Gaming.SE sheep query is quoted by the protected Physics.SE document.
- The retained Engineering.SE passive-house query appears as a protected Physics.SE document.

Also, [`lotte_communities()`](/home/dylan/asymetric-dual-encoders/m8src/protected_filter.py:97) fails open on missing metadata and silently ignores malformed/missing rows.

Smallest repair: require complete metadata coverage and add explicit cross-site/migration/content linkage. Treat ELL/English and highly migratory technical-site clusters as elevated-risk, not “clean community” by hostname alone.

### MINOR — Two of the three “dead slices cannot be rescued” claims are false

Dropping them is conservative and defensible, but impossibility is overstated:

- `science/test` contains biology, math, and physics.
- `technology/test` contains six sites, only Android and Software Engineering being protected.
- Only `writing/test` is entirely the protected English community.

A site-level filter could technically retain the non-overlapping communities. Doing so now would require a newly registered construction; I would not reopen them merely to gain size. Repair the rationale to “conservatively dropped under the registered slice-level rule.”

### BLOCKER — The pin does not enforce what was screened

There is currently no `results/m8_lotte_pin.json`; the project correctly records that pinning is unfinished in [RESULTS.md](/home/dylan/asymetric-dual-encoders/m8/RESULTS.md:206).

Even when run, [`pin()`](/home/dylan/asymetric-dual-encoders/m8src/freeze_lotte.py:118):

- Trusts the mutable remedy artifact’s slice names.
- Permits an arbitrary subset of survivors.
- Hashes whatever bytes happen to occupy the paths at pin time.
- Does not bind those hashes to the content actually screened.
- Does not validate uniqueness, qid-set equality, or complete qrel integrity.

It does hash parsed content rather than merely a filename, but it can bless content modified between screen and pin. The entire raw and remediated LoTTE tree is one guard kind, so a later allowlisted caller can read a dead/raw slice without consulting the pin.

Smallest repair: screen the re-read serialized bytes and record their hashes in the remedy artifact; require `pin()` to match those hashes and the exact registered seven-slice set. The sole shadow loader must accept only pin-listed paths and verify hashes before serving.

### BLOCKER — The protected-path guard has live uncovered routes

I verified that the guard classifies all of these existing cache routes as unprotected:

- `BeIR___fever`
- `BeIR___dbpedia-entity`
- `mteb___cqadupstack-android`
- `mteb___cqadupstack-english`

Likewise, `check_dataset("mteb/cqadupstack-android", "queries")` and the English equivalent return allowed. The code recognizes only the different `BeIR/cqadupstack` + `android` spelling at [`check_dataset()`](/home/dylan/asymetric-dual-encoders/m8src/paths_guard.py:193).

There are two architectural holes:

- If `datasets` is imported after guard installation, it is not wrapped unless the caller remembers [`ensure_loader_guard()`](/home/dylan/asymetric-dual-encoders/m8src/paths_guard.py:285).
- Importing an allowlisted module such as `protected_filter` sets a process-global claim, after which unrelated caller code in that process inherits its access.

Smallest correct repair: do not rely on Python monkeypatching for the one-shot boundary. Put protected labels/caches outside the development account’s readable filesystem, disable network access in development processes, and grant access only to isolated final/remedy subprocesses. Then fix the cache and MTEB identifiers as defense in depth. This walks directly back into CODEMAP pitfall 2.

### MAJOR — The post-remedy filter workflow still excludes LoTTE

[`build_fitlist()`](/home/dylan/asymetric-dual-encoders/m8src/protected_filter.py:711) reads `m8_lotte_overlap.json["kept"]`. That value is empty because S0 rejected all ten slices. It never reads the remedy survivors.

The existing filter artifact confirms `"lotte_slices": []`, and the fit-list manifest has no LoTTE group. Rerunning the standard command will repeat the omission.

Smallest repair: take the exact survivor set from a validated remedy/pin artifact and read the remediated query files, not the raw slices or S0’s `kept`.

### MAJOR — Correlation is unmeasured, and “uncorrelated” is not credible

LoTTE differs from CQADupStack in task—question-to-answer retrieval versus duplicate-question retrieval—which helps. But both share StackExchange prose, site conventions, question construction, and now demonstrably some actual questions. It is useful as a one-time StackExchange catastrophe check, not independent confirmation, and it says little about FEVER/DBpedia.

Without touching the reserved sets, preregister a fixed candidate panel and measure candidate-delta rank correlation across:

- LoTTE dev slices,
- unused CQADupStack subforums,
- non-reserved Wikipedia/entity proxies.

Keep LoTTE test slices untouched for the eventual crossing. This can bound family dependence; nothing short of reserved access can establish correlation with the actual reserved outcomes.

### MINOR — The exact 14,034 count is circular evidence

Those counts were derived descriptively by S0 using the same raw data and detector, then written into the registration. Matching them proves repeatability and lack of input drift. It does not validate coverage or correctness. The two confirmed retained protected-document titles coexist with an exact 14,034 total.

## Bottom line

The correct disposition is:

> **STOP / reopen E10. The current seven-slice artifact is not decontaminated, and its zero re-screen is tautological.**

The most important immediate action is not threshold tuning; it is fixing the missing cross-role and missing-corpus comparisons. Only then does it make sense to replace the self-certifying re-screen with an independent, serialized-output acceptance check.
