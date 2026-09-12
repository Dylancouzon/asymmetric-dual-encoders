# Attribution notice — Kubernetes documentation (ships with derived weights)

This file is the attribution artifact required by CC BY 4.0 for any model weights, dataset
artifact or derived file in this repository that was trained on, or derived from, official
Kubernetes documentation. Ship it (or its text) alongside the released weights, in the model card
and in the release bundle. Recorded 2026-09-11 as part of M17 pre-clock step 2a.

## Notice

> © The Kubernetes Authors. Kubernetes documentation is distributed under
> [CC BY 4.0](https://git.k8s.io/website/LICENSE).

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode ·
repository copy: https://github.com/kubernetes/website/blob/main/LICENSE

## Source

- Work: official Kubernetes documentation, English pages only (`content/en/docs`).
- Origin: the `kubernetes/website` repository, https://github.com/kubernetes/website
- Creator / attribution party: The Kubernetes Authors.
- Acquisition route: `git clone --filter=blob:none` of the repository followed by checkout of the
  pinned revision below. Not a scrape of the rendered kubernetes.io site.

## Revision

- Pinned commit: `17133089068629ec12ca15c1bdf36a60d2671a74` (`refs/heads/main`, resolved by
  `git ls-remote` on 2026-09-11).
- Only content present at that commit was used. Submodules (the Docsy theme) were not fetched and
  contribute nothing.

## Modification statement

The licensed material was modified. Modifications: selection of the English documentation subset
only; removal of YAML front matter, Hugo shortcodes and template directives; conversion to plain
text records in a JSON Lines file; deduplication and near-duplicate removal; tokenization; and use
as training text for a query-encoder model. The released model weights are a derived work in which
no verbatim page is redistributed. No endorsement by The Kubernetes Authors, the CNCF or the Linux
Foundation is stated or implied.

## Rights not granted

CC BY 4.0 §2(b)(2): "Patent and trademark rights are not licensed under this Public License."
"Kubernetes" and related marks remain trademarks of the Linux Foundation and are not licensed by
CC BY 4.0; see https://lfprojects.org/policies/. Nothing in the release may suggest sponsorship or
endorsement (CC BY 4.0 §2(a)(6)).
