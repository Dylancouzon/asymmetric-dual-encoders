# M19 development sensitivity method — Astra review

Date: 2026-09-13

Reviewer: `m19_sensitivity_method_astra` (`gpt-6-astra`, high reasoning)

Final disposition: **GO before result access**

The read-only review covered the frozen sensitivity amendment, implementation and synthetic tests,
plus only the tracked metric, registry, judgment-policy, owner-review and incomplete-result files
needed to assess the method. It did not access `work/m19`, queries, labels, qrels, packets, quality
results, confirmation content or any protected/spent surface.

The first pass found one P1: rounded floating-point Precision aggregates could misclassify an exact
threshold equality. It also recorded one small input-hardening issue: direct query scoring needed to
refuse a ranked artifact outside the pool. The implementation now retains exact rational
Precision@10 arithmetic through aggregation, term signs and gate comparisons, converting only for
JSON presentation, and validates every route top ten against the frozen pool. Synthetic regressions
cover both cases. Astra re-reviewed the changes and returned GO; all four focused tests passed.

Astra otherwise found the method sound: uncertainty is shared across routes, nonlinear metrics use
complete assigned pools, per-query enumeration gives exact separate term-macro extrema, and the
missing supporting-passage audit prevents any claim of full eligibility.
