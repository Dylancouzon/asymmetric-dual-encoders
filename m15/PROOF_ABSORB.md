# Proof: what a mean-pooled lookup table absorbs

A lookup-table query encoder has rows $w_t \in \mathbb{R}^d$ for each token $t$ of a vocabulary $V$.
A query $q$ is a multiset of tokens with counts $c_t(q)$. The encoder pools with nonnegative
weights $\alpha_t(q)$ and normalizes:

$$p(q) = \sum_{t} \alpha_t(q)\, w_t, \qquad v(q) = \frac{p(q)}{\lVert p(q) \rVert}.$$

Mean pooling has $\alpha_t = c_t / \sum_s c_s$. Zero's count-saturated pooling has
$\alpha_t = \sqrt{c_t} / \sum_s \sqrt{c_s}$ (`m11/release/zero_encoder.py`). Both satisfy
$\sum_t \alpha_t(q) = 1$ for every nonempty $q$.

**Lemma 1 (affine maps after pooling).** Let $A \in \mathbb{R}^{d' \times d}$ and $b \in \mathbb{R}^{d'}$,
and let a system apply $x \mapsto Ax + b$ to the pooled vector. Define a new table
$w'_t = A w_t + b$. If $\sum_t \alpha_t(q) = 1$, then for every query

$$\sum_t \alpha_t(q)\, w'_t = A \sum_t \alpha_t(q)\, w_t + b \sum_t \alpha_t(q) = A\,p(q) + b,$$

so the transformed system is the table $w'$ with the same pooling, and after normalization it serves
the same vector. *Proof:* linearity of the sum. $\square$

The condition matters. Under sum pooling, $\sum_t \alpha_t(q) = \sum_t c_t(q) = |q|$, and the right
side becomes $A\,p(q) + |q|\, b$: the offset scales with query length, and no fixed table reproduces
the system unless $b = 0$. Centering ($A = I$, $b = -\mu$), whitening and projections ($b = 0$),
and top-principal-component removal after centering are all instances of Lemma 1.

**Lemma 2 (fixed per-token weights).** Let a system reweight tokens by fixed $g_t > 0$ (for
example IDF or SIF weights), renormalizing the pooling weights:
$\alpha'_t(q) = g_t\,\alpha_t(q) / \sum_s g_s\,\alpha_s(q)$. Define $w''_t = g_t w_t$. Then

$$\sum_t \alpha'_t(q)\, w_t = \frac{1}{Z(q)} \sum_t \alpha_t(q)\, w''_t, \qquad Z(q) = \sum_s g_s\, \alpha_s(q) > 0,$$

so the two pooled vectors differ by the positive scalar $1/Z(q)$, which the final normalization
removes: $v$ is identical for every query. *Proof:* substitute and factor. $\square$

**Corollary.** Any composition of Lemma 1 and Lemma 2 maps (for instance the full SIF recipe:
weights, centering and principal-component removal) is again a table of the same shape. Such
post-processing cannot extend the set of query functions the architecture can represent; a table
trained directly could have learned the result. It can still help as an initialization or prior.

**What the lemmas do not cover.** Pooling weights that depend on the pattern of counts rather than
on token identity (count saturation versus plain mean differ by 0.129 on our check and are not
interconvertible), any nonlinear map applied before normalization, and word order: every such table
is invariant to permuting the query's tokens.

**Numerical check.** `m7src/absorb_check.py` rebuilds each absorbed table explicitly and compares it
with the transformed system on ragged multisets with repeats (vocabulary 500, dimension 64); the
largest absolute difference over all cases is 9.31e-14 (`results/m7_absorb_check.json`).
