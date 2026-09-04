# The Cloud Inference ask — drafted 2026-09-04, not yet sent

Dylan asked for a CTO-ready case for hosting stella's document tower in Qdrant Cloud Inference.
Reviewed by Fable and Codex; Codex's verdict on the first version was **"not defensible as stated"**
and the draft below is the narrowed form. Evidence: `m7/RESULTS.md`, `results/m7_learnability_report.json`,
`results/m7_offfamily_report.json`, `research/teacher-reviews-2026-09-04.md`.

## What may and may not be claimed

- **MAY:** stella is the best teacher of eleven measured, CI-resolved, and survives an
  exposure-free control (SQuAD +0.1652, ESCI +0.0384 vs bge-base).
- **MAY NOT:** *"Cloud Inference must host it."* False — bring-your-own-vectors, local inference and
  self-hosted Superlinked all work. The defensible form is the **absence of a first-party managed
  raw-text path** for the vector contract we published.
- **MAY NOT:** *"−0.0365 is the cost of the hosted alternative."* It is a screening delta; bge never
  received stella's recipe. Six-set cost is **unmeasured**, plausibly 0.045–0.07.
- **MAY NOT:** *"stella is the only logical choice"* at 99% confidence. Retraining against a hosted
  model is a worse option, not an impossible one.

## The draft

> **Subject: Cloud Inference — request to host one document encoder (stella_en_400M_v5)**
>
> We've published `constella-zero` (MIT, on HF): a query encoder with no transformer in it. It's a
> 31 MB lookup table that embeds a search query in **0.023 ms on CPU** — no GPU, no model download,
> no inference call. It's the query half of an asymmetric pair aimed at edge and thin-client retrieval.
>
> It only works against an index built by one specific document model: **`NovaSearch/stella_en_400M_v5`**
> (MIT, 435M params). Qdrant Cloud can't build that index today, so a customer using our own published
> model has to precompute vectors elsewhere or run stella themselves. Bring-your-own-vectors and
> self-hosted options do work — this is about there being no first-party managed raw-text path for
> the contract we shipped.
>
> **The ask:** add the stella document encoder to Cloud Inference as a pilot. We've already exported
> and byte-verified the ONNX graph (parity 4.5e-08 on 1,024 queries), so the model-side work is done.
>
> **Why stella specifically.** We measured 11 candidate document encoders, including three already in
> Cloud Inference, on the metric that actually predicts our product's quality — how well the model can
> be distilled into a lookup table, which does *not* track MTEB rank (correlation ≈ 0 in our data; the
> best-hosted candidate, mxbai-embed-large, ranked 8th of 11). Stella wins by a statistically resolved
> margin, and still wins on held-out data it demonstrably never trained on. The best hosted alternative
> costs ~0.037 on our selection metric, an estimated 0.045–0.07 on the final benchmark, plus a full
> retrain and the re-release of two public models.
>
> **What this rests on, honestly.** It's a screening comparison — we have not given the hosted
> alternatives stella's full training recipe, so treat the gap as directionally solid rather than exact.
>
> **What to gate approval on:** throughput and cost per token, serving economics (the graph is
> currently fp32, 1.75 GB), latency/SLA, security review. There may also be a cheaper configuration —
> stella publishes a 768-d variant we haven't yet evaluated, which would cut index and serving cost ~25%.

## Questions a CTO will ask that we cannot answer today

Customer demand · why bring-your-own-vectors is insufficient · fp32 serving economics ·
latency/throughput/SLA · the 768-d alternative (never ranked; `m7/RESULTS.md` says lower dim distils
better within a family) · security review · upstream FastEmbed ownership (M13) · the measured quality
loss of the cheapest hosted counterfactual.

## Scoped out by Dylan, 2026-09-04

Prompt-pair handling stays **out** of the ask — document hosting only — even though the published
graph is a complete query encoder (`m11/STATUS.md` addendum).
