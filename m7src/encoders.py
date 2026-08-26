"""The encoder registry: one place that knows how to run each candidate document/query tower.

This exists so the repo is a HARNESS rather than a bge-base-shaped script. Everything that varies
between encoders -- repo, revision, pooling, query/doc prompt, whether it needs remote code, the
tokenizer identity that goes into the encode cache key -- lives here and nowhere else.

Before this, `teacher.py` hardcoded BAAI/bge-base-en-v1.5 AND CLS pooling, so the repo could not
encode with a mean-pooled model (stella) at all, and `teacher_probe.py` had to keep a second copy
of the same table. Two copies of a pooling rule is how you get a silently wrong comparison.

Select the active encoder with the M7_ENCODER environment variable. The default is bge-base, and
its spec reproduces the pre-existing cache keys BYTE-IDENTICALLY -- `tokenizer_id` and `pooling`
are the literal strings that are already written into every meta.json under work/enc, so the
~22 GB of existing encodes stay valid. `m7src/test_encoders.py` pins that.

Adding an encoder: add a Spec. If it needs a different pooling or tokenizer, say so here; do not
special-case it at a call site.
"""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    name: str
    repo: str
    revision: str | None
    dim: int
    pooling: str                # "cls" | "mean" -- how to pool last_hidden_state
    query_prefix: str
    doc_prefix: str = ""
    trust_remote_code: bool = False
    max_length: int = 512
    # Sentence-transformers modules applied AFTER pooling and BEFORE L2 normalize. stella's
    # published pipeline is Transformer -> Pooling(mean) -> Dense_1024 (1024->1024, bias,
    # identity), per its modules.json. Omitting it encodes a DIFFERENT model from the one whose
    # MTEB 58.97 justified shortlisting it -- the same class of loader mismatch that was BLOCKER 1
    # of the M6 gate. None means "pooling output is the embedding".
    post_dense: str | None = None
    # Extra kwargs that must land on the model CONFIG (not on from_pretrained -- transformers 4.57
    # forwards unknown from_pretrained kwargs into the model __init__ and raises). stella cannot be
    # instantiated without these: its config ships unpad_inputs=true and
    # use_memory_efficient_attention=true, whose remote code asserts "please install xformers".
    # Its card documents setting both False as the no-xformers path, which also puts stella on the
    # same attention kernel gte-large already uses (both False in its own config), so the teacher
    # probe compares candidates through one kernel rather than two. Feeds the encode cache key
    # conditionally -- see teacher.cache_key.
    config_kwargs: dict = field(default_factory=dict)
    # The identity string that goes into the encode cache key. It must change whenever the token
    # ids would change, and must NOT change for bge-base or existing caches are orphaned.
    # All five registered candidates ship a BYTE-IDENTICAL vocab.txt (sha256 07eced375cec144d,
    # 30,522 lines -- stock bert-base-uncased WordPiece), verified 2026-08-26, so they legitimately
    # share this string. Consequence worth knowing: a teacher swap does NOT change tokenization, so
    # row indexing, the decontamination fingerprints and the frozen preprocessing rule all survive
    # it -- only the vectors change.
    tokenizer_id: str = "bert-wordpiece-30522"
    # Row id used for the degenerate-empty-query fallback. bge/BERT [CLS] is 101.
    # NOTE: table.py still hardcodes CLS_ID=101 and does NOT read this field yet -- see the
    # harness section of m7/CODEMAP.md. test_encoders.py checks it against the real tokenizer
    # only for models whose tokenizer is already cached locally.
    cls_id: int = 101
    # Rows in the released table = TOKENIZER size, not config.json's vocab_size. gte-large and
    # stella report 30528 in config.json -- that is the embedding matrix padded to a multiple of 64
    # for tensor cores, and allocating it would ship 6 dead rows. The shortlist quoted the padded
    # figure; the artifact is 30,522 x dim for every candidate.
    vocab: int = 30522
    notes: str = ""

    @property
    def pooling_key(self):
        """The exact string written into the encode cache key. 'cls-l2' is what every existing
        meta.json says, so it is preserved verbatim for CLS models."""
        return "cls-l2" if self.pooling == "cls" else f"{self.pooling}-l2"


BGE_PREFIX = "Represent this sentence for searching relevant passages: "

REGISTRY = {
    # The M7 default and the teacher every committed result was produced with.
    "bge-base-en-v1.5": Spec(
        name="bge-base-en-v1.5", repo="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        dim=768, pooling="cls", query_prefix=BGE_PREFIX),
    "bge-large-en-v1.5": Spec(
        name="bge-large-en-v1.5", repo="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        dim=1024, pooling="cls", query_prefix=BGE_PREFIX),
    # Alibaba, admissible with justification under CLAUDE.md's relaxed vendor rule.
    # Custom "NewModel" architecture: needs remote code. unpad_inputs and
    # use_memory_efficient_attention are both false in ITS config, so plain sdpa works and xformers
    # is an optional accelerator here. That is a property of this checkpoint's config, not of the
    # architecture: stella ships the same code with both flags true and will not load without
    # config_kwargs.
    "gte-large-en-v1.5": Spec(
        name="gte-large-en-v1.5", repo="Alibaba-NLP/gte-large-en-v1.5",
        revision="104333d6af6f97649377c2afbde10a7704870c7b",
        dim=1024, pooling="cls", query_prefix="", trust_remote_code=True,
        notes="no prompt convention; ships fp32"),
    # MEAN pooling, and an instruction-style query prompt. This is the spec that the old
    # CLS-hardcoded teacher.py could not have run correctly.
    "stella-400M-v5": Spec(
        name="stella-400M-v5", repo="NovaSearch/stella_en_400M_v5",
        revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
        dim=1024, pooling="mean", trust_remote_code=True, post_dense="2_Dense_1024",
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that "
                     "answer the query.\nQuery: ",
        notes="MRL heads at 256/768/1024+; ArguAna and FiQA2018 are on its recorded training "
              "list, which is 2 of our 6 eval datasets -- must be labelled if used"),
    # MEAN-POOLED, and the reason they are here: the learnability probe found that a teacher's
    # approximability by a bag-of-token-vectors tracks its POOLING, not its size or its MTEB score
    # (stella, mean, ratio 0.715 > bge-base, CLS, 0.686 > arctic-l, CLS, 0.526 > gte-large, CLS,
    # 0.431). Mean pooling IS an average over token positions, which is the operation a lookup table
    # performs, so the hypothesis has a mechanism. e5 is the strongest permissive mean-pooled English
    # family at our vocab and dim: MIT, intfloat is a clean vendor, 30,522 BERT WordPiece.
    # e5 REQUIRES its "query: " / "passage: " prefixes; omitting them is a known large regression.
    "e5-large-v2": Spec(
        name="e5-large-v2", repo="intfloat/e5-large-v2",
        revision="f169b11e22de13617baa190a028a32f3493550b6",
        dim=1024, pooling="mean", query_prefix="query: ", doc_prefix="passage: ",
        notes="mean-pooled; tests whether pooling explains approximability"),
    "e5-base-v2": Spec(
        name="e5-base-v2", repo="intfloat/e5-base-v2",
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        dim=768, pooling="mean", query_prefix="query: ", doc_prefix="passage: ",
        notes="mean-pooled 768-d control for e5-large-v2"),
    # DELIBERATELY OFF-SPEC: arctic-embed-l read out with MEAN pooling instead of its published CLS.
    # It separates "mean pooling makes a teacher approximable" from "stella is simply good": if the
    # best-ceiling tower becomes approximable when read out as a mean, the two are separable and we
    # can have both. The doc tower is ours to define, so an off-spec read-out is legitimate as long
    # as queries and documents use the SAME one -- but its quality is NOT the published model's, and
    # validate_encoder.py will and should FAIL it against sentence-transformers, which implements the
    # published CLS pipeline. Do not "fix" that failure.
    "arctic-embed-l-mean": Spec(
        name="arctic-embed-l-mean", repo="Snowflake/snowflake-arctic-embed-l",
        revision="d8fb21ca8d905d2832ee8b96c894d3298964346b",
        dim=1024, pooling="mean", query_prefix=BGE_PREFIX,
        notes="off-spec mean read-out of a CLS-trained tower; expected to fail ST validation"),
    "arctic-embed-l": Spec(
        name="arctic-embed-l", repo="Snowflake/snowflake-arctic-embed-l",
        revision="d8fb21ca8d905d2832ee8b96c894d3298964346b",
        dim=1024, pooling="cls", query_prefix=BGE_PREFIX,
        notes="vendor tier justify-max; kept for completeness, dominated on quality"),
}

DEFAULT = "bge-base-en-v1.5"


def _assert_pinned():
    """Every Spec must carry a revision. An unpinned repo makes the encode cache key ambiguous --
    a silent upstream bump would reuse stale vectors under the same key -- and would have the probe
    that decides our most expensive choice run unpinned remote code."""
    loose = [n for n, s in REGISTRY.items() if not s.revision]
    if loose:
        raise AssertionError(f"unpinned encoder revisions: {loose}")


_assert_pinned()


def active():
    """The encoder this process is using. M7_ENCODER selects it; default is the M7 teacher."""
    n = os.environ.get("M7_ENCODER", DEFAULT)
    if n not in REGISTRY:
        raise KeyError(f"unknown encoder {n!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[n]


def get(name):
    return REGISTRY[name]


def specs_for_repo(repo):
    return [s for s in REGISTRY.values() if s.repo == repo]


def by_repo(repo):
    """Resolve a spec from a HF repo id. The encode cache key is handed a model_id rather than a
    spec name, and it must not guess pooling or tokenizer identity -- guessing is how two encoders
    that produce different vectors end up sharing one cache key.

    A repo can map to MORE THAN ONE Spec: `arctic-embed-l` and `arctic-embed-l-mean` are the same
    weights read out two ways. So the ACTIVE encoder wins whenever its repo matches -- a process
    running one of them must resolve its own repo to itself -- and an ambiguous repo with no active
    match RAISES rather than picking by dict order. Silently picking would write CLS vectors under a
    cache key that claims mean pooling, which is precisely the collision test_encoders.py exists to
    prevent and which it cannot see (it replays keys written before the variant existed)."""
    act = active()
    if act.repo == repo:
        return act
    matches = specs_for_repo(repo)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"repo {repo!r} is not in the encoder registry. Add a Spec for it rather "
                       f"than passing it through: the cache key needs its pooling and tokenizer "
                       f"identity, and defaulting those silently mislabels encodes.")
    raise KeyError(f"repo {repo!r} maps to several Specs ({[m.name for m in matches]}) and none is "
                   f"the active encoder. Set M7_ENCODER to the one you mean -- resolving this by "
                   f"registry order would mislabel the encode.")


BY_REPO = {s.repo: s for s in REGISTRY.values()}   # kept for readers that only need a repo listing
