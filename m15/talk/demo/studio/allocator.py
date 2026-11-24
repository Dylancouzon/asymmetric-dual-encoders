"""Auto routing as an allocation: which share of traffic each query encoder serves.

Every encoder searches the same Stella index, so the choice is only about cost and quality. Each encoder keeps its
own search settings (HNSW ef and oversampling), set by the operator like any per-model setting; the router only
moves traffic. Two resources bound throughput on one machine, and the encoders stress them differently:

  encode   CPU work to turn the query into a vector; Stella is expensive, Nano moderate, Zero nearly free.
  shard    search time on the Qdrant Edge shard, which serves one search at a time in this process; it grows
           with each encoder's ef and oversampling.

A plan picks fractions f (summing to 1) that maximize the average nDCG@10 of served searches, subject to

  rate * sum(f_i / encode_capacity_i)      <= headroom    encode stays below saturation
  rate * sum(f_i * search_ms_i) / 1000     <= headroom    the shard stays below saturation

Headroom (< 1) leaves room for queueing so latency stays under target; the caller tunes it from measured p95.
Every query is served. An encoder the caller marks as slow (its own processing time exceeds the latency budget)
is left out. If no plan carries the whole rate, the plan is the one that carries the most, and the queue grows.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class EncoderCost:
    name: str
    quality: float          # nDCG@10 at this encoder's search settings (measured on all judged NQ queries)
    encode_per_s: float     # encodes per second this machine sustains for this encoder (measured at startup)
    search_ms: float        # shard time per search at its settings (offline sweep scaled by the live slowdown)


@dataclass(frozen=True)
class Plan:
    shares: dict            # encoder -> fraction of traffic
    quality: float          # expected average nDCG@10 of served searches
    encode_util: float      # fraction of encode capacity the plan uses at the offered rate
    shard_util: float       # fraction of shard time the plan uses at the offered rate
    status: str             # "ok", "over capacity", or "over target" (every encoder too slow for the target)


@lru_cache(maxsize=None)
def _grid(step: float) -> np.ndarray:
    """All fractions over three encoders in `step` increments that sum to 1 (5,151 rows at 1%)."""
    k = int(round(1 / step))
    return np.array([(a, b, k - a - b) for a in range(k + 1) for b in range(k + 1 - a)], dtype=np.float64) / k


def plan(rate: float, costs: list, headroom: float, slow: frozenset = frozenset(), step: float = 0.01) -> Plan:
    """costs: [EncoderCost], one per encoder, in a fixed order. slow: encoders too slow for the latency target on
    their own; left out unless every encoder is slow."""
    F = _grid(step)
    quality = F @ np.array([e.quality for e in costs])
    enc = F @ np.array([1.0 / e.encode_per_s for e in costs])
    shard = F @ np.array([e.search_ms / 1000 for e in costs])
    load = np.maximum(enc, shard) / headroom
    usable = ~(F[:, np.array([e.name in slow for e in costs])] > 0).any(axis=1)
    fast = usable.any()                                   # false only when every encoder is slow
    pool = usable if fast else np.ones(len(F), bool)
    fits = pool & (rate * load <= 1)
    if fits.any():
        # Best quality that fits; among equals, the lightest load (more slack for bursts).
        idx = np.flatnonzero(fits)
        i = idx[np.lexsort((load[idx], -quality[idx]))[0]]
        status = "ok" if fast else "over target"
    else:
        # More traffic than any plan carries: take the one that carries the most; among those, the best quality.
        idx = np.flatnonzero(pool)
        i = idx[np.lexsort((-quality[idx], load[idx]))[0]]
        status = "over capacity"
    return Plan(shares={e.name: float(x) for e, x in zip(costs, F[i])}, quality=float(quality[i]),
                encode_util=float(rate * enc[i]), shard_util=float(rate * shard[i]), status=status)


if __name__ == "__main__":
    # Self-check with this laptop's encode capacity (encodes/s: Stella 80, Nano 1,400, Zero 17,000) and the NQ sweep
    # values (nDCG@10, search ms) at the studio's default settings: Stella ef 64, Nano 128, Zero 256, oversampling 4.
    COSTS = [EncoderCost("stella", 0.617, 80, 0.11), EncoderCost("nano", 0.564, 1400, 0.16),
             EncoderCost("zero", 0.493, 17000, 0.25)]
    quiet = plan(40, COSTS, headroom=0.8)
    assert quiet.status == "ok" and quiet.shares["stella"] == 1.0, quiet           # quiet: the best encoder only
    spike = plan(600, COSTS, headroom=0.8)
    assert spike.status == "ok" and 0 < spike.shares["stella"] < 0.1 and spike.shares["zero"] == 0, spike
    assert spike.encode_util > 0.75, spike                                          # fills the CPU it may use (1% grid)
    busy = plan(2000, COSTS, headroom=0.8)
    assert busy.status == "ok" and busy.shares["zero"] > 0.2 and busy.encode_util > 0.79, busy
    flood = plan(8000, COSTS, headroom=0.8)                                        # past what the shard carries
    assert flood.status == "over capacity" and flood.shares["zero"] > 0.5 and flood.shares["stella"] == 0, flood
    stuck = plan(40, COSTS, headroom=0.8, slow=frozenset({"stella", "nano", "zero"}))
    assert stuck.status == "over target" and stuck.shares["stella"] == 1.0, stuck  # target below every encoder
    lean = plan(300, COSTS, headroom=0.8, slow=frozenset({"stella"}))
    assert lean.status == "ok" and lean.shares["stella"] == 0.0 and lean.shares["nano"] == 1.0, lean
    # A wider search for Zero (ef 1024, oversampling 8) costs shard time: 3,000/s no longer fits.
    assert plan(3000, COSTS, headroom=0.8).status == "ok"
    wide = plan(3000, [*COSTS[:2], EncoderCost("zero", 0.510, 17000, 0.74)], headroom=0.8)
    assert wide.status == "over capacity", wide
    print("allocator self-check passed: 2,000/s ->", {k: round(v, 2) for k, v in busy.shares.items()},
          f"nDCG {busy.quality:.3f} encode {busy.encode_util:.0%} shard {busy.shard_util:.0%}")
