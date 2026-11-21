"""Auto routing as an allocation: which share of traffic each query encoder should serve.

Every encoder searches the same Stella index, so the choice is only about cost and quality. Two resources
bound throughput on one machine, and the encoders stress them differently:

  encode   CPU work to turn the query into a vector; Stella is expensive, Nano moderate, Zero nearly free.
  shard    search time on the Qdrant Edge shard, which serves one search at a time in this process; it grows
           with HNSW ef and oversampling, and Zero needs a wider search than Nano to find good neighbors.

A plan picks fractions f (summing to 1) that maximize the average nDCG@10 of served searches, subject to

  average nDCG  >= floor                                  quality comes first
  rate * sum(f_i / encode_capacity_i)      <= headroom    encode stays below saturation
  rate * sum(f_i * search_ms_i) / 1000     <= headroom    the shard stays below saturation

Headroom (< 1) leaves room for queueing so latency stays under target; the caller tunes it from measured p95.
If no mix serves the whole rate at the floor, the plan is the floor-meeting mix that sustains the most traffic,
and the excess is shed. If no mix can reach the floor, the plan says so and serves the best quality it can.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class EncoderCost:
    name: str
    quality: float          # nDCG@10 at its current search setting (measured on judged queries)
    encode_per_s: float     # encodes per second this machine sustains for this encoder (measured at startup)
    search_ms: float        # shard time per search at its current ef / oversampling (measured live)
    service_ms: float       # p95 of its own processing time (encode + search, no queueing) right now


@dataclass(frozen=True)
class Plan:
    shares: dict            # encoder -> fraction of traffic
    quality: float          # expected average nDCG@10 of served searches
    encode_util: float      # fraction of encode capacity used at the planned rate
    shard_util: float       # fraction of shard time used at the planned rate
    serve_rate: float       # searches/s the plan serves (the offered rate, or less when overloaded)
    status: str             # "ok", "overloaded" (shedding to keep latency) or "floor unreachable"


@lru_cache(maxsize=None)
def _grid(n: int, step: float) -> np.ndarray:
    """All fractions over n encoders in `step` increments that sum to 1 (5,151 rows for 3 encoders at 1%)."""
    k = int(round(1 / step))
    rows = [(a, b, k - a - b) for a in range(k + 1) for b in range(k + 1 - a)] if n == 3 else None
    if rows is None:
        raise ValueError("this demo plans over three encoders")
    return np.array(rows, dtype=np.float64) / k


def plan(rate: float, encoders: list, floor: float, target_ms: float, headroom: float, step: float = 0.01) -> Plan:
    F = _grid(len(encoders), step)
    q = np.array([e.quality for e in encoders])
    inv_cap = np.array([1.0 / e.encode_per_s for e in encoders])
    shard_s = np.array([e.search_ms / 1000 for e in encoders])
    # An encoder whose own processing time is already near the target cannot help, whatever the queue does.
    too_slow = np.array([e.service_ms > 0.7 * target_ms for e in encoders])

    quality = F @ q
    load_per_rate = np.maximum(F @ inv_cap, F @ shard_s)                  # the tighter of the two resources
    usable = ~(F[:, too_slow] > 0).any(axis=1) if too_slow.any() else np.ones(len(F), bool)
    meets_floor = (quality >= floor - 1e-9) & usable

    fits = meets_floor & (rate * load_per_rate <= headroom)
    if fits.any():
        # Best quality that fits; among equals, the lightest load (more slack for bursts).
        idx = np.flatnonzero(fits)
        i = idx[np.lexsort((load_per_rate[idx], -quality[idx]))[0]]
        status, served = "ok", rate
    elif meets_floor.any():
        # Overloaded: the floor-meeting mix that sustains the most traffic; the rest is shed.
        idx = np.flatnonzero(meets_floor)
        i = idx[np.argmin(load_per_rate[idx])]
        status, served = "overloaded", min(rate, headroom / load_per_rate[i])
    else:
        # The floor is above what any usable mix delivers: serve the best quality, and say so.
        idx = np.flatnonzero(usable) if usable.any() else np.arange(len(F))
        i = idx[np.lexsort((load_per_rate[idx], -quality[idx]))[0]]
        status, served = "floor unreachable", min(rate, headroom / load_per_rate[i])

    f = F[i]
    return Plan(shares={e.name: float(x) for e, x in zip(encoders, f)}, quality=float(quality[i]),
                encode_util=float(served * (f @ inv_cap)), shard_util=float(served * (f @ shard_s)),
                serve_rate=float(served), status=status)


if __name__ == "__main__":
    # Self-check with this laptop's measured costs (Stella 80/s, Nano ~1,400/s, Zero ~17,000/s encodes).
    def costs(zero_ef_ms, zero_q):
        return [EncoderCost("stella", 0.631, 80, 0.23, 30), EncoderCost("nano", 0.579, 1400, 0.39, 2.5),
                EncoderCost("zero", zero_q, 17000, zero_ef_ms, 0.8)]
    quiet = plan(40, costs(0.74, 0.510), floor=0.50, target_ms=150, headroom=0.8)
    assert quiet.status == "ok" and quiet.shares["stella"] == 1.0, quiet          # quiet: spend it on quality
    spike = plan(600, costs(0.74, 0.510), floor=0.50, target_ms=150, headroom=0.8)
    assert spike.status == "ok" and spike.shares["stella"] < 0.1, spike           # spike: Stella cannot keep up
    wide = plan(2000, costs(0.74, 0.510), floor=0.50, target_ms=150, headroom=0.8)
    assert wide.status == "overloaded", wide                                      # wide Zero: no mix fits 2,000/s
    narrow = plan(2000, costs(0.16, 0.480), floor=0.50, target_ms=150, headroom=0.95)
    assert narrow.status == "ok" and narrow.shares["nano"] > 0.3 and narrow.shares["zero"] > 0.2, narrow   # a mix fits
    high = plan(40, costs(0.74, 0.510), floor=0.70, target_ms=150, headroom=0.8)
    assert high.status == "floor unreachable", high
    tight = plan(40, costs(0.74, 0.510), floor=0.50, target_ms=10, headroom=0.8)
    assert tight.shares["stella"] == 0.0, tight                                   # Stella alone takes 30 ms
    print("allocator self-check passed:", {k: round(v, 2) for k, v in narrow.shares.items()},
          f"encode {narrow.encode_util:.0%} shard {narrow.shard_util:.0%}")
