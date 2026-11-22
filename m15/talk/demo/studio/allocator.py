"""Auto routing as an allocation: which share of traffic each query encoder serves, and how wide they search.

Every encoder searches the same Stella index, so the choice is only about cost and quality. Two resources bound
throughput on one machine, and the encoders stress them differently:

  encode   CPU work to turn the query into a vector; Stella is expensive, Nano moderate, Zero nearly free.
  shard    search time on the Qdrant Edge shard, which serves one search at a time in this process; it grows
           with HNSW ef and oversampling, and Zero needs a wider search than Nano to find good neighbors.

A plan picks one search preset (ef, oversampling) shared by all encoders and fractions f (summing to 1) that
maximize the average nDCG@10 of served searches, subject to

  average nDCG  >= floor                                  quality comes first
  rate * sum(f_i / encode_capacity_i)      <= headroom    encode stays below saturation
  rate * sum(f_i * search_ms_i) / 1000     <= headroom    the shard stays below saturation

Headroom (< 1) leaves room for queueing so latency stays under target; the caller tunes it from measured p95.
Every query is served. An encoder the caller marks as slow (its own processing time exceeds the latency budget)
is left out, unless the floor needs it. If no plan carries the whole rate at the floor, the plan is the floor-meeting one that carries the
most: the queue grows and latency goes over target, because quality comes first. If no plan reaches the floor,
the plan says so and serves the best quality.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class EncoderCost:
    name: str
    quality: float          # nDCG@10 at this search preset (measured on all judged NQ queries)
    encode_per_s: float     # encodes per second this machine sustains for this encoder (measured at startup)
    search_ms: float        # shard time per search at this preset (offline sweep scaled by the live slowdown)


@dataclass(frozen=True)
class Plan:
    preset: tuple           # (ef, oversampling) every encoder searches with
    shares: dict            # encoder -> fraction of traffic
    quality: float          # expected average nDCG@10 of served searches
    encode_util: float      # fraction of encode capacity the plan uses at the offered rate
    shard_util: float       # fraction of shard time the plan uses at the offered rate
    status: str             # "ok", "over target to keep floor", "over capacity at this floor", "floor unreachable"


@lru_cache(maxsize=None)
def _grid(step: float) -> np.ndarray:
    """All fractions over three encoders in `step` increments that sum to 1 (5,151 rows at 1%)."""
    k = int(round(1 / step))
    return np.array([(a, b, k - a - b) for a in range(k + 1) for b in range(k + 1 - a)], dtype=np.float64) / k


def plan(rate: float, presets: dict, floor: float, headroom: float, slow: frozenset = frozenset(),
         step: float = 0.01) -> Plan:
    """presets: (ef, oversampling) -> [EncoderCost] in a fixed encoder order.
    slow: encoders too slow for the latency target on their own; used only when the floor needs them."""
    F, keys = _grid(step), list(presets)
    names = [e.name for e in presets[keys[0]]]
    rows = []                                             # per preset: quality, load per unit rate, encode, shard
    for key in keys:
        costs = presets[key]
        enc = F @ np.array([1.0 / e.encode_per_s for e in costs])
        shard = F @ np.array([e.search_ms / 1000 for e in costs])
        too_slow = np.array([e.name in slow for e in costs])
        rows.append((F @ np.array([e.quality for e in costs]), np.maximum(enc, shard) / headroom, enc, shard,
                     ~(F[:, too_slow] > 0).any(axis=1)))
    quality, load, enc, shard, usable = (np.concatenate(x) for x in zip(*rows))

    meets_floor = quality >= floor - 1e-9
    # Quality first: use slow encoders only when no fast plan reaches the floor.
    fast = (meets_floor & usable).any()
    pool = meets_floor & usable if fast else meets_floor
    fits = pool & (rate * load <= 1)
    if fits.any():
        # Best quality that fits; among equals, the lightest load (more slack for bursts).
        idx = np.flatnonzero(fits)
        i = idx[np.lexsort((load[idx], -quality[idx]))[0]]
        status = "ok" if fast else "over target to keep floor"
    elif pool.any():
        # More traffic than any floor-meeting plan carries: take the one that carries the most; the queue grows.
        idx = np.flatnonzero(pool)
        i = idx[np.argmin(load[idx])]
        status = "over capacity at this floor"
    else:
        # The floor is above what any plan delivers: serve the best quality, and say so.
        i = np.lexsort((load, -quality))[0]
        status = "floor unreachable"

    f = F[i % len(F)]
    return Plan(preset=keys[i // len(F)], shares={n: float(x) for n, x in zip(names, f)}, quality=float(quality[i]),
                encode_util=float(rate * enc[i]), shard_util=float(rate * shard[i]), status=status)


if __name__ == "__main__":
    # Self-check with this laptop's costs (encodes/s: Stella 80, Nano 1,400, Zero 17,000) and NQ sweep values.
    SWEEP = {(128, 4.0): ((0.627, 0.15), (0.564, 0.16), (0.480, 0.16)),
             (256, 4.0): ((0.631, 0.23), (0.573, 0.24), (0.493, 0.25)),
             (512, 4.0): ((0.633, 0.38), (0.579, 0.39), (0.502, 0.40)),
             (1024, 8.0): ((0.636, 0.69), (0.580, 0.70), (0.510, 0.74))}

    def presets():
        return {k: [EncoderCost("stella", v[0][0], 80, v[0][1]), EncoderCost("nano", v[1][0], 1400, v[1][1]),
                    EncoderCost("zero", v[2][0], 17000, v[2][1])] for k, v in SWEEP.items()}

    quiet = plan(40, presets(), floor=0.50, headroom=0.8)
    assert quiet.status == "ok" and quiet.shares["stella"] == 1.0 and quiet.preset == (1024, 8.0), quiet
    spike = plan(600, presets(), floor=0.50, headroom=0.8)
    assert spike.status == "ok" and spike.shares["stella"] < 0.1, spike             # Stella cannot keep up
    busy = plan(2000, presets(), floor=0.50, headroom=0.8)
    assert busy.status == "ok" and busy.shares["zero"] > 0.2, busy                  # nothing dropped: a cheaper plan
    huge = plan(8000, presets(), floor=0.50, headroom=0.8)
    assert huge.status == "over capacity at this floor", huge                       # the queue grows
    keep = plan(96, presets(), floor=0.60, headroom=0.8, slow=frozenset({"stella"}))
    assert keep.status == "over target to keep floor" and keep.shares["stella"] >= 0.4, keep   # floor beats latency
    agent = plan(300, presets(), floor=0.50, headroom=0.8, slow=frozenset({"stella"}))
    assert agent.status == "ok" and agent.shares["stella"] == 0.0, agent            # slow and not needed: left out
    high = plan(40, presets(), floor=0.70, headroom=0.8)
    assert high.status == "floor unreachable" and high.shares["stella"] == 1.0, high
    print("allocator self-check passed: 2,000/s ->", busy.preset, {k: round(v, 2) for k, v in busy.shares.items()},
          f"nDCG {busy.quality:.3f} encode {busy.encode_util:.0%} shard {busy.shard_util:.0%}")
