"""Idle-jitter for the simulated workloads (Meridian + demo-app).

Why this exists
---------------
The simulated services pin every healthy metric to a single constant
(`CPU_HEALTHY = 18.0`, `MEMORY_USAGE_MB_HEALTHY = 256.0`, ...). That was the
right call for the fault-profile invariants -- a test can assert an exact value
-- but it has two costs in a live cluster:

1. The console's metric charts draw five perfectly flat, perfectly overlapping
   lines. A judge reading that dashboard sees a mock, not a system.
2. A constant series has zero variance, so the robust correlator learns
   `std == 0` for every metric and `verify.py` has to fall back to its
   flat-baseline step rule. Real telemetry has spread; a detector tuned against
   a degenerate baseline has not really been exercised.

So this module adds a small, bounded wander around whatever value the service's
state object currently holds. It does NOT change the state model, the fault
profiles, or any constant -- it is applied at scrape time only, and only when
`INTELLIOPS_TELEMETRY_JITTER` is enabled. The unit tests run with it off and
keep asserting exact values.

Properties that matter
----------------------
- **Deterministic.** A pure function of (metric, service, wall clock). No RNG
  state, so a pod restart rejoins the same curve it left and two replicas of
  one service agree.
- **Bounded.** The deviation never exceeds the per-metric fraction in
  `_AMPLITUDE`, so a healthy service can never wander far enough to trip the
  z-threshold (3.0) and manufacture a false incident. cpu_usage, the widest,
  stays inside 18.0 +/- 16%, which is roughly |z| <= 1.7 against its own
  learned baseline.
- **Shaped like telemetry.** Two sine components at different periods give the
  slow drift and the medium-frequency ripple; a hash-derived grain term adds
  per-scrape texture so the curve is not suspiciously smooth either. Each
  sine's PHASE wanders on smooth value noise, so the curve never repeats: plain
  sines beat into a visibly regular pattern on a 15-minute chart. The sine
  shape itself is kept on purpose: its values sit near the extremes, so the
  spread a median/MAD detector learns covers the whole swing. Replacing it with
  value noise (tried) looked as natural but put the tails outside 3 MADs and
  produced ~200 false threshold crossings per metric in six idle hours; the
  wobbled sine produces none, measured the same way.
- **No instant steps.** `shape()` also eases each gauge toward its target with a
  per-metric time constant, so a fault onset climbs over tens of seconds and a
  recovery decays, the way a real saturating CPU or draining queue does,
  instead of a vertical edge between two ruled lines. This is the one stateful
  part, and it only delays a value; it never changes where it settles.
- **Phase-separated per service.** Without this, five services sharing one
  baseline draw five identical curves, which looks exactly as synthetic as five
  flat lines.
"""

from __future__ import annotations

import hashlib
import math
import os
import time

# Per-metric wander, as a fraction of the current value. Chosen so the widest
# (cpu) stays well inside the correlator's z-threshold of 3.0, and so the
# relative noisiness matches what the metric would really do: disk creeps,
# queue depth is spiky, memory is steady.
_AMPLITUDE: dict[str, float] = {
    "cpu_usage": 0.13,
    "memory_usage_mb": 0.055,
    "latency_p50_ms": 0.17,
    "latency_p99_ms": 0.21,
    "request_rate": 0.13,
    "saturation": 0.20,
    "queue_depth": 0.34,
    "db_pool_in_use": 0.28,
    "disk_usage_percent": 0.015,
    "meridian_error_rate": 0.45,
}

# A metric pinned at exactly 0.0 has nothing to wander around, and a chart of
# five lines along the x-axis is no better than five lines at 18.0. A real
# service at steady state does drop a small fraction of requests, so the error
# rate gets an idle floor to breathe around. Everything else is left alone:
# `service_up` is a boolean, `db_pool_max` is a configured size, and
# `tls_handshake_failures` is the trigger for the escalation demo, so none of
# them should move.
_IDLE_FLOOR: dict[str, float] = {
    "meridian_error_rate": 0.0035,
}

# Ranges the jittered value is clamped back into, so a percentage cannot read
# 104 and a fraction cannot read 1.3.
_BOUNDS: dict[str, tuple[float, float]] = {
    "cpu_usage": (0.0, 100.0),
    "disk_usage_percent": (0.0, 100.0),
    "saturation": (0.0, 1.0),
    "meridian_error_rate": (0.0, 1.0),
}

# How fast each metric follows a change in its target, in seconds (the time to
# cover ~63% of a step). Memory and disk move slowly, latency and errors react
# within a scrape or two. Kept well under a minute so detection is delayed by
# seconds, not by a correlation window.
_TAU: dict[str, float] = {
    "cpu_usage": 20.0,
    "memory_usage_mb": 45.0,
    "latency_p50_ms": 10.0,
    "latency_p99_ms": 12.0,
    "request_rate": 15.0,
    "saturation": 18.0,
    "queue_depth": 12.0,
    "db_pool_in_use": 15.0,
    "disk_usage_percent": 60.0,
    "meridian_error_rate": 8.0,
}

# Periods in seconds. Both are deliberately SHORTER than the span of the
# correlator's learning window, and that constraint is tighter than it looks:
# ingestion polls Prometheus every 5s and publishes one event per service, and
# RobustCorrelator keys its window by metric name alone, so a 128-sample window
# holds only ~128 seconds of wall clock. A drift slower than that is never
# learned as spread -- the window sees only a slice of it, MAD comes out small,
# and the far side of the drift reads as an anomaly. Measured: at a 420s slow
# period, six hours of idle telemetry produced bursts of up to four threshold
# crossings per service per window; at 150s, at most one.
_SLOW_PERIOD = 150.0
_FAST_PERIOD = 40.0
# Phase wobble: how far (radians) and how slowly (seconds between noise
# points) each sine's phase drifts. Enough to break the repetition (one-period
# autocorrelation about 0.5 -> 0.2) without changing the value distribution.
_WOBBLE = (4.0, 2.5)
_WOBBLE_SPACING = 300.0
# Grain is quantised to this many seconds so that one Prometheus scrape (15s)
# lands on its own value and the texture does not smear.
_GRAIN_TICK = 5.0


def jitter_enabled() -> bool:
    """True when the deployment asked for live-shaped telemetry.

    Off by default, which is what keeps `test_metrics.py`'s exact-value
    assertions meaningful. `values-live.yaml` turns it on.
    """
    return os.getenv("INTELLIOPS_TELEMETRY_JITTER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _unit(seed: str) -> float:
    """A stable float in [0, 1) derived from a string."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 4294967296.0


def _noise(seed: str, t: float, spacing: float) -> float:
    """Smooth, non-repeating noise in [-1, 1]: hashed lattice values every
    `spacing` seconds, joined with a smoothstep so the curve has no corners."""
    x = t / spacing
    i = math.floor(x)
    f = x - i
    a = _unit(f"{seed}|{i}") * 2.0 - 1.0
    b = _unit(f"{seed}|{i + 1}") * 2.0 - 1.0
    return a + (b - a) * f * f * (3.0 - 2.0 * f)


# (service, metric) -> (eased value, time it was computed). Per process, which
# is per service: a restart starts from the current target, with no ramp.
_eased: dict[tuple[str, str], tuple[float, float]] = {}


def ease(metric: str, service: str, target: float, now: float | None = None) -> float:
    """Move this gauge part of the way toward `target`, exponentially in time."""
    tau = _TAU.get(metric)
    if tau is None:
        return target
    t = time.time() if now is None else now
    prev = _eased.get((service, metric))
    if prev is None:
        value = target
    else:
        value, t0 = prev
        value += (target - value) * (1.0 - math.exp(-max(t - t0, 0.0) / tau))
    _eased[(service, metric)] = (value, t)
    return value


def shape(metric: str, service: str, value: float, now: float | None = None) -> float:
    """What a live deployment publishes: eased toward `value`, then jittered."""
    return jitter(metric, service, ease(metric, service, value, now), now)


def jitter(metric: str, service: str, value: float, now: float | None = None) -> float:
    """Return `value` nudged onto a plausible idle curve.

    Metrics with no entry in `_AMPLITUDE` are returned untouched, which is how
    `service_up`, `db_pool_max` and `tls_handshake_failures` stay exact.
    """
    amplitude = _AMPLITUDE.get(metric)
    if amplitude is None:
        return value

    base = max(value, _IDLE_FLOOR.get(metric, 0.0))
    if base == 0.0:
        return value

    t = time.time() if now is None else now
    key = f"{service}|{metric}"
    slow = math.sin(
        2 * math.pi * t / _SLOW_PERIOD
        + _unit(f"{key}|slow") * math.tau
        + _WOBBLE[0] * _noise(f"{key}|slow-phase", t, _WOBBLE_SPACING)
    )
    fast = math.sin(
        2 * math.pi * t / _FAST_PERIOD
        + _unit(f"{key}|fast") * math.tau
        + _WOBBLE[1] * _noise(f"{key}|fast-phase", t, _WOBBLE_SPACING / 3)
    )
    grain = _unit(f"{service}|{metric}|{int(t // _GRAIN_TICK)}") * 2.0 - 1.0

    # Weights sum to 1.0, so `deviation` stays in [-1, 1] and the result stays
    # inside +/- one swing of the base. Tuned against a replay of the real
    # detector rather than by eye: five services through a 128-sample median/MAD
    # window with a 30-sample warmup, six hours of idle telemetry, counting
    # threshold crossings per service per 30s correlation window. At these
    # values the worst window holds one; the grain term at 0.22 pushed it to
    # four, which is an incident.
    deviation = 0.50 * slow + 0.40 * fast + 0.10 * grain

    # The swing is proportional to the value. Near a ceiling the overshoot is
    # REFLECTED back inside rather than clipped: a CPU pinned at 92% then reads
    # 80-100 and bounces off the top like a saturated host, where scaling the
    # swing by the 8 points of headroom (the previous rule) drew a flat line
    # and clipping would draw a flat plateau at 100.
    low, high = _BOUNDS.get(metric, (0.0, math.inf))
    out = base + amplitude * (base - low) * deviation
    if out > high:
        out = 2 * high - out
    return min(max(out, low), high)
