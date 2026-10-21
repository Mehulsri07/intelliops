"""Drive several live data streams through a running compose stack.

Everything goes through the real services; nothing is mocked. Concurrently:

  - steady user traffic: Meridian gateway submissions + report reads, and
    demo-app /work, each at its own Poisson rate
  - traffic that swells and dips over ~10 minutes, with occasional bursts
  - a staggered fault schedule across five services (saturation, memory leak,
    demo-app break, DB pool exhaustion, latency, and one signal no RCA rule
    knows), so the correlator sees distinct incidents arrive one after another,
    and two of them recover part-way through
  - a deploy marker before the memory leak, so RCA has a rollback hypothesis
  - two approvals decided by the demo operator, so outcomes and the audit
    trail fill in

The first 90 seconds are healthy traffic only, so the detector has a baseline
before anything breaks. Faults are left ON when the run ends, so the incidents
stay open for a demo or screenshots; `--clear` recovers every service.

Usage (against `docker compose -f deploy/docker-compose.yml up`):

    uv run python scripts/live_traffic.py              # 14 minutes
    uv run python scripts/live_traffic.py --minutes 20
    uv run python scripts/live_traffic.py --clear      # recover everything

Override endpoints with GATEWAY_URL, DEMO_URL, GOV_URL, READ_URL. Under
AUTH_MODE=token, set AUTH_TOKEN (sent as a bearer header), as for reset.sh.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from collections.abc import Callable

import httpx

GW = os.environ.get("GATEWAY_URL", "http://localhost:8008")
DEMO = os.environ.get("DEMO_URL", "http://localhost:8080")
GOV = os.environ.get("GOV_URL", "http://localhost:8005")
READ = os.environ.get("READ_URL", "http://localhost:8007")
OPERATOR = os.environ.get("OPERATOR_NAME", "demo-operator")  # an approver in rbac_policy.yaml

_token = os.environ.get("AUTH_TOKEN")
client = httpx.Client(timeout=5.0, headers={"Authorization": f"Bearer {_token}"} if _token else {})
stop = threading.Event()


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def post(url: str, **kw) -> None:
    try:
        r = client.post(url, **kw)
        log(f"POST {url} -> {r.status_code}")
    except httpx.HTTPError as e:
        log(f"POST {url} failed: {e}")


# Traffic is not flat in real life: a slow swell over ~10 minutes, plus the
# odd burst (a batch job, a retry storm) at two to three times the rate for
# half a minute. Shared by every stream, so they rise and fall together.
_burst_until = 0.0


def load_factor(t: float) -> float:
    global _burst_until
    if t >= _burst_until and random.random() < 0.004:  # ~one burst per 4 minutes
        _burst_until = t + random.uniform(20, 45)
    swell = 1.0 + 0.35 * math.sin(2 * math.pi * t / 600) + 0.15 * math.sin(2 * math.pi * t / 137)
    return swell * (random.uniform(2.0, 3.0) if t < _burst_until else 1.0)


def stream(name: str, call: Callable[[], object], rate_hz: float) -> None:
    """Call `call` forever at a Poisson rate shaped by `load_factor`; errors are
    expected under faults."""
    n = 0
    while not stop.is_set():
        try:
            call()
        except httpx.HTTPError:
            pass
        n += 1
        if n % 500 == 0:
            log(f"[{name}] {n} requests")
        time.sleep(random.expovariate(rate_hz * load_factor(time.monotonic())))


def submit() -> None:
    client.post(
        f"{GW}/api/submissions",
        json={
            "client": random.choice(["acme", "globex", "initech", "umbrella", "hooli"]),
            "period": random.choice(["2026-Q3", "2026-09", "2026-10"]),
            "amount": round(random.uniform(120, 98_000), 2),
        },
    )


def fault(service: str, kind: str, magnitude: float = 1.0) -> Callable[[], None]:
    def go() -> None:
        post(
            f"{GW}/api/ops/fault",
            json={"service": service, "spec": {"type": kind, "magnitude": magnitude}},
        )

    return go


def approve_one() -> None:
    try:
        pending = client.get(f"{GOV}/approvals").json()
    except (httpx.HTTPError, ValueError):
        pending = []
    if not pending:
        log("no pending approval to decide yet")
        return
    post(
        f"{GOV}/approvals/{pending[0]['id']}/decide",
        json={"decision": "approved", "decided_by": OPERATOR},
    )


def clear_all() -> None:
    for svc in ("gateway", "validation", "aggregation", "reporting"):
        post(f"{GW}/api/ops/clear", json={"service": svc})
    post(f"{DEMO}/fix")


# (seconds from start, action)
SCHEDULE: list[tuple[int, Callable[[], None]]] = [
    (90, fault("validation", "saturation")),
    (
        150,
        lambda: post(f"{GW}/api/ops/deploy", json={"service": "aggregation", "version": "v2.4.0"}),
    ),
    (160, fault("aggregation", "memory_leak")),
    (230, lambda: post(f"{DEMO}/break")),
    (300, fault("reporting", "db_exhaustion")),
    # Not everything stays broken: validation recovers on its own and the
    # demo-app gets fixed, so the charts show incidents resolving too.
    (330, lambda: post(f"{GW}/api/ops/clear", json={"service": "validation"})),
    (360, approve_one),
    (390, fault("gateway", "latency", 0.8)),
    (450, fault("reporting", "unknown_signal")),
    (480, approve_one),
    (540, lambda: post(f"{DEMO}/fix")),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=14.0, help="how long to run (default 14)")
    parser.add_argument(
        "--clear", action="store_true", help="recover every faulted service and exit"
    )
    args = parser.parse_args()

    if args.clear:
        clear_all()
        return

    for name, call, hz in (
        ("gateway-submit", submit, 6.0),
        ("gateway-reports", lambda: client.get(f"{GW}/api/reports"), 3.0),
        ("demo-work", lambda: client.get(f"{DEMO}/work"), 4.0),
    ):
        threading.Thread(target=stream, args=(name, call, hz), daemon=True).start()
    log("traffic streams running; building a healthy baseline first")

    t0 = time.monotonic()
    for at, action in SCHEDULE:
        if at > args.minutes * 60:
            break
        if stop.wait(max(0.0, at - (time.monotonic() - t0))):
            break
        action()
    stop.wait(max(0.0, args.minutes * 60 - (time.monotonic() - t0)))
    stop.set()

    try:
        n = len(client.get(f"{READ}/situations").json())
        log(f"done: {n} open situations; faults left on (use --clear to recover)")
    except (httpx.HTTPError, ValueError) as e:
        log(f"done (read model unavailable: {e})")


if __name__ == "__main__":
    main()
