from common import telemetry


def test_jitter_is_bounded_deterministic_and_not_periodic():
    series = [telemetry.jitter("cpu_usage", "svc", 18.0, now=1000.0 + 5 * i) for i in range(720)]
    assert all(18.0 * 0.87 - 1e-9 <= v <= 18.0 * 1.13 + 1e-9 for v in series)
    assert series == [
        telemetry.jitter("cpu_usage", "svc", 18.0, now=1000.0 + 5 * i) for i in range(720)
    ]
    # A sine-based curve repeats exactly after one slow period (150s = 30 samples).
    assert series[:30] != series[30:60]


def test_faulted_value_near_a_ceiling_still_moves_and_stays_in_bounds():
    series = [telemetry.jitter("cpu_usage", "svc", 92.0, now=2000.0 + 5 * i) for i in range(360)]
    assert all(0.0 <= v <= 100.0 for v in series)
    assert max(series) - min(series) > 6.0  # not a ruled line at the top


def test_ease_turns_a_step_into_a_ramp_that_settles():
    telemetry._eased.clear()
    assert telemetry.ease("cpu_usage", "svc", 18.0, now=0.0) == 18.0
    first = telemetry.ease("cpu_usage", "svc", 92.0, now=5.0)
    assert 18.0 < first < 50.0
    assert abs(telemetry.ease("cpu_usage", "svc", 92.0, now=300.0) - 92.0) < 0.01
    assert (
        telemetry.ease("service_up", "svc", 0.0, now=301.0) == 0.0
    )  # untracked metrics pass through
