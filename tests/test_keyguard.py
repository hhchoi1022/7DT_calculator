"""Key-guessing guard of the Schedule viewer: free attempts, growing pauses, forgetting, independent scopes."""
from sevendt_calc.keyguard import KeyGuard


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_free_attempts_then_growing_pause():
    clock = Clock()
    g = KeyGuard(free=3, base=10, cap=35, forget=3600, clock=clock)
    for _ in range(3):
        assert g.wait('ip:a') == 0
        g.failed('ip:a')
    assert g.wait('ip:a') == 0                       # the free attempts cost nothing
    g.failed('ip:a')
    assert g.wait('ip:a') == 10                      # first pause
    clock.t += 10
    assert g.wait('ip:a') == 0
    g.failed('ip:a')
    assert g.wait('ip:a') == 20                      # doubles
    g.failed('ip:a')
    assert g.wait('ip:a') == 35                      # capped
    clock.t += 5
    assert g.wait('ip:a') == 30                      # counts down


def test_scopes_are_independent_and_the_longest_wait_wins():
    clock = Clock()
    g = KeyGuard(free=1, base=10, cap=600, forget=3600, clock=clock)
    g.failed('ip:a', 'pid:1')
    g.failed('ip:a')
    assert g.wait('pid:1') == 0 and g.wait('ip:a') == 10 and g.wait('pid:1', 'ip:a') == 10 and g.wait('ip:b') == 0
    g.succeeded('ip:a')
    assert g.wait('ip:a') == 0 and g.failures('ip:a') == 0 and g.failures('pid:1') == 1


def test_failures_are_forgotten():
    clock = Clock()
    g = KeyGuard(free=0, base=10, cap=600, forget=100, clock=clock)
    g.failed('ip:a')
    assert g.wait('ip:a') == 10 and g.failures('ip:a') == 1
    clock.t += 101
    assert g.wait('ip:a') == 0 and g.failures('ip:a') == 0
