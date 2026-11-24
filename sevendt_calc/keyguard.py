"""
Slows down key guessing on the public Schedule viewer page.

The Project key travels inside the Streamlit websocket, so a request-rate limit in nginx never sees the
attempts. This guard counts the wrong keys per scope (the client's IP address, the Project ID tried) and,
once a scope has used up its free attempts, demands a growing pause before the next one. The guard lives
in the Streamlit process, shared by every session, so reconnecting does not reset it.
"""
from __future__ import annotations

import threading
import time

FREE_ATTEMPTS = 8        # wrong keys per scope before a pause is demanded
BASE_DELAY = 10.0        # seconds after the first failure beyond the free ones; doubles with each further failure
MAX_DELAY = 600.0        # longest pause demanded
FORGET_AFTER = 3600.0    # a failure older than this no longer counts


class KeyGuard:
    def __init__(self, free: int = FREE_ATTEMPTS, base: float = BASE_DELAY, cap: float = MAX_DELAY, forget: float = FORGET_AFTER, clock=time.monotonic):
        self.free, self.base, self.cap, self.forget, self.clock = free, base, cap, forget, clock
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, scope: str, now: float) -> list[float]:
        stamps = [t for t in self._failures.get(scope, ()) if now - t < self.forget]
        if stamps:
            self._failures[scope] = stamps
        else:
            self._failures.pop(scope, None)
        return stamps

    def wait(self, *scopes: str) -> float:
        """Seconds before the next attempt in any of the scopes is allowed (0 = now)."""
        now = self.clock()
        longest = 0.0
        with self._lock:
            for scope in scopes:
                stamps = self._recent(scope, now)
                extra = len(stamps) - self.free
                if extra <= 0:
                    continue
                delay = min(self.cap, self.base * 2 ** (extra - 1))
                longest = max(longest, stamps[-1] + delay - now)
        return max(0.0, longest)

    def failed(self, *scopes: str) -> None:
        now = self.clock()
        with self._lock:
            for scope in scopes:
                self._failures.setdefault(scope, []).append(now)

    def succeeded(self, *scopes: str) -> None:
        with self._lock:
            for scope in scopes:
                self._failures.pop(scope, None)

    def failures(self, scope: str) -> int:
        with self._lock:
            return len(self._recent(scope, self.clock()))


GUARD = KeyGuard()      # one per process: Streamlit serves every session from the same process
