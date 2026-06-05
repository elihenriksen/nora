"""Lightweight per-step timing instrumentation.

Enabled by setting NORA_DEBUG_TIMING=1 in the environment. When off, the
timer object is a no-op so there's zero overhead in normal use.

Usage:

    from nora import timing
    t = timing.timer("character turn")
    do_thing()
    t.step("did thing")
    do_other()
    t.step("did other")
    t.done()
"""
from __future__ import annotations

import os
import sys
import time


ENABLED = bool(os.environ.get("NORA_DEBUG_TIMING"))


class _NoopTimer:
    def step(self, label: str) -> None: ...
    def done(self) -> None: ...


class _RealTimer:
    def __init__(self, label: str) -> None:
        self._start = time.monotonic()
        self._last = self._start
        self._label = label
        print(f"\n[timing] === {label} ===", file=sys.stderr, flush=True)

    def step(self, label: str) -> None:
        now = time.monotonic()
        delta = (now - self._last) * 1000
        total = (now - self._start) * 1000
        print(
            f"[timing] {delta:8.1f} ms  (Δ)   {total:8.1f} ms  (total)   {label}",
            file=sys.stderr, flush=True,
        )
        self._last = now

    def done(self) -> None:
        total = (time.monotonic() - self._start) * 1000
        print(
            f"[timing] === {self._label} done ({total:.1f} ms total) ===",
            file=sys.stderr, flush=True,
        )


def timer(label: str = "") -> "_RealTimer | _NoopTimer":
    return _RealTimer(label) if ENABLED else _NoopTimer()
