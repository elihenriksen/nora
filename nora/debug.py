"""Lightweight debug logging.

Gated on NORA_DEBUG=1 in the environment. When off, the calls are no-ops and
cost nothing. Output goes to stderr so it doesn't interleave with character
text on stdout.

Usage:
    from nora import debug
    debug.log("join", f"running check on {len(absent)} candidates")
"""
from __future__ import annotations

import os
import sys


ENABLED = bool(os.environ.get("NORA_DEBUG"))


def log(category: str, message: str) -> None:
    if not ENABLED:
        return
    print(f"[debug:{category}] {message}", file=sys.stderr, flush=True)
