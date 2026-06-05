"""Cross-platform non-blocking stdin reader.

Pattern: a single daemon thread that blocks on `sys.stdin.readline()` and
pushes each line into a `queue.Queue`. The main thread polls the queue with
`get_line_nowait()` and gets immediate response without blocking.

This works identically on macOS, Linux, and Windows — no `select`, no
`fcntl`, no `termios`, no platform-specific imports. Daemon = True so the
thread is killed when the process exits, even though it's blocked on
readline (the alternative — explicit termination — isn't possible from
another thread without closing stdin).

Lifecycle: built as a lazy singleton. First `get_reader()` call starts the
thread; subsequent calls return the same instance. The thread runs until
the process exits or stdin reaches EOF.

Use ONLY in flows that genuinely need non-blocking input (autonomous mode).
Plain blocking input (`Prompt.ask`, `input()`) works fine on every platform
without this and should remain the default.
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Optional


class StdinReader:
    """Drains stdin line-by-line into a queue, in the background.

    Once started, the daemon thread is permanent for the process. Lines
    accumulate in the queue between polls. EOF on stdin sets `at_eof` so
    callers can stop polling.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._eof = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # daemon=True: when the main thread exits, this thread is killed
        # even though it's blocked on readline. We can't cleanly stop a
        # blocking readline from outside the thread.
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="nora-stdin-reader",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except (ValueError, OSError):
                # Stdin closed or invalid — stop reading.
                self._eof = True
                return
            if not line:
                # EOF.
                self._eof = True
                return
            # Strip both kinds of line endings so callers get clean text
            # whether running on Unix or Windows.
            self._queue.put(line.rstrip("\r\n"))

    def get_line_nowait(self) -> Optional[str]:
        """Return the next available line, or None if none queued. Never blocks."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> int:
        """Discard all queued lines. Returns how many were dropped."""
        n = 0
        while True:
            try:
                self._queue.get_nowait()
                n += 1
            except queue.Empty:
                return n

    @property
    def at_eof(self) -> bool:
        return self._eof


# --- Lazy singleton -------------------------------------------------------

_reader: Optional[StdinReader] = None


def get_reader() -> StdinReader:
    """Return the process-wide StdinReader, starting it on first call.

    Subsequent calls return the same instance. Safe to call from any thread.
    """
    global _reader
    if _reader is None:
        _reader = StdinReader()
        _reader.start()
    return _reader
