"""Cross-platform startup helpers.

Importing this module reconfigures stdout/stderr to UTF-8 on Windows so
Rich's box-drawing characters, the ★ used in trait strength, and any
non-ASCII character text render correctly in cmd.exe / PowerShell.

The reconfigure call is wrapped: it requires Python 3.7+ (always true for
us) and the underlying stream must support it (TextIOWrapper does;
redirected pipes might not). Failures are swallowed silently — if encoding
is already correct or the stream isn't reconfigurable, we just skip.
"""
from __future__ import annotations

import sys


IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def setup_console_encoding() -> None:
    """Make stdout/stderr emit UTF-8 on Windows. No-op elsewhere."""
    if not IS_WINDOWS:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            # AttributeError: stream isn't a TextIOWrapper (e.g., redirected).
            # OSError/ValueError: stream isn't reconfigurable in current state.
            pass


# Run on import so any entry point that imports this module gets the fix.
setup_console_encoding()
