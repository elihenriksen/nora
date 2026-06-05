"""Allow `python -m nora` to launch the interactive menu.

For users who prefer the CLI, `nora` (installed by pyproject.scripts) is the
direct entry point and supports the same subcommands as `python -m nora.cli`.

Pass --debug to enable NORA_DEBUG for this session (also propagates to all
spawned subprocesses via env inheritance).
"""
from __future__ import annotations

import sys

# Import-time UTF-8 fix on Windows; no-op elsewhere. Must happen before any
# Rich output to make sure box-drawing characters render correctly.
from . import _platform  # noqa: F401
from .launcher import main


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
