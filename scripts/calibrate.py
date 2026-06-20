"""Entry-point shim for the calibrate command.

The implementation lives in :mod:`calibration.cli`; this thin wrapper keeps the
documented ``uv run python scripts/calibrate.py`` invocation working after the
calibration code moved into the ``scripts/calibration`` package.
"""

from __future__ import annotations

from calibration.cli import main

if __name__ == "__main__":
    main()
