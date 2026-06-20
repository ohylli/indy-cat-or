"""Entry-point shim for the evaluate command.

The implementation lives in :mod:`calibration.evaluate`; this thin wrapper keeps
the documented ``uv run python scripts/evaluate.py`` invocation working, mirroring
``scripts/calibrate.py``.
"""

from __future__ import annotations

from calibration.evaluate import main

if __name__ == "__main__":
    main()
