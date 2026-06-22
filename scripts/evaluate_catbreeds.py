"""Entry-point shim for the cat-breeds evaluate command.

The implementation lives in :mod:`calibration.evaluate_catbreeds`; this thin
wrapper keeps the ``uv run python scripts/evaluate_catbreeds.py`` invocation
working, mirroring ``scripts/evaluate.py``.
"""

from __future__ import annotations

from calibration.evaluate_catbreeds import main

if __name__ == "__main__":
    main()
