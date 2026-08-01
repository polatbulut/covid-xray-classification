"""Allow the package to be run as ``python -m covid_xray``."""

from __future__ import annotations

from covid_xray.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
