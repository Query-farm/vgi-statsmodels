"""Regression and statistical inference as a VGI worker for DuckDB/SQL.

The implementation is split so each concern stays focused:

- ``stats``       -- pure statsmodels logic (OLS, Logit, GLM, t-test, ADF) over
  ``pandas`` frames + Patsy formulas; no Arrow or VGI dependency, directly
  unit-testable.
- ``buffering``   -- the single-bucket Sink+Source plumbing every function
  shares (buffer all input batches, then fit once).
- ``tables``      -- the VGI ``TableBufferingFunction`` wrappers: relation in
  via ``(SELECT ...)`` (``Arg(0)``), the Patsy formula / column roles / options
  as named string args.

``statsmodels_worker.py`` at the repo root assembles these into the
``statsmodels`` catalog and runs the worker over stdio (or HTTP).
"""

from __future__ import annotations

__version__ = "0.1.0"
