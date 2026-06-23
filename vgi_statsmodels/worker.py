"""VGI worker exposing regression / statistical inference to DuckDB/SQL.

Assembles the table functions in ``vgi_statsmodels`` into a single
``statsmodels`` catalog and provides the process entry point. The repo-root
``statsmodels_worker.py`` is a thin shim over this module for ``uv run``;
installed users get the ``vgi-statsmodels`` console script, which calls ``main``
here.

    ATTACH 'statsmodels' (TYPE vgi, LOCATION 'uv run statsmodels_worker.py');
    SELECT * FROM statsmodels.ols((SELECT y, x FROM d), formula := 'y ~ x');
"""

from __future__ import annotations

import sys

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_statsmodels.tables import TABLE_FUNCTIONS

_FUNCTIONS: list[type] = [*TABLE_FUNCTIONS]

_STATSMODELS_CATALOG = Catalog(
    name="statsmodels",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="Regression (OLS/Logit/GLM) and hypothesis tests (t-test, ADF) for SQL",
            functions=list(_FUNCTIONS),
        ),
    ],
)


class StatsmodelsWorker(Worker):
    """Worker process hosting the ``statsmodels`` catalog."""

    catalog = _STATSMODELS_CATALOG


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    StatsmodelsWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    StatsmodelsWorker.main()
