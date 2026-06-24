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

_SCHEMA_EXAMPLE_QUERIES = (
    "SELECT * FROM statsmodels.main.ols("
    "(SELECT * FROM (VALUES (1,5.1),(2,7.9),(3,11.2),(4,13.8)) AS t(x, y)), "
    "formula := 'y ~ x');\n"
    "SELECT * FROM statsmodels.main.model_stats("
    "(SELECT * FROM (VALUES (1,5.1),(2,7.9),(3,11.2),(4,13.8)) AS t(x, y)), "
    "formula := 'y ~ x');\n"
    "SELECT * FROM statsmodels.main.glm("
    "(SELECT * FROM (VALUES (1,1),(2,2),(3,2),(4,4),(5,5),(6,7),(7,9),(8,12)) AS t(x, y)), "
    "formula := 'y ~ x', family := 'poisson');\n"
    "SELECT * FROM statsmodels.main.ttest("
    "(SELECT * FROM (VALUES (10,'a'),(11,'a'),(20,'b'),(22,'b')) AS t(v, g)), "
    "\"column\" := 'v', \"group\" := 'g');"
)

_CATALOG_DESCRIPTION_LLM = (
    "Run regression with full statistical inference and classic hypothesis "
    "tests directly over SQL relations. Fit ordinary least squares (ols), "
    "logistic (logit), and generalized linear models (glm: "
    "gaussian/binomial/poisson/gamma) from a Patsy formula and get a "
    "coefficient table with standard errors, t/z statistics, p-values, and 95% "
    "confidence intervals; get whole-model fit statistics (model_stats: "
    "R-squared, AIC/BIC, F-test, log-likelihood); run a two-sample t-test "
    "(ttest) for a difference in means; and test a time series for a unit root "
    "with the Augmented Dickey-Fuller test (adfuller). Use it to answer 'which "
    "predictors matter and by how much', 'is this effect significant', 'how "
    "well does the model fit', 'do these two groups differ', and 'is this "
    "series stationary' — all in SQL, powered by statsmodels."
)

_CATALOG_DESCRIPTION_MD = (
    "# statsmodels\n\n"
    "Regression with full statistical inference and hypothesis tests for "
    "DuckDB/SQL, powered by [statsmodels](https://www.statsmodels.org/) and "
    "[Patsy](https://patsy.readthedocs.io/).\n\n"
    "Each function takes a whole input relation as a `(SELECT ...)` subquery "
    "plus a Patsy `formula` (or column roles) as named arguments.\n\n"
    "**Regression** (coefficient tables with std error, t/z, p-value, 95% CI): "
    "`ols`, `logit`, `glm`.\n\n"
    "**Model fit:** `model_stats` (R-squared, adjusted R-squared, F-test, "
    "AIC/BIC, log-likelihood).\n\n"
    "**Hypothesis tests:** `ttest` (two-sample t-test), `adfuller` "
    "(Augmented Dickey-Fuller stationarity test)."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Regression (ols, logit, glm) and model-fit (model_stats) functions plus "
    "hypothesis tests (ttest, adfuller). Each consumes a SQL relation and a "
    "Patsy formula or column roles, returning a coefficient/statistic table."
)

_SCHEMA_DESCRIPTION_MD = (
    "Regression (OLS/Logit/GLM), whole-model fit statistics, and hypothesis "
    "tests (two-sample t-test, Augmented Dickey-Fuller) over SQL relations."
)

_STATSMODELS_CATALOG = Catalog(
    name="statsmodels",
    default_schema="main",
    comment=(
        "statsmodels-powered regression and inference for DuckDB/SQL: OLS/Logit/GLM "
        "fits, model statistics, and t-test/ADF hypothesis tests"
    ),
    source_url="https://github.com/Query-farm/vgi-statsmodels",
    tags={
        "vgi.title": "Regression & Statistical Inference",
        "vgi.keywords": (
            "statsmodels, regression, ols, logit, glm, model statistics, t-test, "
            "adfuller, hypothesis test, p-value, confidence interval, inference, "
            "stationarity, statistics, patsy formula"
        ),
        "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-statsmodels/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-statsmodels/blob/main/README.md",
    },
    schemas=[
        Schema(
            name="main",
            comment="Regression (OLS/Logit/GLM) and hypothesis tests (t-test, ADF) for SQL",
            tags={
                "vgi.title": "statsmodels — main",
                "vgi.keywords": (
                    "regression, ols, logit, glm, model_stats, ttest, adfuller, "
                    "inference, hypothesis test, statistics, time series, stationarity"
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "statistics",
                "category": "regression-and-inference",
                "topic": "statistical-modeling",
                "vgi.source_url": ("https://github.com/Query-farm/vgi-statsmodels/blob/main/vgi_statsmodels/worker.py"),
                "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
                "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
            },
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
