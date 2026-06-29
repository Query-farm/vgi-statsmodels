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

import json
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
    "# statsmodels: Regression & Statistical Inference in SQL\n\n"
    "![statsmodels logo]"
    "(https://www.statsmodels.org/stable/_images/"
    "statsmodels-logo-v2-horizontal.svg)\n\n"
    "Run regression with full statistical inference and classic hypothesis "
    "tests directly in DuckDB SQL: fit OLS, logistic, and generalized linear "
    "models and get coefficient tables complete with standard errors, "
    "t/z-statistics, p-values, and 95% confidence intervals — no Python "
    "notebook required.\n\n"
    "This extension brings the proven statistics of the "
    "[statsmodels](https://www.statsmodels.org/stable/index.html) library "
    "([source on GitHub](https://github.com/statsmodels/statsmodels)) to "
    "anyone who already speaks SQL. It is built for data analysts, data "
    "scientists, and engineers who want trustworthy estimation and inference "
    "— effect sizes, significance, model fit, and stationarity checks — "
    "without exporting data to a separate stats environment. Every function "
    "takes a whole input relation as a `(SELECT ...)` subquery, so you model "
    "the result of any DuckDB query: joins, filters, window functions, and "
    "aggregates all flow straight into the fit.\n\n"
    "Under the hood each call buffers the input relation, builds the design "
    "matrices from a "
    "[Patsy](https://patsy.readthedocs.io/en/latest/) "
    "([source on GitHub](https://github.com/pydata/patsy)) `formula` (for the "
    "regressions) or from named column roles (for the tests), then runs the "
    "corresponding statsmodels routine once and returns the results as an "
    "ordinary SQL table you can join, filter, and persist. The familiar "
    "R-style formula syntax (`y ~ x1 + x2 + C(group)`) makes specifying "
    "interactions, transformations, and categorical encodings concise and "
    "readable.\n\n"
    "The function surface covers the everyday inference workflow. "
    "**Regression** — `ols` (ordinary least squares), `logit` (logistic "
    "regression), and `glm` (generalized linear models for the "
    "gaussian/binomial/poisson/gamma families) — returns a coefficient table "
    "with standard errors, t- or z-statistics, p-values, and 95% confidence "
    "intervals. **Model fit** — `model_stats` — reports whole-model "
    "diagnostics such as R-squared, adjusted R-squared, the F-test, AIC/BIC, "
    "and log-likelihood. **Hypothesis tests** — `ttest` (two-sample t-test "
    "for a difference in means) and `adfuller` (Augmented Dickey-Fuller test "
    "for a unit root) — answer whether two groups differ and whether a time "
    "series is stationary. Together they let you ask, in SQL alone, which "
    "predictors matter and by how much, whether an effect is significant, how "
    "well a model fits, and whether a series is stationary."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Regression (ols, logit, glm) and model-fit (model_stats) functions plus "
    "hypothesis tests (ttest, adfuller). Each consumes a SQL relation and a "
    "Patsy formula or column roles, returning a coefficient/statistic table."
)

_SCHEMA_DESCRIPTION_MD = (
    "Regression (OLS/Logit/GLM), whole-model fit statistics, and hypothesis "
    "tests (two-sample t-test, Augmented Dickey-Fuller) over SQL relations. "
    "Each function buffers a whole input relation passed as a `(SELECT ...)` "
    "subquery, then runs the statsmodels routine once from a Patsy formula "
    "(for the regressions) or named column roles (for the tests), returning a "
    "coefficient or statistic table. Use it to estimate effects, judge model "
    "fit, compare two groups, or check a series for stationarity — all in SQL."
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
        "vgi.keywords": json.dumps(
            [
                "statsmodels",
                "regression",
                "ols",
                "logit",
                "glm",
                "model statistics",
                "t-test",
                "adfuller",
                "hypothesis test",
                "p-value",
                "confidence interval",
                "inference",
                "stationarity",
                "statistics",
                "patsy formula",
            ]
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
                "vgi.keywords": json.dumps(
                    [
                        "regression",
                        "ols",
                        "logit",
                        "glm",
                        "model_stats",
                        "ttest",
                        "adfuller",
                        "inference",
                        "hypothesis test",
                        "statistics",
                        "time series",
                        "stationarity",
                    ]
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "statistics",
                "category": "regression-and-inference",
                "topic": "statistical-modeling",
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
