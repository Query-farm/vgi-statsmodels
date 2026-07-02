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

from vgi_statsmodels.tables import (
    _COUNT_REL,
    _GROUP_REL,
    _LINEAR_REL,
    _LOGIT_REL,
    _SERIES_REL,
    TABLE_FUNCTIONS,
)

_FUNCTIONS: list[type] = [*TABLE_FUNCTIONS]

# ---------------------------------------------------------------------------
# Agent-suitability suite (VGI152). Each task's prompt inlines its own data so
# a simulated analyst can solve it end-to-end; the `reference_sql` is the
# grader's canonical solution and is deterministic (fixed data + rounding /
# integer outputs). Column-name-only or row-order differences are tolerated
# per-task via `ignore_column_names` / `unordered`.
# ---------------------------------------------------------------------------

_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "ols_coefficients",
            "prompt": (
                "Using the statsmodels worker, fit an ordinary least squares regression of y "
                "on x for these observations with columns (x, y): "
                "(1,5.1),(2,7.9),(3,11.2),(4,13.8),(5,17.1),(6,19.9),(7,23.2),(8,25.8). "
                "Return one row per model term (the intercept and x) with its estimated "
                "coefficient rounded to 2 decimal places."
            ),
            "reference_sql": (
                f"SELECT term, round(coef, 2) AS coef FROM statsmodels.main.ols({_LINEAR_REL}, formula := 'y ~ x')"
            ),
            "unordered": True,
        },
        {
            "name": "model_fit_r_squared",
            "prompt": (
                "For the same 8 observations with columns (x, y): "
                "(1,5.1),(2,7.9),(3,11.2),(4,13.8),(5,17.1),(6,19.9),(7,23.2),(8,25.8), "
                "how well does an OLS model of y on x fit the data? Report the R-squared "
                "goodness-of-fit statistic, rounded to 3 decimal places."
            ),
            "reference_sql": (
                "SELECT round(value, 3) AS r_squared "
                f"FROM statsmodels.main.model_stats({_LINEAR_REL}, formula := 'y ~ x') "
                "WHERE statistic = 'r_squared'"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "logit_effect_direction",
            "prompt": (
                "Fit a logistic regression of the binary outcome y on x for these rows with "
                "columns (x, y): (1,0),(2,0),(3,0),(4,1),(5,0),(6,1),(7,0),(8,1),(9,1),"
                "(10,1),(11,1),(12,1). Report the estimated log-odds coefficient on x, "
                "rounded to 2 decimal places."
            ),
            "reference_sql": (
                "SELECT round(coef, 2) AS coef "
                f"FROM statsmodels.main.logit({_LOGIT_REL}, formula := 'y ~ x') "
                "WHERE term = 'x'"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "poisson_glm_rate",
            "prompt": (
                "These rows are counts y rising with x, columns (x, y): "
                "(1,1),(2,2),(3,2),(4,4),(5,5),(6,7),(7,9),(8,12). Fit a Poisson "
                "generalized linear model of y on x and report the coefficient on x "
                "(on the log link scale), rounded to 2 decimal places."
            ),
            "reference_sql": (
                "SELECT round(coef, 2) AS coef "
                f"FROM statsmodels.main.glm({_COUNT_REL}, formula := 'y ~ x', family := 'poisson') "
                "WHERE term = 'x'"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "two_group_ttest",
            "prompt": (
                "Two groups 'a' and 'b' have these measurements, columns (v, g): "
                "(10,'a'),(11,'a'),(9,'a'),(12,'a'),(10,'a'),"
                "(20,'b'),(22,'b'),(19,'b'),(21,'b'),(20,'b'). Run a two-sample t-test to "
                "compare the mean of v across the two groups and report the two-sided "
                "p-value, rounded to 6 decimal places."
            ),
            "reference_sql": (
                "SELECT round(p_value, 6) AS p_value "
                f"FROM statsmodels.main.ttest({_GROUP_REL}, \"column\" := 'v', \"group\" := 'g')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "adf_stationarity",
            "prompt": (
                "Run the Augmented Dickey-Fuller unit-root test on this time series. The rows "
                "have columns (t, v) and must be tested in ascending t order: "
                "(0,10.0),(1,13.82),(2,16.66),(3,14.87),(4,14.38),(5,12.71),(6,7.79),"
                "(7,6.64),(8,7.02),(9,6.14),(10,9.6),(11,13.56),(12,13.97),(13,15.99),"
                "(14,16.27),(15,12.06),(16,10.13),(17,8.5),(18,5.1),(19,6.4),(20,9.32),"
                "(21,10.17),(22,13.96),(23,16.72),(24,14.83),(25,14.25),(26,12.54),"
                "(27,7.64),(28,6.56),(29,7.04). Report how many lags the test chose "
                "(used_lag) and how many observations it used (n_obs)."
            ),
            "reference_sql": (
                f"SELECT used_lag, n_obs FROM statsmodels.main.adfuller({_SERIES_REL}, \"column\" := 'v')"
            ),
        },
    ]
)

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
    "The capabilities group into three areas. **Regression** estimates how "
    "predictors drive an outcome — linear for continuous responses, logistic "
    "for binary events, and generalized linear models for counts and other "
    "exponential-family outcomes — always returning coefficients with standard "
    "errors, t- or z-statistics, p-values, and 95% confidence intervals. "
    "**Model fit** reports whole-model diagnostics such as R-squared, adjusted "
    "R-squared, the F-test, AIC/BIC, and log-likelihood so you can judge how "
    "well a linear model explains the data. **Hypothesis tests** compare two "
    "group means and check a time series for a unit root (stationarity). "
    "Together they let you ask, in SQL alone, which predictors matter and by "
    "how much, whether an effect is significant, how well a model fits, and "
    "whether a series is stationary. List the schema to see the individual "
    "functions and their arguments."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Regression (ols, logit, glm) and model-fit (model_stats) functions plus "
    "hypothesis tests (ttest, adfuller). Each consumes a SQL relation and a "
    "Patsy formula or column roles, returning a coefficient/statistic table."
)

_SCHEMA_DESCRIPTION_MD = (
    "## Regression & inference over SQL relations\n\n"
    "This schema fits statistical models and runs classic hypothesis tests "
    "directly over DuckDB relations, powered by "
    "[statsmodels](https://www.statsmodels.org/). Every function takes a whole "
    "input relation as a `(SELECT ...)` subquery plus either a Patsy formula "
    "(`y ~ x1 + x2`) or named column roles, buffers the rows, runs the "
    "routine once, and returns a coefficient or statistic table you can join, "
    "filter, and persist.\n\n"
    "It covers three areas:\n\n"
    "- **Regression** — linear, logistic, and generalized linear fits that "
    "return a coefficient table with standard errors, t/z-statistics, "
    "p-values, and 95% confidence intervals.\n"
    "- **Model fit** — whole-model diagnostics for a linear fit, such as "
    "R-squared, the F-test, and AIC/BIC.\n"
    "- **Hypothesis tests** — compare two group means, and test a time series "
    "for a unit root (stationarity).\n\n"
    "Reach for it to estimate effects, judge model fit, compare two groups, or "
    "check a series for stationarity — all without leaving SQL."
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
        "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
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
                # VGI413 navigation/SEO registry; each function carries a
                # matching vgi.category (see tables.py). Order = display order.
                "vgi.categories": json.dumps(
                    [
                        {
                            "name": "Regression",
                            "description": (
                                "Fit linear, logistic, and generalized linear models and read "
                                "their coefficient tables with full statistical inference."
                            ),
                        },
                        {
                            "name": "Model Fit",
                            "description": (
                                "Whole-model goodness-of-fit diagnostics (R-squared, F-test, "
                                "AIC/BIC) for a fitted linear model."
                            ),
                        },
                        {
                            "name": "Hypothesis Tests",
                            "description": (
                                "Classic significance tests: compare two group means and check "
                                "a time series for a unit root (stationarity)."
                            ),
                        },
                    ]
                ),
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
