"""Regression / inference table functions for DuckDB via VGI.

Each function consumes a *whole* input relation -- passed as a ``(SELECT ...)``
subquery (positional ``Arg(0)``) -- and a Patsy ``formula`` / column roles /
options as NAMED string args (``formula := 'y ~ x1 + x2'``, ``family :=
'poisson'``, ``column := 'value'``, ``"group" := 'arm'``). Because a fit / test
needs every row, these are buffering (Sink+Source) functions: they sink all
input batches, then run the statsmodels routine once in finalize.

    SELECT * FROM statsmodels.ols((SELECT y, x FROM d), formula := 'y ~ x');
    SELECT * FROM statsmodels.model_stats((SELECT y, x FROM d), formula := 'y ~ x');
    SELECT * FROM statsmodels.logit((SELECT y, x FROM d), formula := 'y ~ x');
    SELECT * FROM statsmodels.glm((SELECT y, x FROM d), formula := 'y ~ x', family := 'poisson');
    SELECT * FROM statsmodels.ttest((SELECT v, g FROM d), column := 'v', "group" := 'g');
    SELECT * FROM statsmodels.adfuller((SELECT v FROM d ORDER BY t), column := 'v');

Every column in the relation is available to the Patsy formula by name. See
``vgi_statsmodels.stats`` for the math, the formula syntax, and conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc import OutputCollector

from . import stats
from .buffering import DrainState, SinkBuffer
from .meta import object_tags
from .schema_utils import field as sfield
from .schema_utils import result_columns_schema

# The closed set of GLM error families glm() accepts, sourced from the single
# registry in stats so the discoverable `choices` constraint (VGI317) and the
# runtime behaviour can never drift. Sorted for a stable, deterministic order.
_GLM_FAMILY_NAMES: list[str] = list(stats.GLM_FAMILY_NAMES)

# ---------------------------------------------------------------------------
# Self-contained example relations
#
# Every example below inlines its own data as a `(VALUES ...)` subquery so the
# linter (which EXECUTES examples) can run each one against the attached worker
# with no pre-existing tables. The data is small but sufficient for each fit /
# test to converge cleanly.
# ---------------------------------------------------------------------------

# y ~ x, eight points lying close to y = 2 + 3x (used by ols / model_stats).
_LINEAR_REL = (
    "(SELECT * FROM (VALUES (1,5.1),(2,7.9),(3,11.2),(4,13.8),(5,17.1),(6,19.9),(7,23.2),(8,25.8)) AS t(x, y))"
)

# Binary y with x-overlap so logit converges without perfect separation.
_LOGIT_REL = (
    "(SELECT * FROM (VALUES (1,0),(2,0),(3,0),(4,1),(5,0),(6,1),(7,0),(8,1),(9,1),(10,1),(11,1),(12,1)) AS t(x, y))"
)

# Count y rising with x (Poisson-shaped) for the GLM example.
_COUNT_REL = "(SELECT * FROM (VALUES (1,1),(2,2),(3,2),(4,4),(5,5),(6,7),(7,9),(8,12)) AS t(x, y))"

# Two clearly-separated groups for the two-sample t-test.
_GROUP_REL = (
    "(SELECT * FROM (VALUES "
    "(10,'a'),(11,'a'),(9,'a'),(12,'a'),(10,'a'),"
    "(20,'b'),(22,'b'),(19,'b'),(21,'b'),(20,'b')"
    ") AS t(v, g))"
)

# A 30-point ordered series for the Augmented Dickey-Fuller test.
_SERIES_REL = (
    "(SELECT * FROM (VALUES "
    "(0,10.0),(1,13.82),(2,16.66),(3,14.87),(4,14.38),(5,12.71),(6,7.79),"
    "(7,6.64),(8,7.02),(9,6.14),(10,9.6),(11,13.56),(12,13.97),(13,15.99),"
    "(14,16.27),(15,12.06),(16,10.13),(17,8.5),(18,5.1),(19,6.4),(20,9.32),"
    "(21,10.17),(22,13.96),(23,16.72),(24,14.83),(25,14.25),(26,12.54),"
    "(27,7.64),(28,6.56),(29,7.04)"
    ") AS t(t, v) ORDER BY t)"
)

# ---------------------------------------------------------------------------
# Composed examples: two of this worker's functions in one query
#
# Each is defined once here and referenced from both the driving function's
# `examples` list and the schema-level vgi.example_queries (worker.py), so the
# two carriers can never drift. The data sit in a CTE so both function calls
# read the same relation.
# ---------------------------------------------------------------------------

# Coefficient table (ols) + whole-model fit (model_stats) side by side: every
# term's estimate and p-value next to the model's adjusted R-squared and whether
# its overall F-test is significant. Driven by ols.
COMPOSED_OLS_WITH_FIT = FunctionExample(
    sql=(
        "WITH ads AS (SELECT * FROM (VALUES "
        "(1,8.5,12.6),(2,10.0,12.7),(3,10.4,15.6),(4,8.1,24.8),(5,8.6,28.7),(6,11.7,26.0),"
        "(7,8.3,35.2),(8,8.5,40.6),(9,11.8,36.6),(10,10.5,41.2),(11,9.5,50.5),(12,10.0,51.7)"
        ") AS t(spend, price, sales)), "
        "fit AS (SELECT max(value) FILTER (WHERE statistic = 'adj_r_squared') AS adj_r_squared, "
        "max(value) FILTER (WHERE statistic = 'f_pvalue') AS f_pvalue "
        "FROM statsmodels.main.model_stats((SELECT * FROM ads), formula := 'sales ~ spend + price')) "
        "SELECT o.term, round(o.coef, 3) AS coef, round(o.p_value, 6) AS p_value, "
        "round(fit.adj_r_squared, 4) AS adj_r_squared, fit.f_pvalue < 0.001 AS model_significant "
        "FROM statsmodels.main.ols((SELECT * FROM ads), formula := 'sales ~ spend + price') AS o "
        "CROSS JOIN fit ORDER BY o.term"
    ),
    description=(
        "Regress sales on ad spend and price with ols, and attach model_stats' adjusted "
        "R-squared and overall F-test verdict to every coefficient row"
    ),
)

# Linear vs quadratic fit of the same curved data, compared on adjusted
# R-squared / AIC / BIC (two model_stats calls joined by statistic). Driven by
# model_stats.
COMPOSED_MODEL_COMPARISON = FunctionExample(
    sql=(
        "WITH curve AS (SELECT * FROM (VALUES "
        "(1,2.0),(2,4.8),(3,7.6),(4,10.2),(5,14.0),(6,20.4),(7,27.0),(8,32.4),(9,39.2),(10,48.4)"
        ") AS t(x, y)), "
        "lin AS (SELECT statistic, value FROM statsmodels.main.model_stats("
        "(SELECT * FROM curve), formula := 'y ~ x')), "
        "quad AS (SELECT statistic, value FROM statsmodels.main.model_stats("
        "(SELECT * FROM curve), formula := 'y ~ x + I(x**2)')) "
        "SELECT lin.statistic, round(lin.value, 4) AS linear, round(quad.value, 4) AS quadratic "
        "FROM lin JOIN quad USING (statistic) "
        "WHERE lin.statistic IN ('adj_r_squared', 'aic', 'bic') ORDER BY lin.statistic"
    ),
    description=(
        "Compare a linear and a quadratic model of the same data side by side on adjusted "
        "R-squared, AIC and BIC (the quadratic wins: higher adj R-squared, lower AIC/BIC)"
    ),
)

# Engle-Granger style cointegration check: ols estimates the long-run relation
# between two trending series, then adfuller tests the regression residuals for
# a unit root. Both series are individually non-stationary (ADF p ~ 0.93); the
# residuals are stationary, so the pair moves together. Driven by adfuller.
COMPOSED_RESIDUAL_ADF = FunctionExample(
    sql=(
        "WITH pair AS (SELECT * FROM (VALUES "
        "(0,20.0,41.57),(1,20.3,42.14),(2,20.03,40.82),(3,19.14,39.83),(4,18.68,39.68),"
        "(5,17.69,35.95),(6,17.75,37.52),(7,19.09,39.75),(8,18.6,38.32),(9,17.98,38.66),"
        "(10,18.47,38.9),(11,18.82,38.42),(12,18.93,39.4),(13,18.0,37.85),(14,17.97,37.33),"
        "(15,18.66,39.23),(16,17.32,36.1),(17,16.86,35.62),(18,14.96,32.28),(19,13.67,28.43),"
        "(20,11.83,25.28),(21,11.59,24.4),(22,10.33,22.24),(23,10.6,21.99),(24,10.76,22.67),"
        "(25,10.57,22.52),(26,8.05,18.14),(27,7.51,17.21),(28,7.46,15.63),(29,7.58,16.18),"
        "(30,6.05,13.99),(31,5.57,11.44),(32,4.59,10.4),(33,3.78,9.0),(34,4.84,11.93),"
        "(35,4.04,9.99),(36,4.0,9.3),(37,4.89,11.06),(38,4.3,9.95),(39,4.19,10.79)"
        ") AS t(t, x, y)), "
        "fit AS (SELECT max(coef) FILTER (WHERE term = 'Intercept') AS a, "
        "max(coef) FILTER (WHERE term = 'x') AS b "
        "FROM statsmodels.main.ols((SELECT x, y FROM pair), formula := 'y ~ x')) "
        "SELECT round(statistic, 3) AS adf_statistic, p_value < 0.01 AS residuals_stationary, used_lag, n_obs "
        "FROM statsmodels.main.adfuller("
        "(SELECT pair.y - (fit.a + fit.b * pair.x) AS resid FROM pair, fit ORDER BY pair.t), "
        "\"column\" := 'resid')"
    ),
    description=(
        "Cointegration check in the Engle-Granger style: fit y ~ x with ols, then run adfuller "
        "on the regression residuals; a small p-value means the two trending series move together"
    ),
)

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

_OLS_SCHEMA = pa.schema(
    [
        sfield("term", pa.string(), "Model term (a predictor or 'Intercept').", nullable=False),
        sfield("coef", pa.float64(), "Estimated coefficient."),
        sfield("std_err", pa.float64(), "Standard error of the coefficient."),
        sfield("t_value", pa.float64(), "t-statistic (coef / std_err)."),
        sfield("p_value", pa.float64(), "Two-sided p-value for H0: coef = 0."),
        sfield("ci_lower", pa.float64(), "Lower bound of the 95% confidence interval."),
        sfield("ci_upper", pa.float64(), "Upper bound of the 95% confidence interval."),
    ]
)

_MODEL_STATS_SCHEMA = pa.schema(
    [
        sfield("statistic", pa.string(), "Fit-statistic name.", nullable=False),
        sfield("value", pa.float64(), "Statistic value."),
    ]
)

_LOGIT_SCHEMA = pa.schema(
    [
        sfield("term", pa.string(), "Model term (a predictor or 'Intercept').", nullable=False),
        sfield("coef", pa.float64(), "Estimated log-odds coefficient."),
        sfield("std_err", pa.float64(), "Standard error of the coefficient."),
        sfield("z_value", pa.float64(), "Wald z-statistic (coef / std_err)."),
        sfield("p_value", pa.float64(), "Two-sided p-value for H0: coef = 0."),
        sfield("ci_lower", pa.float64(), "Lower bound of the 95% confidence interval."),
        sfield("ci_upper", pa.float64(), "Upper bound of the 95% confidence interval."),
        sfield("odds_ratio", pa.float64(), "Odds ratio exp(coef); >1 raises odds, <1 lowers."),
    ]
)

_GLM_SCHEMA = pa.schema(
    [
        sfield("term", pa.string(), "Model term (a predictor or 'Intercept').", nullable=False),
        sfield("coef", pa.float64(), "Estimated coefficient (on the link scale)."),
        sfield("std_err", pa.float64(), "Standard error of the coefficient."),
        sfield("z_value", pa.float64(), "Wald z-statistic (coef / std_err)."),
        sfield("p_value", pa.float64(), "Two-sided p-value for H0: coef = 0."),
        sfield("ci_lower", pa.float64(), "Lower bound of the 95% confidence interval."),
        sfield("ci_upper", pa.float64(), "Upper bound of the 95% confidence interval."),
    ]
)

_TTEST_SCHEMA = pa.schema(
    [
        sfield("statistic", pa.float64(), "Two-sample t-statistic.", nullable=False),
        sfield("p_value", pa.float64(), "Two-sided p-value for equal means."),
        sfield("df", pa.float64(), "Degrees of freedom."),
        sfield("mean_diff", pa.float64(), "mean(group0) - mean(group1)."),
    ]
)

_ADF_SCHEMA = pa.schema(
    [
        sfield("statistic", pa.float64(), "Augmented Dickey-Fuller test statistic.", nullable=False),
        sfield("p_value", pa.float64(), "MacKinnon p-value; small => reject unit root (stationary)."),
        sfield("used_lag", pa.int32(), "Number of lags chosen by AIC."),
        sfield("n_obs", pa.int32(), "Observations used after lagging/differencing."),
    ]
)


# ---------------------------------------------------------------------------
# Argument dataclasses -- (SELECT ...) relation as Arg(0), options as named args
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class FormulaArgs:
    """Arguments for the formula-based fits (ols, model_stats, logit)."""

    data: Annotated[TableInput, Arg(0, doc="Relation; every column is available to the formula.")]
    formula: Annotated[str, Arg("formula", default="", doc="Patsy formula, e.g. 'y ~ x1 + x2'.")]


@dataclass(slots=True, frozen=True)
class GlmArgs:
    """Arguments for the glm fit (formula plus an error family)."""

    data: Annotated[TableInput, Arg(0, doc="Relation; every column is available to the formula.")]
    formula: Annotated[str, Arg("formula", default="", doc="Patsy formula, e.g. 'y ~ x1 + x2'.")]
    family: Annotated[
        str,
        Arg(
            "family",
            default="gaussian",
            doc="Error distribution for the response, one of the supported GLM families (default gaussian).",
            choices=_GLM_FAMILY_NAMES,
        ),
    ]


@dataclass(slots=True, frozen=True)
class TTestArgs:
    """Arguments for the two-sample t-test (value column and group column)."""

    data: Annotated[TableInput, Arg(0, doc="Relation containing the value and group columns.")]
    column: Annotated[
        str,
        Arg("column", default="value", doc="Name of the value column whose means are compared across the two groups."),
    ]
    group: Annotated[str, Arg("group", default="group", doc="Grouping column with two levels.")]


@dataclass(slots=True, frozen=True)
class AdfArgs:
    """Arguments for the Augmented Dickey-Fuller test (ordered series column)."""

    data: Annotated[TableInput, Arg(0, doc="Ordered relation containing the series column.")]
    column: Annotated[
        str, Arg("column", default="value", doc="Name of the time-ordered series column to test for a unit root.")
    ]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class Ols(SinkBuffer[FormulaArgs, DrainState]):
    """OLS regression over a buffered relation; one row per model term."""

    FunctionArguments: ClassVar[type] = FormulaArgs

    class Meta:
        """VGI function metadata for ols."""

        name = "ols"
        description = (
            "Ordinary least squares regression with full inference: emits "
            "(term, coef, std_err, t_value, p_value, ci_lower, ci_upper), one "
            "row per term including the Intercept. Pass a Patsy formula."
        )
        categories = ["statistics", "regression"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT term, round(coef, 3) AS coef, round(p_value, 6) AS p_value "
                    f"FROM statsmodels.main.ols({_LINEAR_REL}, formula := 'y ~ x') ORDER BY term"
                ),
                description="Fit y ~ x by OLS and read each term's coefficient and p-value",
            ),
            COMPOSED_OLS_WITH_FIT,
        ]
        tags = {
            **object_tags(
                title="OLS Linear Regression",
                doc_llm=(
                    "# ols\n\n"
                    "Fit an **ordinary least squares** linear regression over a whole SQL "
                    "relation and return its coefficient table with full statistical "
                    "inference. The first positional argument is the data, passed as a "
                    "`(SELECT ...)` subquery; the model is given as a Patsy `formula` named "
                    "argument such as `'y ~ x1 + x2'`. Every column of the relation is "
                    "available to the formula by name.\n\n"
                    "**Use it** to estimate how continuous predictors drive a continuous "
                    "outcome and to judge which terms matter: each row is one model term "
                    "(predictors plus the `Intercept`) with its coefficient, standard error, "
                    "t-statistic, two-sided p-value, and 95% confidence interval. For "
                    "whole-model fit quality (R-squared, F-test, AIC/BIC) use `model_stats` "
                    "with the same formula.\n\n"
                    "**Edge cases:** a malformed formula, a missing or non-numeric column, "
                    "empty input, or a singular (rank-deficient) design raises a clean "
                    "`StatsError` rather than crashing."
                ),
                doc_md=(
                    "## OLS regression\n\n"
                    "Ordinary least squares regression over a SQL relation, powered by "
                    "[statsmodels](https://www.statsmodels.org/). Pass the data as a "
                    "`(SELECT ...)` subquery and the model as a Patsy `formula`.\n\n"
                    "Returns one row per model term (`Intercept` and each predictor) with the "
                    "estimated coefficient, standard error, t-value, p-value, and 95% "
                    "confidence bounds.\n\n"
                    "Call it as `ols(<relation>, formula := 'y ~ x')`; a ready-to-run version "
                    "is in this function's example queries. See `model_stats` for whole-model "
                    "fit statistics on the same formula."
                ),
                keywords=[
                    "ols",
                    "ordinary least squares",
                    "linear regression",
                    "regression",
                    "coefficients",
                    "p-value",
                    "confidence interval",
                    "standard error",
                    "t-statistic",
                    "patsy formula",
                    "fit",
                    "predictors",
                ],
            ),
            "vgi.category": "Regression",
            # VGI307/VGI414: structured static result schema, derived from the output
            # pa.schema so it stays in lockstep with what finalize() emits (VGI910).
            "vgi.result_columns_schema": result_columns_schema(_OLS_SCHEMA),
            # VGI515: carry the example's description through the vgi.example_queries
            # tag (the native duckdb_functions().examples column drops descriptions).
            # Derived from `examples` above so the two carriers can never drift.
            "vgi.example_queries": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
            # VGI509: worker-wide guaranteed-runnable examples. The same
            # self-contained (offline, deterministic) queries, including the composed
            # ols + model_stats one, carried on the vgi.executable_examples tag.
            "vgi.executable_examples": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
        }

    @classmethod
    def on_bind(cls, params: BindParams[FormulaArgs]) -> BindResponse:
        """Declare the fixed output schema for this function."""
        return BindResponse(output_schema=_OLS_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FormulaArgs]) -> DrainState:
        """Start each finalize stream with a fresh single-emit cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FormulaArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the OLS fit over the buffered relation and emit its term table."""
        if state.done:
            out.finish()
            return
        state.done = True
        df = cls.buffered_frame(params)
        result = stats.ols(df, formula=params.args.formula)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class ModelStats(SinkBuffer[FormulaArgs, DrainState]):
    """Whole-model OLS fit statistics; one row per statistic."""

    FunctionArguments: ClassVar[type] = FormulaArgs

    class Meta:
        """VGI function metadata for model_stats."""

        name = "model_stats"
        description = (
            "OLS whole-model fit statistics as (statistic, value) rows: "
            "r_squared, adj_r_squared, f_statistic, f_pvalue, aic, bic, "
            "log_likelihood, nobs, df_model, df_resid. Pass a Patsy formula."
        )
        categories = ["statistics", "regression"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT round(value, 4) AS r_squared "
                    f"FROM statsmodels.main.model_stats({_LINEAR_REL}, formula := 'y ~ x') "
                    "WHERE statistic = 'r_squared'"
                ),
                description="Read the R-squared goodness-of-fit for an OLS model of y on x",
            ),
            COMPOSED_MODEL_COMPARISON,
        ]
        tags = {
            **object_tags(
                title="OLS Whole-Model Fit Statistics",
                doc_llm=(
                    "# model_stats\n\n"
                    "Fit the same **OLS** model as `ols` but return the **whole-model fit "
                    "statistics** instead of the per-term coefficient table. Pass the data as a "
                    "`(SELECT ...)` subquery and a Patsy `formula` named argument.\n\n"
                    "**Use it** to judge how well a linear model explains the data: the result "
                    "is one `(statistic, value)` row per metric, covering `r_squared`, "
                    "`adj_r_squared`, `f_statistic`, `f_pvalue`, `aic`, `bic`, `log_likelihood`, "
                    "`nobs`, `df_model`, and `df_resid`. Reach for `ols` (same formula) when you "
                    "need individual coefficients and their significance instead.\n\n"
                    "**Edge cases:** a bad formula, a missing or non-numeric column, empty input, "
                    "or a singular design raises a clean `StatsError`."
                ),
                doc_md=(
                    "## OLS model fit statistics\n\n"
                    "Whole-model goodness-of-fit metrics for an OLS regression, powered by "
                    "[statsmodels](https://www.statsmodels.org/). Same inputs as `ols` (a "
                    "`(SELECT ...)` relation and a Patsy `formula`), but the output describes "
                    "the model as a whole rather than each term.\n\n"
                    "Returns one row per statistic: R-squared and adjusted R-squared, the "
                    "overall F-test (`f_statistic`, `f_pvalue`), `aic`/`bic`, log-likelihood, "
                    "and the observation / degrees-of-freedom counts.\n\n"
                    "Call it as `model_stats(<relation>, formula := 'y ~ x')` and filter by "
                    "`statistic`; a ready-to-run version is in this function's example queries."
                ),
                keywords=[
                    "model statistics",
                    "r-squared",
                    "adjusted r-squared",
                    "goodness of fit",
                    "f-test",
                    "f-statistic",
                    "aic",
                    "bic",
                    "log-likelihood",
                    "ols",
                    "regression",
                    "model fit",
                    "diagnostics",
                ],
            ),
            "vgi.category": "Model Fit",
            # VGI307/VGI414: structured static result schema derived from the output schema.
            "vgi.result_columns_schema": result_columns_schema(_MODEL_STATS_SCHEMA),
            # VGI515: described examples carried via vgi.example_queries (see ols).
            "vgi.example_queries": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
        }

    @classmethod
    def on_bind(cls, params: BindParams[FormulaArgs]) -> BindResponse:
        """Declare the fixed output schema for this function."""
        return BindResponse(output_schema=_MODEL_STATS_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FormulaArgs]) -> DrainState:
        """Start each finalize stream with a fresh single-emit cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FormulaArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the OLS fit and emit its whole-model statistics."""
        if state.done:
            out.finish()
            return
        state.done = True
        df = cls.buffered_frame(params)
        result = stats.model_stats(df, formula=params.args.formula)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class Logit(SinkBuffer[FormulaArgs, DrainState]):
    """Logistic regression over a buffered relation; one row per model term."""

    FunctionArguments: ClassVar[type] = FormulaArgs

    class Meta:
        """VGI function metadata for logit."""

        name = "logit"
        description = (
            "Logistic (binary-outcome) regression with full inference: emits "
            "(term, coef, std_err, z_value, p_value, ci_lower, ci_upper, "
            "odds_ratio). Response must be binary. Pass a Patsy formula."
        )
        categories = ["statistics", "regression"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT term, round(coef, 3) AS coef, round(odds_ratio, 3) AS odds_ratio "
                    f"FROM statsmodels.main.logit({_LOGIT_REL}, formula := 'y ~ x') WHERE term = 'x'"
                ),
                description="Logistic fit of binary y on x; read the x log-odds coefficient and odds ratio",
            )
        ]
        tags = {
            **object_tags(
                title="Logistic Regression (Logit)",
                doc_llm=(
                    "# logit\n\n"
                    "Fit a **binary logistic regression** over a whole SQL relation and return "
                    "its coefficient table on the log-odds scale, with full inference and a "
                    "convenience odds ratio per term. Pass the data as a `(SELECT ...)` subquery "
                    "and a Patsy `formula` named argument; the response on the left of the "
                    "formula must be binary (two distinct values, e.g. 0/1).\n\n"
                    "**Use it** when the outcome is a yes/no event and you want to know how each "
                    "predictor shifts the odds: each row is one model term with its log-odds "
                    "coefficient, standard error, Wald z-statistic, p-value, 95% confidence "
                    "interval, and `odds_ratio` (= `exp(coef)`; >1 raises the odds, <1 lowers "
                    "them).\n\n"
                    "**Edge cases:** a non-binary response, perfect separation, a singular "
                    "design, a bad formula, a missing/non-numeric column, or empty input raises "
                    "a clean `StatsError`."
                ),
                doc_md=(
                    "## Logistic regression\n\n"
                    "Binary logistic (logit) regression over a SQL relation, powered by "
                    "[statsmodels](https://www.statsmodels.org/). Pass the data as a "
                    "`(SELECT ...)` subquery and the model as a Patsy `formula` whose response "
                    "is binary.\n\n"
                    "Returns one row per model term with the log-odds coefficient, standard "
                    "error, Wald z-value, p-value, 95% confidence interval, and the odds ratio "
                    "`exp(coef)`.\n\n"
                    "Call it as `logit(<relation>, formula := 'y ~ x')` with a binary response; "
                    "a ready-to-run version is in this function's example queries."
                ),
                keywords=[
                    "logit",
                    "logistic regression",
                    "binary classification",
                    "log-odds",
                    "odds ratio",
                    "z-statistic",
                    "p-value",
                    "classification",
                    "probability",
                    "regression",
                    "patsy formula",
                ],
            ),
            "vgi.category": "Regression",
            # VGI307/VGI414: structured static result schema derived from the output schema.
            "vgi.result_columns_schema": result_columns_schema(_LOGIT_SCHEMA),
            # VGI515: described examples carried via vgi.example_queries (see ols).
            "vgi.example_queries": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
        }

    @classmethod
    def on_bind(cls, params: BindParams[FormulaArgs]) -> BindResponse:
        """Declare the fixed output schema for this function."""
        return BindResponse(output_schema=_LOGIT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FormulaArgs]) -> DrainState:
        """Start each finalize stream with a fresh single-emit cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FormulaArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the logistic fit over the buffered relation and emit its term table."""
        if state.done:
            out.finish()
            return
        state.done = True
        df = cls.buffered_frame(params)
        result = stats.logit(df, formula=params.args.formula)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class Glm(SinkBuffer[GlmArgs, DrainState]):
    """Generalized linear model with a selectable error family."""

    FunctionArguments: ClassVar[type] = GlmArgs

    class Meta:
        """VGI function metadata for glm."""

        name = "glm"
        description = (
            "Generalized linear model with a 'family' arg "
            "(gaussian|binomial|poisson|gamma): emits (term, coef, std_err, "
            "z_value, p_value, ci_lower, ci_upper). Pass a Patsy formula."
        )
        categories = ["statistics", "regression"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT term, round(coef, 3) AS coef "
                    f"FROM statsmodels.main.glm({_COUNT_REL}, formula := 'y ~ x', family := 'poisson') "
                    "WHERE term = 'x'"
                ),
                description="Poisson GLM of count y on x; read the log-rate coefficient on x",
            )
        ]
        tags = {
            **object_tags(
                title="Generalized Linear Model (GLM)",
                doc_llm=(
                    "# glm\n\n"
                    "Fit a **generalized linear model** over a whole SQL relation with a "
                    "selectable error `family`, returning a coefficient table on the link scale "
                    "with full inference. Pass the data as a `(SELECT ...)` subquery, a Patsy "
                    "`formula`, and `family := 'gaussian' | 'binomial' | 'poisson' | 'gamma'` "
                    "(default `gaussian`).\n\n"
                    "**Use it** for outcomes that aren't plain continuous Gaussian: `poisson` "
                    "for counts (log link), `binomial` for binary/proportion outcomes (logit "
                    "link), `gamma` for positive skewed responses, and `gaussian` to reproduce "
                    "OLS. Each row is one model term with its coefficient, standard error, Wald "
                    "z-statistic, p-value, and 95% confidence interval.\n\n"
                    "**Edge cases:** an unknown family name, a family/data mismatch, a singular "
                    "design, a bad formula, a missing/non-numeric column, or empty input raises "
                    "a clean `StatsError`."
                ),
                doc_md=(
                    "## Generalized linear model\n\n"
                    "GLM regression over a SQL relation with a configurable error family, "
                    "powered by [statsmodels](https://www.statsmodels.org/). Pass the data as a "
                    "`(SELECT ...)` subquery, a Patsy `formula`, and a `family`.\n\n"
                    "Supported families: `gaussian` (identity link, equals OLS), `binomial` "
                    "(logit link), `poisson` (log link, for counts), and `gamma`. Returns one "
                    "row per term with the link-scale coefficient, standard error, z-value, "
                    "p-value, and 95% confidence bounds.\n\n"
                    "Call it as `glm(<relation>, formula := 'y ~ x', family := 'poisson')`; a "
                    "ready-to-run version is in this function's example queries."
                ),
                keywords=[
                    "glm",
                    "generalized linear model",
                    "poisson regression",
                    "count model",
                    "binomial",
                    "gamma",
                    "gaussian",
                    "log link",
                    "logit link",
                    "regression",
                    "coefficients",
                    "patsy formula",
                ],
            ),
            "vgi.category": "Regression",
            # VGI307/VGI414: structured static result schema derived from the output schema.
            "vgi.result_columns_schema": result_columns_schema(_GLM_SCHEMA),
            # VGI515: described examples carried via vgi.example_queries (see ols).
            "vgi.example_queries": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
        }

    @classmethod
    def on_bind(cls, params: BindParams[GlmArgs]) -> BindResponse:
        """Declare the fixed output schema for this function."""
        return BindResponse(output_schema=_GLM_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[GlmArgs]) -> DrainState:
        """Start each finalize stream with a fresh single-emit cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[GlmArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the GLM fit over the buffered relation and emit its term table."""
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = stats.glm(df, formula=a.formula, family=a.family)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class TTest(SinkBuffer[TTestArgs, DrainState]):
    """Two-sample t-test of a value column across a two-level group column."""

    FunctionArguments: ClassVar[type] = TTestArgs

    class Meta:
        """VGI function metadata for ttest."""

        name = "ttest"
        description = (
            "Two-sample (pooled-variance) t-test of 'column' across the two "
            "levels of 'group'; emits one row (statistic, p_value, df, "
            "mean_diff). 'column' and 'group' are SQL keywords -- double-quote "
            "both arg names in SQL."
        )
        categories = ["statistics", "test"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT round(p_value, 6) AS p_value, round(mean_diff, 2) AS mean_diff "
                    f"FROM statsmodels.main.ttest({_GROUP_REL}, \"column\" := 'v', \"group\" := 'g')"
                ),
                description="Two-sample t-test of v across groups; read the p-value and mean difference",
            )
        ]
        tags = {
            **object_tags(
                title="Two-Sample T-Test",
                doc_llm=(
                    "# ttest\n\n"
                    "Run a **two-sample (pooled-variance) t-test** comparing the mean of a "
                    "numeric column across the two levels of a grouping column. Pass the data as "
                    "a `(SELECT ...)` subquery, the value column as `column`, and the grouping "
                    "column as `group`. Because `column` and `group` are SQL reserved words, "
                    "**double-quote both argument names** at the call site: "
                    "`\"column\" := 'v', \"group\" := 'g'`.\n\n"
                    "**Use it** to answer 'do these two groups differ in mean?'. The result is a "
                    "single row: the t-statistic, the two-sided p-value for equal means, the "
                    "degrees of freedom, and `mean_diff` (mean of the first group minus the "
                    "second). A small p-value is evidence the group means differ.\n\n"
                    "**Edge cases:** a grouping column that does not have exactly two levels, a "
                    "missing/non-numeric value column, or empty input raises a clean "
                    "`StatsError`."
                ),
                doc_md=(
                    "## Two-sample t-test\n\n"
                    "Pooled-variance two-sample t-test over a SQL relation, powered by "
                    "[statsmodels](https://www.statsmodels.org/). Compares the mean of a value "
                    "column across the two levels of a grouping column.\n\n"
                    "Note that `column` and `group` are SQL keywords, so both argument names "
                    "must be double-quoted in SQL.\n\n"
                    "Returns a single row: the t-statistic, the two-sided p-value, the degrees "
                    "of freedom, and the difference in group means.\n\n"
                    "Call it as `ttest(<relation>, \"column\" := 'value', \"group\" := 'arm')` "
                    "with both argument names double-quoted; a ready-to-run version is in this "
                    "function's example queries."
                ),
                keywords=[
                    "ttest",
                    "t-test",
                    "two-sample t-test",
                    "difference in means",
                    "hypothesis test",
                    "p-value",
                    "t-statistic",
                    "ab test",
                    "a/b test",
                    "group comparison",
                    "significance",
                ],
            ),
            "vgi.category": "Hypothesis Tests",
            # VGI307/VGI414: structured static result schema derived from the output schema.
            "vgi.result_columns_schema": result_columns_schema(_TTEST_SCHEMA),
            # VGI515: described examples carried via vgi.example_queries (see ols).
            "vgi.example_queries": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
        }

    @classmethod
    def on_bind(cls, params: BindParams[TTestArgs]) -> BindResponse:
        """Declare the fixed output schema for this function."""
        return BindResponse(output_schema=_TTEST_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[TTestArgs]) -> DrainState:
        """Start each finalize stream with a fresh single-emit cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[TTestArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the two-sample t-test over the buffered relation and emit its row."""
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        df = cls.buffered_frame(params)
        result = stats.ttest(df, column=a.column, group=a.group)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


class Adfuller(SinkBuffer[AdfArgs, DrainState]):
    """Augmented Dickey-Fuller stationarity test over an ordered series."""

    FunctionArguments: ClassVar[type] = AdfArgs

    class Meta:
        """VGI function metadata for adfuller."""

        name = "adfuller"
        description = (
            "Augmented Dickey-Fuller unit-root (stationarity) test over an "
            "already-ordered series; emits one row (statistic, p_value, "
            "used_lag, n_obs). Pass an ORDER BY'd relation so rows are in time "
            "order. Small p_value => reject unit root => stationary."
        )
        categories = ["statistics", "timeseries", "test"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT used_lag, n_obs, round(p_value, 4) AS p_value "
                    f"FROM statsmodels.main.adfuller({_SERIES_REL}, \"column\" := 'v')"
                ),
                description="ADF stationarity test on an ordered series; read the lag, n_obs, and p-value",
            ),
            COMPOSED_RESIDUAL_ADF,
        ]
        tags = {
            **object_tags(
                title="Augmented Dickey-Fuller Stationarity Test",
                doc_llm=(
                    "# adfuller\n\n"
                    "Run the **Augmented Dickey-Fuller (ADF) unit-root test** to decide whether "
                    "a time series is stationary. Pass the series as a `(SELECT ...)` subquery "
                    "and the series column as `column`. Because `column` is a SQL reserved word, "
                    "**double-quote the argument name**: `\"column\" := 'v'`.\n\n"
                    "**Ordering matters:** SQL has no inherent row order, so the input must "
                    "already be time-ordered — pass `(SELECT v FROM series ORDER BY t)`. The "
                    "function treats the rows in the order it receives them.\n\n"
                    "**Use it** for stationarity checks before ARIMA / regression on time series. "
                    "The result is a single row: the test `statistic`, the MacKinnon `p_value` "
                    "(small => reject the unit-root null => the series is stationary), the number "
                    "of lags chosen by AIC (`used_lag`), and the observation count after "
                    "lagging/differencing (`n_obs`).\n\n"
                    "**Edge cases:** a missing/non-numeric column, empty input, or a series too "
                    "short for the chosen lag raises a clean `StatsError`."
                ),
                doc_md=(
                    "## Augmented Dickey-Fuller test\n\n"
                    "Stationarity (unit-root) test for a time series over a SQL relation, "
                    "powered by [statsmodels](https://www.statsmodels.org/).\n\n"
                    "The input must be time-ordered — pass an `ORDER BY`'d subquery, since SQL "
                    "rows have no inherent order. `column` is a SQL keyword, so the argument "
                    "name must be double-quoted.\n\n"
                    "Returns a single row: the test statistic, the MacKinnon p-value (small "
                    "means the series is stationary), the lag count chosen by AIC, and the "
                    "effective observation count.\n\n"
                    "Call it as `adfuller(<ordered-relation>, \"column\" := 'v')` with the "
                    "argument name double-quoted and the rows already time-ordered; a "
                    "ready-to-run version is in this function's example queries."
                ),
                keywords=[
                    "adfuller",
                    "augmented dickey-fuller",
                    "adf",
                    "unit root",
                    "stationarity",
                    "stationary",
                    "time series",
                    "hypothesis test",
                    "p-value",
                    "trend",
                    "arima",
                    "differencing",
                ],
            ),
            "vgi.category": "Hypothesis Tests",
            # VGI307/VGI414: structured static result schema derived from the output schema.
            "vgi.result_columns_schema": result_columns_schema(_ADF_SCHEMA),
            # VGI515: described examples carried via vgi.example_queries (see ols).
            "vgi.example_queries": json.dumps([{"description": e.description, "sql": e.sql} for e in examples]),
        }

    @classmethod
    def on_bind(cls, params: BindParams[AdfArgs]) -> BindResponse:
        """Declare the fixed output schema for this function."""
        return BindResponse(output_schema=_ADF_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[AdfArgs]) -> DrainState:
        """Start each finalize stream with a fresh single-emit cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[AdfArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run the ADF test over the buffered series and emit its row."""
        if state.done:
            out.finish()
            return
        state.done = True
        df = cls.buffered_frame(params)
        result = stats.adfuller(df, column=params.args.column)
        out.emit(pa.RecordBatch.from_pydict(result, schema=params.output_schema))


TABLE_FUNCTIONS: list[type] = [Ols, ModelStats, Logit, Glm, TTest, Adfuller]
