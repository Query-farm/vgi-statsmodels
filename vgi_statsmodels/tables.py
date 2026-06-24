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
            doc="Error family: gaussian | binomial | poisson | gamma.",
        ),
    ]


@dataclass(slots=True, frozen=True)
class TTestArgs:
    """Arguments for the two-sample t-test (value column and group column)."""

    data: Annotated[TableInput, Arg(0, doc="Relation containing the value and group columns.")]
    column: Annotated[str, Arg("column", default="value", doc="Numeric value column to compare.")]
    group: Annotated[str, Arg("group", default="group", doc="Grouping column with two levels.")]


@dataclass(slots=True, frozen=True)
class AdfArgs:
    """Arguments for the Augmented Dickey-Fuller test (ordered series column)."""

    data: Annotated[TableInput, Arg(0, doc="Ordered relation containing the series column.")]
    column: Annotated[str, Arg("column", default="value", doc="Numeric (time-ordered) series column.")]


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
                sql=f"SELECT * FROM statsmodels.main.ols({_LINEAR_REL}, formula := 'y ~ x')",
                description="OLS regression of y on x over an inline relation",
            )
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
                    "```sql\n"
                    "SELECT * FROM statsmodels.main.ols(\n"
                    "  (SELECT y, x FROM observations), formula := 'y ~ x'\n"
                    ");\n"
                    "```\n\n"
                    "See `model_stats` for whole-model fit statistics on the same formula."
                ),
                keywords=(
                    "ols, ordinary least squares, linear regression, regression, coefficients, "
                    "p-value, confidence interval, standard error, t-statistic, patsy formula, "
                    "fit, predictors"
                ),
                relative_path="vgi_statsmodels/tables.py",
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `term` | VARCHAR | Model term (a predictor or `Intercept`). One row per term. |\n"
                "| `coef` | DOUBLE | Estimated coefficient. |\n"
                "| `std_err` | DOUBLE | Standard error of the coefficient. |\n"
                "| `t_value` | DOUBLE | t-statistic (coef / std_err). |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for H0: coef = 0. |\n"
                "| `ci_lower` | DOUBLE | Lower bound of the 95% confidence interval. |\n"
                "| `ci_upper` | DOUBLE | Upper bound of the 95% confidence interval. |"
            ),
            "vgi.executable_examples": json.dumps(
                [
                    {
                        "description": "Fit y ~ x by OLS over an inline relation and read the coefficient table.",
                        "sql": (
                            "SELECT term, round(coef, 3) AS coef, round(p_value, 6) AS p_value "
                            f"FROM statsmodels.main.ols({_LINEAR_REL}, formula := 'y ~ x') ORDER BY term"
                        ),
                    }
                ]
            ),
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
                sql=f"SELECT * FROM statsmodels.main.model_stats({_LINEAR_REL}, formula := 'y ~ x')",
                description="OLS whole-model fit statistics over an inline relation",
            )
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
                    "```sql\n"
                    "SELECT * FROM statsmodels.main.model_stats(\n"
                    "  (SELECT y, x FROM observations), formula := 'y ~ x'\n"
                    ");\n"
                    "```"
                ),
                keywords=(
                    "model statistics, r-squared, adjusted r-squared, goodness of fit, f-test, "
                    "f-statistic, aic, bic, log-likelihood, ols, regression, model fit, diagnostics"
                ),
                relative_path="vgi_statsmodels/tables.py",
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `statistic` | VARCHAR | Fit-statistic name: `r_squared`, `adj_r_squared`, "
                "`f_statistic`, `f_pvalue`, `aic`, `bic`, `log_likelihood`, `nobs`, `df_model`, "
                "`df_resid`. One row per statistic. |\n"
                "| `value` | DOUBLE | Value of the statistic. |"
            ),
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
                sql=f"SELECT * FROM statsmodels.main.logit({_LOGIT_REL}, formula := 'y ~ x')",
                description="Logistic regression of binary y on x over an inline relation",
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
                    "```sql\n"
                    "SELECT * FROM statsmodels.main.logit(\n"
                    "  (SELECT clicked AS y, age AS x FROM events), formula := 'y ~ x'\n"
                    ");\n"
                    "```"
                ),
                keywords=(
                    "logit, logistic regression, binary classification, log-odds, odds ratio, "
                    "z-statistic, p-value, classification, probability, regression, patsy formula"
                ),
                relative_path="vgi_statsmodels/tables.py",
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `term` | VARCHAR | Model term (a predictor or `Intercept`). One row per term. |\n"
                "| `coef` | DOUBLE | Estimated log-odds coefficient. |\n"
                "| `std_err` | DOUBLE | Standard error of the coefficient. |\n"
                "| `z_value` | DOUBLE | Wald z-statistic (coef / std_err). |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for H0: coef = 0. |\n"
                "| `ci_lower` | DOUBLE | Lower bound of the 95% confidence interval. |\n"
                "| `ci_upper` | DOUBLE | Upper bound of the 95% confidence interval. |\n"
                "| `odds_ratio` | DOUBLE | Odds ratio exp(coef); >1 raises odds, <1 lowers. |"
            ),
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
                sql=(f"SELECT * FROM statsmodels.main.glm({_COUNT_REL}, formula := 'y ~ x', family := 'poisson')"),
                description="Poisson GLM of count y on x over an inline relation",
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
                    "```sql\n"
                    "SELECT * FROM statsmodels.main.glm(\n"
                    "  (SELECT events AS y, exposure AS x FROM logs),\n"
                    "  formula := 'y ~ x', family := 'poisson'\n"
                    ");\n"
                    "```"
                ),
                keywords=(
                    "glm, generalized linear model, poisson regression, count model, binomial, "
                    "gamma, gaussian, log link, logit link, regression, coefficients, patsy formula"
                ),
                relative_path="vgi_statsmodels/tables.py",
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `term` | VARCHAR | Model term (a predictor or `Intercept`). One row per term. |\n"
                "| `coef` | DOUBLE | Estimated coefficient (on the link scale). |\n"
                "| `std_err` | DOUBLE | Standard error of the coefficient. |\n"
                "| `z_value` | DOUBLE | Wald z-statistic (coef / std_err). |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for H0: coef = 0. |\n"
                "| `ci_lower` | DOUBLE | Lower bound of the 95% confidence interval. |\n"
                "| `ci_upper` | DOUBLE | Upper bound of the 95% confidence interval. |"
            ),
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
                sql=(f"SELECT * FROM statsmodels.main.ttest({_GROUP_REL}, \"column\" := 'v', \"group\" := 'g')"),
                description="Two-sample t-test across the group column over an inline relation",
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
                    "```sql\n"
                    "SELECT * FROM statsmodels.main.ttest(\n"
                    "  (SELECT value, arm FROM trial),\n"
                    "  \"column\" := 'value', \"group\" := 'arm'\n"
                    ");\n"
                    "```"
                ),
                keywords=(
                    "ttest, t-test, two-sample t-test, difference in means, hypothesis test, "
                    "p-value, t-statistic, ab test, a/b test, group comparison, significance"
                ),
                relative_path="vgi_statsmodels/tables.py",
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `statistic` | DOUBLE | Two-sample t-statistic. One row total. |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for equal means. |\n"
                "| `df` | DOUBLE | Degrees of freedom. |\n"
                "| `mean_diff` | DOUBLE | mean(group0) - mean(group1). |"
            ),
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
                sql=f"SELECT * FROM statsmodels.main.adfuller({_SERIES_REL}, \"column\" := 'v')",
                description="ADF stationarity test on an ordered inline series",
            )
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
                    "```sql\n"
                    "SELECT * FROM statsmodels.main.adfuller(\n"
                    "  (SELECT value AS v FROM prices ORDER BY day),\n"
                    "  \"column\" := 'v'\n"
                    ");\n"
                    "```"
                ),
                keywords=(
                    "adfuller, augmented dickey-fuller, adf, unit root, stationarity, stationary, "
                    "time series, hypothesis test, p-value, trend, arima, differencing"
                ),
                relative_path="vgi_statsmodels/tables.py",
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `statistic` | DOUBLE | Augmented Dickey-Fuller test statistic. One row total. |\n"
                "| `p_value` | DOUBLE | MacKinnon p-value; small => reject unit root (stationary). |\n"
                "| `used_lag` | INTEGER | Number of lags chosen by AIC. |\n"
                "| `n_obs` | INTEGER | Observations used after lagging/differencing. |"
            ),
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
