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
from .schema_utils import field as sfield

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
                sql="SELECT * FROM statsmodels.ols((SELECT y, x FROM d), formula := 'y ~ x')",
                description="OLS regression of y on x",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `term` | VARCHAR | Model term (a predictor or `Intercept`). One row per term. |\n"
                "| `coef` | DOUBLE | Estimated coefficient. |\n"
                "| `std_err` | DOUBLE | Standard error of the coefficient. |\n"
                "| `t_value` | DOUBLE | t-statistic (coef / std_err). |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for H0: coef = 0. |\n"
                "| `ci_lower` | DOUBLE | Lower bound of the 95% confidence interval. |\n"
                "| `ci_upper` | DOUBLE | Upper bound of the 95% confidence interval. |"
            )
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
                sql="SELECT * FROM statsmodels.model_stats((SELECT y, x FROM d), formula := 'y ~ x')",
                description="OLS model fit statistics",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `statistic` | VARCHAR | Fit-statistic name: `r_squared`, `adj_r_squared`, "
                "`f_statistic`, `f_pvalue`, `aic`, `bic`, `log_likelihood`, `nobs`, `df_model`, "
                "`df_resid`. One row per statistic. |\n"
                "| `value` | DOUBLE | Value of the statistic. |"
            )
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
                sql="SELECT * FROM statsmodels.logit((SELECT y, x FROM d), formula := 'y ~ x')",
                description="Logistic regression of binary y on x",
            )
        ]
        tags = {
            "vgi.columns_md": (
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
            )
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
                sql=("SELECT * FROM statsmodels.glm((SELECT y, x FROM d), formula := 'y ~ x', family := 'poisson')"),
                description="Poisson GLM of count y on x",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `term` | VARCHAR | Model term (a predictor or `Intercept`). One row per term. |\n"
                "| `coef` | DOUBLE | Estimated coefficient (on the link scale). |\n"
                "| `std_err` | DOUBLE | Standard error of the coefficient. |\n"
                "| `z_value` | DOUBLE | Wald z-statistic (coef / std_err). |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for H0: coef = 0. |\n"
                "| `ci_lower` | DOUBLE | Lower bound of the 95% confidence interval. |\n"
                "| `ci_upper` | DOUBLE | Upper bound of the 95% confidence interval. |"
            )
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
                sql=("SELECT * FROM statsmodels.ttest((SELECT v, g FROM d), \"column\" := 'v', \"group\" := 'g')"),
                description="Two-sample t-test across the group column",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `statistic` | DOUBLE | Two-sample t-statistic. One row total. |\n"
                "| `p_value` | DOUBLE | Two-sided p-value for equal means. |\n"
                "| `df` | DOUBLE | Degrees of freedom. |\n"
                "| `mean_diff` | DOUBLE | mean(group0) - mean(group1). |"
            )
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
                sql=("SELECT * FROM statsmodels.adfuller((SELECT v FROM d ORDER BY t), \"column\" := 'v')"),
                description="ADF stationarity test on an ordered series",
            )
        ]
        tags = {
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `statistic` | DOUBLE | Augmented Dickey-Fuller test statistic. One row total. |\n"
                "| `p_value` | DOUBLE | MacKinnon p-value; small => reject unit root (stationary). |\n"
                "| `used_lag` | INTEGER | Number of lags chosen by AIC. |\n"
                "| `n_obs` | INTEGER | Observations used after lagging/differencing. |"
            )
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
