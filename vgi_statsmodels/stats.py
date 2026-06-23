"""Pure regression / statistical-inference logic over statsmodels.

This module is the framework-free core: it takes a ``pandas.DataFrame`` (the
buffered input relation) plus a Patsy ``formula`` / column roles / options, runs
the statsmodels routine, and returns plain ``dict[str, list]`` column blocks
ready to hand to pyarrow. No VGI, no Arrow, no DuckDB here -- so every function
is directly unit-testable.

Formula syntax (Patsy)
----------------------
Regression functions take a Patsy ``formula`` string such as
``'y ~ x1 + x2'``. The left-hand side is the response column; the right-hand
side lists the predictors. Patsy supports rich syntax:

- ``y ~ x1 + x2``        -- additive terms (an Intercept is added by default)
- ``y ~ x1 + x2 - 1``    -- drop the Intercept
- ``y ~ x1 * x2``        -- main effects plus the ``x1:x2`` interaction
- ``y ~ C(arm)``         -- treat ``arm`` as categorical (one-hot, reference level)
- ``y ~ np.log(x)``      -- numpy transforms are available inside the formula

**Every column in the input relation is available to the formula** by name --
select exactly the columns you reference (plus the response) in the
``(SELECT ...)`` relation.

statsmodels and patsy are BSD-licensed; numpy/pandas are BSD-licensed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Importing statsmodels is expensive (pulls in scipy/patsy); do it once at
# module import so the per-call path is cheap. The worker imports this module at
# startup, so the cost is paid before the first SQL call.
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import PatsyError
from statsmodels.tsa.stattools import adfuller as sm_adfuller

__all__ = [
    "StatsError",
    "adfuller",
    "glm",
    "logit",
    "model_stats",
    "ols",
    "ttest",
]

# GLM family name -> (statsmodels family constructor, default link). The named
# 'family' arg picks the error distribution for glm().
_GLM_FAMILIES = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
    "poisson": sm.families.Poisson,
    "gamma": sm.families.Gamma,
}


class StatsError(ValueError):
    """Raised for user-facing input problems (bad formula, missing/non-numeric
    column, empty relation, singular design, unknown family).

    A plain, explicit error so the worker surfaces a clear message to SQL
    instead of crashing with an opaque statsmodels/patsy traceback.
    """


def _require_nonempty(df: pd.DataFrame, what: str) -> None:
    if len(df) == 0:
        raise StatsError(f"{what} requires a non-empty input relation")


def _require_column(df: pd.DataFrame, column: str, *, role: str) -> None:
    if column not in df.columns:
        raise StatsError(
            f"{role} column '{column}' not found; input relation has columns: "
            f"{', '.join(map(str, df.columns))}"
        )


def _numeric(df: pd.DataFrame, column: str, *, role: str) -> np.ndarray:
    """Coerce a column to a float64 numpy array or raise a clear error."""
    series = df[column]
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any() and not series.isna().any():
        raise StatsError(
            f"{role} column '{column}' must be numeric, but contains "
            f"non-numeric values (dtype {series.dtype})"
        )
    return np.asarray(coerced, dtype=float)


def _fit_linearish(builder, formula: str, df: pd.DataFrame, what: str):
    """Build + fit a formula model, translating patsy/statsmodels failures.

    ``builder`` is a callable ``(formula, data) -> results`` (already wrapping
    ``.fit()`` for the model in question). Any patsy/linear-algebra failure is
    converted into a clear :class:`StatsError`.
    """
    try:
        return builder(formula, df)
    except PatsyError as exc:
        raise StatsError(f"invalid formula '{formula}': {exc}") from exc
    except (np.linalg.LinAlgError, ValueError) as exc:
        msg = str(exc)
        if "must be" in msg or "non-numeric" in msg:
            raise
        raise StatsError(
            f"could not fit model for formula '{formula}': {msg} "
            f"(check for a singular/collinear design or non-numeric predictors)"
        ) from exc


def _inference_block(result, *, stat_name: str) -> dict[str, list]:
    """Common coefficient table: term, coef, std_err, stat, p, CI.

    ``stat_name`` is ``'t_value'`` for OLS, ``'z_value'`` for Logit/GLM. The
    statistic column itself is keyed by ``stat_name`` so callers can build the
    right output schema.
    """
    conf = result.conf_int()
    terms = [str(t) for t in result.params.index]
    return {
        "term": terms,
        "coef": [float(x) for x in result.params],
        "std_err": [float(x) for x in result.bse],
        stat_name: [float(x) for x in result.tvalues],
        "p_value": [float(x) for x in result.pvalues],
        "ci_lower": [float(x) for x in conf.iloc[:, 0]],
        "ci_upper": [float(x) for x in conf.iloc[:, 1]],
    }


def ols(df: pd.DataFrame, *, formula: str) -> dict[str, list]:
    """Ordinary least squares regression with full inference.

    Args:
        df: Input relation; every column is available to the Patsy formula.
        formula: Patsy formula, e.g. ``'y ~ x1 + x2'``.

    Returns:
        Column block with keys ``term``, ``coef``, ``std_err``, ``t_value``,
        ``p_value``, ``ci_lower``, ``ci_upper`` -- one entry per model term
        (including the Intercept unless dropped with ``- 1``).

    Raises:
        StatsError: On empty input, bad formula, or a singular design.
    """
    _require_nonempty(df, "ols")
    result = _fit_linearish(lambda f, d: smf.ols(f, data=d).fit(), formula, df, "ols")
    return _inference_block(result, stat_name="t_value")


def model_stats(df: pd.DataFrame, *, formula: str) -> dict[str, list]:
    """Whole-model OLS fit statistics, one row per statistic.

    Args:
        df: Input relation; every column is available to the Patsy formula.
        formula: Patsy formula, e.g. ``'y ~ x1 + x2'``.

    Returns:
        Column block with keys ``statistic`` (str) and ``value`` (float),
        covering r_squared, adj_r_squared, f_statistic, f_pvalue, aic, bic,
        nobs, df_resid, log_likelihood.

    Raises:
        StatsError: On empty input, bad formula, or a singular design.
    """
    _require_nonempty(df, "model_stats")
    result = _fit_linearish(lambda f, d: smf.ols(f, data=d).fit(), formula, df, "model_stats")
    stats = {
        "r_squared": float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "f_statistic": float(result.fvalue),
        "f_pvalue": float(result.f_pvalue),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "nobs": float(result.nobs),
        "df_model": float(result.df_model),
        "df_resid": float(result.df_resid),
    }
    return {"statistic": list(stats.keys()), "value": list(stats.values())}


def logit(df: pd.DataFrame, *, formula: str) -> dict[str, list]:
    """Logistic (binary-outcome) regression with full inference + odds ratios.

    The response must be binary (0/1, boolean, or two distinct values).

    Args:
        df: Input relation; every column is available to the Patsy formula.
        formula: Patsy formula, e.g. ``'y ~ x1 + x2'``.

    Returns:
        Column block with keys ``term``, ``coef``, ``std_err``, ``z_value``,
        ``p_value``, ``ci_lower``, ``ci_upper``, ``odds_ratio`` (= exp(coef)).

    Raises:
        StatsError: On empty input, bad formula, a non-binary response, perfect
            separation, or a singular design.
    """
    _require_nonempty(df, "logit")

    def _build(f: str, d: pd.DataFrame):
        try:
            return smf.logit(f, data=d).fit(disp=0)
        except np.linalg.LinAlgError as exc:
            raise StatsError(
                f"could not fit logit for formula '{f}': {exc} (singular design or perfect separation)"
            ) from exc
        except Exception as exc:  # statsmodels PerfectSeparationError etc.
            name = type(exc).__name__
            if "Separation" in name or "Singular" in name:
                raise StatsError(
                    f"could not fit logit for formula '{f}': {exc} "
                    f"(perfect separation -- the predictors split the outcome exactly)"
                ) from exc
            raise

    result = _fit_linearish(_build, formula, df, "logit")
    block = _inference_block(result, stat_name="z_value")
    block["odds_ratio"] = [float(np.exp(c)) for c in block["coef"]]
    return block


def glm(df: pd.DataFrame, *, formula: str, family: str = "gaussian") -> dict[str, list]:
    """Generalized linear model with a selectable error family.

    Args:
        df: Input relation; every column is available to the Patsy formula.
        formula: Patsy formula, e.g. ``'y ~ x1 + x2'``.
        family: Error distribution: ``'gaussian'``, ``'binomial'``,
            ``'poisson'``, or ``'gamma'`` (case-insensitive). Default
            ``'gaussian'``.

    Returns:
        Column block with keys ``term``, ``coef``, ``std_err``, ``z_value``,
        ``p_value``, ``ci_lower``, ``ci_upper``.

    Raises:
        StatsError: On empty input, bad formula, unknown family, or a singular
            design.
    """
    _require_nonempty(df, "glm")
    key = (family or "gaussian").strip().lower()
    if key not in _GLM_FAMILIES:
        raise StatsError(f"unknown family '{family}'; choose one of: {', '.join(sorted(_GLM_FAMILIES))}")
    fam = _GLM_FAMILIES[key]()
    result = _fit_linearish(lambda f, d: smf.glm(f, data=d, family=fam).fit(), formula, df, "glm")
    return _inference_block(result, stat_name="z_value")


def ttest(df: pd.DataFrame, *, column: str, group: str) -> dict[str, list]:
    """Two-sample (Welch-style independent) t-test across a group column.

    Compares the mean of ``column`` between the two distinct levels of
    ``group``. Uses the pooled-variance two-sample t-test (``usevar='pooled'``),
    matching ``scipy.stats.ttest_ind`` defaults.

    Args:
        df: Input relation.
        column: Numeric value column to compare.
        group: Grouping column with exactly two distinct levels.

    Returns:
        Single-row column block with keys ``statistic`` (float), ``p_value``
        (float), ``df`` (float, degrees of freedom), ``mean_diff`` (float,
        mean(level0) - mean(level1)).

    Raises:
        StatsError: On empty input, a missing column, a non-numeric value
            column, or a group column that does not have exactly two levels.
    """
    _require_nonempty(df, "ttest")
    _require_column(df, column, role="value")
    _require_column(df, group, role="group")

    levels = list(pd.unique(df[group].dropna()))
    if len(levels) != 2:
        raise StatsError(
            f"ttest needs exactly two distinct groups in '{group}', found {len(levels)}: {levels}"
        )

    values = _numeric(df, column, role="value")
    g = df[group].to_numpy()
    a = values[g == levels[0]]
    b = values[g == levels[1]]
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise StatsError(
            f"ttest needs at least two observations per group; got "
            f"{len(a)} for '{levels[0]}' and {len(b)} for '{levels[1]}'"
        )

    from statsmodels.stats.weightstats import ttest_ind

    statistic, p_value, dof = ttest_ind(a, b, usevar="pooled")
    return {
        "statistic": [float(statistic)],
        "p_value": [float(p_value)],
        "df": [float(dof)],
        "mean_diff": [float(np.mean(a) - np.mean(b))],
    }


def adfuller(df: pd.DataFrame, *, column: str) -> dict[str, list]:
    """Augmented Dickey-Fuller test for a unit root (non-stationarity).

    The input is treated as an *already-ordered* series -- pass an ordered
    ``(SELECT ... ORDER BY ...)`` relation so the rows are in time order. The
    null hypothesis is that the series has a unit root (is non-stationary); a
    small p-value rejects it in favour of stationarity.

    Args:
        df: Input relation containing the ordered series.
        column: Numeric series column.

    Returns:
        Single-row column block with keys ``statistic`` (float, the ADF test
        statistic), ``p_value`` (float), ``used_lag`` (int, lags chosen by AIC),
        ``n_obs`` (int, observations used after differencing/lagging).

    Raises:
        StatsError: On empty input, a missing column, a non-numeric column, or
            too few observations to run the test.
    """
    _require_nonempty(df, "adfuller")
    _require_column(df, column, role="series")
    series = _numeric(df, column, role="series")
    series = series[~np.isnan(series)]
    if len(series) < 6:
        raise StatsError(
            f"adfuller needs at least ~6 observations to estimate a unit root; got {len(series)}"
        )
    try:
        statistic, p_value, used_lag, n_obs, _crit, _icbest = sm_adfuller(series, autolag="AIC")
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise StatsError(f"adfuller could not run on column '{column}': {exc}") from exc
    return {
        "statistic": [float(statistic)],
        "p_value": [float(p_value)],
        "used_lag": [int(used_lag)],
        "n_obs": [int(n_obs)],
    }
