"""Unit tests for the pure statsmodels logic, validated against known data.

Uses constructed datasets with a fixed seed so the recovered coefficients are
predictable (e.g. ``y = 2 + 3*x + noise`` => Intercept ~ 2, x ~ 3), plus checks
against statsmodels' own fit and the error edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from vgi_statsmodels import stats
from vgi_statsmodels.stats import StatsError

# --------------------------------------------------------------------------
# Fixtures: deterministic constructed data
# --------------------------------------------------------------------------


def _linear_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """y = 2 + 3*x + small noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y})


def _binary_data(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Logistic outcome driven strongly by x (positive sign)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    logits = -0.5 + 2.0 * x
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"x": x, "y": y})


def _count_data(n: int = 800, seed: int = 11) -> pd.DataFrame:
    """Poisson counts with log-mean = 0.5 + 0.8*x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    mu = np.exp(0.5 + 0.8 * x)
    y = rng.poisson(mu)
    return pd.DataFrame({"x": x, "y": y})


# --------------------------------------------------------------------------
# OLS — known-data recovery
# --------------------------------------------------------------------------


def test_ols_recovers_known_coefficients() -> None:
    df = _linear_data()
    out = stats.ols(df, formula="y ~ x")
    by_term = dict(zip(out["term"], range(len(out["term"])), strict=False))

    i_int, i_x = by_term["Intercept"], by_term["x"]
    # Recovers Intercept ~ 2 and slope x ~ 3.
    assert out["coef"][i_int] == pytest.approx(2.0, abs=0.1)
    assert out["coef"][i_x] == pytest.approx(3.0, abs=0.1)
    # Strongly significant slope.
    assert out["p_value"][i_x] < 1e-10
    # True coefficients fall inside their confidence intervals.
    assert out["ci_lower"][i_int] <= 2.0 <= out["ci_upper"][i_int]
    assert out["ci_lower"][i_x] <= 3.0 <= out["ci_upper"][i_x]
    # std_err > 0, t_value = coef / std_err.
    assert out["std_err"][i_x] > 0
    assert out["t_value"][i_x] == pytest.approx(out["coef"][i_x] / out["std_err"][i_x])


def test_ols_matches_statsmodels_fit() -> None:
    df = _linear_data()
    out = stats.ols(df, formula="y ~ x")
    fit = smf.ols("y ~ x", data=df).fit()
    i_x = out["term"].index("x")
    assert out["coef"][i_x] == pytest.approx(float(fit.params["x"]), rel=1e-9)
    assert out["p_value"][i_x] == pytest.approx(float(fit.pvalues["x"]), rel=1e-9)
    assert out["std_err"][i_x] == pytest.approx(float(fit.bse["x"]), rel=1e-9)


def test_ols_multiple_predictors() -> None:
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    out = stats.ols(df, formula="y ~ x1 + x2")
    by = dict(zip(out["term"], out["coef"], strict=False))
    assert by["x1"] == pytest.approx(2.0, abs=0.1)
    assert by["x2"] == pytest.approx(-1.5, abs=0.1)


# --------------------------------------------------------------------------
# model_stats
# --------------------------------------------------------------------------


def test_model_stats_high_r_squared() -> None:
    df = _linear_data()
    out = stats.model_stats(df, formula="y ~ x")
    s = dict(zip(out["statistic"], out["value"], strict=False))
    assert s["r_squared"] > 0.95
    assert 0.0 <= s["adj_r_squared"] <= 1.0
    assert s["f_pvalue"] < 1e-10
    assert s["nobs"] == float(len(df))
    assert s["df_resid"] == float(len(df) - 2)
    # Matches statsmodels.
    fit = smf.ols("y ~ x", data=df).fit()
    assert s["aic"] == pytest.approx(float(fit.aic), rel=1e-9)
    assert s["r_squared"] == pytest.approx(float(fit.rsquared), rel=1e-9)


# --------------------------------------------------------------------------
# Logit — recovers sign + significance
# --------------------------------------------------------------------------


def test_logit_recovers_sign_and_significance() -> None:
    df = _binary_data()
    out = stats.logit(df, formula="y ~ x")
    i_x = out["term"].index("x")
    assert out["coef"][i_x] > 0  # positive driver
    assert out["odds_ratio"][i_x] > 1.0
    assert out["p_value"][i_x] < 1e-10
    # z_value = coef / std_err.
    assert out["z_value"][i_x] == pytest.approx(out["coef"][i_x] / out["std_err"][i_x])
    # Matches statsmodels' own fit.
    fit = smf.logit("y ~ x", data=df).fit(disp=0)
    assert out["coef"][i_x] == pytest.approx(float(fit.params["x"]), rel=1e-6)


# --------------------------------------------------------------------------
# GLM
# --------------------------------------------------------------------------


def test_glm_poisson_recovers_coefficients() -> None:
    df = _count_data()
    out = stats.glm(df, formula="y ~ x", family="poisson")
    by = dict(zip(out["term"], out["coef"], strict=False))
    assert by["Intercept"] == pytest.approx(0.5, abs=0.1)
    assert by["x"] == pytest.approx(0.8, abs=0.1)
    i_x = out["term"].index("x")
    assert out["p_value"][i_x] < 1e-10


def test_glm_gaussian_matches_ols() -> None:
    df = _linear_data()
    g = stats.glm(df, formula="y ~ x", family="gaussian")
    o = stats.ols(df, formula="y ~ x")
    gi = g["term"].index("x")
    oi = o["term"].index("x")
    assert g["coef"][gi] == pytest.approx(o["coef"][oi], rel=1e-6)


def test_glm_family_case_insensitive() -> None:
    df = _count_data()
    out = stats.glm(df, formula="y ~ x", family="POISSON")
    assert "x" in out["term"]


def test_glm_unknown_family_errors() -> None:
    df = _count_data()
    with pytest.raises(StatsError, match="unknown family"):
        stats.glm(df, formula="y ~ x", family="banana")


# --------------------------------------------------------------------------
# t-test
# --------------------------------------------------------------------------


def test_ttest_clearly_different_groups_small_p() -> None:
    rng = np.random.default_rng(3)
    a = pd.DataFrame({"value": rng.normal(0.0, 1.0, 100), "arm": "a"})
    b = pd.DataFrame({"value": rng.normal(5.0, 1.0, 100), "arm": "b"})
    df = pd.concat([a, b], ignore_index=True)
    out = stats.ttest(df, column="value", group="arm")
    assert out["p_value"][0] < 1e-20
    assert abs(out["mean_diff"][0]) == pytest.approx(5.0, abs=0.5)
    assert out["df"][0] == pytest.approx(198.0)


def test_ttest_identical_groups_large_p() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 1.0, 200)
    df = pd.DataFrame({"value": np.concatenate([base, base]), "arm": ["a"] * 200 + ["b"] * 200})
    out = stats.ttest(df, column="value", group="arm")
    assert out["p_value"][0] > 0.9


def test_ttest_matches_scipy_via_statsmodels() -> None:
    from statsmodels.stats.weightstats import ttest_ind

    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 50), rng.normal(1, 1, 60)]),
            "arm": ["x"] * 50 + ["y"] * 60,
        }
    )
    out = stats.ttest(df, column="value", group="arm")
    a = df[df.arm == "x"]["value"].to_numpy()
    b = df[df.arm == "y"]["value"].to_numpy()
    stat, p, dof = ttest_ind(a, b)
    assert out["statistic"][0] == pytest.approx(float(stat))
    assert out["p_value"][0] == pytest.approx(float(p))


# --------------------------------------------------------------------------
# ADF
# --------------------------------------------------------------------------


def test_adfuller_random_walk_is_nonstationary() -> None:
    rng = np.random.default_rng(123)
    walk = np.cumsum(rng.normal(0.0, 1.0, 300))  # unit root => non-stationary
    df = pd.DataFrame({"value": walk})
    out = stats.adfuller(df, column="value")
    assert out["p_value"][0] > 0.10  # fail to reject unit root


def test_adfuller_white_noise_is_stationary() -> None:
    rng = np.random.default_rng(321)
    noise = rng.normal(0.0, 1.0, 300)  # stationary
    df = pd.DataFrame({"value": noise})
    out = stats.adfuller(df, column="value")
    assert out["p_value"][0] < 0.05  # reject unit root
    assert out["n_obs"][0] > 0


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


def test_ols_bad_formula_errors() -> None:
    df = _linear_data(n=10)
    with pytest.raises(StatsError, match="invalid formula"):
        stats.ols(df, formula="y ~~ x")


def test_ols_missing_column_errors() -> None:
    df = _linear_data(n=10)
    with pytest.raises(StatsError):
        stats.ols(df, formula="y ~ nope")


def test_ols_empty_errors() -> None:
    df = pd.DataFrame({"x": pd.Series([], dtype=float), "y": pd.Series([], dtype=float)})
    with pytest.raises(StatsError, match="non-empty"):
        stats.ols(df, formula="y ~ x")


def test_ols_singular_design_does_not_crash() -> None:
    # x2 is an exact duplicate of x1 -> perfectly collinear / singular design.
    # statsmodels uses a Moore-Penrose pseudo-inverse, so this does NOT crash:
    # it splits the slope across the collinear terms and returns a clean fit.
    # The contract we guarantee is *robustness* -- a finite result, never a
    # worker crash or an opaque traceback to SQL.
    rng = np.random.default_rng(9)
    x1 = rng.normal(size=50)
    df = pd.DataFrame({"y": x1 + rng.normal(0, 0.1, 50), "x1": x1, "x2": x1})
    out = stats.ols(df, formula="y ~ x1 + x2")
    # All coefficients finite; the two collinear terms share the recovered slope.
    assert all(np.isfinite(c) for c in out["coef"])
    by = dict(zip(out["term"], out["coef"], strict=False))
    assert by["x1"] + by["x2"] == pytest.approx(1.0, abs=0.1)


def test_ttest_missing_column_errors() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0], "arm": ["a", "b"]})
    with pytest.raises(StatsError, match="not found"):
        stats.ttest(df, column="nope", group="arm")


def test_ttest_needs_two_groups() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "arm": ["a", "a", "a"]})
    with pytest.raises(StatsError, match="exactly two distinct groups"):
        stats.ttest(df, column="value", group="arm")


def test_ttest_non_numeric_value_errors() -> None:
    df = pd.DataFrame({"value": ["p", "q", "r", "s"], "arm": ["a", "a", "b", "b"]})
    with pytest.raises(StatsError, match="must be numeric"):
        stats.ttest(df, column="value", group="arm")


def test_adfuller_missing_column_errors() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    with pytest.raises(StatsError, match="not found"):
        stats.adfuller(df, column="nope")


def test_adfuller_too_short_errors() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    with pytest.raises(StatsError, match="at least"):
        stats.adfuller(df, column="value")


def test_logit_empty_errors() -> None:
    df = pd.DataFrame({"x": pd.Series([], dtype=float), "y": pd.Series([], dtype=float)})
    with pytest.raises(StatsError, match="non-empty"):
        stats.logit(df, formula="y ~ x")
