"""Table-function tests via the in-process buffering harness.

Drive each function through the real bind -> process(sink) -> combine ->
finalize lifecycle (no subprocess), checking the emitted Arrow result and that
the named formula / column-role args resolve correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from vgi_statsmodels.tables import Adfuller, Glm, Logit, ModelStats, Ols, TTest

from .harness import run_buffering


def _arrow(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)


def _linear() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, 500)
    return pd.DataFrame({"x": x, "y": y})


def test_ols_function_recovers_coefficients() -> None:
    out = run_buffering(Ols, _arrow(_linear()), named={"formula": "y ~ x"})
    d = out.to_pydict()
    assert out.schema.names == [
        "term",
        "coef",
        "std_err",
        "t_value",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    i = d["term"].index("x")
    assert d["coef"][i] == pytest.approx(3.0, abs=0.1)
    assert d["p_value"][i] < 1e-9


def test_model_stats_function() -> None:
    out = run_buffering(ModelStats, _arrow(_linear()), named={"formula": "y ~ x"})
    d = out.to_pydict()
    s = dict(zip(d["statistic"], d["value"], strict=False))
    assert s["r_squared"] > 0.95


def test_logit_function() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 600)
    p = 1.0 / (1.0 + np.exp(-(2.0 * x)))
    y = (rng.uniform(size=600) < p).astype(int)
    df = pd.DataFrame({"x": x, "y": y})
    out = run_buffering(Logit, _arrow(df), named={"formula": "y ~ x"})
    d = out.to_pydict()
    assert "odds_ratio" in out.schema.names
    i = d["term"].index("x")
    assert d["odds_ratio"][i] > 1.0
    assert d["p_value"][i] < 1e-9


def test_glm_poisson_function() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 600)
    y = rng.poisson(np.exp(0.5 + 0.8 * x))
    df = pd.DataFrame({"x": x, "y": y})
    out = run_buffering(Glm, _arrow(df), named={"formula": "y ~ x", "family": "poisson"})
    d = out.to_pydict()
    i = d["term"].index("x")
    assert d["coef"][i] == pytest.approx(0.8, abs=0.1)


def test_ttest_function() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 80), rng.normal(5, 1, 80)]),
            "arm": ["a"] * 80 + ["b"] * 80,
        }
    )
    out = run_buffering(TTest, _arrow(df), named={"column": "value", "group": "arm"})
    d = out.to_pydict()
    assert out.num_rows == 1
    assert d["p_value"][0] < 1e-20


def test_adfuller_function_random_walk() -> None:
    rng = np.random.default_rng(123)
    df = pd.DataFrame({"value": np.cumsum(rng.normal(0, 1, 300))})
    out = run_buffering(Adfuller, _arrow(df), named={"column": "value"})
    d = out.to_pydict()
    assert d["p_value"][0] > 0.1
    assert pa.types.is_int32(out.schema.field("used_lag").type)


def test_bad_formula_raises() -> None:
    with pytest.raises(Exception, match="invalid formula"):
        run_buffering(Ols, _arrow(_linear()), named={"formula": "y ~~ x"})


def test_missing_column_raises() -> None:
    tbl = pa.table({"value": [1.0, 2.0], "arm": ["a", "b"]})
    with pytest.raises(Exception, match="not found"):
        run_buffering(TTest, tbl, named={"column": "nope", "group": "arm"})
