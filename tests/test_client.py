"""End-to-end tests driving statsmodels_worker.py as a real subprocess.

These spawn the worker via ``vgi.client.Client`` and invoke each function
through the real ``table_buffering_function`` RPC path -- exactly how DuckDB
drives a buffering function after ``ATTACH`` -- exercising bind, the sink
process RPC per batch, combine, and the finalize source stream over the wire.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client, ClientError

_WORKER = str(Path(__file__).resolve().parent.parent / "statsmodels_worker.py")


@pytest.fixture(scope="module")
def client() -> Iterator[Client]:
    with Client(f"{sys.executable} {_WORKER}", worker_limit=1) as c:
        yield c


def _run(client: Client, name: str, table: pa.Table, **named: str) -> pa.Table:
    batches = list(
        client.table_buffering_function(
            function_name=name,
            input=iter(table.to_batches()),
            arguments=Arguments(named={k: pa.scalar(v) for k, v in named.items()}),
        )
    )
    return pa.Table.from_batches(batches)


def _linear_table() -> pa.Table:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, 500)
    return pa.Table.from_pandas(pd.DataFrame({"x": x, "y": y}), preserve_index=False)


def test_ols_e2e(client: Client) -> None:
    out = _run(client, "ols", _linear_table(), formula="y ~ x")
    d = out.to_pydict()
    i = d["term"].index("x")
    assert d["coef"][i] == pytest.approx(3.0, abs=0.1)
    assert d["p_value"][i] < 1e-9


def test_model_stats_e2e(client: Client) -> None:
    out = _run(client, "model_stats", _linear_table(), formula="y ~ x")
    s = dict(zip(out.to_pydict()["statistic"], out.to_pydict()["value"], strict=False))
    assert s["r_squared"] > 0.95


def test_logit_e2e(client: Client) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 600)
    p = 1.0 / (1.0 + np.exp(-(2.0 * x)))
    y = (rng.uniform(size=600) < p).astype(int)
    tbl = pa.Table.from_pandas(pd.DataFrame({"x": x, "y": y}), preserve_index=False)
    out = _run(client, "logit", tbl, formula="y ~ x")
    d = out.to_pydict()
    i = d["term"].index("x")
    assert d["odds_ratio"][i] > 1.0


def test_glm_poisson_e2e(client: Client) -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 600)
    y = rng.poisson(np.exp(0.5 + 0.8 * x))
    tbl = pa.Table.from_pandas(pd.DataFrame({"x": x, "y": y}), preserve_index=False)
    out = _run(client, "glm", tbl, formula="y ~ x", family="poisson")
    d = out.to_pydict()
    i = d["term"].index("x")
    assert d["coef"][i] == pytest.approx(0.8, abs=0.1)


def test_ttest_e2e(client: Client) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 80), rng.normal(5, 1, 80)]),
            "arm": ["a"] * 80 + ["b"] * 80,
        }
    )
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    out = _run(client, "ttest", tbl, column="value", group="arm")
    assert out.to_pydict()["p_value"][0] < 1e-20


def test_adfuller_e2e(client: Client) -> None:
    rng = np.random.default_rng(321)
    tbl = pa.Table.from_pandas(
        pd.DataFrame({"value": rng.normal(0, 1, 300)}), preserve_index=False
    )
    out = _run(client, "adfuller", tbl, column="value")
    assert out.to_pydict()["p_value"][0] < 0.05


def test_bad_formula_errors_e2e(client: Client) -> None:
    with pytest.raises(ClientError):
        _run(client, "ols", _linear_table(), formula="y ~~ x")
