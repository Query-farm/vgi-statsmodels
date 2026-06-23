<p align="center">
  <img src="docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-statsmodels

A [VGI](https://query.farm) worker that brings **regression with full
statistical inference** and **hypothesis tests** to DuckDB/SQL: OLS / logistic /
GLM regression (coefficients, standard errors, p-values, confidence intervals),
whole-model fit statistics, two-sample t-tests, and the Augmented Dickey-Fuller
stationarity test — backed by [statsmodels](https://www.statsmodels.org/) and
[patsy](https://patsy.readthedocs.io/) (both BSD).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'statsmodels' (TYPE vgi, LOCATION 'uv run statsmodels_worker.py');

-- OLS regression: one row per term (incl. Intercept) with full inference
SELECT * FROM statsmodels.ols((SELECT y, x1, x2 FROM d),
                              formula := 'y ~ x1 + x2');

-- Whole-model fit statistics (r_squared, aic, f_pvalue, ...)
SELECT * FROM statsmodels.model_stats((SELECT y, x FROM d), formula := 'y ~ x');

-- Logistic regression (binary outcome) with odds ratios
SELECT * FROM statsmodels.logit((SELECT y, x FROM d), formula := 'y ~ x');

-- GLM with a selectable error family
SELECT * FROM statsmodels.glm((SELECT y, x FROM d),
                              formula := 'y ~ x', family := 'poisson');

-- Two-sample t-test ("column"/"group" are double-quoted: both SQL keywords)
SELECT * FROM statsmodels.ttest((SELECT value, arm FROM d),
                                "column" := 'value', "group" := 'arm');

-- Augmented Dickey-Fuller stationarity test on an ORDERED series
SELECT * FROM statsmodels.adfuller((SELECT v FROM d ORDER BY t), "column" := 'v');
```

## Data flow: one relation in, a result set out

Every function is a **table function** that consumes a *whole input relation* —
passed as a single `(SELECT ...)` subquery (the positional argument) — and emits
a result set. The model specification is passed as **named string arguments**: a
Patsy `formula` for the regressions, or `column` / `group` roles for the tests.

| named arg | meaning |
|-----------|---------|
| `formula := 'y ~ x1 + x2'` | Patsy formula (response `~` predictors) |
| `family := 'poisson'`      | (glm only) `gaussian` \| `binomial` \| `poisson` \| `gamma` |
| `"column" := 'col'`        | (ttest/adfuller) the numeric value/series column — `column` is a SQL keyword, so **double-quote the arg name** |
| `"group" := 'col'`         | (ttest only) the two-level grouping column — `group` is a SQL keyword, so **double-quote the arg name** |

Because a fit / test needs **every row** before it can produce output, these are
buffering (Sink+Source) functions — they buffer all input batches, then run the
statsmodels routine once.

## Formula syntax (Patsy)

The regression functions take a [Patsy](https://patsy.readthedocs.io/) formula.
**Every column in the input relation is available to the formula by name** —
select exactly the columns you reference (plus the response):

| formula | meaning |
|---------|---------|
| `y ~ x1 + x2`     | additive terms; an Intercept is added by default |
| `y ~ x1 + x2 - 1` | drop the Intercept |
| `y ~ x1 * x2`     | main effects plus the `x1:x2` interaction |
| `y ~ C(arm)`      | treat `arm` as categorical (one-hot vs a reference level) |
| `y ~ np.log(x)`   | numpy transforms work inside the formula |

## Functions

| function | returns |
|----------|---------|
| `ols(rel, formula)` | `(term, coef, std_err, t_value, p_value, ci_lower, ci_upper)` — one row per term |
| `model_stats(rel, formula)` | `(statistic, value)` — r_squared, adj_r_squared, f_statistic, f_pvalue, aic, bic, log_likelihood, nobs, df_model, df_resid |
| `logit(rel, formula)` | `(term, coef, std_err, z_value, p_value, ci_lower, ci_upper, odds_ratio)` |
| `glm(rel, formula, family)` | `(term, coef, std_err, z_value, p_value, ci_lower, ci_upper)` |
| `ttest(rel, column, group)` | `(statistic, p_value, df, mean_diff)` — one row |
| `adfuller(rel, column)` | `(statistic, p_value, used_lag, n_obs)` — one row |

`ols` reports the per-term coefficient table; `model_stats` reports the
whole-model fit statistics for the same OLS specification. `logit` adds
`odds_ratio = exp(coef)`. `adfuller` treats its input as an **already-ordered**
series — pass `(SELECT ... ORDER BY t)` so the rows are in time order.

## Robustness

A bad/unparseable formula, a missing or non-numeric column, an empty relation, a
singular/collinear design, an unknown GLM family, a non-binary logit response,
or a t-test group column without exactly two levels all surface a **clear error**
rather than crashing the worker.

## Development

```sh
uv sync --extra dev
uv run --no-sync pytest -q                 # unit + in-proc tables + Client RPC E2E
make test-sql                              # haybarn-unittest SQL E2E (authoritative)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_statsmodels/
```

## Licensing

This worker is MIT. statsmodels and patsy are **BSD**; numpy and pandas are
**BSD** — all permissive, no copyleft obligations.

---

## Authorship & License

Written by [Query.Farm](https://query.farm) — every VGI worker is designed and built by Query.Farm.

Copyright 2026 Query Farm LLC - https://query.farm

