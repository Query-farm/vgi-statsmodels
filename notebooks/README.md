# Statsmodels notebooks

[01 · Pooled and within-site slopes](01_pooled_and_within_site_slopes.ipynb) fits two OLS specifications to a disclosed synthetic two-site generator. It contrasts pooled and adjusted associations, reconstructs the within-site slope and classical standard error in SQL, and interprets 95% intervals conditionally on model assumptions.

## Run from this directory alone

Install [uv](https://docs.astral.sh/uv/) and Python 3.14. Initial package, tagged Git worker and VGI extension downloads need internet access. All analytical data are generated locally; no credentials, external dataset, local worker checkout or globally installed kernel is required.

```sh
cd notebooks
uv sync --locked
uv run --locked python execute.py
uv run --locked python execute.py --parameters '{"SITE_OFFSET":10.0}' --output-dir validation/offset-10
uv run --locked python -m ipykernel install --user --name vgi-statsmodels-notebooks --display-name 'Python (vgi-statsmodels notebooks)'
```

Use that kernel in Jupyter with this directory as cwd. Parameters and SQL stay visible; setup, plotting and interpretation implementations carry portable folding metadata. Restart and Run All after editing. The runner uses its own environment's temporary kernel, saves outputs atomically only on success, rejects unknown parameters and requires a separate destination for overrides. Experiment copies include assets, README, standard, runner, constraints and lock.

The literal ATTACH is `uvx --python 3.14 --from git+https://github.com/Query-farm/vgi-statsmodels.git@v0.4.0 --constraints worker-constraints.txt vgi-statsmodels`. The published tag resolves to `03c255e3439bda6224ab7e9424d2494526069ddf`. Tested constraints match the source runtime: statsmodels 0.15.0, Patsy 1.0.3, Formulaic 1.2.2, SDK 0.51.0/RPC 0.53.1, Arrow 26.0.1, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1. Notebook plotting has a separate lock. Upgrade the visible tag and constraints deliberately, then execute both variants. Root source and locks remain untouched.

## Data and statistical interpretation

The visible SQL generator creates 120 observations, 60 per site. Site A inputs span 2–12 hours; site B spans 8–18 hours. The mean response is `20 - 1.5*x + SITE_OFFSET*(site == 'B')`; seed 1729 supplies 120 independent normal error draws with standard deviation 2 response units. `SITE_OFFSET` is bounded to 0–40 units, default 30; the experiment uses 10. The noise fingerprint and input population remain identical.

The real worker receives the entire relation, fits `y ~ x` and `y ~ x + C(site)` and returns coefficients/inference plus separate whole-model statistics. Results are materialized before DataFrame capture to avoid redundant fits. An ordinary staging table bridges the native connection's cursor-local DataFrame registration. The categorical term uses site A as reference; the shared slope is an assumption, not a conclusion tested by this notebook.

Inference is the worker's ordinary nonrobust OLS inference. The interval is for a coefficient, not an individual response or a probability that a fixed coefficient lies in the interval. Classical standard errors assume independent constant-variance errors; correct linear mean specification and normal errors support the finite-sample t calculation. The pooled model deliberately omits a generator term. No causal or out-of-sample performance claim follows from p-values or training R². Primary references: [OLS fit](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html), [coefficient confidence intervals](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.conf_int.html).

## Validation and baseline limitation

The unchanged worker passes 39 pytest tests, 14 stdio SQL assertions, 16 HTTP assertions and 14 Unix-socket assertions (one source SQL case each), Ruff lint/format, mypy (7 files), and isolated pydoclint. SQL transport checks use published Haybarn 2.0.0rc11, scoped ephemeral ports/sockets, and the repository's CI harness.

The existing strict metadata gate fails on one **VGI417 warning** for oversized `statsmodels.main` metadata. Quality score is 100/100 but assurance is L0 because of that structural warning; behavioral checks also ran. This is an unchanged baseline issue, preserved rather than hidden or repaired in worker code. The unit suite also emits existing ADF future-result and singular-design warnings. No live data or Docker service is required for this notebook.

Validation logs, charts, experiment copies and runtime audit JSON are ignored. The main notebook retains successful embedded outputs. See [STANDARD.md](STANDARD.md).
