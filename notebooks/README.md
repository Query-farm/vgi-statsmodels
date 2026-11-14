# Statsmodels notebooks

Read saved outputs on GitHub or use the **Open in Colab** button at the top of each notebook.

| Notebook | Question | Default → experiment |
|---|---|---|
| [01 · Pooled and within-site slopes](01_pooled_and_within_site_slopes.ipynb) | Does pooling answer the within-site question? | Site B offset 30 → 10 response units |
| [02 · Odds and probability changes](02_odds_are_not_probability_changes.ipynb) | What does an odds multiplier say about a probability difference? | Model-input unit 60 → 30 minutes |
| [03 · Mean shifts and unit roots](03_a_mean_shift_is_not_a_unit_root.ipynb) | Why are these two p-values answering different questions? | Independent batch shift 1 → 0 units |

The first walkthrough retains the original 120-row synthetic OLS analysis, SQL slope/standard-error reconstruction and conditional-inference discussion. The additions use 600 seeded Bernoulli observations, two independent 80-row normal batches, and two 240-point ordered traces. Every observation is synthetic and generated visibly; no credentials, real measurements, external data service or account writes are involved. All fits/tests run through the released worker via actual Haybarn/VGI; Python generates disclosed random draws and plots returned results.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/) and use this directory as the working directory. Initial package, Git-tag worker and community extension downloads need internet. The kernel accepts Python3.11–3.14; uv provisions the released worker's separate Python>=3.13 environment. No source checkout, system Statsmodels installation or manually installed global kernel is needed.

```sh
uv sync --locked --extra runner
uv run --locked --extra runner vgi-notebooks
uv run --locked --extra runner vgi-notebooks 01_pooled_and_within_site_slopes.ipynb --parameters '{"SITE_OFFSET":10.0}' --output-dir validation/offset-10
uv run --locked --extra runner vgi-notebooks 02_odds_are_not_probability_changes.ipynb --parameters '{"MINUTES_PER_UNIT":30}' --output-dir validation/unit-30
uv run --locked --extra runner vgi-notebooks 03_a_mean_shift_is_not_a_unit_root.ipynb --parameters '{"BATCH_SHIFT":0.0}' --output-dir validation/shift-0
```

Interactive Jupyter can use this environment's Python kernel. Restart and Run All after changing the visible parameter. Colab installs only the shared base package, preserving compatible imported plotting/data and kernel libraries; `[runner]` is for local publication only. Shared [v0.3.24](https://github.com/Query-farm/vgi-example-notebooks/releases/tag/v0.3.24) provides canonical branding, isolated SQL sessions, SVG styling, the atomic runner, and coverage validation. No domain helpers or downloaded fixture companions are needed. The runner requires a separate destination for overrides and preserves the last successful output when execution fails.

All notebooks visibly use:

```sql
INSTALL vgi FROM community;
LOAD vgi;
ATTACH 'statsmodels' (TYPE vgi,
 LOCATION 'uvx --from git+https://github.com/Query-farm/vgi-statsmodels@v0.4.0 vgi-statsmodels');
```

Release v0.4.0 resolves to `03c255e3439bda6224ab7e9424d2494526069ddf`. The clean release launcher works without the old notebook constraint file or transitive overrides. The worker release declares version ranges, so the notebook lock fixes kernel/publication packages but does not freeze every worker transitive dependency. Upgrade the visible release deliberately and revalidate results. Each session creates an empty temporary SQL-secret directory with persistence disabled and verifies an empty inventory before ATTACH; cleanup closes the connection and removes that directory.

## Interpretation and coverage

OLS uses ordinary constant-variance, nonrobust inference. Site adjustment assumes a common slope; pooled/adjusted coefficients answer different questions. Training R² is not out-of-sample performance, and coefficient intervals are not prediction intervals or causal evidence.

The binary example compares logit with binomial/logit GLM. It checks coefficient/standard-error agreement, odds-ratio arithmetic and probability reconstruction. A fixed-hour odds multiplier can correspond to different probability changes. Descriptive event bins from the same training rows are not a calibration validation. Halving the model unit halves its slope and standard error while preserving per-hour contrasts and probabilities.

The t-test is **pooled equal-variance**, not Welch. It subtracts groups in first-observed order, made explicit by ordered A/B input; SQL independently reconstructs the statistic and standard error. The release's original docstring mislabels it Welch-style, but its code uses `usevar='pooled'`; current main corrects only that description. ADF uses a constant and AIC lag selection, requires ordered input, and tests a unit-root null. It does not establish full stationarity or cointegration, and failure to reject is not proof of the null. Missingness/populations are checked before either test. See primary [pooled t-test](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttest_ind.html) and [ADF documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html).

The complete released runtime catalog contains **6 table functions and 1 reference view**, all 7 meaningfully demonstrated. `coverage.json` records executed SQL cells, result interpretation, complete native function/table/view exports, and raw saved-output receipts. Current public main has the same seven names/signatures and unchanged runtime; its separate source review retains composed-example metadata and the docstring correction. Current-source review is not current-revision runtime execution.

## Validation and scope

Every default and its independent parameter experiment is executed, and one fresh notebook-only Python3.12 run with compatible libraries already imported tests this shared launcher path. Actual hosted Colab is not claimed. SVG charts remain embedded for Jupyter/Colab and have content-addressed GitHub image links. Review PNGs, execution logs, experiment notebooks and audit JSON are ignored.

Unchanged source CI [43585812310](https://github.com/Query-farm/vgi-statsmodels/actions/runs/43585812310) at `aaac9b3b3c1350329fbaf006bace75e413095fc8` is reused; source builds, unit/transport suites and container publishing are not repeated for notebook-only work. The earlier rollout recorded a stricter local VGI417 metadata warning (oversized schema metadata, 100/100 but L0) and existing source-suite warnings; those historical observations have not been silently claimed fixed. This update's saved notebook outputs must be free of errors and unexpected warnings.

The full [shared standard](https://github.com/Query-farm/vgi-example-notebooks/blob/main/docs/standard.md) and [coverage format](https://github.com/Query-farm/vgi-example-notebooks/blob/main/docs/coverage.md) are maintained centrally.
