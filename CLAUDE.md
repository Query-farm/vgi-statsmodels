# CLAUDE.md — vgi-statsmodels

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion. Sibling style/tooling
to `vgi-conform` (structure) and `vgi-survival` / `vgi-causal` (the
whole-relation buffering data-flow).

## What this is

A [VGI](https://query.farm) worker exposing **regression with full statistical
inference** and **hypothesis tests** to DuckDB/SQL via
[statsmodels](https://www.statsmodels.org/) + [patsy](https://patsy.readthedocs.io/)
(both BSD): OLS / Logit / GLM regression, OLS fit statistics, two-sample
t-tests, and the Augmented Dickey-Fuller stationarity test.
`statsmodels_worker.py` assembles every function into one `statsmodels` catalog
(single `main` schema) over stdio.

## Layout

```
statsmodels_worker.py   repo-root stdio entry shim; PEP 723 inline deps; main()
vgi_statsmodels/
  stats.py              pure statsmodels logic over pandas frames + Patsy formulas; no Arrow/VGI; unit-testable
  buffering.py          SinkBuffer (single-bucket sink/combine) + Arrow<->pandas plumbing
  tables.py             the six TableBufferingFunction wrappers + output schemas + arg classes
  schema_utils.py       pa.Field comment / column-doc helper
  worker.py             assembles the catalog; main() / main_http()
tests/                  pytest: test_stats (pure), test_tables (in-proc harness), test_client (Client RPC)
test/sql/*.test         haybarn-unittest sqllogictest — authoritative E2E
Makefile                test / test-unit / test-sql / lint
```

To add a function: implement the math in `stats.py` (pure, takes a pandas frame
+ formula/role kwargs, returns a `dict[str, list]`, raises `StatsError` on bad
input), add a `pa.schema` + `@dataclass` args class + a `SinkBuffer` subclass in
`tables.py`, append it to `TABLE_FUNCTIONS`.

## THE core convention (read first): one relation in, named formula/role args

These are **table functions**, not scalars. Each takes the whole input relation
as a single `(SELECT ...)` subquery — `Arg(0)`, typed `TableInput` — and the
model specification as NAMED string args: a Patsy `formula := 'y ~ x1 + x2'` for
the regressions, or `column` / `"group"` roles for the tests, plus `family` for
glm. The relation's columns *are* the data; **every column is available to the
formula by name**. This mirrors `vgi-survival`'s `cox_hazard_ratios(...,
duration := 't', event := 'e')`.

**`column` and `group` are SQL keywords** — in SQL the t-test/adfuller arg names
must be double-quoted: `"column" := 'value'`, `"group" := 'arm'`. The framework
arg keys themselves are plain `column` / `group` (matched to the Python
attribute); only the SQL call site needs the quoting. The in-proc / Client tests
pass the bare names; the `.test` file double-quotes them.

Fits/tests need **every row** before any output, so every function is a
`TableBufferingFunction` (Sink+Source), routed through the C++
`PhysicalVgiTableBuffering` operator:

- `process(batch)` — sink each input batch to execution-scoped `BoundStorage`.
- `combine(state_ids)` — collapse to a single finalize key (one bucket).
- `finalize(...)` — reassemble the full table (`buffered_frame()` → pandas),
  run the statsmodels routine once, emit one result batch, then `out.finish()`.

`SinkBuffer` in `buffering.py` implements `process`/`combine`/`buffered_frame`;
each function only writes `on_bind` (its output schema) + `finalize`. A
`DrainState(done: bool)` cursor makes finalize emit exactly once.

## Sharp edges (learned the hard way)

1. **`haybarn-unittest` silently SKIPS `require vgi`.** Under haybarn the
   extension isn't autoloaded for `require`, so a `.test` using `require vgi` is
   SKIPPED, not run. Use an explicit `statement ok` / `LOAD vgi;` (the `.test`
   here does).
2. **Buffering needs the input schema at bind.** The `(SELECT ...)` relation's
   schema arrives via `bind_call.input_schema`; `buffered_frame()` uses it to
   reassemble even when zero batches were sunk (uniform empty-input handling).
   `Client.table_buffering_function` peeks the first batch to learn that schema,
   so the E2E tests always feed at least the typed columns.
3. **Expensive import paid once.** `import statsmodels.api`/`statsmodels.formula.api`
   happens at `stats.py` module load (worker startup), not per call — it pulls in
   scipy/patsy. Keep heavy imports at module top.
4. **Errors must be clean, never a crash.** `stats.py` wraps patsy `PatsyError`,
   `LinAlgError`, perfect-separation, unknown-family, missing/non-numeric column,
   empty input, and a t-test group that isn't exactly two levels into a
   `StatsError` with an explicit message. The `.test` asserts `statement error`
   with the `invalid formula` substring.
5. **`ols` vs `model_stats` share one OLS fit spec.** `ols` returns the per-term
   coefficient table; `model_stats` returns the whole-model statistics for the
   same formula. They re-fit independently (cheap relative to the buffering /
   RPC round-trip); no shared state.
6. **`adfuller` assumes its input is already ordered.** SQL has no inherent row
   order, so callers must pass `(SELECT ... ORDER BY t)`. Documented in the
   `Meta`, README, and `stats.adfuller`.
7. **The unit suite can pass while the RPC path is broken.** `test_stats.py`
   calls pure functions; only `test_tables.py` (in-proc bind→process→finalize),
   `test_client.py` (real `vgi.client.Client` subprocess), and `test/sql/*.test`
   exercise the framework/wire. **Run the SQL suite** — it's authoritative.

## Known-data validation

Tests build constructed data with a fixed seed so the recovered parameters are
predictable:

- **`y = 2 + 3*x + noise`** → `ols` recovers `Intercept ≈ 2`, `x ≈ 3` with
  `p < 1e-10`, the true coefficient inside the CI, and matches statsmodels' own
  `smf.ols(...).fit()` exactly. `model_stats` reports `r_squared > 0.95`.
- **Logistic data** with a positive driver → `logit` recovers a positive coef,
  `odds_ratio > 1`, `p < 1e-10`.
- **Poisson counts** with log-mean `0.5 + 0.8*x` → `glm(..., family:='poisson')`
  recovers `x ≈ 0.8`. Gaussian GLM matches OLS.
- **Two clearly-different groups** → `ttest` `p` is tiny; identical groups → `p`
  near 1.
- **Random walk** (unit root) → `adfuller` `p` large (non-stationary); **white
  noise** → `p` small (stationary).

Edges: bad formula, missing column, empty input, singular design, non-numeric
column, unknown family, and a single-group t-test all raise `StatsError`.

## Licensing

statsmodels and patsy are **BSD**; numpy and pandas are **BSD** — all permissive,
no copyleft. The worker's own code is MIT. No vendoring, no patched deps.

## Testing

```sh
uv sync --extra dev
uv run --no-sync pytest -q     # pure logic + in-proc tables + Client RPC E2E
make test-sql                  # haybarn-unittest over test/sql/*  (authoritative)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_statsmodels/
```

`make test-sql` sets `VGI_STATSMODELS_WORKER="uv run --python 3.13
statsmodels_worker.py"`, puts `~/.local/bin` on PATH, and runs `haybarn-unittest
--test-dir . "test/sql/*"`. Install the runner once with
`uv tool install haybarn-unittest`. Everything is offline/hermetic.
