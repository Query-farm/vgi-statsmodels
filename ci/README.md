# CI: the vgi-statsmodels worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-statsmodels
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

Rather than building the vgi DuckDB extension from source, CI drives a
**prebuilt** standalone `haybarn-unittest` (the DuckDB/Haybarn sqllogictest
runner, published in Haybarn's releases) and installs the **signed** `vgi`
extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen --extra dev` into a venv.
   `.venv/bin/vgi-statsmodels` is the installed console-script stdio worker the
   extension can spawn. The main dependency is `vgi-python[http]`, so the sync
   also installs `waitress` (the HTTP server used for the `http` transport leg).
2. **Download the runner** — the matching `haybarn_unittest-*` asset per
   platform from the latest Haybarn release.
3. **Preprocess** — the standalone runner links none of the extensions the
   tests gate on, so [`preprocess-require.awk`](preprocess-require.awk) rewrites
   each `require <ext>` into an explicit signed `INSTALL <ext> FROM
   {community,core}; LOAD <ext>;`. These tests skip `require vgi` (haybarn
   silently SKIPs it) and `LOAD vgi;` directly, so the awk also injects an
   `INSTALL vgi FROM community;` right before each bare `LOAD vgi;`. `require-env`
   and everything else pass through untouched.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, resolves `VGI_STATSMODELS_WORKER` per transport (see below), warms the
   extension cache once, then runs the suite in a single `haybarn-unittest`
   invocation. Any failed assertion exits non-zero and fails the job.

## Three transports (one suite)

The integration job is a matrix of `transport: [subprocess, http, unix]` ×
`os: [ubuntu-latest, macos-latest]`. The SAME `test/sql/*.test` suite runs over
every VGI transport — the vgi extension picks the transport from the ATTACH
LOCATION string that `run-integration.sh` sets in `VGI_STATSMODELS_WORKER`:

- **subprocess** (default) — `LOCATION` is the bare stdio command
  (`.venv/bin/vgi-statsmodels`); the extension spawns the worker per query and
  talks Arrow IPC over stdin/stdout.
- **http** — the script boots `vgi-statsmodels --http --port 0 --port-file <f>`
  with cwd = the stage dir, polls the port-file for the auto-selected port, and
  sets `LOCATION = http://127.0.0.1:<port>` (bare `scheme://host:port`, no path
  suffix — the extension POSTs each RPC method at `<LOCATION>/<method>`). The
  HTTP transport rides DuckDB's `httpfs`, so the script injects
  `INSTALL httpfs FROM core; LOAD httpfs;` after each `LOAD vgi;` in the staged
  tests for this leg only — without it the ATTACH fails with "VGI HTTP transport
  requires the httpfs extension." The statsmodels/scipy import is heavy, so the
  port poll runs up to ~90s.
- **unix** — the script boots `vgi-statsmodels --unix <sock>` (cwd = stage dir),
  polls for the socket file, and sets `LOCATION = unix://<sock>`.

All six table functions are whole-relation *buffering* (Sink+Source) functions
that produce their entire result in a single `finalize` emit, so they work over
the stateless HTTP transport unchanged: the single result batch is small (a
coefficient/statistic table), and the `DrainState` cursor flips `done` **before**
the emit, so an HTTP continuation snapshot never re-emits. No tests are gated.

### The silent-skip guard

The DuckDB/Haybarn sqllogictest runner SKIPS (exit 0, not a failure) any test
whose error message contains "HTTP" or "Unable to connect". A broken http leg
would therefore report "All tests were skipped" and the job would go GREEN
having run nothing — a fake pass. `run-integration.sh` captures the run log and
fails the leg if it sees "All tests were skipped", surfacing the real reason.

## Run it locally

```bash
uv sync --extra dev --python 3.13           # install the worker + deps (+ waitress)
# point HAYBARN_UNITTEST at a haybarn-unittest binary (or a local DuckDB
# `unittest` built with the vgi extension), and the worker at the stdio command:
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_STATSMODELS_WORKER="$PWD/.venv/bin/vgi-statsmodels" \
TRANSPORT=subprocess \
  ci/run-integration.sh        # or TRANSPORT=http / TRANSPORT=unix
```

Or use the Makefile target `make test-sql` (subprocess only).
