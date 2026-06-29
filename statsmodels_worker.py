# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.8",
#     "statsmodels>=0.14",
#     "patsy>=0.5",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""Stdio entry shim for the statsmodels VGI worker.

Lets the worker run straight from a source checkout (``uv run
statsmodels_worker.py``) and keeps ``import statsmodels_worker`` working for
tests. The implementation lives in ``vgi_statsmodels.worker``; installed users
invoke the ``vgi-statsmodels`` console script (which points at
``vgi_statsmodels.worker:main``).

    ATTACH 'statsmodels' (TYPE vgi, LOCATION 'uv run statsmodels_worker.py');
    SELECT * FROM statsmodels.ols((SELECT y, x FROM data), formula := 'y ~ x');
"""

from vgi_statsmodels.worker import StatsmodelsWorker, main

__all__ = ["StatsmodelsWorker", "main"]

if __name__ == "__main__":
    main()
