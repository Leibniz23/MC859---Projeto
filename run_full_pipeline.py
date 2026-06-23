"""Compatibility shim for the most-documented entry point.

The pipeline orchestrator now lives in ``supplychain.cli.run_pipeline``.
Running ``python run_full_pipeline.py ...`` (or the ``run-pipeline`` console
script declared in pyproject.toml) both dispatch to the same ``main()``.
"""
from supplychain.cli.run_pipeline import main

if __name__ == "__main__":
    main()
