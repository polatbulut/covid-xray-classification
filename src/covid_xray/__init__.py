"""Chest X-ray classification with PyTorch.

The package is organised around a small number of focused modules:

* :mod:`covid_xray.config` -- typed schema for the YAML experiment files.
* :mod:`covid_xray.data` -- dataset, transforms and dataloader construction.
* :mod:`covid_xray.models` -- the baseline CNN and the transfer-learning factory.
* :mod:`covid_xray.training` -- the training loop, class weighting and checkpointing.
* :mod:`covid_xray.evaluation` -- test-set metrics and diagnostic figures.
* :mod:`covid_xray.cli` -- the ``covid-xray`` command-line entry point.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
