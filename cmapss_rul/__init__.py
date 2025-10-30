# ===============================================================
# cmapss_rul/__init__.py
# ===============================================================
# This file turns the 'cmapss_rul' folder into a Python package.
#
# Purpose:
#   - Allows you to import functions and modules like:
#         from cmapss_rul import config, model_tcn
#   - Provides a clean, centralized import interface so that
#     your main.py or notebooks can use one-line imports.
#
# Why it matters:
#   - Keeps your project modular and organized.
#   - Makes importing submodules easy and consistent.
# ===============================================================

# Re-export submodules for convenient imports in main.py
from . import (
    config,
    download,
    load,
    explore,
    preprocess,
    regimes,
    sequences,
    eval,
    model_tcn,
    model_lstm,
    model_cnn,
)

# __all__ defines which names are publicly available when someone imports
#   from cmapss_rul import *
# It’s a good practice to list only the intended public modules here.
__all__ = [
    "config",
    "download",
    "load",
    "explore",
    "preprocess",
    "regimes",
    "sequences",
    "eval",
    "model_tcn",
    "model_lstm",
    "model_cnn",
]
