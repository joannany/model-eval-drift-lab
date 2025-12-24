"""
Drift Detection Module

Statistical methods for detecting when production data
no longer matches your training distribution.
"""

__version__ = "0.1.0"

from .ks_test import KSTest, KSResult
from .psi import PSI, PSIResult
from .mmd import MMD, MMDResult
from .utils import (
    DriftReport,
    validate_inputs,
    bootstrap_confidence_interval,
    aggregate_drift_results,
)

__all__ = [
    "__version__",
    "KSTest",
    "KSResult",
    "PSI",
    "PSIResult",
    "MMD",
    "MMDResult",
    "DriftReport",
    "validate_inputs",
    "bootstrap_confidence_interval",
    "aggregate_drift_results",
]
