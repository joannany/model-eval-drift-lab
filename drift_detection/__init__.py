"""
Drift Detection Module

Statistical methods for detecting when production data
no longer matches your training distribution.
"""

from .ks_test import KSTest
from .psi import PSI
from .mmd import MMD
from .utils import (
    DriftReport,
    validate_inputs,
    bootstrap_confidence_interval,
    aggregate_drift_results,
)

__all__ = [
    "KSTest",
    "PSI",
    "MMD",
    "DriftReport",
    "validate_inputs",
    "bootstrap_confidence_interval",
    "aggregate_drift_results",
]
