"""
Shared utilities for drift detection.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class DriftReport:
    """Summary report across multiple drift detectors."""
    feature_name: str
    ks_pvalue: Optional[float]
    psi_score: Optional[float]
    mmd_pvalue: Optional[float]
    overall_status: str  # "stable", "warning", "drift"
    
    def __repr__(self) -> str:
        status_indicator = {
            "stable": "[STABLE]", 
            "warning": "[WARNING]", 
            "drift": "[DRIFT]"
        }
        
        ks_str = f"{self.ks_pvalue:.4f}" if self.ks_pvalue is not None else "N/A"
        psi_str = f"{self.psi_score:.4f}" if self.psi_score is not None else "N/A"
        mmd_str = f"{self.mmd_pvalue:.4f}" if self.mmd_pvalue is not None else "N/A"
        
        return (
            f"{status_indicator[self.overall_status]} {self.feature_name}: {self.overall_status.upper()}\n"
            f"   KS p-value:  {ks_str}\n"
            f"   PSI:         {psi_str}\n"
            f"   MMD p-value: {mmd_str}"
        )


def validate_inputs(
    reference: np.ndarray, 
    current: np.ndarray,
    min_samples: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert inputs to numpy arrays.
    
    Args:
        reference: Reference distribution samples
        current: Current distribution samples
        min_samples: Minimum required samples per distribution
        
    Returns:
        Tuple of validated numpy arrays
        
    Raises:
        ValueError: If inputs are invalid
    """
    reference = np.asarray(reference)
    current = np.asarray(current)
    
    if len(reference) < min_samples:
        raise ValueError(
            f"Reference data has {len(reference)} samples, need at least {min_samples}"
        )
    
    if len(current) < min_samples:
        raise ValueError(
            f"Current data has {len(current)} samples, need at least {min_samples}"
        )
    
    if reference.ndim > 2 or current.ndim > 2:
        raise ValueError("Data must be 1D or 2D arrays")
    
    if np.any(~np.isfinite(reference)):
        raise ValueError("Reference data contains NaN or Inf values")
    
    if np.any(~np.isfinite(current)):
        raise ValueError("Current data contains NaN or Inf values")
    
    return reference, current


def bootstrap_confidence_interval(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    reference: np.ndarray,
    current: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        statistic_fn: Function that takes (reference, current) and returns a float
        reference: Reference distribution samples
        current: Current distribution samples
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default: 0.95)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    
    point_estimate = statistic_fn(reference, current)
    
    bootstrap_stats = []
    n_ref, n_cur = len(reference), len(current)
    
    for _ in range(n_bootstrap):
        ref_boot = reference[rng.choice(n_ref, n_ref, replace=True)]
        cur_boot = current[rng.choice(n_cur, n_cur, replace=True)]
        bootstrap_stats.append(statistic_fn(ref_boot, cur_boot))
    
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    
    return point_estimate, lower, upper


def aggregate_drift_results(
    ks_result=None,
    psi_result=None, 
    mmd_result=None,
    feature_name: str = "feature"
) -> DriftReport:
    """
    Aggregate results from multiple drift detectors into a single report.
    
    Args:
        ks_result: Optional KSResult object
        psi_result: Optional PSIResult object
        mmd_result: Optional MMDResult object
        feature_name: Name for the feature being analyzed
        
    Returns:
        DriftReport with overall status based on all detectors
    """
    drift_detected = False
    warning = False
    
    ks_pvalue = psi_score = mmd_pvalue = None
    
    if ks_result is not None:
        ks_pvalue = ks_result.p_value
        if ks_result.drift_detected:
            drift_detected = True
    
    if psi_result is not None:
        psi_score = psi_result.psi
        if psi_result.drift_level == "significant":
            drift_detected = True
        elif psi_result.drift_level == "moderate":
            warning = True
    
    if mmd_result is not None:
        mmd_pvalue = mmd_result.p_value
        if mmd_result.drift_detected:
            drift_detected = True
    
    if drift_detected:
        status = "drift"
    elif warning:
        status = "warning"
    else:
        status = "stable"
    
    return DriftReport(
        feature_name=feature_name,
        ks_pvalue=ks_pvalue,
        psi_score=psi_score,
        mmd_pvalue=mmd_pvalue,
        overall_status=status
    )
