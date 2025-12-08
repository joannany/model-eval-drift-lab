"""
Population Stability Index (PSI)

PSI is the standard drift metric in credit risk and insurance.
It bins feature values and compares the proportion in each bin
between reference and current distributions.

Interpretation:
  PSI < 0.1:  No significant drift
  PSI 0.1-0.2: Moderate drift, investigate
  PSI > 0.2:  Significant drift, action required
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class PSIResult:
    """Results from PSI calculation."""
    psi: float
    bin_psis: np.ndarray
    expected_percents: np.ndarray
    actual_percents: np.ndarray
    drift_level: str  # "stable", "moderate", "significant"
    
    def __repr__(self):
        emoji = {"stable": "🟢", "moderate": "🟡", "significant": "🔴"}
        return (
            f"PSIResult({emoji[self.drift_level]} {self.drift_level.upper()})\n"
            f"  psi: {self.psi:.4f}\n"
            f"  interpretation: PSI {'<' if self.psi < 0.1 else '>' if self.psi > 0.2 else '~'} "
            f"{'0.1 (stable)' if self.psi < 0.1 else '0.2 (significant)' if self.psi > 0.2 else '0.1-0.2 (moderate)'}"
        )


class PSI:
    """
    Population Stability Index calculator.
    
    PSI measures how much a distribution has shifted by comparing
    the proportion of observations in each bin.
    
    Formula:
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    
    Example:
        >>> calculator = PSI(n_bins=10)
        >>> reference = np.random.normal(100, 15, 10000)
        >>> current = np.random.normal(105, 18, 10000)  # shifted
        >>> result = calculator.calculate(reference, current)
        >>> print(result)
    """
    
    def __init__(
        self, 
        n_bins: int = 10, 
        binning: Literal["quantile", "uniform"] = "quantile"
    ):
        self.n_bins = n_bins
        self.binning = binning
    
    def calculate(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        eps: float = 1e-4
    ) -> PSIResult:
        reference = np.asarray(reference).flatten()
        current = np.asarray(current).flatten()
        
        # Create bins based on reference distribution
        if self.binning == "quantile":
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(reference, percentiles)
            bin_edges = np.unique(bin_edges)
        else:
            bin_edges = np.linspace(reference.min(), reference.max(), self.n_bins + 1)
        
        # Extend edges to include all values
        bin_edges[0] = min(bin_edges[0], current.min()) - eps
        bin_edges[-1] = max(bin_edges[-1], current.max()) + eps
        
        # Calculate proportions in each bin
        expected_counts = np.histogram(reference, bins=bin_edges)[0]
        actual_counts = np.histogram(current, bins=bin_edges)[0]
        
        expected_percents = expected_counts / len(reference)
        actual_percents = actual_counts / len(current)
        
        # Avoid division by zero
        expected_percents = np.clip(expected_percents, eps, 1)
        actual_percents = np.clip(actual_percents, eps, 1)
        
        # PSI per bin
        bin_psis = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(bin_psis)
        
        # Interpret
        if psi < 0.1:
            drift_level = "stable"
        elif psi < 0.2:
            drift_level = "moderate"
        else:
            drift_level = "significant"
        
        return PSIResult(
            psi=psi,
            bin_psis=bin_psis,
            expected_percents=expected_percents,
            actual_percents=actual_percents,
            drift_level=drift_level
        )
    
    def calculate_multivariate(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        feature_names: Optional[list] = None
    ) -> dict:
        reference = np.asarray(reference)
        current = np.asarray(current)
        
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
            current = current.reshape(-1, 1)
        
        n_features = reference.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        results = {}
        for i, name in enumerate(feature_names):
            results[name] = self.calculate(reference[:, i], current[:, i])
        
        return results


if __name__ == "__main__":
    np.random.seed(42)
    
    reference = np.random.normal(loc=120, scale=10, size=5000)
    current_stable = np.random.normal(loc=120, scale=10, size=5000)
    current_moderate = np.random.normal(loc=123, scale=11, size=5000)
    current_drifted = np.random.normal(loc=128, scale=14, size=5000)
    
    calculator = PSI(n_bins=10, binning="quantile")
    
    print("Scenario 1: Same conditions")
    print(calculator.calculate(reference, current_stable))
    print()
    print("Scenario 2: Minor calibration shift")
    print(calculator.calculate(reference, current_moderate))
    print()
    print("Scenario 3: Population shift")
    print(calculator.calculate(reference, current_drifted))