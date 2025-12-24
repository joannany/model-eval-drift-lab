"""
Kolmogorov-Smirnov Test for Distribution Drift

The KS test compares the empirical CDFs of two samples.
Good for: Single features, interpretable results, fast.
Bad for: High-dimensional data, subtle shifts.
"""

import numpy as np
from scipy.stats import ks_2samp
from dataclasses import dataclass
from typing import Optional, Dict

from .utils import validate_inputs


@dataclass
class KSResult:
    """Results from a KS test."""
    statistic: float
    p_value: float
    drift_detected: bool
    threshold: float
    
    def __repr__(self) -> str:
        status = "[DRIFT]" if self.drift_detected else "[STABLE]"
        return (
            f"KSResult({status})\n"
            f"  statistic: {self.statistic:.4f}\n"
            f"  p_value:   {self.p_value:.4e}\n"
            f"  threshold: {self.threshold}"
        )


class KSTest:
    """
    Kolmogorov-Smirnov two-sample test for drift detection.
    
    The KS statistic measures the maximum distance between the 
    empirical cumulative distribution functions of two samples.
    
    Example:
        >>> detector = KSTest(alpha=0.05)
        >>> reference = np.random.normal(0, 1, 1000)
        >>> current = np.random.normal(0.5, 1, 1000)  # shifted!
        >>> result = detector.detect(reference, current)
        >>> print(result)
        KSResult([DRIFT])
          statistic: 0.1820
          p_value:   2.1e-15
          threshold: 0.05
    """
    
    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
    
    def detect(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        feature_name: Optional[str] = None
    ) -> KSResult:
        """
        Perform KS test on two samples.
        
        Args:
            reference: Reference (training) distribution samples
            current: Current (production) distribution samples
            feature_name: Optional name for reporting
            
        Returns:
            KSResult with test statistic, p-value, and drift detection flag
        """
        reference, current = validate_inputs(reference, current, min_samples=2)
        
        reference = reference.flatten()
        current = current.flatten()
        
        statistic, p_value = ks_2samp(reference, current)
        drift_detected = p_value < self.alpha
        
        return KSResult(
            statistic=float(statistic),
            p_value=float(p_value),
            drift_detected=drift_detected,
            threshold=self.alpha
        )
    
    def detect_multivariate(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        feature_names: Optional[list] = None
    ) -> Dict[str, KSResult]:
        """
        Apply KS test to each feature independently.
        
        Args:
            reference: Reference data (n_samples, n_features)
            current: Current data (n_samples, n_features)
            feature_names: Optional list of feature names
            
        Returns:
            Dict mapping feature names to KSResult objects
        """
        reference = np.asarray(reference)
        current = np.asarray(current)
        
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
            current = current.reshape(-1, 1)
        
        n_features = reference.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if len(feature_names) != n_features:
            raise ValueError(f"Expected {n_features} feature names, got {len(feature_names)}")
        
        results = {}
        for i, name in enumerate(feature_names):
            results[name] = self.detect(reference[:, i], current[:, i])
        
        return results


if __name__ == "__main__":
    np.random.seed(42)
    
    ref = np.random.normal(0, 1, 1000)
    cur_stable = np.random.normal(0, 1, 1000)
    cur_drifted = np.random.normal(0.3, 1.2, 1000)
    
    detector = KSTest(alpha=0.05)
    
    print("Testing stable distribution:")
    print(detector.detect(ref, cur_stable))
    print()
    print("Testing drifted distribution:")
    print(detector.detect(ref, cur_drifted))
