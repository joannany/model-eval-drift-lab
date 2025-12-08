"""
Maximum Mean Discrepancy (MMD)

MMD compares distributions by mapping samples to a reproducing kernel
Hilbert space (RKHS) and comparing their mean embeddings. This captures
differences in the full joint distribution, not just marginals.

Good for:
- High-dimensional data (images, embeddings)
- Detecting subtle multivariate shifts
- When per-feature tests miss correlated changes
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class MMDResult:
    """Results from MMD calculation."""
    mmd: float
    mmd_squared: float
    p_value: Optional[float]
    drift_detected: bool
    threshold: float
    
    def __repr__(self):
        status = "🔴 DRIFT" if self.drift_detected else "🟢 STABLE"
        p_str = f"{self.p_value:.4f}" if self.p_value is not None else "N/A"
        return (
            f"MMDResult({status})\n"
            f"  mmd:     {self.mmd:.4f}\n"
            f"  mmd²:    {self.mmd_squared:.6f}\n"
            f"  p_value: {p_str}\n"
            f"  threshold: {self.threshold}"
        )


class MMD:
    """
    Maximum Mean Discrepancy for distribution comparison.
    
    Uses RBF (Gaussian) kernel by default.
    
    Example:
        >>> detector = MMD(sigma=1.0)
        >>> X_ref = np.random.multivariate_normal([0, 0], np.eye(2), 500)
        >>> X_cur = np.random.multivariate_normal([0.3, 0.3], np.eye(2), 500)
        >>> result = detector.detect(X_ref, X_cur, permutation_test=True)
        >>> print(result)
    """
    
    def __init__(
        self, 
        sigma: Optional[float] = None,
        kernel: Literal["rbf", "linear"] = "rbf"
    ):
        self.sigma = sigma
        self.kernel = kernel
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        X_sqnorms = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sqnorms = np.sum(Y ** 2, axis=1, keepdims=True)
        sq_distances = X_sqnorms + Y_sqnorms.T - 2 * X @ Y.T
        return np.exp(-sq_distances / (2 * sigma ** 2))
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return X @ Y.T
    
    def _median_heuristic(self, X: np.ndarray, Y: np.ndarray) -> float:
        n_subsample = min(1000, len(X), len(Y))
        X_sub = X[np.random.choice(len(X), n_subsample, replace=False)]
        Y_sub = Y[np.random.choice(len(Y), n_subsample, replace=False)]
        
        combined = np.vstack([X_sub, Y_sub])
        sq_dists = np.sum(combined ** 2, axis=1, keepdims=True) + \
                   np.sum(combined ** 2, axis=1) - 2 * combined @ combined.T
        
        sq_dists = sq_dists[np.triu_indices_from(sq_dists, k=1)]
        return np.sqrt(np.median(sq_dists[sq_dists > 0]))
    
    def _compute_mmd_squared(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        sigma: float
    ) -> float:
        m, n = len(X), len(Y)
        
        if self.kernel == "rbf":
            K_XX = self._rbf_kernel(X, X, sigma)
            K_YY = self._rbf_kernel(Y, Y, sigma)
            K_XY = self._rbf_kernel(X, Y, sigma)
        else:
            K_XX = self._linear_kernel(X, X)
            K_YY = self._linear_kernel(Y, Y)
            K_XY = self._linear_kernel(X, Y)
        
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        mmd_sq = (
            np.sum(K_XX) / (m * (m - 1)) +
            np.sum(K_YY) / (n * (n - 1)) -
            2 * np.sum(K_XY) / (m * n)
        )
        
        return mmd_sq
    
    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        permutation_test: bool = True,
        n_permutations: int = 100,
        alpha: float = 0.05
    ) -> MMDResult:
        reference = np.asarray(reference)
        current = np.asarray(current)
        
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
            current = current.reshape(-1, 1)
        
        sigma = self.sigma if self.sigma is not None else self._median_heuristic(reference, current)
        
        mmd_squared = self._compute_mmd_squared(reference, current, sigma)
        mmd = np.sqrt(max(0, mmd_squared))
        
        p_value = None
        if permutation_test:
            combined = np.vstack([reference, current])
            n_ref = len(reference)
            
            null_mmds = []
            for _ in range(n_permutations):
                perm = np.random.permutation(len(combined))
                X_perm = combined[perm[:n_ref]]
                Y_perm = combined[perm[n_ref:]]
                null_mmd_sq = self._compute_mmd_squared(X_perm, Y_perm, sigma)
                null_mmds.append(null_mmd_sq)
            
            p_value = np.mean(np.array(null_mmds) >= mmd_squared)
            drift_detected = p_value < alpha
        else:
            drift_detected = False
        
        return MMDResult(
            mmd=mmd,
            mmd_squared=mmd_squared,
            p_value=p_value,
            drift_detected=drift_detected,
            threshold=alpha
        )


if __name__ == "__main__":
    np.random.seed(42)
    
    n_samples = 500
    n_features = 5
    
    mean_ref = np.zeros(n_features)
    cov_ref = np.eye(n_features)
    X_ref = np.random.multivariate_normal(mean_ref, cov_ref, n_samples)
    
    X_stable = np.random.multivariate_normal(mean_ref, cov_ref, n_samples)
    
    mean_shifted = np.array([0.3, 0.3, 0, 0, 0])
    X_shifted = np.random.multivariate_normal(mean_shifted, cov_ref, n_samples)
    
    cov_changed = np.eye(n_features)
    cov_changed[0, 1] = cov_changed[1, 0] = 0.7
    X_cov_changed = np.random.multivariate_normal(mean_ref, cov_changed, n_samples)
    
    detector = MMD()
    
    print("Scenario 1: No drift")
    print(detector.detect(X_ref, X_stable, n_permutations=100))
    print()
    print("Scenario 2: Mean shift")
    print(detector.detect(X_ref, X_shifted, n_permutations=100))
    print()
    print("Scenario 3: Covariance change (same means)")
    print(detector.detect(X_ref, X_cov_changed, n_permutations=100))