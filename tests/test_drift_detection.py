"""Tests for drift detection module."""

import numpy as np
import pytest

from drift_detection import KSTest, PSI, MMD, validate_inputs


class TestKSTest:
    """Tests for KSTest class."""
    
    def test_raises_on_insufficient_samples(self) -> None:
        """Should raise if either distribution has too few samples."""
        detector = KSTest(alpha=0.05)
        with pytest.raises(ValueError, match="need at least"):
            detector.detect(np.array([1]), np.array([1, 2, 3]))
    
    def test_raises_on_invalid_alpha(self) -> None:
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            KSTest(alpha=0)
        with pytest.raises(ValueError, match="alpha must be in"):
            KSTest(alpha=1)
    
    def test_detects_no_drift_for_same_distribution(self) -> None:
        """Same distribution should not trigger drift detection."""
        detector = KSTest(alpha=0.05)
        rng = np.random.default_rng(42)
        
        data = rng.normal(0, 1, 1000)
        ref = data[:500]
        cur = data[500:]
        
        result = detector.detect(ref, cur)
        
        # p-value should be high (no significant difference)
        assert result.p_value > 0.05
        assert not result.drift_detected
    
    def test_detects_drift_for_shifted_distribution(self) -> None:
        """Shifted distribution should trigger drift detection."""
        detector = KSTest(alpha=0.05)
        rng = np.random.default_rng(42)
        
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(1, 1, 1000)  # shifted mean
        
        result = detector.detect(ref, cur)
        
        assert result.p_value < 0.05
        assert result.drift_detected
    
    def test_multivariate_returns_dict(self) -> None:
        """detect_multivariate should return dict with results per feature."""
        detector = KSTest(alpha=0.05)
        rng = np.random.default_rng(42)
        
        ref = rng.normal(0, 1, (500, 3))
        cur = rng.normal(0, 1, (500, 3))
        
        results = detector.detect_multivariate(ref, cur, ["a", "b", "c"])
        
        assert len(results) == 3
        assert all(name in results for name in ["a", "b", "c"])


class TestPSI:
    """Tests for PSI class."""
    
    def test_raises_on_invalid_n_bins(self) -> None:
        """n_bins < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bins must be at least 2"):
            PSI(n_bins=1)
    
    def test_stable_distribution_has_low_psi(self) -> None:
        """Same distribution should have low PSI."""
        calculator = PSI(n_bins=10)
        rng = np.random.default_rng(42)
        
        ref = rng.normal(100, 15, 5000)
        cur = rng.normal(100, 15, 5000)
        
        result = calculator.calculate(ref, cur)
        
        assert result.psi < 0.1
        assert result.drift_level == "stable"
    
    def test_shifted_distribution_has_high_psi(self) -> None:
        """Shifted distribution should have high PSI."""
        calculator = PSI(n_bins=10)
        rng = np.random.default_rng(42)
        
        ref = rng.normal(100, 15, 5000)
        cur = rng.normal(120, 20, 5000)  # shifted
        
        result = calculator.calculate(ref, cur)
        
        assert result.psi > 0.2
        assert result.drift_level == "significant"
    
    def test_n_bins_used_tracked(self) -> None:
        """Result should track actual number of bins used."""
        calculator = PSI(n_bins=10)
        rng = np.random.default_rng(42)
        
        ref = rng.normal(100, 15, 1000)
        cur = rng.normal(100, 15, 1000)
        
        result = calculator.calculate(ref, cur)
        
        assert result.n_bins_used > 0
        assert result.n_bins_used <= 10


class TestMMD:
    """Tests for MMD class."""
    
    def test_raises_on_insufficient_samples(self) -> None:
        """Should raise if either distribution has < 2 samples."""
        detector = MMD()
        with pytest.raises(ValueError):
            detector.detect(np.array([[1]]), np.array([[1], [2], [3]]))
    
    def test_raises_on_invalid_sigma(self) -> None:
        """sigma <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            MMD(sigma=0)
        with pytest.raises(ValueError, match="sigma must be positive"):
            MMD(sigma=-1)
    
    def test_detects_no_drift_for_same_distribution(self) -> None:
        """Same distribution should not trigger drift detection."""
        detector = MMD()
        rng = np.random.default_rng(42)
        
        ref = rng.multivariate_normal([0, 0], np.eye(2), 300)
        cur = rng.multivariate_normal([0, 0], np.eye(2), 300)
        
        result = detector.detect(ref, cur, n_permutations=50)
        
        # p-value should be relatively high
        assert result.p_value > 0.01
    
    def test_detects_drift_for_shifted_distribution(self) -> None:
        """Shifted distribution should trigger drift detection."""
        detector = MMD()
        rng = np.random.default_rng(42)
        
        ref = rng.multivariate_normal([0, 0], np.eye(2), 300)
        cur = rng.multivariate_normal([1, 1], np.eye(2), 300)  # shifted
        
        result = detector.detect(ref, cur, n_permutations=50)
        
        assert result.p_value < 0.05
        assert result.drift_detected
    
    def test_detects_correlation_drift(self) -> None:
        """Should detect changes in correlation structure."""
        detector = MMD()
        rng = np.random.default_rng(42)
        
        ref = rng.multivariate_normal([0, 0], np.eye(2), 500)
        
        cov_correlated = np.array([[1, 0.8], [0.8, 1]])
        cur = rng.multivariate_normal([0, 0], cov_correlated, 500)
        
        result = detector.detect(ref, cur, n_permutations=100)
        
        # MMD should detect correlation change
        assert result.mmd > 0
    
    def test_mmd_is_non_negative(self) -> None:
        """MMD should always be non-negative."""
        detector = MMD()
        rng = np.random.default_rng(42)
        
        ref = rng.normal(0, 1, (200, 5))
        cur = rng.normal(0.5, 1.2, (200, 5))
        
        result = detector.detect(ref, cur, permutation_test=False)
        
        assert result.mmd >= 0


class TestValidateInputs:
    """Tests for validate_inputs function."""
    
    def test_raises_on_nan_values(self) -> None:
        """Should raise if data contains NaN."""
        ref = np.array([1, 2, np.nan, 4, 5] * 10)
        cur = np.array([1, 2, 3, 4, 5] * 10)
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_inputs(ref, cur, min_samples=10)
    
    def test_raises_on_inf_values(self) -> None:
        """Should raise if data contains Inf."""
        ref = np.array([1, 2, np.inf, 4, 5] * 10)
        cur = np.array([1, 2, 3, 4, 5] * 10)
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_inputs(ref, cur, min_samples=10)
    
    def test_raises_on_3d_array(self) -> None:
        """Should raise if data is more than 2D."""
        ref = np.ones((10, 10, 10))
        cur = np.ones((10, 10, 10))
        
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            validate_inputs(ref, cur, min_samples=5)
    
    def test_returns_numpy_arrays(self) -> None:
        """Should convert inputs to numpy arrays."""
        ref = [1, 2, 3, 4, 5] * 10
        cur = [1, 2, 3, 4, 5] * 10
        
        ref_out, cur_out = validate_inputs(ref, cur, min_samples=10)
        
        assert isinstance(ref_out, np.ndarray)
        assert isinstance(cur_out, np.ndarray)
