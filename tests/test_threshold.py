"""Tests for threshold optimization module."""

import numpy as np
import pytest

from evaluation.threshold import ThresholdOptimizer


class TestThresholdOptimizer:
    """Tests for ThresholdOptimizer class."""
    
    def test_raises_on_empty_input(self) -> None:
        """Empty arrays should raise ValueError."""
        opt = ThresholdOptimizer()
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            opt.optimize_youden(np.array([]), np.array([]))
    
    def test_raises_on_length_mismatch(self) -> None:
        """Mismatched lengths should raise ValueError."""
        opt = ThresholdOptimizer()
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.1, 0.9])
        with pytest.raises(ValueError, match="Length mismatch"):
            opt.optimize_youden(y_true, y_prob)
    
    def test_raises_on_single_class(self) -> None:
        """Single class in y_true should raise ValueError."""
        opt = ThresholdOptimizer()
        y_true = np.zeros(10, dtype=int)
        y_prob = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="must contain both positive and negative"):
            opt.optimize_youden(y_true, y_prob)
    
    def test_optimize_sensitivity_achieves_target(self) -> None:
        """Should achieve target sensitivity when possible."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=500)
        y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.15, 500), 0, 1)
        
        result = opt.optimize_sensitivity(y_true, y_prob, min_sensitivity=0.90)
        
        assert result is not None
        assert result.sensitivity >= 0.90
    
    def test_optimize_specificity_achieves_target(self) -> None:
        """Should achieve target specificity when possible."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=500)
        y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.15, 500), 0, 1)
        
        result = opt.optimize_specificity(y_true, y_prob, min_specificity=0.90)
        
        assert result is not None
        assert result.specificity >= 0.90
    
    def test_youden_threshold_in_valid_range(self) -> None:
        """Optimal threshold should be in [0, 1]."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=1000)
        y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.2, 1000), 0, 1)
        
        result = opt.optimize_youden(y_true, y_prob)
        
        assert 0.0 <= result.optimal_threshold <= 1.0
    
    def test_metrics_in_valid_range(self) -> None:
        """All metrics should be in [0, 1]."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=1000)
        y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.2, 1000), 0, 1)
        
        result = opt.optimize_youden(y_true, y_prob)
        
        assert 0.0 <= result.sensitivity <= 1.0
        assert 0.0 <= result.specificity <= 1.0
        assert 0.0 <= result.ppv <= 1.0
        assert 0.0 <= result.npv <= 1.0
    
    def test_optimize_f1_returns_valid_result(self) -> None:
        """F1 optimization should return valid result."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=1000)
        y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.2, 1000), 0, 1)
        
        result = opt.optimize_f1(y_true, y_prob)
        
        assert 0.0 <= result.optimal_threshold <= 1.0
        assert "F1" in result.metric_name
    
    def test_optimize_cost_returns_valid_result(self) -> None:
        """Cost optimization should return valid result."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=2000)
        y_prob = np.clip(0.2 + 0.6 * y_true + rng.normal(0, 0.15, 2000), 0, 1)
        
        result = opt.optimize_cost(y_true, y_prob, fp_cost=1.0, fn_cost=10.0)
        
        assert 0.0 <= result.optimal_threshold <= 1.0
        assert 0.0 <= result.sensitivity <= 1.0
        assert 0.0 <= result.specificity <= 1.0
    
    def test_high_fn_cost_lowers_threshold(self) -> None:
        """Higher FN cost should result in lower threshold (more sensitive)."""
        opt = ThresholdOptimizer()
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=2000)
        y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.2, 2000), 0, 1)
        
        result_balanced = opt.optimize_cost(y_true, y_prob, fp_cost=1.0, fn_cost=1.0)
        result_fn_heavy = opt.optimize_cost(y_true, y_prob, fp_cost=1.0, fn_cost=10.0)
        
        # Higher FN cost should lead to lower threshold (catch more positives)
        assert result_fn_heavy.optimal_threshold <= result_balanced.optimal_threshold
    
    def test_repr_contains_key_info(self) -> None:
        """String representation should contain threshold and metrics."""
        opt = ThresholdOptimizer()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        
        result = opt.optimize_youden(y_true, y_prob)
        repr_str = repr(result)
        
        assert "threshold" in repr_str.lower()
        assert "Sensitivity" in repr_str
        assert "Specificity" in repr_str
