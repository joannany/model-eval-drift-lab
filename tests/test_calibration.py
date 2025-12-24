"""Tests for calibration module."""

import numpy as np
import pytest

from evaluation.calibration import CalibrationAnalyzer


class TestCalibrationAnalyzer:
    """Tests for CalibrationAnalyzer class."""
    
    def test_raises_on_empty_input(self) -> None:
        """Empty arrays should raise ValueError."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            analyzer.analyze(np.array([]), np.array([]))
    
    def test_raises_on_length_mismatch(self) -> None:
        """Mismatched lengths should raise ValueError."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.1, 0.9])
        with pytest.raises(ValueError, match="Length mismatch"):
            analyzer.analyze(y_true, y_prob)
    
    def test_raises_on_out_of_range_probs(self) -> None:
        """Probabilities outside [0, 1] should raise ValueError."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 1.2, 0.4, -0.1])
        with pytest.raises(ValueError, match="range \\[0, 1\\]"):
            analyzer.analyze(y_true, y_prob)
    
    def test_raises_on_invalid_n_bins(self) -> None:
        """n_bins < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bins must be at least 1"):
            CalibrationAnalyzer(n_bins=0)
    
    def test_ece_mce_in_valid_range(self) -> None:
        """ECE and MCE should be in [0, 1]."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=2000)
        y_prob = rng.random(size=2000)
        
        result = analyzer.analyze(y_true, y_prob)
        
        assert 0.0 <= result.ece <= 1.0
        assert 0.0 <= result.mce <= 1.0
    
    def test_bin_counts_sum_to_total(self) -> None:
        """Bin counts should sum to total number of samples."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=2000)
        y_prob = rng.random(size=2000)
        
        result = analyzer.analyze(y_true, y_prob)
        
        assert int(result.bin_counts.sum()) == len(y_true)
    
    def test_bin_arrays_have_correct_shape(self) -> None:
        """Bin arrays should have shape (n_bins,)."""
        n_bins = 10
        analyzer = CalibrationAnalyzer(n_bins=n_bins)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=1000)
        y_prob = rng.random(size=1000)
        
        result = analyzer.analyze(y_true, y_prob)
        
        assert result.bin_accuracies.shape == (n_bins,)
        assert result.bin_confidences.shape == (n_bins,)
        assert result.bin_counts.shape == (n_bins,)
    
    def test_perfect_calibration_has_low_ece(self) -> None:
        """Perfectly calibrated predictions should have ECE close to 0."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        rng = np.random.default_rng(42)
        
        # Generate well-calibrated predictions
        y_prob = rng.random(size=5000)
        y_true = (rng.random(size=5000) < y_prob).astype(int)
        
        result = analyzer.analyze(y_true, y_prob)
        
        # ECE should be relatively low for well-calibrated model
        assert result.ece < 0.1
    
    def test_repr_contains_key_info(self) -> None:
        """String representation should contain ECE and interpretation."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        result = analyzer.analyze(y_true, y_prob)
        repr_str = repr(result)
        
        assert "ECE" in repr_str
        assert "MCE" in repr_str
