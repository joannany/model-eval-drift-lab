"""
Model Calibration Analysis

A well-calibrated model's predicted probabilities match empirical frequencies.
When a model says "80% chance of cancer", ~80% of those cases should actually be cancer.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    ece: float
    mce: float
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    
    def __repr__(self) -> str:
        return (
            f"CalibrationResult\n"
            f"  ECE: {self.ece:.4f} (lower is better)\n"
            f"  MCE: {self.mce:.4f}\n"
            f"  Interpretation: "
            f"{'Well-calibrated' if self.ece < 0.05 else 'Moderate miscalibration' if self.ece < 0.1 else 'Poorly calibrated'}"
        )


class CalibrationAnalyzer:
    """
    Analyze and visualize model calibration.
    
    Example:
        >>> analyzer = CalibrationAnalyzer(n_bins=10)
        >>> result = analyzer.analyze(y_true, y_prob)
        >>> print(result)
        CalibrationResult
          ECE: 0.0234 (lower is better)
          MCE: 0.0891
          Interpretation: Well-calibrated
        >>> analyzer.plot_reliability_diagram(y_true, y_prob)
    """
    
    def __init__(self, n_bins: int = 10) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        self.n_bins = n_bins
    
    def analyze(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> CalibrationResult:
        """
        Compute calibration metrics.
        
        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_prob: Predicted probabilities for the positive class
            
        Returns:
            CalibrationResult with ECE, MCE, and per-bin statistics
            
        Raises:
            ValueError: If inputs are empty or have mismatched lengths
        """
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        # Input validation
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty")
        if len(y_true) != len(y_prob):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_prob ({len(y_prob)})")
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            raise ValueError("y_prob must be in range [0, 1]")
        
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        
        # Fix: np.digitize returns 1-indexed values, clip to valid range
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
        # Now bin_indices is in range [0, n_bins-1] for values in [0, 1]
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        bin_accuracies = np.zeros(self.n_bins)
        bin_confidences = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            bin_counts[i] = np.sum(mask)
            
            if bin_counts[i] > 0:
                bin_accuracies[i] = np.mean(y_true[mask])
                bin_confidences[i] = np.mean(y_prob[mask])
        
        # ECE: weighted average of calibration gaps
        weights = bin_counts / len(y_prob)
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(weights * gaps)
        
        # MCE: maximum calibration gap
        non_empty = bin_counts > 0
        mce = np.max(gaps[non_empty]) if np.any(non_empty) else 0.0
        
        return CalibrationResult(
            ece=ece,
            mce=mce,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts
        )
    
    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot reliability diagram with prediction distribution.
        
        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: If provided, save figure to this path
            figsize: Figure size as (width, height)
            
        Returns:
            matplotlib Figure object
        """
        result = self.analyze(y_true, y_prob)
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, 
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        bin_centers = np.linspace(0.05, 0.95, self.n_bins)
        width = 0.8 / self.n_bins
        
        # Reliability diagram
        ax1.bar(
            bin_centers, result.bin_accuracies, width=width, 
            color='steelblue', edgecolor='black', alpha=0.7, label='Accuracy'
        )
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title(f'{title}\nECE = {result.ece:.4f}, MCE = {result.mce:.4f}', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Prediction distribution
        ax2.bar(
            bin_centers, result.bin_counts, width=width, 
            color='gray', edgecolor='black', alpha=0.7
        )
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Prediction Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate well-calibrated predictions
    n_samples = 5000
    true_prob = np.random.beta(2, 2, n_samples)
    y_true_good = (np.random.random(n_samples) < true_prob).astype(int)
    y_prob_good = true_prob + np.random.normal(0, 0.05, n_samples)
    y_prob_good = np.clip(y_prob_good, 0, 1)
    
    # Generate overconfident predictions
    y_prob_overconf = y_prob_good ** 0.5
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    print("Well-calibrated model:")
    print(analyzer.analyze(y_true_good, y_prob_good))
    print()
    print("Overconfident model:")
    print(analyzer.analyze(y_true_good, y_prob_overconf))
