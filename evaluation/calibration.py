"""
Model Calibration Analysis

A well-calibrated model's predicted probabilities match empirical frequencies.
When a model says "80% chance of cancer", ~80% of those cases should actually be cancer.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    ece: float
    mce: float
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    
    def __repr__(self):
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
        >>> analyzer.plot_reliability_diagram(y_true, y_prob)
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def analyze(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> CalibrationResult:
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
        
        bin_accuracies = np.zeros(self.n_bins)
        bin_confidences = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            bin_counts[i] = np.sum(mask)
            
            if bin_counts[i] > 0:
                bin_accuracies[i] = np.mean(y_true[mask])
                bin_confidences[i] = np.mean(y_prob[mask])
        
        weights = bin_counts / len(y_prob)
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(weights * gaps)
        
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
        result = self.analyze(y_true, y_prob)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        bin_centers = np.linspace(0.05, 0.95, self.n_bins)
        width = 0.08
        
        ax1.bar(bin_centers, result.bin_accuracies, width=width, 
                color='steelblue', edgecolor='black', alpha=0.7, label='Accuracy')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title(f'{title}\nECE = {result.ece:.4f}, MCE = {result.mce:.4f}', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(bin_centers, result.bin_counts, width=width, 
                color='gray', edgecolor='black', alpha=0.7)
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
    
    n_samples = 5000
    true_prob = np.random.beta(2, 2, n_samples)
    y_true_good = (np.random.random(n_samples) < true_prob).astype(int)
    y_prob_good = true_prob + np.random.normal(0, 0.05, n_samples)
    y_prob_good = np.clip(y_prob_good, 0, 1)
    
    y_prob_overconf = y_prob_good ** 0.5
    
    analyzer = CalibrationAnalyzer(n_bins=10)
    
    print("Well-calibrated model:")
    print(analyzer.analyze(y_true_good, y_prob_good))
    print()
    print("Overconfident model:")
    print(analyzer.analyze(y_true_good, y_prob_overconf))