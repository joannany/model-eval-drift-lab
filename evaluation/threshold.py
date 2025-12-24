"""
Threshold Selection for High-Stakes Decisions

Your model outputs probabilities. You need to make binary decisions.
Where you draw the line matters enormously.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


@dataclass 
class ThresholdResult:
    """Results from threshold optimization."""
    optimal_threshold: float
    metric_name: str
    metric_value: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    
    def __repr__(self) -> str:
        return (
            f"ThresholdResult\n"
            f"  Optimal threshold: {self.optimal_threshold:.3f}\n"
            f"  Optimized for: {self.metric_name} = {self.metric_value:.4f}\n"
            f"  ─────────────────────────────\n"
            f"  Sensitivity (TPR): {self.sensitivity:.3f}\n"
            f"  Specificity (TNR): {self.specificity:.3f}\n"
            f"  PPV (Precision):   {self.ppv:.3f}\n"
            f"  NPV:               {self.npv:.3f}"
        )


class ThresholdOptimizer:
    """
    Find optimal classification thresholds under various criteria.
    
    Example:
        >>> optimizer = ThresholdOptimizer()
        >>> result = optimizer.optimize_sensitivity(y_true, y_prob, min_sensitivity=0.95)
        >>> print(result)
        ThresholdResult
          Optimal threshold: 0.127
          Optimized for: Max specificity @ sensitivity>=0.95 = 0.8342
          ─────────────────────────────
          Sensitivity (TPR): 0.952
          Specificity (TNR): 0.834
          PPV (Precision):   0.241
          NPV:               0.997
    """
    
    def __init__(self) -> None:
        pass
    
    def _validate_inputs(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs to numpy arrays."""
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty")
        if len(y_true) != len(y_prob):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_prob ({len(y_prob)})")
        
        n_positive = np.sum(y_true)
        n_negative = len(y_true) - n_positive
        if n_positive == 0 or n_negative == 0:
            raise ValueError("y_true must contain both positive and negative samples")
        
        return y_true, y_prob
    
    def _compute_metrics_at_threshold(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        threshold: float
    ) -> dict:
        """Compute all metrics at a given threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    
    def optimize_youden(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> ThresholdResult:
        """
        Find threshold that maximizes Youden's J = Sensitivity + Specificity - 1.
        
        This balances sensitivity and specificity equally.
        
        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities for the positive class
            
        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        y_true, y_prob = self._validate_inputs(y_true, y_prob)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr  # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
        best_idx = np.argmax(j_scores)
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=float(optimal_threshold),
            metric_name="Youden's J",
            metric_value=float(j_scores[best_idx]),
            sensitivity=metrics['sensitivity'],
            specificity=metrics['specificity'],
            ppv=metrics['ppv'],
            npv=metrics['npv']
        )
    
    def optimize_sensitivity(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        min_sensitivity: float = 0.95
    ) -> Optional[ThresholdResult]:
        """
        Find highest specificity threshold that achieves minimum sensitivity.
        
        Use this for screening applications where missing positives is costly.
        
        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities
            min_sensitivity: Minimum required sensitivity (default: 0.95)
            
        Returns:
            ThresholdResult, or None if target sensitivity is not achievable
        """
        y_true, y_prob = self._validate_inputs(y_true, y_prob)
        
        if not 0 < min_sensitivity <= 1:
            raise ValueError("min_sensitivity must be in (0, 1]")
        
        # Use ROC curve thresholds for efficiency
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find all thresholds where sensitivity >= min_sensitivity
        valid_mask = tpr >= min_sensitivity
        
        if not np.any(valid_mask):
            # Target sensitivity not achievable
            return None
        
        # Among valid thresholds, find the one with highest specificity (lowest FPR)
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmin(fpr[valid_indices])]
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=float(optimal_threshold),
            metric_name=f"Max specificity @ sensitivity>={min_sensitivity}",
            metric_value=metrics['specificity'],
            sensitivity=metrics['sensitivity'],
            specificity=metrics['specificity'],
            ppv=metrics['ppv'],
            npv=metrics['npv']
        )
    
    def optimize_specificity(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        min_specificity: float = 0.95
    ) -> Optional[ThresholdResult]:
        """
        Find highest sensitivity threshold that achieves minimum specificity.
        
        Use this when false positives are costly (e.g., confirmatory testing).
        
        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities
            min_specificity: Minimum required specificity (default: 0.95)
            
        Returns:
            ThresholdResult, or None if target specificity is not achievable
        """
        y_true, y_prob = self._validate_inputs(y_true, y_prob)
        
        if not 0 < min_specificity <= 1:
            raise ValueError("min_specificity must be in (0, 1]")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        specificity = 1 - fpr
        
        # Find all thresholds where specificity >= min_specificity
        valid_mask = specificity >= min_specificity
        
        if not np.any(valid_mask):
            return None
        
        # Among valid thresholds, find the one with highest sensitivity (TPR)
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmax(tpr[valid_indices])]
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=float(optimal_threshold),
            metric_name=f"Max sensitivity @ specificity>={min_specificity}",
            metric_value=metrics['sensitivity'],
            sensitivity=metrics['sensitivity'],
            specificity=metrics['specificity'],
            ppv=metrics['ppv'],
            npv=metrics['npv']
        )
    
    def optimize_f1(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> ThresholdResult:
        """
        Find threshold that maximizes F1 score.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities
            
        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        y_true, y_prob = self._validate_inputs(y_true, y_prob)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Avoid division by zero
        f1_scores = np.divide(
            2 * precision * recall, 
            precision + recall, 
            out=np.zeros_like(precision), 
            where=(precision + recall) > 0
        )
        
        # precision_recall_curve returns n+1 values, last threshold is implicit
        best_idx = np.argmax(f1_scores[:-1])
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=float(optimal_threshold),
            metric_name="F1 Score",
            metric_value=float(f1_scores[best_idx]),
            sensitivity=metrics['sensitivity'],
            specificity=metrics['specificity'],
            ppv=metrics['ppv'],
            npv=metrics['npv']
        )
    
    def optimize_cost(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fp_cost: float = 1.0,
        fn_cost: float = 1.0
    ) -> ThresholdResult:
        """
        Find threshold that minimizes weighted misclassification cost.
        
        Total cost = fp_cost * FP + fn_cost * FN
        
        Args:
            y_true: Ground truth binary labels
            y_prob: Predicted probabilities
            fp_cost: Cost of a false positive (default: 1.0)
            fn_cost: Cost of a false negative (default: 1.0)
            
        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        y_true, y_prob = self._validate_inputs(y_true, y_prob)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        n_positive = np.sum(y_true)
        n_negative = len(y_true) - n_positive
        
        # FP = FPR * N_negative, FN = (1 - TPR) * N_positive
        costs = fp_cost * fpr * n_negative + fn_cost * (1 - tpr) * n_positive
        best_idx = np.argmin(costs)
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=float(optimal_threshold),
            metric_name=f"Min cost (FP={fp_cost}, FN={fn_cost})",
            metric_value=float(costs[best_idx]),
            sensitivity=metrics['sensitivity'],
            specificity=metrics['specificity'],
            ppv=metrics['ppv'],
            npv=metrics['npv']
        )


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Generate imbalanced classification data
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        weights=[0.95, 0.05], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    optimizer = ThresholdOptimizer()
    
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION COMPARISON")
    print("=" * 60)
    
    print("\n[1] Youden's J (balanced):")
    print(optimizer.optimize_youden(y_test, y_prob))
    
    print("\n[2] Sensitivity >= 95% (screening):")
    print(optimizer.optimize_sensitivity(y_test, y_prob, 0.95))
    
    print("\n[3] Specificity >= 95% (confirmatory):")
    print(optimizer.optimize_specificity(y_test, y_prob, 0.95))
    
    print("\n[4] Max F1 Score:")
    print(optimizer.optimize_f1(y_test, y_prob))
    
    print("\n[5] Cost-weighted (FN costs 10x more than FP):")
    print(optimizer.optimize_cost(y_test, y_prob, fp_cost=1, fn_cost=10))
