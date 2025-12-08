"""
Threshold Selection for High-Stakes Decisions

Your model outputs probabilities. You need to make binary decisions.
Where you draw the line matters enormously.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
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
    
    def __repr__(self):
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
        >>> print(f"Use threshold: {result.optimal_threshold}")
    """
    
    def __init__(self):
        pass
    
    def _compute_metrics_at_threshold(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        threshold: float
    ) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
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
        """Find threshold that maximizes Youden's J = Sensitivity + Specificity - 1"""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=optimal_threshold,
            metric_name="Youden's J",
            metric_value=j_scores[best_idx],
            **{k: metrics[k] for k in ['sensitivity', 'specificity', 'ppv', 'npv']}
        )
    
    def optimize_sensitivity(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        min_sensitivity: float = 0.95
    ) -> ThresholdResult:
        """Find highest specificity threshold that achieves minimum sensitivity."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        thresholds = np.linspace(0.99, 0.01, 200)
        
        best_threshold = 0.5
        best_specificity = 0
        
        for thresh in thresholds:
            metrics = self._compute_metrics_at_threshold(y_true, y_prob, thresh)
            
            if metrics['sensitivity'] >= min_sensitivity:
                if metrics['specificity'] > best_specificity:
                    best_specificity = metrics['specificity']
                    best_threshold = thresh
        
        final_metrics = self._compute_metrics_at_threshold(y_true, y_prob, best_threshold)
        
        return ThresholdResult(
            optimal_threshold=best_threshold,
            metric_name=f"Max specificity @ sensitivity≥{min_sensitivity}",
            metric_value=final_metrics['specificity'],
            **{k: final_metrics[k] for k in ['sensitivity', 'specificity', 'ppv', 'npv']}
        )
    
    def optimize_specificity(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        min_specificity: float = 0.95
    ) -> ThresholdResult:
        """Find highest sensitivity threshold that achieves minimum specificity."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        thresholds = np.linspace(0.01, 0.99, 200)
        
        best_threshold = 0.5
        best_sensitivity = 0
        
        for thresh in thresholds:
            metrics = self._compute_metrics_at_threshold(y_true, y_prob, thresh)
            
            if metrics['specificity'] >= min_specificity:
                if metrics['sensitivity'] > best_sensitivity:
                    best_sensitivity = metrics['sensitivity']
                    best_threshold = thresh
        
        final_metrics = self._compute_metrics_at_threshold(y_true, y_prob, best_threshold)
        
        return ThresholdResult(
            optimal_threshold=best_threshold,
            metric_name=f"Max sensitivity @ specificity≥{min_specificity}",
            metric_value=final_metrics['sensitivity'],
            **{k: final_metrics[k] for k in ['sensitivity', 'specificity', 'ppv', 'npv']}
        )
    
    def optimize_f1(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> ThresholdResult:
        """Find threshold that maximizes F1 score."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        
        optimal_threshold = thresholds[best_idx]
        metrics = self._compute_metrics_at_threshold(y_true, y_prob, optimal_threshold)
        
        return ThresholdResult(
            optimal_threshold=optimal_threshold,
            metric_name="F1 Score",
            metric_value=f1_scores[best_idx],
            **{k: metrics[k] for k in ['sensitivity', 'specificity', 'ppv', 'npv']}
        )


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        weights=[0.95, 0.05], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    optimizer = ThresholdOptimizer()
    
    print("Youden's J:")
    print(optimizer.optimize_youden(y_test, y_prob))
    print()
    print("Sensitivity >= 95%:")
    print(optimizer.optimize_sensitivity(y_test, y_prob, 0.95))