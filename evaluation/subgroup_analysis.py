"""
Subgroup Performance Analysis

Models can perform well overall but fail on specific subgroups.
This is a fairness issue, a safety issue, and often a regulatory issue.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


@dataclass
class SubgroupResult:
    """Performance metrics for a single subgroup."""
    subgroup_name: str
    n_samples: int
    n_positive: int
    prevalence: float
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    auc: Optional[float]
    
    def __repr__(self):
        return (
            f"Subgroup: {self.subgroup_name} (n={self.n_samples}, prevalence={self.prevalence:.1%})\n"
            f"  Accuracy:    {self.accuracy:.3f}\n"
            f"  Sensitivity: {self.sensitivity:.3f}\n"
            f"  Specificity: {self.specificity:.3f}\n"
            f"  PPV:         {self.ppv:.3f}\n"
            f"  AUC:         {self.auc:.3f if self.auc else 'N/A'}"
        )


class SubgroupAnalyzer:
    """
    Analyze model performance across demographic and clinical subgroups.
    
    Example:
        >>> analyzer = SubgroupAnalyzer()
        >>> results = analyzer.analyze(
        ...     y_true, y_pred, y_prob,
        ...     subgroups={'age_group': age_labels, 'sex': sex_labels}
        ... )
        >>> analyzer.print_report(results)
    """
    
    def __init__(self, min_subgroup_size: int = 30):
        self.min_subgroup_size = min_subgroup_size
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> dict:
        if len(y_true) == 0:
            return None
        
        n_positive = np.sum(y_true)
        n_negative = len(y_true) - n_positive
        
        if n_positive == 0 or n_negative == 0:
            return None
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics = {
            'n_samples': len(y_true),
            'n_positive': int(n_positive),
            'prevalence': n_positive / len(y_true),
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        }
        
        if y_prob is not None and n_positive > 0 and n_negative > 0:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = None
        else:
            metrics['auc'] = None
        
        return metrics
    
    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        subgroups: Dict[str, np.ndarray] = None
    ) -> Dict[str, Dict[str, SubgroupResult]]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
        
        results = {}
        
        overall_metrics = self._compute_metrics(y_true, y_pred, y_prob)
        results['_overall'] = {
            'all': SubgroupResult(subgroup_name='Overall', **overall_metrics)
        }
        
        if subgroups is None:
            return results
        
        for subgroup_type, subgroup_labels in subgroups.items():
            subgroup_labels = np.asarray(subgroup_labels)
            unique_labels = np.unique(subgroup_labels)
            
            results[subgroup_type] = {}
            
            for label in unique_labels:
                mask = subgroup_labels == label
                n_samples = np.sum(mask)
                
                if n_samples < self.min_subgroup_size:
                    continue
                
                y_true_sub = y_true[mask]
                y_pred_sub = y_pred[mask]
                y_prob_sub = y_prob[mask] if y_prob is not None else None
                
                metrics = self._compute_metrics(y_true_sub, y_pred_sub, y_prob_sub)
                
                if metrics is not None:
                    results[subgroup_type][str(label)] = SubgroupResult(
                        subgroup_name=f"{subgroup_type}={label}",
                        **metrics
                    )
        
        return results
    
    def print_report(
        self,
        results: Dict[str, Dict[str, SubgroupResult]],
        gap_threshold: float = 0.1
    ):
        print("=" * 70)
        print(" SUBGROUP PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        if '_overall' in results:
            print("\n📊 OVERALL PERFORMANCE")
            print("-" * 40)
            print(results['_overall']['all'])
        
        for subgroup_type, subgroup_results in results.items():
            if subgroup_type == '_overall':
                continue
            
            print(f"\n📊 BY {subgroup_type.upper()}")
            print("-" * 40)
            
            for label, result in subgroup_results.items():
                print(f"\n{result}")


def create_age_groups(
    ages: np.ndarray,
    bins: List[int] = [0, 50, 65, 80, 120],
    labels: List[str] = ['<50', '50-64', '65-79', '80+']
) -> np.ndarray:
    return pd.cut(ages, bins=bins, labels=labels, right=False).astype(str)


if __name__ == "__main__":
    np.random.seed(42)
    
    n_samples = 2000
    ages = np.concatenate([
        np.random.normal(55, 10, 1200),
        np.random.normal(75, 8, 800)
    ])
    ages = np.clip(ages, 30, 95)
    sex = np.random.choice(['M', 'F'], n_samples)
    
    base_prob = 0.05
    age_effect = (ages - 50) / 100 * 0.1
    true_prob = np.clip(base_prob + age_effect, 0.01, 0.3)
    y_true = (np.random.random(n_samples) < true_prob).astype(int)
    
    noise = np.where(ages > 70, 0.3, 0.1)
    y_prob = true_prob + np.random.normal(0, noise)
    y_prob = np.clip(y_prob, 0, 1)
    y_pred = (y_prob > 0.15).astype(int)
    
    analyzer = SubgroupAnalyzer(min_subgroup_size=50)
    age_groups = create_age_groups(ages)
    
    results = analyzer.analyze(
        y_true, y_pred, y_prob,
        subgroups={'age_group': age_groups, 'sex': sex}
    )
    
    analyzer.print_report(results)