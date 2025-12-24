"""Tests for subgroup analysis module."""

import numpy as np
import pytest

from evaluation.subgroup_analysis import SubgroupAnalyzer, create_age_groups


class TestSubgroupAnalyzer:
    """Tests for SubgroupAnalyzer class."""
    
    def test_raises_on_empty_y_true(self) -> None:
        """Empty y_true should raise ValueError."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=30)
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            analyzer.analyze(np.array([]), np.array([]))
    
    def test_raises_on_length_mismatch(self) -> None:
        """Mismatched lengths should raise ValueError."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=30)
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        with pytest.raises(ValueError, match="must have same length"):
            analyzer.analyze(y_true, y_pred)
    
    def test_raises_on_subgroup_labels_length_mismatch(self) -> None:
        """Subgroup labels with wrong length should raise ValueError."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=1)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        subgroups = {"sex": np.array(["F", "M"])}  # wrong length
        with pytest.raises(ValueError, match="has 2 labels"):
            analyzer.analyze(y_true, y_pred, subgroups=subgroups)
    
    def test_raises_on_invalid_min_subgroup_size(self) -> None:
        """min_subgroup_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_subgroup_size must be at least 1"):
            SubgroupAnalyzer(min_subgroup_size=0)
    
    def test_skips_small_groups(self) -> None:
        """Groups smaller than min_subgroup_size should be skipped."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=3)
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])
        
        # group A has 4, group B has 2 -> B should be skipped
        grp = np.array(["A", "A", "A", "A", "B", "B"])
        results = analyzer.analyze(y_true, y_pred, y_prob=y_prob, subgroups={"grp": grp})
        
        assert "A" in results["grp"]
        assert "B" not in results["grp"]
    
    def test_includes_overall_results(self) -> None:
        """Results should always include _overall key."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=1)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        results = analyzer.analyze(y_true, y_pred)
        
        assert "_overall" in results
        assert "all" in results["_overall"]
    
    def test_metrics_in_valid_range(self) -> None:
        """All metrics should be in [0, 1]."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=10)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        y_prob = rng.random(size=200)
        
        results = analyzer.analyze(y_true, y_pred, y_prob=y_prob)
        overall = results["_overall"]["all"]
        
        assert 0.0 <= overall.accuracy <= 1.0
        assert 0.0 <= overall.sensitivity <= 1.0
        assert 0.0 <= overall.specificity <= 1.0
        assert 0.0 <= overall.ppv <= 1.0
        assert 0.0 <= overall.npv <= 1.0
    
    def test_auc_computed_when_prob_provided(self) -> None:
        """AUC should be computed when y_prob is provided."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=10)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        y_prob = rng.random(size=200)
        
        results = analyzer.analyze(y_true, y_pred, y_prob=y_prob)
        overall = results["_overall"]["all"]
        
        assert overall.auc is not None
        assert 0.0 <= overall.auc <= 1.0
    
    def test_auc_none_when_prob_not_provided(self) -> None:
        """AUC should be None when y_prob is not provided."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=10)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        
        results = analyzer.analyze(y_true, y_pred)
        overall = results["_overall"]["all"]
        
        assert overall.auc is None
    
    def test_compute_disparity_returns_gaps(self) -> None:
        """compute_disparity should return max, min, and gap."""
        analyzer = SubgroupAnalyzer(min_subgroup_size=10)
        rng = np.random.default_rng(42)
        
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        
        results = analyzer.analyze(y_true, y_pred, subgroups={"group": groups})
        disparities = analyzer.compute_disparity(results, metric="sensitivity")
        
        assert "group" in disparities
        assert "max" in disparities["group"]
        assert "min" in disparities["group"]
        assert "gap" in disparities["group"]


class TestCreateAgeGroups:
    """Tests for create_age_groups function."""
    
    def test_creates_correct_labels(self) -> None:
        """Should create correct age group labels."""
        ages = np.array([30, 55, 70, 85])
        groups = create_age_groups(ages)
        
        assert groups[0] == "<50"
        assert groups[1] == "50-64"
        assert groups[2] == "65-79"
        assert groups[3] == "80+"
    
    def test_raises_on_mismatched_bins_labels(self) -> None:
        """Should raise if bins and labels don't match."""
        ages = np.array([30, 40, 70])
        with pytest.raises(ValueError, match="Need .* labels"):
            create_age_groups(ages, bins=[0, 50, 100], labels=["<50"])
    
    def test_custom_bins_and_labels(self) -> None:
        """Should work with custom bins and labels."""
        ages = np.array([25, 45, 65])
        groups = create_age_groups(
            ages, 
            bins=[0, 40, 60, 100], 
            labels=["young", "middle", "senior"]
        )
        
        assert groups[0] == "young"
        assert groups[1] == "middle"
        assert groups[2] == "senior"
