"""
Model Evaluation Module

Beyond accuracy - tools for understanding model behavior
in production settings.
"""

from .calibration import CalibrationAnalyzer, CalibrationResult
from .threshold import ThresholdOptimizer, ThresholdResult
from .subgroup_analysis import SubgroupAnalyzer, SubgroupResult, create_age_groups

__all__ = [
    "CalibrationAnalyzer",
    "CalibrationResult", 
    "ThresholdOptimizer",
    "ThresholdResult",
    "SubgroupAnalyzer",
    "SubgroupResult",
    "create_age_groups"
]