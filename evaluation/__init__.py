"""
Model Evaluation Module

Beyond accuracy - tools for understanding model behavior
in production settings.
"""

__version__ = "0.1.0"

from .calibration import CalibrationAnalyzer, CalibrationResult
from .threshold import ThresholdOptimizer, ThresholdResult
from .subgroup_analysis import SubgroupAnalyzer, SubgroupResult, create_age_groups

__all__ = [
    "__version__",
    "CalibrationAnalyzer",
    "CalibrationResult", 
    "ThresholdOptimizer",
    "ThresholdResult",
    "SubgroupAnalyzer",
    "SubgroupResult",
    "create_age_groups",
]
