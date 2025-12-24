"""
Drift Detection Demo

Usage:
    python -m drift_detection.demo
"""

import numpy as np
from .ks_test import KSTest
from .psi import PSI
from .mmd import MMD


def print_header(text: str) -> None:
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def scenario_medical_imaging() -> None:
    print_header("SCENARIO: Medical Imaging - Scanner Calibration Drift")
    
    np.random.seed(42)
    
    # Two imaging features (intensity & contrast)
    ref_intensity = np.random.normal(loc=0.5, scale=0.15, size=2000)
    ref_contrast = np.random.normal(loc=0.6, scale=0.12, size=2000)
    reference = np.column_stack([ref_intensity, ref_contrast])
    
    cur_intensity = np.random.normal(loc=0.53, scale=0.16, size=2000)
    cur_contrast = np.random.normal(loc=0.58, scale=0.13, size=2000)
    current = np.column_stack([cur_intensity, cur_contrast])
    
    feature_names = ["pixel_intensity", "contrast_ratio"]
    
    print("\n[KS Test] (per feature):")
    ks = KSTest(alpha=0.05)
    for name, result in ks.detect_multivariate(reference, current, feature_names).items():
        status = "DRIFT" if result.drift_detected else "OK"
        print(f"  {name}: p={result.p_value:.4f} [{status}]")
    
    print("\n[PSI] (per feature):")
    psi = PSI(n_bins=10)
    for name, result in psi.calculate_multivariate(reference, current, feature_names).items():
        print(f"  {name}: PSI={result.psi:.4f} [{result.drift_level}]")
    
    print("\n[MMD] (joint distribution):")
    mmd = MMD()
    result = mmd.detect(reference, current, n_permutations=100)
    p_str = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
    print(f"  MMD={result.mmd:.4f}, p={p_str}")
    print(f"  Verdict: {'DRIFT DETECTED' if result.drift_detected else 'No significant drift'}")


def scenario_demographic_shift() -> None:
    print_header("SCENARIO: Credit Scoring - Demographic Drift")
    
    np.random.seed(123)
    
    # Income, debt ratio, age
    ref_income = np.random.lognormal(mean=10.5, sigma=0.8, size=3000)
    ref_debt_ratio = np.random.beta(2, 5, size=3000)
    ref_age = np.random.normal(loc=42, scale=12, size=3000)
    reference = np.column_stack([ref_income, ref_debt_ratio, ref_age])
    
    cur_income = np.random.lognormal(mean=10.7, sigma=0.9, size=3000)
    cur_debt_ratio = np.random.beta(2.5, 4.5, size=3000)
    cur_age = np.random.normal(loc=40, scale=13, size=3000)
    current = np.column_stack([cur_income, cur_debt_ratio, cur_age])
    
    feature_names = ["income", "debt_ratio", "age"]
    
    print("\n[KS Test] (per feature):")
    ks = KSTest(alpha=0.05)
    for name, result in ks.detect_multivariate(reference, current, feature_names).items():
        status = "DRIFT" if result.drift_detected else "OK"
        print(f"  {name}: p={result.p_value:.4f} [{status}]")
    
    print("\n[PSI] (per feature):")
    psi = PSI(n_bins=10)
    for name, result in psi.calculate_multivariate(reference, current, feature_names).items():
        print(f"  {name}: PSI={result.psi:.4f} [{result.drift_level}]")
    
    print("\n[MMD] (joint distribution):")
    mmd = MMD()
    result = mmd.detect(reference, current, n_permutations=100)
    p_str = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
    print(f"  MMD={result.mmd:.4f}, p={p_str}")
    print(f"  Verdict: {'DRIFT DETECTED' if result.drift_detected else 'No significant drift'}")


def scenario_correlation_change() -> None:
    print_header("SCENARIO: Fraud Detection - Correlation Drift (Tricky!)")
    
    np.random.seed(456)
    
    n_samples = 1500
    
    # Same marginals, different joint structure
    reference = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=np.eye(3),
        size=n_samples
    )
    
    cov_current = np.array([
        [1.0, 0.6, 0.0],
        [0.6, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    current = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=cov_current,
        size=n_samples
    )
    
    feature_names = ["txn_amount", "txn_frequency", "account_age"]
    
    print("\n[KS Test] (per feature):")
    print("  (These SHOULD miss the drift - marginals are identical)")
    ks = KSTest(alpha=0.05)
    for name, result in ks.detect_multivariate(reference, current, feature_names).items():
        status = "DRIFT" if result.drift_detected else "OK"
        print(f"  {name}: p={result.p_value:.4f} [{status}]")
    
    print("\n[PSI] (per feature):")
    print("  (These SHOULD also miss it)")
    psi = PSI(n_bins=10)
    for name, result in psi.calculate_multivariate(reference, current, feature_names).items():
        print(f"  {name}: PSI={result.psi:.4f} [{result.drift_level}]")
    
    print("\n[MMD] (joint distribution):")
    print("  (This SHOULD catch the correlation change)")
    mmd = MMD()
    result = mmd.detect(reference, current, n_permutations=100)
    p_str = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
    print(f"  MMD={result.mmd:.4f}, p={p_str}")
    print(f"  Verdict: {'DRIFT DETECTED' if result.drift_detected else 'No significant drift'}")


def main() -> None:
    print("\n" + "=" * 60)
    print(" DRIFT DETECTION DEMO")
    print(" Tools for catching ML models before they fail silently")
    print("=" * 60)
    
    scenario_medical_imaging()
    scenario_demographic_shift()
    scenario_correlation_change()
    
    print_header("SUMMARY")
    print("""
Key takeaways:

1. KS Test: Fast, interpretable, univariate only.
   Use for quick QA checks and regulatory documentation.

2. PSI: Business-friendly, stable for dashboards.
   Use for monitoring model score drift or risk-score migration.

3. MMD: Detects multivariate & correlation drift.
   Use when features interact or subtle distribution shifts matter.

Best practice in production systems:
-> Use all three. Drift is multidimensional.
""")


if __name__ == "__main__":
    main()
