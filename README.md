# model-eval-drift-lab

**Practical tools for detecting distribution drift and evaluating ML model behavior.**

This toolkit provides lightweight, interpretable utilities for drift detection, calibration analysis, threshold optimization, and subgroup evaluation. It was developed while working on deployed medical AI systems, where catching small performance degradations early can meaningfully affect outcomes.

## Why This Exists

Production ML systems fail silently. Your validation accuracy tells you nothing about what happens when:

- patient demographics shift,
- imaging equipment is recalibrated,
- input pipelines change upstream, or
- user behavior drifts over months.

These tools help surface statistical signals of distributional change—they don't prescribe automated enforcement, but they make model behavior measurable, auditable, and hard to ignore.

## Quickstart

```bash
git clone https://github.com/joannany/model-eval-drift-lab.git
cd model-eval-drift-lab
pip install -e .

# Run tests
pytest -q

# Run drift detection demo
python -m drift_detection.demo
```

## What's Included

### Drift Detection (`drift_detection/`)

| Method | Best For | Limitations |
|--------|----------|-------------|
| **KS Test** | Fast univariate checks, regulatory docs | Misses multivariate/correlation drift |
| **PSI** | Business dashboards, score migration | Sensitive to binning choices |
| **MMD** | High-dimensional data, correlation changes | Slower, less interpretable |

```python
from drift_detection import KSTest, PSI, MMD

ks = KSTest(alpha=0.05)
result = ks.detect(reference_data, current_data)
print(result)  # KSResult([STABLE]) or KSResult([DRIFT])
```

### Model Evaluation (`evaluation/`)

- **Calibration**: ECE/MCE, reliability diagrams
- **Threshold optimization**: Youden's J, sensitivity-constrained, cost-weighted
- **Subgroup analysis**: Per-group metrics, disparity gaps

```python
from evaluation import ThresholdOptimizer, SubgroupAnalyzer

opt = ThresholdOptimizer()
result = opt.optimize_sensitivity(y_true, y_prob, min_sensitivity=0.95)
print(result.optimal_threshold)
```

### Notebooks (`notebooks/`)

- `01_covariate_shift.ipynb` — Detecting input distribution changes
- `02_threshold_selection.ipynb` — Operating point selection for screening
- `03_monitoring_dashboard.ipynb` — Building a weekly drift report

## Design Principles

- **Evaluation-first**: Model behavior is a measurable object, not an anecdote.
- **Minimal dependencies**: Easy to audit, adapt, and integrate.
- **Failure-oriented defaults**: Validate inputs, surface edge cases, skip unreliable slices.

## Limitations

- These are **reference implementations**. For production monitoring, you'll need logging, alerting, data contracts, and environment pinning.
- Subgroup metrics can be unstable with small samples—use `min_subgroup_size` and interpret with care.
- ECE depends on binning; this repo uses equal-width bins for clarity.

## Repository Structure

```
model-eval-drift-lab/
├── drift_detection/
│   ├── ks_test.py      # Kolmogorov-Smirnov test
│   ├── psi.py          # Population Stability Index
│   ├── mmd.py          # Maximum Mean Discrepancy
│   ├── utils.py        # Validation, reporting
│   └── demo.py         # End-to-end examples
│
├── evaluation/
│   ├── calibration.py      # ECE, MCE, reliability diagrams
│   ├── threshold.py        # Operating point optimization
│   └── subgroup_analysis.py # Demographic/fairness evaluation
│
├── notebooks/
│   ├── 01_covariate_shift.ipynb
│   ├── 02_threshold_selection.ipynb
│   └── 03_monitoring_dashboard.ipynb
│
├── tests/
│   ├── test_calibration.py
│   ├── test_threshold.py
│   ├── test_subgroup_analysis.py
│   └── test_drift_detection.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

MIT

## Citation

```bibtex
@software{model_eval_drift_lab,
  title={model-eval-drift-lab: Tools for Detecting Distribution Drift in ML Systems},
  author={Jo, Anna},
  year={2025},
  url={https://github.com/joannany/model-eval-drift-lab}
}
```
