# model-eval-drift-lab

![Run Notebooks](https://github.com/joannany/model-eval-drift-lab/actions/workflows/run-notebooks.yml/badge.svg)

**Tools for catching ML models before they fail silently in production.**

Production models degrade. Your validation accuracy tells you nothing about what happens when:

- patient demographics shift,
- imaging equipment is recalibrated,
- input pipelines change, or
- user behavior drifts over months.

I built this toolkit while working on deployed cancer detection systems, where noticing a 1–2% performance drop early can meaningfully affect clinical outcomes. These are the exact drift-detection and evaluation utilities I used in production — no heavy frameworks, no unnecessary abstractions.

---

## Quickstart
```bash
git clone https://github.com/joannany/model-eval-drift-lab.git
cd model-eval-drift-lab
pip install -r requirements.txt

# Run drift detection demo
python -m drift_detection.demo
```

---

## Repository Structure
```
model-eval-drift-lab/
├── README.md
├── requirements.txt
├── LICENSE
│
├── drift_detection/
│   ├── __init__.py
│   ├── ks_test.py              # Kolmogorov–Smirnov test (univariate shift)
│   ├── psi.py                  # Population Stability Index
│   ├── mmd.py                  # Maximum Mean Discrepancy (multivariate shift)
│   ├── utils.py                # Validation helpers, drift reporting
│   └── demo.py                 # End-to-end drift detection scenarios
│
├── evaluation/
│   ├── __init__.py
│   ├── calibration.py          # Reliability diagrams, ECE/MCE
│   ├── threshold.py            # Operating-point selection for high-stakes domains
│   └── subgroup_analysis.py    # Demographic / FDA-required subgroup evaluations
│
├── notebooks/
│   ├── 01_covariate_shift.ipynb
│   ├── 02_threshold_selection.ipynb
│   └── 03_monitoring_dashboard.ipynb
│
└── examples/
    └── README.md
```

---

## Drift Detection Tools

### KS Test (Kolmogorov–Smirnov)

Univariate distribution comparison.  
✔ Fast  ✔ Interpretable  ✖ Misses correlated / high-dimensional changes

### PSI (Population Stability Index)

Industry standard in credit scoring and regulated risk modeling.

- PSI < 0.1 → stable
- PSI 0.1–0.2 → moderate drift (investigate)
- PSI > 0.2 → significant drift (take action)

### MMD (Maximum Mean Discrepancy)

Kernel-based distribution comparison. Detects multivariate or correlation drift that per-feature tests miss.

If your dataset is high-dimensional or structured (embeddings, images), MMD is essential.

---

## Model Evaluation Tools

### Calibration Analysis (ECE, MCE)

Ensures predicted probabilities reflect empirical outcomes — crucial for clinical use and decision support.

### Threshold Optimization

Supports:

- Youden's J
- Sensitivity-constrained thresholds
- Specificity-constrained thresholds
- F1 / cost-weighted optimization

Useful for screening programs and asymmetric risk environments.

### Subgroup Performance Analysis

Breaks down metrics by:

- age group
- sex
- any demographic label

Used for:

- FDA submissions
- fairness audits
- post-market surveillance

---

## Why This Repo Exists

Deployed ML systems do not operate in the same environment they were trained in.

- Small shifts compound over time.
- Model behavior changes long before anyone notices.
- Traditional metrics give a false sense of safety.
- Regulators (FDA, MDR) increasingly require continuous monitoring.

This repository reflects the practical, reliable toolkit I built to detect these issues early.

If you're not monitoring for drift, you're hoping — and hope is not a strategy.

---

## License

MIT