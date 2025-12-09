# model-eval-drift-lab

**Lightweight, practical tools for detecting when ML models begin to fail silently in production.**

Validation accuracy only reflects your training and validation distribution.  
Real-world systems drift — sometimes slowly, sometimes suddenly — due to:

- demographic shifts in users or patients,
- changes in acquisition hardware or sensor calibration,
- upstream pipeline modifications,
- seasonal or behavioral changes,
- long-term distribution shift that accumulates unnoticed.

This toolkit contains the exact drift-detection and evaluation utilities I built for real deployed medical AI systems, where catching a **1–2% performance decline early** can meaningfully affect clinical outcomes.

No heavy frameworks. No unnecessary abstractions.  
Just reliable, interpretable, production-ready tools.

---

## 🚀 Quickstart

```bash
git clone https://github.com/joannany/model-eval-drift-lab.git
cd model-eval-drift-lab

pip install -r requirements.txt

# Run drift detection demo
python -m drift_detection.demo
```

---

## 📁 Repository Structure

```
model-eval-drift-lab/
├── README.md
├── requirements.txt
├── LICENSE
│
├── drift_detection/
│   ├── __init__.py
│   ├── ks_test.py              # Kolmogorov–Smirnov (univariate shift)
│   ├── psi.py                  # Population Stability Index
│   ├── mmd.py                  # Maximum Mean Discrepancy (multi-dimensional)
│   ├── utils.py                # Helpers for drift scoring + reporting
│   └── demo.py                 # End-to-end drift detection examples
│
├── evaluation/
│   ├── __init__.py
│   ├── calibration.py          # ECE, MCE, reliability diagrams
│   ├── threshold.py            # Operating point selection in high-risk settings
│   └── subgroup_analysis.py    # Demographic / fairness / FDA subgroup evaluation
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

## 🔎 Drift Detection Tools

### 1. **Kolmogorov–Smirnov Test (KS Test)**  
Univariate distribution comparison.

- ✔ Fast  
- ✔ Easy to interpret  
- ✖ Does not detect correlated or multi-dimensional drift  

Useful for monitoring single scalar signals (confidence scores, model logits, per-feature distributions).

---

### 2. **Population Stability Index (PSI)**  
Widely used in regulated industries (finance, credit scoring).

Interpretation:

| PSI Value | Meaning |
|----------|---------|
| < 0.1 | Stable distribution |
| 0.1–0.2 | Moderate drift — investigate |
| > 0.2 | Significant drift — take action |

Good for **monitoring categorical or histogram-based features** where scale is known.

---

### 3. **Maximum Mean Discrepancy (MMD)**  
Kernel-based method for multi-dimensional drift.

- Detects changes that KS/PSI miss  
- Effective for **embeddings**, **feature vectors**, **representations**, or any high-dimensional data  
- Works without assumptions about the distribution  

If your model uses embeddings or structured features, MMD should be part of your monitoring pipeline.

---

## 📊 Model Evaluation Tools

### **Calibration Analysis (ECE, MCE)**  
Reliability matters — especially in medicine, safety systems, and regulated environments.

This module computes:

- **ECE (Expected Calibration Error)**  
- **MCE (Maximum Calibration Error)**  
- **Reliability diagrams**

Useful when model confidence is used downstream for clinical decisions or triaging.

---

### **Threshold Optimization**  
Helpful for:

- Highly imbalanced datasets  
- Screening workflows  
- Regulatory submissions  
- Risk-weighted decision systems  

Supports:

- **Youden’s J statistic**  
- **Sensitivity-constrained thresholds**  
- **Specificity-constrained thresholds**  
- **F1 / cost-sensitive optimization**

---

### **Subgroup Performance Analysis**  
Splits evaluation metrics across:

- age  
- sex  
- any demographic label  

Frequently required for:

- FDA submissions  
- post-market surveillance  
- fairness audits  
- longitudinal monitoring  

---

## 💡 Why This Repo Exists

Machine learning systems rarely fail catastrophically all at once.  
They fail **quietly**, **slowly**, and **silently** — until someone finally notices.

Some truths:

- Data shifts constantly — even in controlled environments.  
- A small shift in feature distribution can cascade into measurable performance decline.  
- Most teams rely solely on validation metrics, which do not reflect real-world data.  
- Regulators increasingly mandate continuous monitoring (FDA, MDR, ISO 13485).  

This repository represents a **practical, field-tested toolkit** for detecting these issues early — before they impact users or clinical outcomes.

If you're not monitoring for drift, you're relying on hope.  
And **hope is not a strategy**.

---

## 📄 License

MIT

## Citation

If you use this package in academic or applied research, please cite:

```bibtex
@software{model_eval_drift_lab,
  title={model-eval-drift-lab: Tools for Detecting Distribution Drift in Deployed ML Systems},
  author={Jo, Anna},
  year={2025},
  url={https://github.com/joannany/model-eval-drift-lab}
}
```

