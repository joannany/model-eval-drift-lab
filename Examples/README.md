# Examples

This directory contains example visual outputs generated from the accompanying notebooks in `/notebooks`.

These artifacts illustrate what the monitoring and evaluation tools look like when applied to real or simulated data.

---

## 📈 Included Examples

### `drift_dashboard.png`

A weekly drift monitoring visualization that highlights:

- Population Stability Index (PSI) over time
- KS test p-values on a log scale
- Warning and critical drift periods
- Feature-specific drift timelines

This corresponds to the `03_monitoring_dashboard.ipynb` notebook.

---

### `calibration_plot.png`

A reliability diagram showing:

- Empirical accuracy vs. predicted confidence
- Expected Calibration Error (ECE)
- Distribution of predictions across probability bins

Generated from `02_threshold_selection.ipynb` or `calibration.py`.

---

## 🔧 Reproducing These Outputs

To regenerate all example figures:
```bash
cd notebooks
jupyter notebook
```

Then run the relevant notebook cells that produce the plots.