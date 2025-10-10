# Bootstrap Regression (Fall 2025)

This repository is the **final project** for a Linear Regression course.  
It explores **bootstrap methods for linear regression** and demonstrates how resampling can reveal the stability and uncertainty of regression estimates.

The project includes:
- A **blog post** introducing and explaining bootstrap regression.
- An **interactive Dash app** to build intuition through visualizations.
- **Reproducible Python code** for simulations, bootstrapping, and evaluation.

---

## Motivation

Ordinary Least Squares (OLS) regression relies on strong assumptions (normality, homoskedasticity, large sample size).  
When these assumptions fail, standard errors and confidence intervals may be misleading.  
**Bootstrapping** offers a flexible alternative by resampling the data, refitting the model many times, and using the variation in estimates to measure uncertainty.

---

## Goals

1. **Blog Post** – A deep dive into bootstrap methods: parametric (normal), pairs, residual, and wild bootstraps.  
2. **Interactive App / Visualization** – Visualize bootstrap regression lines, coefficient distributions, and confidence intervals.  
3. **Clean Codebase** – Reproducible simulations, modular functions, and easy-to-run notebooks and apps.

---

## Repository Structure

```
README.md
requirements.txt
app.py                 # Interactive Dash app (main entry point)
experiments.ipynb      # Simulation & analysis notebook
src/
├─ simulate.py          # Data simulation helpers
├─ bootstrap_methods.py # Bootstrap implementations
├─ evaluate.py          # Evaluation, summaries, and plots
data/                   # Saved datasets & simulation results
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/bootstrap-regression-fall2025.git
cd bootstrap-regression-fall2025
```

### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# or
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the App

Start the interactive Dash app locally:
```bash
python app.py
```

Then open the link shown in the terminal (usually `http://127.0.0.1:8050/`) to explore bootstrap regression interactively.

---

## App Features

The interactive **Dash app** allows users to:
- Choose among **bootstrap methods** (Parametric Normal, Pairs, Residuals, Wild).  
- Adjust simulation parameters such as:
  - Number of data points  
  - True intercept and slope (β₀, β₁)  
  - Error variance (σ²)  
  - Degree of heteroskedasticity  
- Visualize:
  - Bootstrap regression lines (overlayed on scatter)  
  - Histogram / density of bootstrap coefficient draws  
  - OLS estimate vs. bootstrap distribution and percentile CI  

Notes:
- The app renders matplotlib figures to PNG and embeds them as base64 images.
- Default sampling uses NumPy's Generator for reproducible simulation.

---

## How it works (brief)

1. **Simulate** synthetic linear data with `src/simulate.py` (options for heteroskedasticity and heavy-tailed noise).  
2. **Fit OLS** once to obtain fitted values and residuals using `statsmodels` (`src/bootstrap_methods.py`).  
3. **Bootstrap**:
   - Parametric: simulate residuals from estimated normal variance.
   - Pairs: resample (X, y) pairs with replacement.
   - Residuals: resample residuals and add to fitted values.
   - Wild: multiply residuals by random weights (Rademacher or Normal).
4. **Refit** model for each bootstrap replicate and collect coefficient draws.  
5. **Evaluate & Visualize** using `src/evaluate.py` — compute percentile CIs, plot bootstrap lines, and histogram distributions.

---

## Core Modules

- `src/simulate.py` — Data generation helpers (`generate_linear_data`, `save_df`, `plot_data`).  
- `src/bootstrap_methods.py` — Bootstrap implementations and a `fit_ols` wrapper around `statsmodels.OLS`.  
- `src/evaluate.py` — Visualization helpers (`plot_bootstrap_lines`, `plot_coef_histogram`) and summary functions (`compute_ci`, `compute_bias_var`).

---

## Authors

- **Mani Atwal**  
- **Patrick Crouch**  
- **Shiman Zhang**
