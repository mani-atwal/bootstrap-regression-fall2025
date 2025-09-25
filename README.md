# Bootstrap Regression (Fall 2025)

This repository is the **final project** for a Linear Regression course.  
It explores **bootstrap methods for linear regression** and demonstrates how resampling can reveal the stability and uncertainty of regression estimates.

The project includes:
- A **blog post** that introduces and explains bootstrap regression.
- An **interactive Dash app** to build intuition through visualizations.
- **Reproducible Python code** for simulations, bootstrapping, and evaluation.

---

## Motivation
Ordinary Least Squares (OLS) regression relies on strong assumptions (normality, homoskedasticity, large sample size).  
When these fail, standard errors and confidence intervals may be misleading.  
**Bootstrapping** offers a flexible alternative by resampling the data, refitting the model many times, and using the variation in estimates to measure uncertainty.

---

## Goals
1. **Blog Post** – A deep dive into bootstrap methods: pairs, residual, wild, and parametric bootstraps.  
2. **Interactive App** – Visualize bootstrap lines, coefficient distributions, and confidence intervals.  
3. **Clean Codebase** – Reproducible simulations, clear modular functions, and easy-to-run notebooks.

---

## Repository Structure
README.md
requirements.txt
app.py # Interactive Dash app
experiments.ipynb # Notebook with simulation & analysis
src/
├─ simulate.py # Data simulation helpers
├─ bootstrap_methods.py # Bootstrap implementations
├─ evaluate.py # Evaluation helpers (summaries, plots, coverage tests)
data/ # Saved simulation datasets

---

## Authors
- **Mani Atwal**  
- **Patrick Crouch**  
- **Shiman Zhang**  

---