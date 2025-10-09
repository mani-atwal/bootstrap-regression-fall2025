"""
evaluate.py

Helper functions for evaluating bootstrap regression experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from typing import Dict, Optional


def _flatten_to_1d(X):
    """Ensure X is a 1D array (for single predictor problems)."""
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 1:
        return X.ravel()
    if X.ndim == 1:
        return X
    raise ValueError("Expected a single predictor (1D X).")


def compute_ci(draws: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute bootstrap percentile confidence intervals, mean, and std.
    """
    lower = draws.quantile(alpha / 2)
    upper = draws.quantile(1 - alpha / 2)
    return pd.DataFrame({
        "lower": lower,
        "upper": upper,
        "mean": draws.mean(),
        "std": draws.std(ddof=1),
    })


def compute_bias_var(draws: pd.DataFrame, true_coefs: Dict[str, float]) -> pd.DataFrame:
    """
    Compare bootstrap results against true coefficients.
    Returns true value, mean, bias, variance, std, and CI width for each coefficient.

    true_coefs : dict, e.g. {"const": 2.0, "x1": 3.0}
    """
    summary = compute_ci(draws)
    true_vals = pd.Series(true_coefs)

    bias = summary["mean"] - true_vals
    var = draws.var(ddof=1)
    ci_width = summary["upper"] - summary["lower"]

    return pd.DataFrame({
        "true": true_vals,
        "mean": summary["mean"],
        "bias": bias,
        "var": var,
        "std": summary["std"],
        "ci_width": ci_width,
    })


def plot_bootstrap_lines(X, y, draws: pd.DataFrame, n_lines: int = 50, alpha: float = 0.15):
    """
    Plot scatter of (X, y) and overlay regression lines from bootstrap samples.
    Assumes model has intercept 'const' and one predictor (e.g. 'x1').
    """
    X = _flatten_to_1d(X)

    if "const" not in draws.columns or len(draws.columns) < 2:
        raise ValueError(
            "Expected bootstrap draws with columns ['const', 'x1'].")

    slope_name = draws.columns[1]

    # Sample bootstrap lines to overlay
    sampled_draws = draws.sample(min(len(draws), n_lines), replace=False)

    xs = np.linspace(X.min(), X.max(), 200)

    fig = plt.figure(figsize=(6, 4))
    plt.scatter(X, y, s=25, alpha=0.7, edgecolor="k", linewidth=0.25)

    # Plot sampled bootstrap regression lines
    for _, coefs in sampled_draws.iterrows():
        intercept, slope = coefs["const"], coefs[slope_name]
        plt.plot(xs, intercept + slope * xs, alpha=alpha, linewidth=1)

    # Plot mean bootstrap regression line
    mean_intercept = draws["const"].mean()
    mean_slope = draws[slope_name].mean()
    plt.plot(xs, mean_intercept + mean_slope * xs,
             linestyle="--", linewidth=2, label="bootstrap mean")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bootstrap regression lines")
    plt.legend()
    plt.tight_layout()
    #plt.show()

    return fig


def plot_coef_histogram(draws: pd.DataFrame, coef: str = "x1", alpha: float = 0.05, ols_res: Optional[any] = None):
    """
    Plot histogram of bootstrap draws for a given coefficient.
    Optionally overlay OLS estimate (if statsmodels results are provided).
    """
    if coef not in draws.columns:
        raise ValueError(f"{coef} not found in bootstrap draws")

    values = draws[coef].values
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    mean_val = values.mean()

    fig = plt.figure(figsize=(6, 3.5))
    plt.hist(values, bins=30, density=True, alpha=0.8, edgecolor="k")

    # Overlay bootstrap statistics
    plt.axvline(mean_val, color="k", linestyle="--", label="bootstrap mean")
    plt.axvline(lower, color="red", linestyle=":",
                label=f"{100*(alpha/2):.1f}%")
    plt.axvline(upper, color="red", linestyle=":",
                label=f"{100*(1-alpha/2):.1f}%")

    # Optionally overlay OLS estimate
    if ols_res is not None:
        try:
            plt.axvline(ols_res.params[coef], color="blue",
                        linestyle="-.", label="OLS estimate")
        except Exception:
            pass

    plt.xlabel(coef)
    plt.ylabel("density")
    plt.title(f"Bootstrap distribution of {coef}")
    plt.legend()
    plt.tight_layout()
    #plt.show()

    return fig
