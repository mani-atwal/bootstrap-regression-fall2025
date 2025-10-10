"""
bootstrap_methods.py

Bootstrap implementations for linear regression.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional


def _to_2d_array(X):
    """Convert input into a 2D NumPy array (adds column dimension if 1D)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def fit_ols(X, y):
    """Fit an OLS regression and return the fitted model results."""
    X = _to_2d_array(X)
    X = pd.DataFrame(X)
    X_with_const = sm.add_constant(X, has_constant="add")

    y = pd.Series(y)
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    return results


def bootstrap_parametric_normal(
    X, y, n_boot: int = 1000, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Parametric (Normal) Bootstrap:
    1. Fit model once and estimate residual variance.
    2. Generate new residuals ~ N(0, sigma_hat^2).
    3. Create new responses y* = fitted + residuals.
    4. Refit model and store coefficients.
    """
    rng = np.random.default_rng(seed)
    X = _to_2d_array(X)

    # Fit once to get baseline model
    base_fit = fit_ols(X, y)
    fitted_vals = base_fit.fittedvalues
    residuals = base_fit.resid
    n_obs = X.shape[0]

    # Estimate residual standard deviation
    dof_resid = int(base_fit.df_resid)
    sigma_hat = np.sqrt(np.sum(residuals**2) / dof_resid)

    coef_names = base_fit.params.index.tolist()
    bootstrap_coefs = []

    for _ in range(n_boot):
        simulated_resid = rng.normal(0, sigma_hat, size=n_obs)
        y_star = fitted_vals + simulated_resid
        refit = fit_ols(X, y_star)
        bootstrap_coefs.append(refit.params.values)

    return pd.DataFrame(bootstrap_coefs, columns=coef_names)


def bootstrap_pairs(
    X, y, n_boot: int = 1000, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Pairs Bootstrap:
    1. Resample (X, y) pairs with replacement.
    2. Refit model on resampled data.
    3. Collect coefficients.
    """
    rng = np.random.default_rng(seed)
    X = _to_2d_array(X)
    n_obs = X.shape[0]

    base_fit = fit_ols(X, y)
    coef_names = base_fit.params.index.tolist()
    bootstrap_coefs = []

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n_obs, size=n_obs)
        X_resampled = X[sample_idx]
        y_resampled = np.asarray(y)[sample_idx]
        refit = fit_ols(X_resampled, y_resampled)
        bootstrap_coefs.append(refit.params.values)

    return pd.DataFrame(bootstrap_coefs, columns=coef_names)


def bootstrap_residuals(
    X, y, n_boot: int = 1000, seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Residual bootstrap:
    - Fit model once.
    - Resample residuals with replacement.
    - Form y* = fitted + resampled_resid.
    - Refit and collect coefficients.
    """
    rng = np.random.default_rng(seed)

    model = fit_ols(X, y)
    fitted = model.fittedvalues
    resid = model.resid
    coef_names = model.params.index

    boot_coefs = np.zeros((n_boot, len(coef_names)))

    for b in range(n_boot):
        resampled_resid = rng.choice(resid, size=len(resid), replace=True)
        y_star = fitted + resampled_resid
        model_star = fit_ols(X, y_star)
        boot_coefs[b, :] = model_star.params.values

    return pd.DataFrame(boot_coefs, columns=coef_names)


def bootstrap_wild(
    X, y, n_boot: int = 1000, seed: Optional[int] = None, wild: str = "rademacher"
) -> pd.DataFrame:
    """
    Wild bootstrap:
    - Fit model once.
    - Multiply residuals by random weights v_i.
    - Form y* = fitted + v_i * resid.
    - Refit and collect coefficients.

    wild : {"rademacher", "normal"}
        Choice of distribution for the weights.
    """
    rng = np.random.default_rng(seed)

    model = fit_ols(X, y)
    fitted = model.fittedvalues
    resid = model.resid
    coef_names = model.params.index

    boot_coefs = np.zeros((n_boot, len(coef_names)))

    for b in range(n_boot):
        if wild == "rademacher":
            v = rng.choice([-1, 1], size=len(resid))
        elif wild == "normal":
            v = rng.normal(loc=0.0, scale=1.0, size=len(resid))
        else:
            raise ValueError("wild must be 'rademacher' or 'normal'")

        y_star = fitted + resid * v
        model_star = fit_ols(X, y_star)
        boot_coefs[b, :] = model_star.params.values

    return pd.DataFrame(boot_coefs, columns=coef_names)


def bootstrap_summary(
    boot_coefs: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Summarize bootstrap results with:
    - Mean of coefficients
    - Standard deviation
    - Percentile confidence interval (alpha/2, 1 - alpha/2)
    """
    lower_bound = boot_coefs.quantile(alpha / 2)
    upper_bound = boot_coefs.quantile(1 - alpha / 2)

    summary = pd.DataFrame(
        {
            "mean": boot_coefs.mean(),
            "std": boot_coefs.std(ddof=1),
            f"{100 * alpha/2:.1f}%": lower_bound,
            f"{100 * (1 - alpha/2):.1f}%": upper_bound,
        }
    )
    return summary
