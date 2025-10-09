"""
Generate and inspect simulated linear regression data.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd


def generate_linear_data(
    n: int = 30,
    beta0: float = 2.0,
    beta1: float = 3.0,
    sigma: float = 1.0,
    heteroskedastic: bool = False,
    hetero_strength: float = 1.0,
    heavy_tails: bool = False,
    df_t: int = 1,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate data from the model y = beta0 + beta1 * x + eps.
    Options:
      - heteroskedastic: variance increases with |x|
      - heavy_tails: use Student-t noise instead of normal
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y_true = beta0 + beta1 * x

    if heavy_tails:
        eps = rng.standard_t(df_t, size=n) * sigma
    elif heteroskedastic:
        scale = sigma * (1.0 + hetero_strength * np.abs(x))
        eps = rng.normal(loc=0.0, scale=scale, size=n)
    else:
        eps = rng.normal(loc=0.0, scale=sigma, size=n)

    y = y_true + eps

    df = pd.DataFrame({"x": x, "y": y, "y_true": y_true})
    params = {
        "beta0": beta0,
        "beta1": beta1,
        "sigma": sigma,
        "heteroskedastic": heteroskedastic,
        "hetero_strength": hetero_strength,
        "heavy_tails": heavy_tails,
        "df_t": df_t,
        "n": n,
        "seed": seed,
    }
    return df, params


def save_df(df: pd.DataFrame, path: str = "data/simulated.csv") -> None:
    """Save dataframe to CSV (simple helper)."""
    df.to_csv(path, index=False)


def plot_data(df: pd.DataFrame, params: Dict | None = None, show_true: bool = True) -> None:
    """
    Quick scatter plot of the simulated data and (optionally) the true line.

    This uses matplotlib and is intentionally minimal.
    """
    import matplotlib.pyplot as plt
    xs = df["x"].values
    ys = df["y"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, alpha=0.7, edgecolor="k", linewidth=0.25)
    if show_true and params is not None:
        grid = np.linspace(xs.min(), xs.max(), 200)
        true_y = params["beta0"] + params["beta1"] * grid
        ax.plot(grid, true_y, lw=2, label="true line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated linear data")
    if show_true and params is not None:
        ax.legend()
    plt.tight_layout()
    #plt.show()

    return fig
