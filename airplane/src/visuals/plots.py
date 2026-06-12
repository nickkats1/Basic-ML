"""Exploratory data analysis plots.

Generic, config-driven figures saved to ``images/``. Imported lazily so the
serving app and training pipeline don't require matplotlib at runtime.
"""

from pathlib import Path

import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def correlation_heatmap(
    data: pd.DataFrame,
    config: Config | None = None,
    out_path: str | Path = "images/heatmap.png",
) -> Path:
    """Save a correlation heatmap of the numeric columns.

    Returns:
        The path the figure was written to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    config = config or get_config()
    numeric = data.select_dtypes(include="number")

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Feature correlation heatmap")
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("saved correlation heatmap to %s", out)
    return out


def feature_target_scatter(
    data: pd.DataFrame,
    config: Config | None = None,
    out_path: str | Path = "images/feature_target.png",
) -> Path:
    """Save a grid of scatter plots of each feature against the target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = config or get_config()
    features = list(config.features)
    ncols = 3
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

    for ax, feature in zip(axes.flat, features, strict=False):
        ax.scatter(data[feature], data[config.target], s=10, alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel(config.target)
    for ax in list(axes.flat)[len(features) :]:
        ax.axis("off")

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("saved feature/target scatter to %s", out)
    return out
