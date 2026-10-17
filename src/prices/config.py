from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


class Cfg:
    """Configuration for the house-price dataset pipeline."""

    data_path: str = str(_ROOT / "data" / "dataset.csv")

    # Schema (post feature-engineering column names).
    features: tuple[str, ...] = (
        "year",
        "age",
        "beds",
        "baths",
        "home_size",
        "parcel_size",
        "pool",
        "dist_cbd",
        "dist_lakes",
        "x_coord",
        "y_coord",
    )
    target: str = "price"

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 4
    n_iter: int = 20