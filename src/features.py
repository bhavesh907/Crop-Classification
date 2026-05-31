"""Vegetation-index feature engineering on the preprocessed DataFrame."""
import numpy as np
import pandas as pd

from .config import N_BANDS, N_TIMESTAMPS, NIR_BAND_IDX, RED_BAND_IDX


def compute_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-timestamp NDVI columns (``t{t}_ndvi``) to *df*.

    NDVI = (NIR - Red) / (NIR + Red).  A small epsilon avoids division by zero.
    """
    df = df.copy()
    for t in range(N_TIMESTAMPS):
        red = df[f"t{t}_b{RED_BAND_IDX}"].astype(float)
        nir = df[f"t{t}_b{NIR_BAND_IDX}"].astype(float)
        df[f"t{t}_ndvi"] = (nir - red) / (nir + red + 1e-8)
    return df


def get_feature_array(
    df: pd.DataFrame,
    include_ndvi: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(X, y)`` where X has shape ``(n, n_timestamps * features_per_step)``.

    Feature order within each timestep: [b0, b1, …, b{N_BANDS-1}, ndvi?].
    The flat layout means ``X.reshape(n, N_TIMESTAMPS, features_per_step)``
    correctly reconstructs the temporal tensor without data re-ordering.
    """
    step_blocks = []
    for t in range(N_TIMESTAMPS):
        band_cols = [df[f"t{t}_b{b}"].values for b in range(N_BANDS)]
        if include_ndvi and f"t{t}_ndvi" in df.columns:
            band_cols.append(df[f"t{t}_ndvi"].values)
        step_blocks.append(np.stack(band_cols, axis=1))  # (n, features_per_step)

    X = np.hstack(step_blocks).astype(np.float32)  # (n, T * F)
    y = df["class"].values
    return X, y
