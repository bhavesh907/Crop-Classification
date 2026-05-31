import numpy as np
import pandas as pd
import pytest

from src.config import N_BANDS, N_TIMESTAMPS, NIR_BAND_IDX, RED_BAND_IDX
from src.features import compute_ndvi, get_feature_array


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {
        f"t{t}_b{b}": rng.integers(500, 12000, n).astype(float)
        for t in range(N_TIMESTAMPS)
        for b in range(N_BANDS)
    }
    cols["class"] = rng.choice([36, 69, 75, 121, 225], n)
    return pd.DataFrame(cols)


def test_compute_ndvi_adds_columns():
    df = _make_df()
    result = compute_ndvi(df)
    for t in range(N_TIMESTAMPS):
        assert f"t{t}_ndvi" in result.columns, f"Missing t{t}_ndvi"


def test_ndvi_range():
    df = compute_ndvi(_make_df())
    for t in range(N_TIMESTAMPS):
        col = df[f"t{t}_ndvi"]
        assert col.between(-1.0, 1.0).all(), f"NDVI at t{t} out of [-1, 1]"


def test_ndvi_formula():
    """Spot-check NDVI = (NIR - Red) / (NIR + Red) for a known pixel."""
    df = _make_df(n=1)
    red = float(df[f"t0_b{RED_BAND_IDX}"].iloc[0])
    nir = float(df[f"t0_b{NIR_BAND_IDX}"].iloc[0])
    expected = (nir - red) / (nir + red + 1e-8)
    result = compute_ndvi(df)[f"t0_ndvi"].iloc[0]
    assert abs(result - expected) < 1e-6


def test_ndvi_does_not_mutate_input():
    df = _make_df()
    original_cols = set(df.columns)
    compute_ndvi(df)
    assert set(df.columns) == original_cols


def test_get_feature_array_shape():
    df = compute_ndvi(_make_df(n=100))
    X, y = get_feature_array(df, include_ndvi=True)
    features_per_step = N_BANDS + 1  # +1 for NDVI
    assert X.shape == (100, N_TIMESTAMPS * features_per_step)
    assert y.shape == (100,)


def test_get_feature_array_without_ndvi():
    df = _make_df(n=50)  # no NDVI columns added
    X, y = get_feature_array(df, include_ndvi=False)
    assert X.shape == (50, N_TIMESTAMPS * N_BANDS)


def test_feature_array_temporal_reshape():
    """Flat array must reshape cleanly to (n, T, F) without reordering data."""
    df = compute_ndvi(_make_df(n=10))
    X_flat, _ = get_feature_array(df, include_ndvi=True)
    features_per_step = N_BANDS + 1
    X_3d = X_flat.reshape(10, N_TIMESTAMPS, features_per_step)
    # First feature of timestamp 1 should match original column value
    expected = df["t1_b0"].values
    np.testing.assert_allclose(X_3d[:, 1, 0], expected, rtol=1e-5)
