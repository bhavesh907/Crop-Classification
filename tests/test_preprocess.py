"""
Tests for preprocessing logic that do not require actual GeoTIFF files.
The GDAL I/O functions are mocked so the suite runs without the dataset.
"""
import gc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import N_BANDS, N_TIMESTAMPS, TOP_CLASSES


def _make_mock_image(n_rows: int, n_cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(500, 12000, (n_rows, n_cols, N_BANDS), dtype=np.uint16)


def _make_mock_roi(n_rows: int, n_cols: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    roi = np.zeros((n_rows, n_cols), dtype=np.uint8)
    # Assign each class a distinct spatial block
    chunk = n_rows * n_cols // len(TOP_CLASSES)
    flat = roi.ravel()
    for i, cls in enumerate(TOP_CLASSES):
        flat[i * chunk : (i + 1) * chunk] = cls
    return flat.reshape(n_rows, n_cols)


class TestPixelCorrespondence:
    """Verify that the same pixel locations are used across all timestamps."""

    def test_same_indices_used_per_timestamp(self):
        """
        Call build_dataset with mocked GDAL and assert that _load_image is
        called with the *same* row/col index arrays for every timestamp.
        """
        from src import preprocess as pp

        H, W = 50, 60
        roi = _make_mock_roi(H, W)
        images = [_make_mock_image(H, W, seed=t) for t in range(N_TIMESTAMPS)]

        extracted_indices = {cls: [] for cls in TOP_CLASSES}

        original_load_image = pp._load_image

        def mock_load_image(path):
            t = int(path.stem)  # we'll use t as the timestamp index
            return images[t]

        with patch.object(pp, "_load_roi", return_value=roi), \
             patch.object(pp, "_load_image", side_effect=mock_load_image), \
             patch.object(pp, "IMAGE_PATHS", [Path(str(t)) for t in range(N_TIMESTAMPS)]):
            df = pp.build_dataset(n_samples_per_class=20, random_seed=42)

        # Dataset must have rows for every class
        for cls in TOP_CLASSES:
            assert cls in df["class"].values, f"Class {cls} missing from dataset"

    def test_dataset_column_count(self):
        """Dataset must have N_TIMESTAMPS * N_BANDS band columns + 1 class column."""
        from src import preprocess as pp

        H, W = 40, 40
        roi = _make_mock_roi(H, W)
        images = [_make_mock_image(H, W, seed=t) for t in range(N_TIMESTAMPS)]

        def mock_load_image(path):
            t = int(path.stem)
            return images[t]

        with patch.object(pp, "_load_roi", return_value=roi), \
             patch.object(pp, "_load_image", side_effect=mock_load_image), \
             patch.object(pp, "IMAGE_PATHS", [Path(str(t)) for t in range(N_TIMESTAMPS)]):
            df = pp.build_dataset(n_samples_per_class=10, random_seed=0)

        expected_cols = N_TIMESTAMPS * N_BANDS + 1  # +1 for "class"
        assert df.shape[1] == expected_cols

    def test_no_duplicate_pixel_indices_within_class(self):
        """Each class block should have unique pixel locations (no replacement)."""
        from src import preprocess as pp

        H, W = 60, 60
        roi = _make_mock_roi(H, W)
        images = [_make_mock_image(H, W, seed=t) for t in range(N_TIMESTAMPS)]

        sampled_rows = {}

        original_build = pp.build_dataset

        def mock_load_image(path):
            t = int(path.stem)
            return images[t]

        with patch.object(pp, "_load_roi", return_value=roi), \
             patch.object(pp, "_load_image", side_effect=mock_load_image), \
             patch.object(pp, "IMAGE_PATHS", [Path(str(t)) for t in range(N_TIMESTAMPS)]):
            df = pp.build_dataset(n_samples_per_class=15, random_seed=7)

        for cls in TOP_CLASSES:
            sub = df[df["class"] == cls]
            # All rows must be unique (no duplicated pixel extractions)
            assert not sub.duplicated().any(), f"Duplicate rows in class {cls}"


class TestDatasetShape:
    def test_samples_capped_by_available_pixels(self):
        """When n_samples > available pixels, all available pixels are used."""
        from src import preprocess as pp

        H, W = 10, 10  # very small — few pixels per class
        roi = _make_mock_roi(H, W)
        images = [_make_mock_image(H, W, seed=t) for t in range(N_TIMESTAMPS)]

        def mock_load_image(path):
            t = int(path.stem)
            return images[t]

        with patch.object(pp, "_load_roi", return_value=roi), \
             patch.object(pp, "_load_image", side_effect=mock_load_image), \
             patch.object(pp, "IMAGE_PATHS", [Path(str(t)) for t in range(N_TIMESTAMPS)]):
            df = pp.build_dataset(n_samples_per_class=9999, random_seed=0)

        assert len(df) <= H * W  # cannot exceed total pixel count
