"""
Build the pixel-level dataset from raw GeoTIFFs.

Critical fix vs. original code: pixel indices are sampled ONCE per class,
then that same fixed set of pixels is extracted from every timestamp.
The original notebook re-sampled independently per timestamp, so the
temporal sequences contained data from different spatial locations and had
no valid temporal signal.
"""
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CDL_PATH,
    IMAGE_PATHS,
    N_SAMPLES_PER_CLASS,
    PROCESSED_CSV,
    RANDOM_SEED,
    TOP_CLASSES,
)

logger = logging.getLogger(__name__)


def _load_roi(path: Path) -> np.ndarray:
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    return ds.GetRasterBand(1).ReadAsArray()


def _load_image(path: Path) -> np.ndarray:
    from osgeo import gdal, gdal_array
    gdal.UseExceptions()
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)
    img = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=dtype)
    for b in range(img.shape[2]):
        img[:, :, b] = ds.GetRasterBand(b + 1).ReadAsArray()
    return img


def build_dataset(
    n_samples_per_class: int = N_SAMPLES_PER_CLASS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Return a DataFrame with shape (n_classes * n_samples, n_timestamps*n_bands + 1).

    Columns are named ``t{t}_b{b}`` for timestamp t, band b, plus a ``class`` column.
    Each row is one pixel whose band values are drawn from the *same spatial location*
    across all timestamps.
    """
    rng = np.random.default_rng(random_seed)
    logger.info("Loading CDL label raster …")
    roi = _load_roi(CDL_PATH)

    n_timestamps = len(IMAGE_PATHS)
    n_bands: int | None = None
    all_dfs = []

    for cls in TOP_CLASSES:
        logger.info("Sampling class %d (%s) …", cls, cls)
        pixel_rows, pixel_cols = np.where(roi == cls)
        n_available = len(pixel_rows)

        n = min(n_samples_per_class, n_available)
        chosen = rng.choice(n_available, size=n, replace=False)
        row_idx = pixel_rows[chosen]
        col_idx = pixel_cols[chosen]

        # Extract the *same* pixels from every timestamp
        timestamp_arrays = []
        for t, img_path in enumerate(IMAGE_PATHS):
            logger.info("  timestamp %d/%d — %s", t + 1, n_timestamps, img_path.name)
            img = _load_image(img_path)
            if n_bands is None:
                n_bands = img.shape[2]
            timestamp_arrays.append(img[row_idx, col_idx, :])  # (n, n_bands)
            del img
            gc.collect()

        # Stack into (n, n_timestamps * n_bands) — row-major so reshape is clean
        features = np.hstack(timestamp_arrays)

        cols = [f"t{t}_b{b}" for t in range(n_timestamps) for b in range(n_bands)]
        df_cls = pd.DataFrame(features, columns=cols)
        df_cls["class"] = cls
        all_dfs.append(df_cls)
        logger.info("  → %d pixels sampled", n)

    final = pd.concat(all_dfs, ignore_index=True)
    logger.info("Dataset shape: %s", final.shape)
    return final


def save_dataset(df: pd.DataFrame, path: Path = PROCESSED_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved to %s", path)


def load_dataset(path: Path = PROCESSED_CSV) -> pd.DataFrame:
    return pd.read_csv(path)
