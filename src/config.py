from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "Dataset"
MODELS_DIR = ROOT / "models_saved"
RESULTS_DIR = ROOT / "results"

IMAGE_PATHS = [
    DATA_DIR / "20170306.tiff",
    DATA_DIR / "20170410.tiff",
    DATA_DIR / "20170601.tiff",
    DATA_DIR / "20170615.tiff",
    DATA_DIR / "20170708.tiff",
    DATA_DIR / "20170807.tiff",
    DATA_DIR / "20170905.tiff",
    DATA_DIR / "20170923.tiff",
    DATA_DIR / "20171015.tiff",
    DATA_DIR / "20171207.tiff",
]

CDL_PATH = DATA_DIR / "cdl2017.tiff"

# USDA CDL class codes and human-readable names
TOP_CLASSES = [36, 69, 75, 121, 225]
CLASS_NAMES = {
    36: "Alfalfa",
    69: "Grapes",
    75: "Almonds",
    121: "Developed/Open Space",
    225: "Dbl Crop WinWht/Corn",
}

# RapidEye band order: Blue(0), Green(1), Red(2), RedEdge(3), NIR(4)
N_BANDS = 5
N_TIMESTAMPS = len(IMAGE_PATHS)
RED_BAND_IDX = 2
NIR_BAND_IDX = 4

N_SAMPLES_PER_CLASS = 100_000
RANDOM_SEED = 42

PROCESSED_CSV = DATA_DIR / "processed.csv"
