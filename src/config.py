from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

STANDARD_FILE = "log_standard_4_08_to_4_21_pure.csv"
RANDOM_FILE = "log_random_4_22_to_5_08_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"

PERIODS = {
    "train": (20220409, 20220421),
    "validation": (20220422, 20220428),
    "test": (20220429, 20220508),
}
MIN_HISTORY = 5
MIN_CANDIDATES = 20
TOP_K = 10
NEIGHBOR_SIZES = [20, 50, 100]
CF_WEIGHTS = [0.25, 0.50, 0.75]