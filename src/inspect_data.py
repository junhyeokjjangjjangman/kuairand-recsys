from pathlib import Path

import pandas as pd


# Locate the project folder and raw data.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"

FILES = [
    "log_standard_4_08_to_4_21_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
]

for filename in FILES:
    path = RAW_DIR / filename

    # Read only the first five rows to inspect the structure.
    df = pd.read_csv(path, nrows=5)

    print("\n" + "=" * 60)
    print("File:", filename)
    print("Columns:", df.columns.tolist())

    # Show the fields relevant to our recommendation system.
    preview_columns = [
        column
        for column in ["user_id", "video_id", "date", "long_view", "tag"]
        if column in df.columns
    ]

    print("\nSample:")
    if preview_columns:
        print(df[preview_columns].to_string(index=False))
    else:
        print(df.to_string(index=False))