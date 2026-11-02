import pandas as pd

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    STANDARD_FILE,
    RANDOM_FILE,
    PERIODS,
)


PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

columns = ["user_id", "video_id", "date", "long_view", "is_rand"]

standard = pd.read_csv(RAW_DIR / STANDARD_FILE, usecols=columns)
random = pd.read_csv(RAW_DIR / RANDOM_FILE, usecols=columns)

splits = {}

# Split exposure records by date.
for name, (start_date, end_date) in PERIODS.items():
    source = standard if name == "train" else random

    data = source.loc[
        source["date"].between(start_date, end_date)
    ].copy()

    if data.empty:
        raise ValueError(f"{name}: no records in the selected period")

    expected_flag = 0 if name == "train" else 1
    if not data["is_rand"].eq(expected_flag).all():
        raise ValueError(f"{name}: unexpected exposure type")

    splits[name] = data

# Check that no source records were lost during this split.
if len(splits["train"]) != len(standard):
    raise ValueError("Some standard records fall outside the training period")

if len(splits["validation"]) + len(splits["test"]) != len(random):
    raise ValueError("Some random records fall outside the evaluation periods")

# Save the split logs and their basic statistics.
summary = []

for name, data in splits.items():
    output_path = PROCESSED_DIR / f"{name}_log.csv"
    data.to_csv(output_path, index=False)

    summary.append({
        "split": name,
        "rows": len(data),
        "users": data["user_id"].nunique(),
        "videos": data["video_id"].nunique(),
        "start_date": data["date"].min(),
        "end_date": data["date"].max(),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(RESULT_DIR / "split_summary.csv", index=False)

print(summary_df.to_string(index=False))
print("\nSaved split logs to:", PROCESSED_DIR)
print("Saved statistics to:", RESULT_DIR / "split_summary.csv")