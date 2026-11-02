from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

report = []


def show(text):
    """Print a line and keep it for the report."""
    print(text)
    report.append(text)


# Read only the columns needed for this inspection.
log_columns = ["user_id", "video_id", "date", "long_view", "is_rand"]

standard = pd.read_csv(
    RAW_DIR / "log_standard_4_08_to_4_21_pure.csv",
    usecols=log_columns,
)

random = pd.read_csv(
    RAW_DIR / "log_random_4_22_to_5_08_pure.csv",
    usecols=log_columns,
)

videos = pd.read_csv(
    RAW_DIR / "video_features_basic_pure.csv",
    usecols=["video_id", "tag"],
    dtype={"tag": "string"},
)

# Inspect both log files before preprocessing.
for name, df in [("STANDARD LOG", standard), ("RANDOM LOG", random)]:
    show(f"\n[{name}]")
    show(f"Rows: {len(df):,}")
    show(f"Users: {df['user_id'].nunique():,}")
    show(f"Videos: {df['video_id'].nunique():,}")
    show(f"Date range: {df['date'].min()} - {df['date'].max()}")
    show(f"long_view counts: {df['long_view'].value_counts(dropna=False).to_dict()}")
    show(f"is_rand counts: {df['is_rand'].value_counts(dropna=False).to_dict()}")
    show(f"Missing values: {df.isna().sum().to_dict()}")

# Select the training period before removing duplicate positive pairs.
train = standard.loc[
    standard["date"].between(20220409, 20220421)
    & standard["is_rand"].eq(0)
]

positive_pairs = (
    train.loc[train["long_view"].eq(1), ["user_id", "video_id"]]
    .drop_duplicates()
)

positive_counts = positive_pairs.groupby("user_id").size()

show("\n[TRAINING HISTORY: APRIL 9-21]")
show(f"Exposure rows: {len(train):,}")
show(f"Distinct positive user-video pairs: {len(positive_pairs):,}")
show(f"Users with at least 5 positive videos: {(positive_counts >= 5).sum():,}")
show(f"Videos with positive training history: {positive_pairs['video_id'].nunique():,}")

# Split comma-separated tags and remove empty tokens.
tag_lists = (
    videos["tag"]
    .fillna("")
    .apply(lambda value: [
        tag.strip() for tag in value.split(",") if tag.strip()
    ])
)

unique_tags = {tag for tags in tag_lists for tag in tags}
missing_tags = tag_lists.apply(len).eq(0)

show("\n[VIDEO INFORMATION]")
show(f"Video rows: {len(videos):,}")
show(f"Distinct video IDs: {videos['video_id'].nunique():,}")
show(f"Duplicate video IDs: {videos['video_id'].duplicated().sum():,}")
show(f"Videos with tags: {(~missing_tags).sum():,}")
show(f"Videos without tags: {missing_tags.sum():,}")
show(f"Distinct tags: {len(unique_tags):,}")
show(f"Tag IDs: {sorted(unique_tags)}")

# This total covers only the two log files used above.
show(f"\nTwo-log rows: {len(standard) + len(random):,}")

output_path = RESULT_DIR / "data_summary.txt"
output_path.write_text("\n".join(report), encoding="utf-8")
print("\nSaved:", output_path)