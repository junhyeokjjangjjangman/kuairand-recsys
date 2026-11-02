import pandas as pd

from config import PROCESSED_DIR, RESULT_DIR


RESULT_DIR.mkdir(parents=True, exist_ok=True)

id_columns = ["user_id", "video_id"]
columns = id_columns + ["long_view"]
summary = []

for name in ["train", "validation", "test"]:
    input_path = PROCESSED_DIR / f"{name}_log.csv"
    data = pd.read_csv(input_path, usecols=columns)

    # Check the fields required for aggregation.
    if data[columns].isna().any().any():
        raise ValueError(f"{name}: missing values in required columns")

    if not data["long_view"].isin([0, 1]).all():
        raise ValueError(f"{name}: long_view must contain only 0 or 1")

    # Aggregate repeated exposures within this split.
    # max() returns 1 if at least one exposure has long_view = 1.
    pairs = (
        data.groupby(id_columns, as_index=False)["long_view"]
        .max()
        .rename(columns={"long_view": "label"})
    )

    pairs.to_csv(
        PROCESSED_DIR / f"{name}_pairs.csv",
        index=False,
    )

    positive_count = int(pairs["label"].eq(1).sum())
    zero_count = int(pairs["label"].eq(0).sum())

    summary.append({
        "split": name,
        "input_rows": len(data),
        "unique_pairs": len(pairs),
        "removed_rows": len(data) - len(pairs),
        "positive_pairs": positive_count,
        "zero_pairs": zero_count,
    })

    if name == "train":
        # Positive history is used to calculate recommendation scores.
        train_positive = pairs.loc[
            pairs["label"].eq(1), id_columns
        ]
        train_positive.to_csv(
            PROCESSED_DIR / "train_positive.csv",
            index=False,
        )

        # All previously exposed videos are excluded from candidates.
        train_seen = pairs[id_columns]
        train_seen.to_csv(
            PROCESSED_DIR / "train_seen.csv",
            index=False,
        )

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    RESULT_DIR / "preprocess_summary.csv",
    index=False,
)

print(summary_df.to_string(index=False))
print("\nSaved preprocessed files to:", PROCESSED_DIR)