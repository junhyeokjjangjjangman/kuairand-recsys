import pandas as pd

from config import (
    PROCESSED_DIR,
    RESULT_DIR,
    MIN_HISTORY,
    MIN_CANDIDATES,
)


def build_candidates(pairs, seen, history_counts):
    # Match exposures using both user_id and video_id.
    merged = pairs.merge(
        seen,
        on=["user_id", "video_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    candidates = merged.loc[
        merged["_merge"].eq("left_only"),
        ["user_id", "video_id", "label"],
    ].copy()

    # Include users whose candidates were all removed.
    counts = candidates.groupby("user_id")["label"].agg(
        candidate_count="size",
        positive_count="sum",
    )
    period_users = pd.Index(
        sorted(pairs["user_id"].unique()),
        name="user_id",
    )

    status = counts.reindex(period_users, fill_value=0).astype("int64")
    status["zero_count"] = (
        status["candidate_count"] - status["positive_count"]
    )
    status["train_positive_count"] = history_counts.reindex(
        period_users, fill_value=0
    )

    # Assign the first applicable exclusion reason.
    status["status"] = "evaluated"

    rules = [
        ("history_lt_min",
         status["train_positive_count"].lt(MIN_HISTORY)),
        ("candidates_lt_min",
         status["candidate_count"].lt(MIN_CANDIDATES)),
        ("no_positive", status["positive_count"].eq(0)),
        ("no_zero", status["zero_count"].eq(0)),
    ]

    for reason, condition in rules:
        status.loc[
            status["status"].eq("evaluated") & condition,
            "status",
        ] = reason

    selected = status.index[status["status"].eq("evaluated")]

    final = candidates.loc[
        candidates["user_id"].isin(selected)
    ].sort_values(["user_id", "video_id"])

    return final, status


RESULT_DIR.mkdir(parents=True, exist_ok=True)

train_positive = pd.read_csv(
    PROCESSED_DIR / "train_positive.csv"
)
train_seen = pd.read_csv(
    PROCESSED_DIR / "train_seen.csv"
)

history_counts = train_positive.groupby("user_id").size()

eligible_users = (
    history_counts.loc[history_counts.ge(MIN_HISTORY)]
    .rename("positive_count")
    .reset_index()
)
eligible_users.to_csv(
    PROCESSED_DIR / "eligible_users.csv",
    index=False,
)

print("Users meeting the training-history condition:", len(eligible_users))

reason_order = [
    "history_lt_min",
    "candidates_lt_min",
    "no_positive",
    "no_zero",
    "evaluated",
]
summary = []

for name in ["validation", "test"]:
    pairs = pd.read_csv(
        PROCESSED_DIR / f"{name}_pairs.csv"
    )

    final, status = build_candidates(
        pairs, train_seen, history_counts
    )

    reason_counts = (
        status["status"]
        .value_counts()
        .reindex(reason_order, fill_value=0)
    )

    final.to_csv(
        PROCESSED_DIR / f"{name}_candidates.csv",
        index=False,
    )
    status.reset_index().to_csv(
        RESULT_DIR / f"{name}_user_status.csv",
        index=False,
    )

    summary.append({
        "split": name,
        "period_users": len(status),
        **reason_counts.to_dict(),
        "candidate_pairs": len(final),
    })

    eligible_in_period = int(
        status["train_positive_count"].ge(MIN_HISTORY).sum()
    )

    print(f"\n[{name.upper()}]")
    print("Users in the period:", len(status))
    print(
        "History-eligible users without exposure in this period:",
        len(eligible_users) - eligible_in_period,
    )
    print(reason_counts.to_string())
    print("Final candidate pairs:", len(final))

pd.DataFrame(summary).to_csv(
    RESULT_DIR / "candidate_summary.csv",
    index=False,
)