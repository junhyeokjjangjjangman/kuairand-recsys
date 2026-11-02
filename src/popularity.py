import pandas as pd

from config import PROCESSED_DIR, RESULT_DIR, TOP_K
from evaluation import evaluate_ranking


def fit_popularity(train_positive):
    """Count distinct positive users for each video."""
    return (
        train_positive.groupby("video_id")["user_id"]
        .nunique()
        .rename("score")
    )


def score_popularity(candidates, popularity):
    """Assign a popularity score to every candidate."""
    scored = candidates.copy()

    # Videos without positive training history receive score 0.
    scored["score"] = (
        scored["video_id"]
        .map(popularity)
        .fillna(0)
        .astype("int64")
    )

    return scored


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate popularity using training positives only.
    train_positive = pd.read_csv(
        PROCESSED_DIR / "train_positive.csv"
    )
    popularity = fit_popularity(train_positive)

    popularity.reset_index().to_csv(
        PROCESSED_DIR / "popularity_scores.csv",
        index=False,
    )

    # Evaluate on the fixed validation candidates.
    candidates = pd.read_csv(
        PROCESSED_DIR / "validation_candidates.csv"
    )
    scored = score_popularity(candidates, popularity)

    metrics, per_user = evaluate_ranking(
        scored, top_k=TOP_K
    )

    summary = pd.DataFrame([{
        "model": "Popularity",
        "split": "validation",
        **metrics,
    }])

    summary.to_csv(
        RESULT_DIR / "popularity_validation_metrics.csv",
        index=False,
    )
    per_user.to_csv(
        RESULT_DIR / "popularity_validation_users.csv",
        index=False,
    )

    # Save the top recommendations using the same tie-breaking rule.
    top10 = (
        scored.sort_values(
            ["user_id", "score", "video_id"],
            ascending=[True, False, True],
        )
        .groupby("user_id", sort=False)
        .head(TOP_K)
        .copy()
    )

    top10["rank"] = (
        top10.groupby("user_id").cumcount() + 1
    )
    top10 = top10[
        ["user_id", "rank", "video_id", "score", "label"]
    ]

    top10.to_csv(
        RESULT_DIR / "popularity_validation_top10.csv",
        index=False,
    )

    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    example_user = top10["user_id"].iloc[0]
    print(f"\nExample user: {example_user}")
    print(
        top10.loc[
            top10["user_id"].eq(example_user),
            ["rank", "video_id", "score", "label"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()