import numpy as np
import pandas as pd

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    VIDEO_FILE,
    TOP_K,
    NEIGHBOR_SIZES,
)
from evaluation import evaluate_ranking
from item_knn import (
    fit_item_knn,
    score_item_knn,
    explain_item_knn,
    save_item_knn,
)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sizes = sorted(set(NEIGHBOR_SIZES))

    if not sizes or any(
        not isinstance(k, int) or k < 1
        for k in sizes
    ):
        raise ValueError(
            "NEIGHBOR_SIZES must contain positive integers"
        )

    train_positive = pd.read_csv(
        PROCESSED_DIR / "train_positive.csv"
    )
    catalog = pd.read_csv(
        RAW_DIR / VIDEO_FILE,
        usecols=["video_id"],
    )
    candidates = pd.read_csv(
        PROCESSED_DIR / "validation_candidates.csv"
    )

    # Fit once, keeping enough neighbors for all settings.
    print(
        "Fitting from all training positives...",
        flush=True,
    )

    model = fit_item_knn(
        train_positive,
        catalog["video_id"],
        max_neighbors=max(sizes),
    )

    save_item_knn(model, PROCESSED_DIR)

    print("Training matrix shape:", model["history"].shape)
    print("Training positive pairs:", model["history"].nnz)

    example_k = 50 if 50 in sizes else sizes[0]
    example_scores = None
    rows = []

    # Compare settings on validation data only.
    for n_neighbors in sizes:
        print(
            f"\nValidation: neighbors={n_neighbors}",
            flush=True,
        )

        scored = score_item_knn(
            candidates,
            model,
            n_neighbors,
        )

        metrics, per_user = evaluate_ranking(
            scored,
            top_k=TOP_K,
        )

        rows.append({
            "model": "Item-KNN",
            "split": "validation",
            "neighbors": n_neighbors,
            **metrics,
        })

        scored.to_csv(
            PROCESSED_DIR
            / f"item_knn_validation_k{n_neighbors}_scores.csv",
            index=False,
        )

        per_user.to_csv(
            RESULT_DIR
            / f"item_knn_validation_k{n_neighbors}_users.csv",
            index=False,
        )

        if n_neighbors == example_k:
            example_scores = scored

    summary = pd.DataFrame(rows)

    summary.to_csv(
        RESULT_DIR / "item_knn_validation_metrics.csv",
        index=False,
    )

    print("\n[ITEM-KNN VALIDATION]")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    # Show one user's recommendations at a fixed example setting.
    user_id = example_scores["user_id"].iloc[0]

    top = (
        example_scores.loc[
            example_scores["user_id"].eq(user_id)
        ]
        .sort_values(
            ["score", "video_id"],
            ascending=[False, True],
        )
        .head(TOP_K)
        .copy()
    )

    top["rank"] = np.arange(1, len(top) + 1)
    top = top[
        ["user_id", "rank", "video_id", "score", "label"]
    ]

    top.to_csv(
        RESULT_DIR / "item_knn_example_top10.csv",
        index=False,
    )

    print(
        f"\nExample user: {user_id}, neighbors={example_k}"
    )
    print(top.drop(columns="user_id").to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    # Explain the first recommendation.
    target = top["video_id"].iloc[0]

    explanation = explain_item_knn(
        model,
        user_id,
        target,
        example_k,
    )

    explanation.to_csv(
        RESULT_DIR / "item_knn_example_contributions.csv",
        index=False,
    )

    total = float(explanation["similarity"].sum())

    if not np.isclose(total, top["score"].iloc[0]):
        raise ValueError(
            "Explanation does not match the recommendation score"
        )

    print(f"\nScore explanation for video {target}:")

    if explanation.empty:
        print("No matching history neighbors; score = 0.")
    else:
        print(explanation.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ))
        print(
            "Contributing history videos:",
            len(explanation),
        )
        print(f"Total similarity sum: {total:.6f}")


if __name__ == "__main__":
    main()