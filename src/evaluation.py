import numpy as np
import pandas as pd

from config import TOP_K


def evaluate_ranking(scored_candidates, top_k=TOP_K):
    """Return average metrics and the metrics for each user."""
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    columns = ["user_id", "video_id", "label", "score"]

    if not set(columns).issubset(scored_candidates.columns):
        raise ValueError(f"Required columns: {columns}")

    data = scored_candidates[columns].copy()

    if data.empty or data.isna().any().any():
        raise ValueError("Evaluation data is empty or has missing values")

    if data.duplicated(["user_id", "video_id"]).any():
        raise ValueError("Duplicate user-video pairs")

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    data["score"] = pd.to_numeric(data["score"], errors="raise")

    if not np.isfinite(data["score"].to_numpy(dtype=float)).all():
        raise ValueError("Scores must be finite")

    # Higher scores first; smaller video IDs break ties.
    ranked = data.sort_values(
        ["user_id", "score", "video_id"],
        ascending=[True, False, True],
    )

    # Rank 1 receives weight 1; lower ranks receive smaller weights.
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2))
    rows = []

    for user_id, group in ranked.groupby("user_id", sort=False):
        labels = group["label"].to_numpy(dtype=int)
        positives = int(labels.sum())

        if positives == 0:
            raise ValueError(f"User {user_id} has no positive candidate")

        top_labels = labels[:top_k]
        hits = int(top_labels.sum())

        recall = hits / positives

        dcg = float(
            np.dot(top_labels, discounts[:len(top_labels)])
        )
        idcg = float(
            discounts[:min(positives, top_k)].sum()
        )
        ndcg = dcg / idcg

        rows.append({
            "user_id": user_id,
            "candidate_count": len(group),
            "positive_count": positives,
            "hits": hits,
            f"Recall@{top_k}": recall,
            f"NDCG@{top_k}": ndcg,
        })

    per_user = pd.DataFrame(rows)

    metrics = {
        "users": len(per_user),
        f"Recall@{top_k}": float(
            per_user[f"Recall@{top_k}"].mean()
        ),
        f"NDCG@{top_k}": float(
            per_user[f"NDCG@{top_k}"].mean()
        ),
    }

    return metrics, per_user


if __name__ == "__main__":
    # Small example: check the top 3 instead of the actual top 10.
    example = pd.DataFrame({
        "user_id": [1] * 5 + [2] * 5,
        "video_id": [1, 2, 3, 4, 5] * 2,
        "label": [1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
        "score": [5, 4, 3, 2, 1] * 2,
    })

    metrics, per_user = evaluate_ranking(example, top_k=3)

    # User 1 finds 2 of 3 positives; user 2 finds all 3.
    expected_recall = (2 / 3 + 1) / 2
    expected_ndcg = (
        1.5 / (1 + 1 / np.log2(3) + 0.5) + 1
    ) / 2

    assert np.isclose(metrics["Recall@3"], expected_recall)
    assert np.isclose(metrics["NDCG@3"], expected_ndcg)

    # Check that equal scores are ordered by video ID.
    ties = pd.DataFrame({
        "user_id": [1, 1, 1],
        "video_id": [30, 10, 20],
        "label": [1, 0, 1],
        "score": [0, 0, 0],
    })

    tie_metrics, _ = evaluate_ranking(ties, top_k=2)

    expected_tie_ndcg = (
        (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    )
    assert np.isclose(tie_metrics["Recall@2"], 0.5)
    assert np.isclose(tie_metrics["NDCG@2"], expected_tie_ndcg)

    print(per_user.to_string(index=False))
    print(f"\nRecall@3: {metrics['Recall@3']:.6f}")
    print(f"NDCG@3: {metrics['NDCG@3']:.6f}")
    print("Evaluation checks passed")