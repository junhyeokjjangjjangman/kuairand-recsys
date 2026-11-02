import json

import pandas as pd

from config import PROCESSED_DIR, RESULT_DIR, TOP_K
from evaluation import evaluate_ranking
from popularity import score_popularity
from item_knn import load_item_knn, score_item_knn
from content import load_content_model, score_content
from hybrid import combine_scores, MINMAX_EPS


def save_evaluation(name, scored, candidates, top_k):
    """Check the shared cohort and save one model's test results."""
    keys = ["user_id", "video_id", "label"]

    expected = (
        candidates[keys]
        .sort_values(keys[:2])
        .reset_index(drop=True)
    )
    actual = (
        scored[keys]
        .sort_values(keys[:2])
        .reset_index(drop=True)
    )

    if not actual.equals(expected):
        raise ValueError(
            f"{name}: test candidates or labels were changed"
        )

    metrics, per_user = evaluate_ranking(
        scored,
        top_k=top_k,
    )

    prefix = name.lower().replace("-", "_")

    scored.to_csv(
        PROCESSED_DIR / f"{prefix}_test_scores.csv",
        index=False,
    )
    per_user.to_csv(
        RESULT_DIR / f"{prefix}_test_users.csv",
        index=False,
    )

    top = (
        scored.sort_values(
            ["user_id", "score", "video_id"],
            ascending=[True, False, True],
        )
        .groupby("user_id", sort=False)
        .head(top_k)
        .copy()
    )

    top["rank"] = (
        top.groupby("user_id").cumcount() + 1
    )

    top.to_csv(
        RESULT_DIR / f"{prefix}_test_top10.csv",
        index=False,
    )

    return {
        "model": name,
        "split": "test",
        **metrics,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    settings = json.loads(
        (RESULT_DIR / "best_settings.json").read_text(
            encoding="utf-8"
        )
    )

    # Keep the protocol selected during validation.
    if settings["selection_split"] != "validation":
        raise ValueError(
            "Settings must have been selected on validation"
        )

    top_k = int(settings["top_k"])

    if top_k != TOP_K:
        raise ValueError(
            "TOP_K differs from the saved validation setting"
        )

    if settings["selection_metric"] != f"NDCG@{top_k}":
        raise ValueError(
            "Unexpected validation selection metric"
        )

    if settings["minmax_epsilon"] != MINMAX_EPS:
        raise ValueError(
            "Normalization differs from the validation setting"
        )

    if (
        settings["ranking_tie_break"]
        != "score_desc_video_id_asc"
    ):
        raise ValueError(
            "Unexpected ranking tie-breaking rule"
        )

    candidates = pd.read_csv(
        PROCESSED_DIR / "test_candidates.csv"
    )

    print("[FIXED TEST SETTINGS]")
    print(json.dumps(settings, indent=2))
    print(
        "\nTest users:",
        candidates["user_id"].nunique(),
    )
    print("Test candidate pairs:", len(candidates))

    rows = []

    # Popularity: use the saved training-period counts.
    print("\nEvaluating Popularity...", flush=True)

    popularity = (
        pd.read_csv(
            PROCESSED_DIR / "popularity_scores.csv"
        )
        .set_index("video_id")["score"]
    )

    scored = score_popularity(candidates, popularity)

    rows.append(
        save_evaluation(
            "Popularity", scored, candidates, top_k
        )
    )

    # Item-KNN: load the original training model.
    print("Evaluating Item-KNN...", flush=True)

    model = load_item_knn(PROCESSED_DIR)
    cf_neighbors = int(
        settings["item_knn"]["neighbors"]
    )

    cf_scored = score_item_knn(
        candidates,
        model,
        cf_neighbors,
    )

    rows.append(
        save_evaluation(
            "Item-KNN", cf_scored, candidates, top_k
        )
    )

    # Hybrid: use the fixed neighborhood size and CF weight.
    print("Evaluating Hybrid...", flush=True)

    hybrid_neighbors = int(
        settings["hybrid"]["neighbors"]
    )
    alpha = float(settings["hybrid"]["alpha"])

    if hybrid_neighbors == cf_neighbors:
        hybrid_cf = cf_scored
    else:
        hybrid_cf = score_item_knn(
            candidates,
            model,
            hybrid_neighbors,
        )

    content_model = load_content_model(
        PROCESSED_DIR / "content_model.npz"
    )

    content_scored = score_content(
        candidates,
        content_model,
    )

    hybrid_scored = combine_scores(
        hybrid_cf,
        content_scored,
        alpha,
    )

    rows.append(
        save_evaluation(
            "Hybrid", hybrid_scored, candidates, top_k
        )
    )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        RESULT_DIR / "final_test_metrics.csv",
        index=False,
    )

    (RESULT_DIR / "final_test_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print("\n[FINAL TEST RESULTS]")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))


if __name__ == "__main__":
    main()