"""Validate the nine planned hybrid settings and freeze the best settings."""

import json

import pandas as pd

from config import (
    PROCESSED_DIR, RESULT_DIR, TOP_K, NEIGHBOR_SIZES, CF_WEIGHTS,
)
from evaluation import evaluate_ranking
from hybrid import combine_scores, MINMAX_EPS, HYBRID_SCORING_RULE


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(NEIGHBOR_SIZES))
    weights = sorted(set(CF_WEIGHTS))
    if not sizes or not weights:
        raise ValueError("Neighborhood sizes and weights must not be empty")

    keys = ["user_id", "video_id", "label"]
    reference = pd.read_csv(PROCESSED_DIR / "validation_candidates.csv")
    reference = reference[keys].sort_values(keys[:2]).reset_index(drop=True)
    content = pd.read_csv(
        PROCESSED_DIR / "content_validation_scores.csv",
        dtype={"has_item_tags": "boolean", "has_user_tags": "boolean"},
    )

    metric = f"NDCG@{TOP_K}"
    rows = []
    best_key = None
    best_row = None
    best_scores = None

    for neighbors in sizes:
        cf = pd.read_csv(
            PROCESSED_DIR / f"item_knn_validation_k{neighbors}_scores.csv"
        )
        for alpha in weights:
            print(f"Validation: neighbors={neighbors}, alpha={alpha}", flush=True)
            scored = combine_scores(cf, content, alpha)
            if not scored[keys].equals(reference):
                raise ValueError("Scores do not match the fixed validation cohort")
            metrics, per_user = evaluate_ranking(scored, top_k=TOP_K)
            row = {
                "model": "Hybrid", "split": "validation",
                "neighbors": neighbors, "alpha": alpha, **metrics,
            }
            rows.append(row)
            alpha_tag = str(alpha).replace(".", "p")
            per_user.to_csv(
                RESULT_DIR / f"hybrid_validation_k{neighbors}_a{alpha_tag}_users.csv",
                index=False,
            )

            # NDCG first; exact ties prefer smaller k, then smaller alpha.
            key = (-metrics[metric], neighbors, alpha)
            if best_key is None or key < best_key:
                best_key, best_row, best_scores = key, row, scored

    summary = pd.DataFrame(rows).sort_values(
        [metric, "neighbors", "alpha"], ascending=[False, True, True],
    )
    summary.to_csv(RESULT_DIR / "hybrid_validation_metrics.csv", index=False)
    print("\n[HYBRID VALIDATION]")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    cf_summary = pd.read_csv(RESULT_DIR / "item_knn_validation_metrics.csv")
    cf_best = cf_summary.sort_values(
        [metric, "neighbors"], ascending=[False, True],
    ).iloc[0]
    settings = {
        "selection_split": "validation",
        "selection_metric": metric,
        "top_k": TOP_K,
        "item_knn": {"neighbors": int(cf_best["neighbors"])},
        "hybrid": {
            "neighbors": int(best_row["neighbors"]),
            "alpha": float(best_row["alpha"]),
        },
        "minmax_epsilon": MINMAX_EPS,
        "hybrid_scoring_rule": HYBRID_SCORING_RULE,
        "ranking_tie_break": "score_desc_video_id_asc",
    }
    (RESULT_DIR / "best_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8",
    )
    best_scores.to_csv(
        PROCESSED_DIR / "hybrid_validation_best_scores.csv", index=False,
    )
    print("\n[BEST SETTINGS]")
    print(json.dumps(settings, indent=2))

    top = best_scores.sort_values(
        ["user_id", "score", "video_id"], ascending=[True, False, True],
    ).groupby("user_id", sort=False).head(TOP_K).copy()
    top["rank"] = top.groupby("user_id").cumcount() + 1
    top.to_csv(RESULT_DIR / "hybrid_validation_best_top10.csv", index=False)
    user_id = top["user_id"].iloc[0]
    print(f"\nExample user: {user_id}")
    print(top.loc[top["user_id"].eq(user_id), [
        "rank", "video_id", "score", "cf_norm", "content_norm", "label",
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
