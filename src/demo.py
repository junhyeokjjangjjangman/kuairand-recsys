"""Recommend up to ten videos using the frozen, evaluated hybrid model.

Demo candidates are the whole video catalog minus the user's TRAINING-period
exposures. They differ from the exposure-based evaluation candidate sets.
Known users with no positive training history receive popularity recommendations.
"""

import json

import pandas as pd

from config import RAW_DIR, PROCESSED_DIR, RESULT_DIR, RANDOM_FILE
from item_knn import load_item_knn, score_item_knn
from content import load_content_model, score_content
from popularity import score_popularity
from hybrid import minmax_by_user, MINMAX_EPS, HYBRID_SCORING_RULE


def load_demo():
    """Load the frozen models and IDs; never fit on evaluation feedback."""
    settings = json.loads(
        (RESULT_DIR / "final_test_settings.json").read_text(encoding="utf-8")
    )
    if settings.get("hybrid_scoring_rule") != HYBRID_SCORING_RULE:
        raise ValueError(
            "The saved test settings use another hybrid scoring rule. "
            "Complete validation and evaluation of the revised experiment "
            "before running this demo."
        )
    if settings["minmax_epsilon"] != MINMAX_EPS:
        raise ValueError("Normalization differs from the evaluated model")
    if settings["ranking_tie_break"] != "score_desc_video_id_asc":
        raise ValueError("Unexpected ranking tie-breaking rule")

    seen = pd.read_csv(PROCESSED_DIR / "train_seen.csv")
    # Only IDs are read here to recognize existing users, not their feedback.
    other_ids = pd.read_csv(RAW_DIR / RANDOM_FILE, usecols=["user_id"])
    known_users = set(seen["user_id"]) | set(other_ids["user_id"])

    return {
        "settings": settings,
        "knn": load_item_knn(PROCESSED_DIR),
        "content": load_content_model(PROCESSED_DIR / "content_model.npz"),
        "popularity": pd.read_csv(
            PROCESSED_DIR / "popularity_scores.csv"
        ).set_index("video_id")["score"],
        "seen": seen,
        "known_users": known_users,
    }


def recommend(user_id, state):
    """Return ranked video IDs and scores, plus the recommendation method."""
    if user_id not in state["known_users"]:
        raise ValueError("Unknown user ID")

    seen = state["seen"].loc[
        state["seen"]["user_id"].eq(user_id), "video_id"
    ]
    catalog = state["knn"]["video_ids"]
    unseen = catalog[~catalog.isin(seen)]
    candidates = pd.DataFrame({
        "user_id": user_id,
        "video_id": unseen.to_numpy(),
    })
    if candidates.empty:
        return pd.DataFrame(columns=["rank", "video_id", "score"]), "None"

    if user_id in state["knn"]["user_ids"]:
        settings = state["settings"]["hybrid"]
        cf = score_item_knn(candidates, state["knn"], settings["neighbors"])
        content = score_content(candidates, state["content"])
        keys = ["user_id", "video_id"]
        if not cf[keys].equals(content[keys]):
            raise ValueError("CF and content candidates do not match")

        # Same normalization and uniform CF weight as hybrid.combine_scores.
        # Demo candidates have no observed labels; only scores are calculated.
        scored = cf.rename(columns={"score": "cf_score"}).copy()
        scored["content_score"] = content["score"].to_numpy()
        valid = content["has_item_tags"] & content["has_user_tags"]
        scored["cf_norm"] = minmax_by_user(scored, "cf_score")
        scored["content_norm"] = minmax_by_user(
            scored, "content_score", valid=valid,
        )
        alpha = float(settings["alpha"])
        scored["score"] = (
            alpha * scored["cf_norm"] + (1 - alpha) * scored["content_norm"]
        )
        method = "Hybrid"
    else:
        scored = score_popularity(candidates, state["popularity"])
        method = "Popularity (no positive training history)"

    top = scored.sort_values(
        ["score", "video_id"], ascending=[False, True],
    ).head(int(state["settings"]["top_k"])).copy()
    top["rank"] = range(1, len(top) + 1)
    return top[["rank", "video_id", "score"]], method


def main():
    print("Loading saved models...", flush=True)
    state = load_demo()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("Enter an existing user ID, for example 5. Enter q to exit.")

    while True:
        text = input("\nUser ID: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            user_id = int(text)
        except ValueError:
            print("Enter an integer user ID or q.")
            continue

        try:
            top, method = recommend(user_id, state)
        except ValueError as error:
            print(error)
            continue
        if top.empty:
            print("No unseen videos available.")
            continue

        print("Method:", method)
        print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        output_path = RESULT_DIR / f"demo_user_{user_id}.csv"
        top.to_csv(output_path, index=False)
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
