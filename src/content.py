import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    VIDEO_FILE,
    TOP_K,
)
from item_knn import load_item_knn


def fit_content_model(history_model, videos):
    """Build binary item vectors and training-only user profiles."""
    if (
        videos["video_id"].isna().any()
        or videos["video_id"].duplicated().any()
    ):
        raise ValueError(
            "Video IDs must be nonmissing and unique"
        )

    # Match the video order used in the training matrix.
    tags = (
        videos.set_index("video_id")
        .loc[history_model["video_ids"], "tag"]
        .astype("string")
        .fillna("")
    )

    tag_lists = tags.apply(
        lambda value: [
            tag.strip()
            for tag in value.split(",")
            if tag.strip()
        ]
    )

    encoder = MultiLabelBinarizer()
    item_vectors = encoder.fit_transform(
        tag_lists
    ).astype(np.float64)

    if item_vectors.shape[1] == 0:
        raise ValueError(
            "No usable tags in the video catalog"
        )

    # Average tag vectors over all positive training videos.
    history = history_model["history"]
    counts = np.asarray(
        history.sum(axis=1)
    ).ravel()

    profiles = np.asarray(history @ item_vectors)
    profiles = profiles / np.maximum(
        counts[:, None], 1
    )

    return {
        "user_ids": history_model["user_ids"].to_numpy(),
        "video_ids": history_model["video_ids"].to_numpy(),
        "tag_ids": encoder.classes_.astype(str),
        "item_vectors": item_vectors,
        "user_profiles": profiles,
        "item_unit": normalize(
            item_vectors, norm="l2", axis=1
        ),
        "user_unit": normalize(
            profiles, norm="l2", axis=1
        ),
    }


def score_content(candidates, model):
    """Calculate content scores without using candidate labels."""
    keys = ["user_id", "video_id"]

    if (
        candidates[keys].isna().any().any()
        or candidates.duplicated(keys).any()
    ):
        raise ValueError(
            "Candidate keys must be nonmissing and unique"
        )

    scored = candidates.copy().reset_index(drop=True)

    users = pd.Index(
        model["user_ids"]
    ).get_indexer(scored["user_id"])

    items = pd.Index(
        model["video_ids"]
    ).get_indexer(scored["video_id"])

    if (users < 0).any() or (items < 0).any():
        raise ValueError(
            "Candidate ID is missing from the content model"
        )

    # Dot products of L2-normalized vectors give cosine similarities.
    scores = np.einsum(
        "ij,ij->i",
        model["user_unit"][users],
        model["item_unit"][items],
    )

    scored["score"] = np.clip(scores, 0.0, 1.0)

    # Keep availability flags for the hybrid fallback rules.
    scored["has_item_tags"] = np.any(
        model["item_vectors"] > 0, axis=1
    )[items]

    scored["has_user_tags"] = np.any(
        model["user_profiles"] > 0, axis=1
    )[users]

    return scored


def load_content_model(path):
    """Load saved vectors and profiles without fitting again."""
    with np.load(path, allow_pickle=False) as data:
        return {
            key: data[key]
            for key in data.files
        }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the positive training matrix saved in Step 8.
    history_model = load_item_knn(PROCESSED_DIR)

    videos = pd.read_csv(
        RAW_DIR / VIDEO_FILE,
        usecols=["video_id", "tag"],
        dtype={"tag": "string"},
    )

    model = fit_content_model(
        history_model, videos
    )

    np.savez_compressed(
        PROCESSED_DIR / "content_model.npz",
        **model,
    )

    candidates = pd.read_csv(
        PROCESSED_DIR / "validation_candidates.csv"
    )

    scored = score_content(candidates, model)

    scored.to_csv(
        PROCESSED_DIR / "content_validation_scores.csv",
        index=False,
    )

    no_tags = ~np.any(
        model["item_vectors"] > 0, axis=1
    )
    no_profile = ~np.any(
        model["user_profiles"] > 0, axis=1
    )

    summary = {
        "videos": len(model["video_ids"]),
        "tags": len(model["tag_ids"]),
        "videos_without_tags": int(no_tags.sum()),
        "training_profile_users": len(model["user_ids"]),
        "training_users_without_usable_tags": int(
            no_profile.sum()
        ),
        "validation_users": scored["user_id"].nunique(),
        "validation_candidate_pairs": len(scored),
        "validation_pairs_without_item_tags": int(
            (~scored["has_item_tags"]).sum()
        ),
        "validation_users_without_usable_tags": scored.loc[
            ~scored["has_user_tags"], "user_id"
        ].nunique(),
    }

    pd.DataFrame([summary]).to_csv(
        RESULT_DIR / "content_summary.csv",
        index=False,
    )

    print("[CONTENT SUMMARY]")
    for key, value in summary.items():
        print(f"{key}: {value:,}")

    # Show one user's preferences and content ranking.
    user_id = scored["user_id"].iloc[0]

    user_row = pd.Index(
        model["user_ids"]
    ).get_loc(user_id)

    profile = pd.DataFrame({
        "tag_id": model["tag_ids"],
        "preference": model["user_profiles"][user_row],
    }).sort_values(
        ["preference", "tag_id"],
        ascending=[False, True],
    )

    profile.to_csv(
        RESULT_DIR / "content_example_profile.csv",
        index=False,
    )

    print(f"\nExample user: {user_id}")
    print("Top profile tags:")
    print(profile.head(5).to_string(index=False))

    top = (
        scored.loc[scored["user_id"].eq(user_id)]
        .sort_values(
            ["score", "video_id"],
            ascending=[False, True],
        )
        .head(TOP_K)
        .copy()
    )

    top["rank"] = np.arange(1, len(top) + 1)

    top.to_csv(
        RESULT_DIR / "content_example_top10.csv",
        index=False,
    )

    print("\nContent ranking example:")
    print(
        top[
            ["rank", "video_id", "score", "label", "has_item_tags"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()