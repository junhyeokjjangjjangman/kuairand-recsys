from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics.pairwise import cosine_similarity


def _check_pairs(pairs):
    """Check that user-video pairs are unique and have no missing keys."""
    keys = ["user_id", "video_id"]

    if not set(keys).issubset(pairs.columns):
        raise ValueError("user_id and video_id are required")

    if pairs[keys].isna().any().any():
        raise ValueError("Missing user-video keys")

    if pairs.duplicated(keys).any():
        raise ValueError("Duplicate user-video pairs")


def fit_item_knn(
    train_positive,
    video_ids,
    max_neighbors=100,
    block_size=256,
    verbose=True,
):
    """Build item neighborhoods using all positive training interactions."""
    _check_pairs(train_positive)

    if train_positive.empty:
        raise ValueError("Training positives are empty")

    if max_neighbors < 1 or block_size < 1:
        raise ValueError(
            "max_neighbors and block_size must be positive"
        )

    users = pd.Index(
        sorted(train_positive["user_id"].unique())
    )
    videos = pd.Index(sorted(video_ids))

    if videos.empty or videos.has_duplicates or videos.hasnans:
        raise ValueError(
            "The video catalog must be nonempty and unique"
        )

    user_rows = users.get_indexer(train_positive["user_id"])
    item_cols = videos.get_indexer(train_positive["video_id"])

    if (item_cols < 0).any():
        raise ValueError(
            "Training video is missing from the catalog"
        )

    # Build a sparse binary user-video matrix.
    # Float64 avoids overflow during similarity calculations.
    history = csr_matrix(
        (
            np.ones(len(train_positive), dtype=np.float64),
            (user_rows, item_cols),
        ),
        shape=(len(users), len(videos)),
    )

    item_vectors = history.T.tocsr()
    n_items = len(videos)

    neighbor_cols = np.full(
        (n_items, max_neighbors),
        -1,
        dtype=np.int32,
    )
    neighbor_sims = np.zeros(
        (n_items, max_neighbors),
        dtype=np.float64,
    )

    catalog_ids = videos.to_numpy()

    # Calculate similarities in blocks to limit memory usage.
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)

        similarity = cosine_similarity(
            item_vectors[start:end],
            item_vectors,
            dense_output=False,
        ).tocsr()

        for offset, target in enumerate(range(start, end)):
            left, right = similarity.indptr[offset:offset + 2]

            cols = similarity.indices[left:right]
            values = similarity.data[left:right]

            # Exclude the item itself and zero similarities.
            keep = (cols != target) & (values > 0)
            cols = cols[keep]
            values = values[keep]

            # Higher similarity first; smaller video ID breaks ties.
            order = np.lexsort(
                (catalog_ids[cols], -values)
            )[:max_neighbors]

            count = len(order)
            neighbor_cols[target, :count] = cols[order]
            neighbor_sims[target, :count] = values[order]

        if verbose and (end % 1024 == 0 or end == n_items):
            print(
                f"Similarity: {end:,}/{n_items:,} videos",
                flush=True,
            )

    return {
        "history": history,
        "user_ids": users,
        "video_ids": videos,
        "neighbor_cols": neighbor_cols,
        "neighbor_sims": neighbor_sims,
    }


def make_neighbor_matrix(model, n_neighbors):
    """Create a matrix with target items as rows and neighbors as columns."""
    max_neighbors = model["neighbor_cols"].shape[1]

    if not 1 <= n_neighbors <= max_neighbors:
        raise ValueError(
            f"n_neighbors must be between 1 and {max_neighbors}"
        )

    cols = model["neighbor_cols"][:, :n_neighbors]
    values = model["neighbor_sims"][:, :n_neighbors]
    valid = cols >= 0

    rows = np.broadcast_to(
        np.arange(len(cols))[:, None],
        cols.shape,
    )

    n_items = len(model["video_ids"])

    return csr_matrix(
        (
            values[valid],
            (rows[valid], cols[valid]),
        ),
        shape=(n_items, n_items),
    )


def score_item_knn(candidates, model, n_neighbors=50):
    """Score every supplied candidate using positive training history."""
    _check_pairs(candidates)

    scored = candidates.copy().reset_index(drop=True)

    user_rows = model["user_ids"].get_indexer(
        scored["user_id"]
    )
    item_cols = model["video_ids"].get_indexer(
        scored["video_id"]
    )

    if (user_rows < 0).any():
        raise ValueError(
            "Candidate user has no positive training history"
        )

    if (item_cols < 0).any():
        raise ValueError(
            "Candidate video is missing from the catalog"
        )

    neighbors = make_neighbor_matrix(model, n_neighbors)
    scored["score"] = 0.0

    for user_id, group in scored.groupby(
        "user_id", sort=False
    ):
        user_row = model["user_ids"].get_loc(user_id)

        candidate_cols = model["video_ids"].get_indexer(
            group["video_id"]
        )

        profile = model["history"].getrow(user_row)

        # Sum similarities to history items in each candidate's neighborhood.
        scores = (
            neighbors[candidate_cols] @ profile.T
        ).toarray().ravel()

        scored.loc[group.index, "score"] = scores

    return scored


def explain_item_knn(
    model,
    user_id,
    video_id,
    n_neighbors=50,
):
    """List the history videos contributing to one candidate's score."""
    user_row = model["user_ids"].get_loc(user_id)
    item_col = model["video_ids"].get_loc(video_id)

    neighbors = make_neighbor_matrix(model, n_neighbors)

    contribution = neighbors.getrow(item_col).multiply(
        model["history"].getrow(user_row)
    ).tocsr()

    contribution.eliminate_zeros()

    explanation = pd.DataFrame({
        "history_video_id": model["video_ids"].to_numpy()[
            contribution.indices
        ],
        "similarity": contribution.data,
    })

    return explanation.sort_values(
        ["similarity", "history_video_id"],
        ascending=[False, True],
    )


def save_item_knn(model, output_dir):
    """Save training history and ranked neighborhoods for reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_dir / "item_knn_history.npz",
        model["history"],
    )

    np.savez_compressed(
        output_dir / "item_knn_metadata.npz",
        user_ids=model["user_ids"].to_numpy(),
        video_ids=model["video_ids"].to_numpy(),
        neighbor_cols=model["neighbor_cols"],
        neighbor_sims=model["neighbor_sims"],
    )


def load_item_knn(input_dir):
    """Load a saved model without fitting it again."""
    input_dir = Path(input_dir)

    with np.load(
        input_dir / "item_knn_metadata.npz",
        allow_pickle=False,
    ) as data:
        model = {
            key: data[key]
            for key in data.files
        }

    model["user_ids"] = pd.Index(model["user_ids"])
    model["video_ids"] = pd.Index(model["video_ids"])

    model["history"] = load_npz(
        input_dir / "item_knn_history.npz"
    )

    return model