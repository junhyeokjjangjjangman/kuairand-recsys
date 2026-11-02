import numpy as np
import pandas as pd


MINMAX_EPS = 1e-12


def minmax_by_user(data, column, valid=None):
    """Scale scores within each user's candidate set."""
    values = data[column]

    if valid is not None:
        values = values.where(valid)

    groups = values.groupby(data["user_id"])
    lower = groups.transform("min")
    upper = groups.transform("max")

    span = upper - lower

    # Constant or numerically indistinguishable scores become zero.
    result = (
        (values - lower)
        / span.where(span > MINMAX_EPS)
    )

    return result.fillna(0.0).clip(0.0, 1.0)


def combine_scores(cf_scores, content_scores, alpha):
    """Combine scores for identical candidate sets."""
    alpha = float(alpha)

    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    keys = ["user_id", "video_id"]
    cf_columns = keys + ["label", "score"]
    content_columns = cf_columns + [
        "has_item_tags",
        "has_user_tags",
    ]

    cf = cf_scores[cf_columns].copy()
    content = content_scores[content_columns].copy()

    if cf.isna().any().any() or content.isna().any().any():
        raise ValueError(
            "Missing keys, labels, scores, or tag flags"
        )

    cf = cf.rename(columns={
        "label": "cf_label",
        "score": "cf_score",
    })
    content = content.rename(columns={
        "label": "content_label",
        "score": "content_score",
    })

    # Check exact correspondence between CF and content candidates.
    data = cf.merge(
        content,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not data["_merge"].eq("both").all():
        raise ValueError(
            "CF and content candidate sets do not match"
        )

    if not data["cf_label"].eq(data["content_label"]).all():
        raise ValueError(
            "CF and content labels do not match"
        )

    if not data["cf_label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    data["label"] = data["cf_label"].astype("int64")

    for column in ["cf_score", "content_score"]:
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

    if not np.isfinite(
        data[["cf_score", "content_score"]].to_numpy()
    ).all():
        raise ValueError("Scores must be finite")

    for column in ["has_item_tags", "has_user_tags"]:
        if not data[column].isin([True, False]).all():
            raise ValueError(
                "Tag flags must be boolean values"
            )
        data[column] = data[column].astype(bool)

    valid_content = (
        data["has_item_tags"]
        & data["has_user_tags"]
    )

    data["cf_norm"] = minmax_by_user(
        data, "cf_score"
    )
    data["content_norm"] = minmax_by_user(
        data,
        "content_score",
        valid=valid_content,
    )

    data["score"] = (
        alpha * data["cf_norm"]
        + (1 - alpha) * data["content_norm"]
    )

    # Use normalized CF alone when content is unavailable.
    data.loc[~valid_content, "score"] = data.loc[
        ~valid_content, "cf_norm"
    ]

    output_columns = keys + [
        "label",
        "cf_score",
        "content_score",
        "cf_norm",
        "content_norm",
        "has_item_tags",
        "has_user_tags",
        "score",
    ]

    return (
        data[output_columns]
        .sort_values(keys)
        .reset_index(drop=True)
    )