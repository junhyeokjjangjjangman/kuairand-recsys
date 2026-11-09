"""Analyze fixed models without changing training or selected settings."""

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap

from config import PROCESSED_DIR, RESULT_DIR, TOP_K
from content import load_content_model, score_content
from evaluation import evaluate_ranking
from hybrid import combine_scores, HYBRID_SCORING_RULE
from item_knn import (
    load_item_knn,
    score_item_knn,
    explain_item_knn,
)
from popularity import fit_popularity, score_popularity


OUT = RESULT_DIR / "analysis_k750"
METRICS = ["Recall@10", "NDCG@10"]
GROUPS = ["5-9", "10-19", "20-49", "50+"]

EPS = 1e-12
SEED = 42
RESAMPLES = 20000


def save(data, name):
    data.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig",
    )


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def check_inputs(train, candidates, model, content):
    if TOP_K != 10 or HYBRID_SCORING_RULE != "uniform_cf_weight_v2":
        raise ValueError("Use TOP_K=10 and uniform_cf_weight_v2")

    if candidates.user_id.nunique() != 7737 or len(candidates) != 527231:
        raise ValueError(
            "The test cohort differs from the original experiment"
        )

    keys = ["user_id", "video_id"]
    seen = read(PROCESSED_DIR / "train_seen.csv")

    if pd.MultiIndex.from_frame(candidates[keys]).isin(
        pd.MultiIndex.from_frame(seen[keys])
    ).any():
        raise ValueError(
            "Training exposures remain in test candidates"
        )

    counts = candidates.groupby("user_id").label.agg(["size", "sum"])
    history = train.groupby("user_id").size().reindex(counts.index)

    if (
        history.isna().any()
        or history.lt(5).any()
        or counts["size"].lt(20).any()
        or counts["sum"].lt(1).any()
        or counts["sum"].ge(counts["size"]).any()
    ):
        raise ValueError("Invalid evaluation eligibility")

    rows = model["user_ids"].get_indexer(train.user_id)
    cols = model["video_ids"].get_indexer(train.video_id)

    if (
        (rows < 0).any()
        or (cols < 0).any()
        or model["history"].nnz != len(train)
        or model["neighbor_cols"].shape[1] < 1000
    ):
        raise ValueError(
            "Rebuild the training model by completing Step 1"
        )

    if not np.equal(
        np.asarray(model["history"][rows, cols]).ravel(), 1
    ).all():
        raise ValueError(
            "The model does not match training positives"
        )

    pd.testing.assert_index_equal(
        model["video_ids"],
        pd.Index(content["video_ids"]),
    )
    pd.testing.assert_index_equal(
        model["user_ids"],
        pd.Index(content["user_ids"]),
    )

    totals = np.asarray(model["history"].sum(axis=1)).ravel()
    expected = (
        np.asarray(model["history"] @ content["item_vectors"])
        / totals[:, None]
    )

    np.testing.assert_allclose(
        expected,
        content["user_profiles"],
        rtol=1e-10,
        atol=EPS,
    )


def evaluate_models(train, candidates, model, content_model):
    content = score_content(candidates, content_model)

    scores = {
        "Popularity": score_popularity(
            candidates,
            fit_popularity(train),
        )
    }

    for k in [100, 750, 1000]:
        print(f"Scoring k={k}...", flush=True)

        scores[f"CF{k}"] = score_item_knn(
            candidates, model, k
        )

        alpha = 0.75 if k == 100 else 0.8
        scores[f"HY{k}"] = combine_scores(
            scores[f"CF{k}"], content, alpha
        )

    records = []
    users = {}

    expected = (
        candidates[["user_id", "video_id", "label"]]
        .sort_values(["user_id", "video_id"])
        .reset_index(drop=True)
    )

    for name, frame in scores.items():
        actual = (
            frame[expected.columns]
            .sort_values(["user_id", "video_id"])
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(actual, expected)

        metrics, detail = evaluate_ranking(frame, top_k=10)

        users[name] = (
            detail.sort_values("user_id")
            .set_index("user_id")
        )

        save(detail, f"{name}_test_users.csv")
        records.append({"model": name, **metrics})

    table = pd.DataFrame(records).set_index("model")

    save(table.reset_index(), "all_test_metrics.csv")

    save(
        table.loc[["CF750", "HY750"]].reset_index(),
        "same_k750_metrics.csv",
    )

    save(
        table.loc[
            ["Popularity", "CF750", "HY1000"]
        ].reset_index(),
        "best_models_metrics.csv",
    )

    # alpha=1 must preserve the CF ranking.
    for k in [750, 1000]:
        _, control = evaluate_ranking(
            combine_scores(
                scores[f"CF{k}"], content, 1.0
            )
        )

        np.testing.assert_allclose(
            control.sort_values("user_id")[METRICS],
            users[f"CF{k}"][METRICS],
            rtol=0,
            atol=EPS,
        )

    return scores, users, table


def paired_statistics(users):
    comparisons = [
        ("HY750", "CF750"),
        ("HY1000", "CF1000"),
        ("HY1000", "CF750"),
        ("HY1000", "Popularity"),
    ]

    records = []
    differences = []

    for hybrid, baseline in comparisons:
        a = users[hybrid]
        b = users[baseline]

        pd.testing.assert_frame_equal(
            a[["candidate_count", "positive_count"]],
            b[["candidate_count", "positive_count"]],
        )

        for metric in ["NDCG@10", "Recall@10"]:
            delta = (a[metric] - b[metric]).to_numpy()
            differences.append(delta)

            records.append({
                "comparison": f"{hybrid} - {baseline}",
                "metric": metric,
                "primary": (
                    hybrid == "HY750"
                    and metric == "NDCG@10"
                ),
                "difference": delta.mean(),
                "relative_gain_percent": (
                    100 * delta.mean() / b[metric].mean()
                ),
                "wins_percent": 100 * (delta > EPS).mean(),
                "ties_percent": (
                    100 * (np.abs(delta) <= EPS).mean()
                ),
                "losses_percent": 100 * (delta < -EPS).mean(),
            })

    print(
        "Paired-user bootstrap: 20,000 resamples...",
        flush=True,
    )

    differences = np.stack(differences)
    varying = np.ptp(differences, axis=1) > 1e-15

    low = differences.mean(axis=1).copy()
    high = low.copy()

    if varying.any():
        result = bootstrap(
            (differences[varying],),
            np.mean,
            axis=-1,
            vectorized=True,
            method="BCa",
            confidence_level=0.95,
            n_resamples=RESAMPLES,
            batch=64,
            rng=np.random.default_rng(SEED),
        )

        low[varying], high[varying] = (
            result.confidence_interval
        )

    if not np.isfinite(np.r_[low, high]).all():
        raise ValueError(
            "Nonfinite bootstrap confidence interval"
        )

    table = pd.DataFrame(records)
    table["ci95_low"] = low
    table["ci95_high"] = high

    table["conclusion"] = np.select(
        [low > 0, high < 0],
        ["Hybrid higher", "Hybrid lower"],
        default="Difference not established",
    )

    save(table, "paired_bootstrap_results.csv")

    detail = pd.DataFrame({
        "user_id": users["HY750"].index
    })

    for row, delta in zip(records, differences):
        column = (
            row["comparison"] + " / " + row["metric"]
        )
        detail[column] = delta

    save(detail, "paired_user_differences.csv")

    return table


def history_groups(train, users):
    cohort = users["CF750"][
        ["candidate_count", "positive_count"]
    ].copy()

    cohort["history_count"] = (
        train.groupby("user_id")
        .size()
        .reindex(cohort.index)
    )

    cohort["history_group"] = pd.cut(
        cohort.history_count,
        [4, 9, 19, 49, np.inf],
        labels=GROUPS,
    )

    save(
        cohort.reset_index(),
        "test_user_characteristics.csv",
    )

    rows = []

    for name in [
        "Popularity", "CF750", "HY750", "CF1000", "HY1000"
    ]:
        data = users[name].join(
            cohort[["history_group"]]
        )

        for label, group in data.groupby(
            "history_group", observed=True
        ):
            rows.append({
                "model": name,
                "history_group": str(label),
                "users": len(group),
                **group[METRICS].mean().to_dict(),
            })

    table = pd.DataFrame(rows)
    save(table, "performance_by_history.csv")

    delta = (
        users["HY750"]["NDCG@10"]
        - users["CF750"]["NDCG@10"]
    )

    changes = []

    for label, group in cohort.join(
        delta.rename("difference")
    ).groupby("history_group", observed=True):
        d = group.difference

        changes.append({
            "history_group": str(label),
            "users": len(group),
            "NDCG_difference": d.mean(),
            "wins": int((d > EPS).sum()),
            "ties": int((d.abs() <= EPS).sum()),
            "losses": int((d < -EPS).sum()),
        })

    save(
        pd.DataFrame(changes),
        "same_k750_changes_by_history.csv",
    )

    return table


def example_cases(scores, users, model, content):
    delta = (
        users["HY750"]["NDCG@10"]
        - users["CF750"]["NDCG@10"]
    )

    details = []

    for kind, mask in [
        ("improved", delta > EPS),
        ("worsened", delta < -EPS),
    ]:
        values = delta[mask]

        if values.empty:
            continue

        user = int(
            (values - values.median())
            .abs()
            .sort_values(kind="stable")
            .index[0]
        )

        ordered = {}

        for name in ["Popularity", "CF750", "HY750"]:
            frame = (
                scores[name]
                .loc[scores[name].user_id.eq(user)]
                .sort_values(
                    ["score", "video_id"],
                    ascending=[False, True],
                )
                .copy()
            )

            frame["rank"] = np.arange(
                1, len(frame) + 1
            )
            ordered[name] = frame

        top = pd.concat([
            frame.head(10).assign(model=name)
            for name, frame in ordered.items()
        ])

        save(top, f"case_{kind}_top10.csv")

        cf_ranks = (
            ordered["CF750"]
            .set_index("video_id")["rank"]
        )

        promoted = ordered["HY750"].head(10).copy()
        promoted["CF_rank"] = promoted.video_id.map(
            cf_ranks
        )

        label = 1 if kind == "improved" else 0

        targets = promoted.loc[
            promoted.label.eq(label)
            & promoted.CF_rank.gt(promoted["rank"])
        ]

        row = (
            targets if not targets.empty else promoted
        ).iloc[0]

        video = int(row.video_id)

        item_row = pd.Index(
            content["video_ids"]
        ).get_loc(video)

        profile_row = pd.Index(
            content["user_ids"]
        ).get_loc(user)

        profile = pd.DataFrame({
            "tag_id": content["tag_ids"],
            "preference": (
                content["user_profiles"][profile_row]
            ),
        })

        save(
            profile.sort_values(
                "preference", ascending=False
            ),
            f"case_{kind}_profile.csv",
        )

        save(
            explain_item_knn(model, user, video, 750),
            f"case_{kind}_CF_contributions.csv",
        )

        details.append({
            "kind": kind,
            "user_id": user,
            "CF_NDCG": float(
                users["CF750"].loc[user, "NDCG@10"]
            ),
            "Hybrid_NDCG": float(
                users["HY750"].loc[user, "NDCG@10"]
            ),
            "video_id": video,
            "label": int(row.label),
            "tag_ids": content["tag_ids"][
                content["item_vectors"][item_row] > 0
            ].tolist(),
            "CF_rank": int(row.CF_rank),
            "Hybrid_rank": int(row["rank"]),
            "cf_norm": float(row.cf_norm),
            "content_norm": float(row.content_norm),
        })

    (OUT / "case_descriptions.json").write_text(
        json.dumps(
            details,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return details


def figures(
    metrics,
    statistics,
    validation_cf,
    validation_hybrid,
    groups,
):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })

    def finish(fig, name):
        fig.savefig(
            OUT / f"{name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        fig.savefig(
            OUT / f"{name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    def bars(initial, final, labels, name, title, legends):
        fig, axes = plt.subplots(
            1, 2,
            figsize=(11, 4),
            layout="constrained",
        )

        x = np.arange(len(labels))

        for ax, metric in zip(axes, METRICS):
            series = [
                (-0.17, initial, "#356abd", legends[0]),
                (0.17, final, "#138572", legends[1]),
            ]

            for shift, names, color, legend in series:
                values = metrics.loc[
                    names, metric
                ].to_numpy()

                ax.bar(
                    x + shift,
                    values,
                    0.32,
                    label=legend,
                    color=color,
                )

                for position, value in zip(
                    x + shift, values
                ):
                    ax.text(
                        position,
                        value + 0.004,
                        f"{value:.6f}",
                        ha="center",
                        fontsize=8,
                    )

            ax.set_xticks(x, labels)
            ax.set_ylim(
                0,
                float(metrics[metric].max()) * 1.25,
            )
            ax.set_title(metric)
            ax.legend(fontsize=8)

        fig.suptitle(title)
        finish(fig, name)

    bars(
        ["CF750"],
        ["HY750"],
        ["k=750"],
        "01_same_k750",
        "Tag effect with matched neighborhoods",
        ["Item-KNN", "Hybrid"],
    )

    bars(
        ["CF100", "HY100"],
        ["CF750", "HY1000"],
        ["Item-KNN", "Hybrid"],
        "05_initial_vs_final",
        "Initial and validation-selected final settings",
        ["Initial", "Final"],
    )

    rows = statistics.loc[
        statistics.metric.eq("NDCG@10")
    ].iloc[::-1]

    fig, ax = plt.subplots(
        figsize=(9, 4),
        layout="constrained",
    )

    mean = rows.difference.to_numpy()

    ax.errorbar(
        mean,
        np.arange(len(rows)),
        fmt="o",
        capsize=4,
        xerr=np.array([
            mean - rows.ci95_low,
            rows.ci95_high - mean,
        ]),
    )

    ax.set_yticks(
        np.arange(len(rows)),
        rows.comparison,
    )
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel(
        "Mean NDCG@10 difference, BCa 95% CI"
    )

    finish(fig, "02_NDCG_confidence_intervals")

    fig, ax = plt.subplots(
        figsize=(8, 4),
        layout="constrained",
    )

    data = validation_cf.sort_values("neighbors")

    ax.plot(
        data.neighbors,
        data["NDCG@10"],
        "o-",
    )

    chosen = data.loc[
        data.neighbors.eq(750), "NDCG@10"
    ].iloc[0]

    ax.scatter(
        [750],
        [chosen],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
    )

    ax.annotate(
        "Selected k=750",
        (750, chosen),
        xytext=(-30, -30),
        textcoords="offset points",
    )

    ax.set(
        xlabel="Neighbors (k)",
        ylabel="Validation NDCG@10",
        title="Item-KNN: recomputed validation grid",
    )

    finish(fig, "03_validation_neighbors")

    grid = (
        validation_hybrid.pivot(
            index="neighbors",
            columns="alpha",
            values="NDCG@10",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5),
        layout="constrained",
    )

    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="YlGnBu",
    )

    ax.set_xticks(
        range(len(grid.columns)),
        [f"{a:g}" for a in grid.columns],
    )
    ax.set_yticks(
        range(len(grid.index)),
        grid.index,
    )

    threshold = (
        grid.to_numpy().min()
        + grid.to_numpy().max()
    ) / 2

    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            value = grid.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.5f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    ax.set(
        xlabel="CF weight (alpha)",
        ylabel="Neighbors (k)",
        title="Hybrid: recomputed validation NDCG@10",
    )

    i = grid.index.get_loc(1000)
    j = grid.columns.get_loc(0.8)

    ax.add_patch(
        plt.Rectangle(
            (j - 0.48, i - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor="orange",
            linewidth=3,
        )
    )

    fig.colorbar(im, ax=ax)

    finish(fig, "04_validation_hybrid")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 4),
        layout="constrained",
    )

    for ax, metric in zip(axes, METRICS):
        for name in [
            "Popularity", "CF750", "HY1000"
        ]:
            values = (
                groups.loc[groups.model.eq(name)]
                .set_index("history_group")[metric]
            )

            ax.plot(
                GROUPS,
                values.reindex(GROUPS),
                "o-",
                label=name,
            )

        ax.set(
            xlabel="Positive training videos",
            ylabel=metric,
        )
        ax.legend()

    finish(fig, "06_history_groups")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    settings_path = RESULT_DIR / "best_settings.json"
    frozen = settings_path.read_bytes()
    settings = json.loads(frozen)

    if (
        settings.get("item_knn") != {"neighbors": 750}
        or settings.get("hybrid")
        != {"neighbors": 1000, "alpha": 0.8}
        or settings.get("selection_split") != "validation"
        or settings.get("selection_metric") != "NDCG@10"
        or settings.get("hybrid_scoring_rule")
        != HYBRID_SCORING_RULE
    ):
        raise ValueError(
            "Complete Step 1 before this analysis"
        )

    validation_cf = read(
        RESULT_DIR / "item_knn_validation_metrics.csv"
    )
    validation_hybrid = read(
        RESULT_DIR / "hybrid_validation_metrics.csv"
    )

    if (
        len(validation_cf) != 8
        or len(validation_hybrid) != 48
    ):
        raise ValueError(
            "Complete the expanded validation grid in Step 1"
        )

    best_cf = validation_cf.sort_values(
        ["NDCG@10", "neighbors"],
        ascending=[False, True],
    ).iloc[0]

    best_hy = validation_hybrid.sort_values(
        ["NDCG@10", "neighbors", "alpha"],
        ascending=[False, True, True],
    ).iloc[0]

    if (
        int(best_cf.neighbors) != 750
        or int(best_hy.neighbors) != 1000
        or float(best_hy.alpha) != 0.8
    ):
        raise ValueError(
            "The settings differ from the validation winners"
        )

    train = read(
        PROCESSED_DIR / "train_positive.csv"
    )
    candidates = read(
        PROCESSED_DIR / "test_candidates.csv"
    )

    model = load_item_knn(PROCESSED_DIR)
    content = load_content_model(
        PROCESSED_DIR / "content_model.npz"
    )

    check_inputs(
        train, candidates, model, content
    )

    scores, users, metrics = evaluate_models(
        train, candidates, model, content
    )

    statistics = paired_statistics(users)
    groups = history_groups(train, users)

    cases = example_cases(
        scores, users, model, content
    )

    save(
        validation_cf,
        "validation_item_knn_metrics.csv",
    )
    save(
        validation_hybrid,
        "validation_hybrid_metrics.csv",
    )

    figures(
        metrics,
        statistics,
        validation_cf,
        validation_hybrid,
        groups,
    )

    primary = statistics.loc[
        statistics.primary
    ].iloc[0]

    report = (
        "대표 비교: Item-KNN 750 vs Hybrid 750, alpha=0.8\n"
        "모델별 선택 설정: Item-KNN 750 vs Hybrid 1000, alpha=0.8\n\n"
        + metrics.to_string()
        + "\n\n"
        + statistics.to_string(index=False)
        + "\n\n사용자별 NDCG: 개선 "
        + f"{primary.wins_percent:.2f}%, "
        + f"동률 {primary.ties_percent:.2f}%, "
        + f"하락 {primary.losses_percent:.2f}%\n\n"
        + "추천 사례: 같은 k=750 비교의 개선/하락 집단별 중앙값 근처 사용자\n"
        + json.dumps(
            cases, ensure_ascii=False, indent=2
        )
        + "\n\n해석: 양의 NDCG 신뢰구간은 평균 개선을 뒷받침하나 개선 폭은 작습니다.\n"
        + "Recall 차이의 구간에 0이 포함되면 확실한 개선으로 단정하지 않습니다.\n"
        + "신뢰구간은 비교별 구간이며 여러 비교의 동시 95% 보장을 뜻하지 않습니다.\n"
        + "검증표는 명시한 범위를 재계산한 자료이며 원래 전체 탐색 기록은 아닙니다.\n"
        + "초기 테스트 확인 이후의 추가 분석이며 새로운 미사용 테스트의 확증이 아닙니다.\n"
        + "평가 사용자와 무작위 노출 후보 안에서의 성능으로 해석합니다.\n"
        + "집단별 평균은 설명 분석이며 집단 간 유의성 검정 결과는 아닙니다.\n"
        + "정답 0은 비호감을 뜻하지 않으며 사례 원점수는 모델 간 직접 비교하지 않습니다.\n"
    )

    (OUT / "analysis_summary.txt").write_text(
        report,
        encoding="utf-8-sig",
    )

    sources = [
        Path(__file__),
        PROCESSED_DIR / "train_positive.csv",
        PROCESSED_DIR / "test_candidates.csv",
        settings_path,
    ]

    meta = {
        "comparison_k": 750,
        "settings": settings,
        "resamples": RESAMPLES,
        "seed": SEED,
        "prior_test_seen": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "sha256": {
            str(p.name): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
    }

    (OUT / "analysis_settings.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    if settings_path.read_bytes() != frozen:
        raise ValueError(
            "The selected settings were unexpectedly changed"
        )

    print("\n[SAME k=750]")
    print(
        metrics.loc[
            ["CF750", "HY750"]
        ].to_string()
    )

    print("\n[PRIMARY STATISTICS]")
    print(
        statistics.loc[
            statistics.primary
        ].to_string(index=False)
    )

    print("\nSaved analyses to:", OUT)


if __name__ == "__main__":
    main()