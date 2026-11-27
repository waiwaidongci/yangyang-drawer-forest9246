"""Train the inspection classifier using only features known at measurement time."""

from sklearn.ensemble import RandomForestClassifier

from .records import MEASUREMENTS


FEATURES = {
    **{key: spec[0] for key, spec in MEASUREMENTS.items()},
    "gap_imbalance_mm": "左右间隙差绝对值（mm）",
}
FOREST_SEED = 51125
TREE_COUNT = 160


def _features(rows: list[dict]) -> list[list[float]]:
    return [
        [row[key] for key in MEASUREMENTS] + [abs(row["left_gap_mm"] - row["right_gap_mm"])]
        for row in rows
    ]


def fit_forest(training_rows: list[dict]) -> RandomForestClassifier:
    labels = [row["failed"] for row in training_rows]
    if set(labels) != {0, 1}:
        raise ValueError("训练集须同时包含末检通过和失败记录才能拟合随机森林")
    forest = RandomForestClassifier(
        n_estimators=TREE_COUNT,
        criterion="gini",
        max_depth=7,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=FOREST_SEED,
        n_jobs=1,
    )
    forest.fit(_features(training_rows), labels)
    return forest


def failure_scores(forest: RandomForestClassifier, rows: list[dict]) -> list[float]:
    failed_column = list(forest.classes_).index(1)
    return forest.predict_proba(_features(rows))[:, failed_column].tolist()
