"""Domain entry point for the slide retest review workflow."""

from .audit import feature_table, holdout_diagnostics, split_table, threshold_table
from .chronology import batch_rows, partition_batches
from .fixtures import DATASET_NAME, demo_rows
from .forest import TREE_COUNT, failure_scores, fit_forest
from .policy import SETTINGS, parse_costs, select_threshold
from .records import COLUMNS, FIELDS, normalize_rows


SPEC = {
    "columns": COLUMNS,
    "fields": FIELDS,
    "settings": SETTINGS,
    "modules": {"datasets": "装配来数", "runs": "分诊运行", "results": "返测顺序", "reviews": "检验复核"},
    "limitation": "适用于带末检确认的归档批次回放。得分未经概率校准，厂线部署前须另做前瞻验证。",
    "dataset_name": DATASET_NAME,
    "recommendation": "按留出批次返测顺序单核查优先返测件，交班时复核漏报与误报代价。",
}


def analyze(rows: list[dict], settings: dict) -> dict:
    costs = parse_costs(settings)
    partition = partition_batches(normalize_rows(rows))
    training = batch_rows(partition.training)
    validation = batch_rows(partition.validation)

    forest = fit_forest(training)
    decision = select_threshold(failure_scores(forest, validation), [row["failed"] for row in validation], costs)

    holdout = batch_rows(partition.holdout)
    metrics, tables, warnings = holdout_diagnostics(
        holdout, failure_scores(forest, holdout), decision.chosen, costs,
    )
    if decision.chosen.threshold is None:
        warnings.append("验证集最省代价的规则为全不标记，留出装配件均进入常规末检队列。")
    return {
        "summary": (
            f"使用 {len(partition.training)} 个早期批次的 {len(training)} 条记录训练 {TREE_COUNT} 棵树。"
            f"在随后 {len(partition.validation)} 个批次的 {len(validation)} 条记录中按误报和漏报代价定阈，"
            f"验证集最低代价为 {decision.chosen.cost} 等效分钟。"
            f"最后 {len(partition.holdout)} 个批次的 {len(holdout)} 条记录用于留出评估和返测对账。"
        ),
        "metrics": metrics,
        "tables": [*tables, split_table(partition), threshold_table(decision), feature_table(forest)],
        "warnings": warnings,
    }
