"""Export the cut, validation decision and independent holdout errors for review."""

from collections import Counter, defaultdict

from sklearn.metrics import roc_auc_score

from .forest import FEATURES


OUTCOMES = {
    "tp": "正确优先返测（TP）",
    "fp": "误报（FP）",
    "tn": "正确常规末检（TN）",
    "fn": "漏报（FN）",
}


def table(key, title, labels, rows):
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def split_table(partition):
    rows = [
        {
            "stage": stage,
            "batch_id": batch.batch_id,
            "first_measurement": batch.first_measurement,
            "last_confirmation": batch.last_confirmation,
            "count": len(batch.rows),
            "failed": sum(row["failed"] for row in batch.rows),
        }
        for stage, batches in partition.stages()
        for batch in batches
    ]
    return table("split_batches", "整批时间切分台账", {
        "stage": "数据用途", "batch_id": "装配批次", "first_measurement": "最早装配测读",
        "last_confirmation": "最晚末检确认", "count": "记录数", "failed": "失败数",
    }, rows)


def threshold_table(decision):
    rows = [{
        "selected": "是" if point == decision.chosen else "否",
        "rule": "全不标记" if point.threshold is None else "得分大于或等于阈值",
        "threshold": point.threshold,
        "tp": point.true_positive,
        "fp": point.false_positive,
        "tn": point.true_negative,
        "fn": point.false_negative,
        "cost": point.cost,
    } for point in decision.candidates]
    rows.sort(key=lambda row: row["selected"] != "是")
    return table("threshold_audit", "验证集阈值与代价审计", {
        "selected": "采用", "rule": "优先返测规则", "threshold": "得分阈值", **OUTCOMES,
        "cost": "验证集代价（等效分钟）",
    }, rows)


def feature_table(forest):
    contributions = sorted(
        zip(FEATURES, forest.feature_importances_.tolist(), strict=True), key=lambda pair: (-pair[1], pair[0]),
    )
    rows = [{"feature": FEATURES[key], "importance": importance} for key, importance in contributions]
    return table("feature_importance", "训练集特征重要性", {"feature": "测读特征", "importance": "基尼重要性"}, rows)


def holdout_diagnostics(rows, scores, policy, costs):
    queue = []
    by_batch = defaultdict(Counter)
    error_cost = {"tp": 0, "fp": costs.unnecessary_retest, "tn": 0, "fn": costs.missed_failure}
    ranked = sorted(
        zip(rows, scores, strict=True), key=lambda pair: (-pair[1], pair[0]["measured_at"], pair[0]["id"]),
    )
    for row, score in ranked:
        flagged = policy.flags(score)
        failed = row["failed"]
        outcome = ("tp" if failed else "fp") if flagged else ("fn" if failed else "tn")
        by_batch[row["batch_id"]][outcome] += 1
        queue.append({
            "rank": len(queue) + 1,
            "id": row["id"],
            "batch_id": row["batch_id"],
            "measured_at": row["measured_at"],
            "score": score,
            "action": "优先返测" if flagged else "常规末检",
            "actual": "失败" if failed else "通过",
            "outcome": OUTCOMES[outcome],
            "cost": error_cost[outcome],
        })
    counts = sum(by_batch.values(), Counter())
    true_positive, false_positive, false_negative = (counts[key] for key in ("tp", "fp", "fn"))
    actual_failures = true_positive + false_negative
    flagged_count = true_positive + false_positive
    labels = [row["failed"] for row in rows]
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    metrics = {
        "优先返测阈值": "全不标记" if policy.threshold is None else policy.threshold,
        "留出优先返测数": flagged_count,
        "留出失败召回率": round(true_positive / actual_failures, 4) if actual_failures else None,
        "留出 ROC AUC": round(auc, 4) if auc is not None else None,
        "留出总代价（等效分钟）": costs.total(false_negative, false_positive),
    }
    warnings = []
    if auc is None:
        warnings.append("留出集只含一种末检结果，无法计算 ROC AUC。没有失败记录时，失败召回率留空。")
    if false_negative:
        warnings.append(f"留出集有 {false_negative} 条漏报，请在返测顺序单中复核相应总成。")
    warnings.append("随机森林得分未经概率校准，特征重要性反映训练树的基尼分裂贡献，供排查线索参考。")
    confusion_rows = [
        {"outcome": label, "count": counts[key], "unit_cost": error_cost[key], "cost": counts[key] * error_cost[key]}
        for key, label in OUTCOMES.items()
    ]
    batch_audits = []
    for batch_id, batch_counts in by_batch.items():
        batch_audits.append({
            "batch_id": batch_id,
            "count": batch_counts.total(),
            **{key: batch_counts[key] for key in OUTCOMES},
            "cost": costs.total(batch_counts["fn"], batch_counts["fp"]),
        })
    batch_audits.sort(key=lambda row: (-row["cost"], row["batch_id"]))
    tables = [
        table("retest_queue", "留出批次返测顺序单", {
            "rank": "返测顺序", "id": "滑轨总成编号", "batch_id": "装配批次", "measured_at": "装配测读时间",
            "score": "随机森林失败得分", "action": "分诊建议", "actual": "末检结果",
            "outcome": "分类结果", "cost": "代价（等效分钟）",
        }, queue),
        table("confusion", "留出集分类误差与代价", {
            "outcome": "分类结果", "count": "记录数", "unit_cost": "单件代价（等效分钟）",
            "cost": "合计代价（等效分钟）",
        }, confusion_rows),
        table("batch_errors", "留出批次误报与漏报对账", {
            "batch_id": "装配批次", "count": "记录数", **OUTCOMES, "cost": "代价（等效分钟）",
        }, batch_audits),
    ]
    return metrics, tables, warnings
