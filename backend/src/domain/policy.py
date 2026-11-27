"""Choose the retest policy against known outcomes in the middle time window."""

import math
from dataclasses import dataclass
from itertools import groupby


SETTINGS = [
    {"key": "miss_cost", "label": "漏报代价（等效工时分钟）", "default": 12, "min": 1, "max": 100, "step": 1},
    {"key": "retest_cost", "label": "误报代价（等效工时分钟）", "default": 3, "min": 1, "max": 100, "step": 1},
]


@dataclass(frozen=True)
class Costs:
    missed_failure: int
    unnecessary_retest: int

    def total(self, false_negative: int, false_positive: int) -> int:
        return false_negative * self.missed_failure + false_positive * self.unnecessary_retest


@dataclass(frozen=True)
class Candidate:
    threshold: float | None
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    cost: int

    def flags(self, score: float) -> bool:
        return self.threshold is not None and score >= self.threshold


@dataclass(frozen=True)
class Decision:
    chosen: Candidate
    candidates: tuple[Candidate, ...]


def parse_costs(settings: dict) -> Costs:
    if not isinstance(settings, dict):
        raise ValueError("返测设置须为键值对象")
    if settings.keys() - {setting["key"] for setting in SETTINGS}:
        raise ValueError("返测设置包含未知参数")
    values = []
    for setting in SETTINGS:
        raw = settings.get(setting["key"], setting["default"])
        message = f'{setting["label"]}须为 {setting["min"]} 至 {setting["max"]} 的整数'
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(message)
        try:
            value = float(raw)
        except (OverflowError, ValueError) as exc:
            raise ValueError(message) from exc
        if not math.isfinite(value) or not value.is_integer() or not setting["min"] <= value <= setting["max"]:
            raise ValueError(message)
        values.append(int(value))
    return Costs(*values)


def select_threshold(validation_scores: list[float], validation_labels: list[int], costs: Costs) -> Decision:
    if set(validation_labels) != {0, 1}:
        raise ValueError("验证定阈集须同时包含末检通过和失败记录，请补齐该时段的已确认样本")
    failed = sum(validation_labels)
    passed = len(validation_labels) - failed
    true_positive = false_positive = 0
    candidates = [Candidate(None, 0, 0, passed, failed, costs.total(failed, 0))]
    pairs = sorted(zip(validation_scores, validation_labels, strict=True), reverse=True)
    for score, matches in groupby(pairs, key=lambda pair: pair[0]):
        for _, label in matches:
            true_positive += label
            false_positive += 1 - label
        false_negative = failed - true_positive
        candidates.append(Candidate(
            score,
            true_positive,
            false_positive,
            passed - false_positive,
            false_negative,
            costs.total(false_negative, false_positive),
        ))
    chosen = min(candidates, key=lambda point: (point.cost, point.false_negative, point.false_positive))
    return Decision(chosen, tuple(candidates))
