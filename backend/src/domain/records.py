"""CSV and seed schema for archived assembly measurements and confirmed outcomes."""

import math
import re
from datetime import datetime


MEASUREMENTS = {
    "left_gap_mm": ("左侧间隙（mm）", 0, 8),
    "right_gap_mm": ("右侧间隙（mm）", 0, 8),
    "parallel_error_mm": ("双轨平行偏差（mm）", 0, 5),
    "push_force_n": ("推入力（N）", 0, 120),
    "travel_jitter_mm": ("行程抖动（mm）", 0, 8),
    "screw_torque_nm": ("固定螺钉扭矩（N·m）", 0, 6),
    "drawer_load_kg": ("抽屉试装载荷（kg）", 0, 75),
}
FIELDS = {
    "id": "滑轨总成编号",
    "batch_id": "装配批次",
    "measured_at": "装配测读时间",
    **{key: unit[0] for key, unit in MEASUREMENTS.items()},
    "confirmed_at": "末检确认时间",
    "failed": "末检失败（0/1）",
}
COLUMNS = list(FIELDS)
LOCAL_MINUTE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}")
MIN_BATCHES_PER_STAGE = 2
MIN_ROWS_PER_STAGE = 8


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空")
    return value.strip()


def _timestamp(value, label):
    if isinstance(value, str) and LOCAL_MINUTE.fullmatch(value.strip()):
        text = value.strip()
        try:
            datetime.fromisoformat(text)
        except ValueError:
            pass
        else:
            return text
    raise ValueError(f"{label}须为有效的工厂本地时间（YYYY-MM-DD HH:MM）")


def _measurement(value, label, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label}须为数字")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label}须为数字") from exc
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{label}须在 {lower} 到 {upper} 之间且为有限数值")
    return number


def _failure(value):
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value)
    raise ValueError("末检失败须填写 0（通过）或 1（失败）")


def normalize_rows(rows: list[dict]) -> list[dict]:
    minimum_rows = 3 * MIN_ROWS_PER_STAGE
    if not isinstance(rows, list) or len(rows) < minimum_rows:
        raise ValueError(f"至少需要 {minimum_rows} 条已完成末检的装配记录")
    records = []
    identifiers = set()
    batches = set()
    expected = set(COLUMNS)
    for line, row in enumerate(rows, 2):
        if not isinstance(row, dict):
            raise ValueError(f"第 {line} 行须为一条装配记录")
        missing = [key for key in COLUMNS if key not in row]
        extra = sorted(str(key) for key in row.keys() - expected)
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"缺少列 {', '.join(missing)}")
            if extra:
                problems.append(f"存在额外列 {', '.join(extra)}")
            raise ValueError(f"第 {line} 行{'，'.join(problems)}")
        try:
            record_id = _identifier(row["id"], "滑轨总成编号")
            if record_id in identifiers:
                raise ValueError(f"滑轨总成编号重复（{record_id}）")
            batch_id = _identifier(row["batch_id"], "装配批次")
            measured_at = _timestamp(row["measured_at"], "装配测读时间")
            confirmed_at = _timestamp(row["confirmed_at"], "末检确认时间")
            if confirmed_at < measured_at:
                raise ValueError("末检确认时间早于装配测读时间")
            measurements = {
                key: _measurement(row[key], *spec) for key, spec in MEASUREMENTS.items()
            }
            failed = _failure(row["failed"])
        except ValueError as exc:
            raise ValueError(f"第 {line} 行：{exc}") from exc
        identifiers.add(record_id)
        batches.add(batch_id)
        records.append({
            "id": record_id,
            "batch_id": batch_id,
            "measured_at": measured_at,
            **measurements,
            "confirmed_at": confirmed_at,
            "failed": failed,
        })
    minimum_batches = 3 * MIN_BATCHES_PER_STAGE
    if len(batches) < minimum_batches:
        raise ValueError(f"至少需要 {minimum_batches} 个不同装配批次")
    return records
