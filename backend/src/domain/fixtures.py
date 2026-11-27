"""Reproducible slide batches with overlapping risk patterns and measurement drift."""

from datetime import datetime, timedelta
from random import Random

from .records import normalize_rows


DATASET_NAME = "YC-滑轨装配末检-20261012_1104.csv"
SEED = 261125


def demo_rows() -> list[dict]:
    rng = Random(SEED)
    rows = []
    first_shift = datetime(2026, 10, 12, 8, 10)
    for batch_index in range(18):
        week, weekday = divmod(batch_index, 5)
        shift = first_shift + timedelta(days=week * 7 + weekday)
        batch_id = f"YC-HG-{shift:%m%d}-A"
        setpoint = rng.uniform(-0.3, 0.34)
        frame_bias = rng.gauss(0, 0.24)
        for unit in range(1, 13):
            measured_at = shift + timedelta(minutes=(unit - 1) * 9)
            confirmed_at = measured_at + timedelta(minutes=rng.randint(34, 104))
            load = rng.choice((12, 18, 24, 30, 36)) + rng.uniform(-1.4, 1.4)
            nominal_gap = rng.gauss(1.55 + setpoint, 0.3)
            left_gap = round(max(0.25, nominal_gap + rng.gauss(frame_bias, 0.34)), 3)
            right_gap = round(max(0.25, nominal_gap + rng.gauss(-frame_bias, 0.33)), 3)
            skew = round(max(0.04, rng.gauss(0.59 + abs(setpoint), 0.28)), 3)
            force = round(max(9, rng.gauss(24 + 0.41 * load + 9 * skew, 6.1)), 2)
            jitter = round(max(0.06, rng.gauss(0.44 + 0.21 * skew, 0.23)), 3)
            torque = round(max(0.45, rng.gauss(1.81 + setpoint, 0.31)), 3)
            if batch_index >= 14:
                force = round(force + 2.7, 2)
            gap_imbalance = abs(left_gap - right_gap)
            fault_pressure = (
                1.35 * (gap_imbalance > 0.55 and skew > 0.68)
                + 1.50 * (force > 42 and jitter > 0.56)
                + 1.10 * (torque < 1.55 and load > 25)
                + 0.75 * (min(left_gap, right_gap) < 0.95 or skew > 1.02)
                + rng.gauss(0, 0.72)
            )
            rows.append({
                "id": f"{batch_id}-{unit:02d}",
                "batch_id": batch_id,
                "measured_at": measured_at.isoformat(sep=" ", timespec="minutes"),
                "left_gap_mm": left_gap,
                "right_gap_mm": right_gap,
                "parallel_error_mm": skew,
                "push_force_n": force,
                "travel_jitter_mm": jitter,
                "screw_torque_nm": torque,
                "drawer_load_kg": round(load, 2),
                "confirmed_at": confirmed_at.isoformat(sep=" ", timespec="minutes"),
                "failed": int(fault_pressure > 1.30),
            })
    return normalize_rows(rows)
