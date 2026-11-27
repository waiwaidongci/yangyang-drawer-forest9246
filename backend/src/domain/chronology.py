"""Partition complete assembly batches using label availability timestamps."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

from .records import MIN_BATCHES_PER_STAGE, MIN_ROWS_PER_STAGE


@dataclass(frozen=True)
class Batch:
    batch_id: str
    first_measurement: str
    last_confirmation: str
    rows: tuple[dict, ...]


@dataclass(frozen=True)
class Partition:
    training: tuple[Batch, ...]
    validation: tuple[Batch, ...]
    holdout: tuple[Batch, ...]

    def stages(self) -> tuple[tuple[str, tuple[Batch, ...]], ...]:
        return (
            ("训练", self.training),
            ("验证定阈", self.validation),
            ("留出评估", self.holdout),
        )


def batch_rows(batches: tuple[Batch, ...]) -> list[dict]:
    return [row for batch in batches for row in batch.rows]


def _ordered_batches(rows: list[dict]) -> list[Batch]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["batch_id"]].append(row)
    batches = [
        Batch(
            batch_id,
            min(row["measured_at"] for row in members),
            max(row["confirmed_at"] for row in members),
            tuple(sorted(members, key=lambda row: (row["measured_at"], row["id"]))),
        )
        for batch_id, members in grouped.items()
    ]
    return sorted(batches, key=lambda batch: (batch.first_measurement, batch.batch_id))


def _cut_positions(batches: list[Batch]) -> tuple[int, int]:
    row_totals = [0]
    boundaries = []
    latest_confirmation = ""
    for cut, batch in enumerate(batches):
        if cut and latest_confirmation < batch.first_measurement:
            boundaries.append(cut)
        latest_confirmation = max(latest_confirmation, batch.last_confirmation)
        row_totals.append(row_totals[-1] + len(batch.rows))

    group_min = MIN_BATCHES_PER_STAGE
    row_min = MIN_ROWS_PER_STAGE
    batch_count = len(batches)
    training_cuts = [
        cut for cut in boundaries if cut >= group_min and row_totals[cut] >= row_min
    ]
    training_position = 0
    best_training = None
    best_pair = None
    for second in boundaries:
        if second > batch_count - group_min or row_totals[-1] - row_totals[second] < row_min:
            break
        # Feasible first cuts only expand as the second cut moves forward.
        row_limit = bisect_right(row_totals, row_totals[second] - row_min) - 1
        first_limit = min(second - group_min, row_limit)
        while training_position < len(training_cuts):
            first = training_cuts[training_position]
            if first > first_limit:
                break
            training_rank = (abs(5 * first - 3 * batch_count), first)
            if best_training is None or training_rank < best_training:
                best_training = training_rank
            training_position += 1
        if best_training is not None:
            error, first = best_training
            rank = (error + abs(5 * second - 4 * batch_count), first, second)
            if best_pair is None or rank < best_pair:
                best_pair = rank
    if best_pair is None:
        raise ValueError(
            "无法按整批时间划分训练、验证和留出集。每段至少 2 批、8 条记录，"
            "且上一段最晚末检确认须早于下一段最早装配测读。"
        )
    return best_pair[1:]


def partition_batches(rows: list[dict]) -> Partition:
    batches = _ordered_batches(rows)
    first, second = _cut_positions(batches)
    return Partition(tuple(batches[:first]), tuple(batches[first:second]), tuple(batches[second:]))
