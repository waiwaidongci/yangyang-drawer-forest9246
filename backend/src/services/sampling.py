"""两波抽检服务：从分诊运行/台账建立计划，持久化聚合快照，串行化并发领单与回执。"""

from __future__ import annotations

import threading

from fastapi import HTTPException
from sqlalchemy import select

from src.domain.sampling import (
    MISS_ROUTINE_FAILURE,
    SamplingError,
    SamplingPlan,
    frame_from_run_result,
)
from src.models.run import Run
from src.models.sampling import SamplingPlanRecord
from src.services.catalog import get_dataset
from src.services.execution import calculate
from src.utils.time import utc_stamp

_locks_guard = threading.Lock()
_plan_locks: dict[int, threading.RLock] = {}


def plan_lock(plan_id: int) -> threading.RLock:
    with _locks_guard:
        lock = _plan_locks.get(plan_id)
        if lock is None:
            lock = threading.RLock()
            _plan_locks[plan_id] = lock
        return lock


def _pd(payload) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "data"):
        return payload.data()
    return {k: getattr(payload, k) for k in vars(payload)}


def _first(data: dict, names, default=None):
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return default


def _resolve_frame(session, data: dict):
    """返回 (frame_items, config, dataset_id)。"""
    run_id = _first(data, ("run_id", "analysis_run_id"))
    dataset_id = data.get("dataset_id")
    settings = dict(data.get("settings") or {})
    if run_id is not None:
        run = session.get(Run, int(run_id))
        if run is None:
            raise LookupError("找不到分诊运行")
        if run.result is None:
            raise HTTPException(409, "该分诊运行尚未成功完成，不能沿用其留出判定")
        dataset_id = run.dataset_id
        result = run.result
        rows = get_dataset(session, dataset_id).rows
    else:
        if dataset_id is None:
            raise SamplingError("须提供 run_id 或 dataset_id")
        dataset = get_dataset(session, int(dataset_id))
        checked_settings = settings or {"miss_cost": 12, "retest_cost": 3}
        result, error = calculate(dataset.rows, checked_settings)
        if error is not None:
            raise SamplingError(error)
        rows = dataset.rows
    items = frame_from_run_result(result, rows)
    first_wave = int(_first(data, ("first_wave_size", "initial_size", "first_wave", "per_batch"), 3))
    config = {
        "first_wave_size": first_wave,
        "seed": int(_first(data, ("seed", "random_seed"), 261125)),
        "miss_basis": _first(data, ("miss_basis",), MISS_ROUTINE_FAILURE),
        "total_quota": _first(data, ("total_quota", "quota", "budget")),
        "extra_quota": _first(data, ("extra_quota", "remaining_quota", "extra")),
    }
    return items, config, int(dataset_id)


def create_plan(session, payload, actor) -> dict:
    data = _pd(payload)
    items, config, dataset_id = _resolve_frame(session, data)
    label = data.get("label") or f"抽检计划-{utc_stamp()}"
    plan = SamplingPlan(items, **config)
    record = SamplingPlanRecord(
        label=label,
        run_id=_first(data, ("run_id", "analysis_run_id")),
        dataset_id=dataset_id,
        created_by=actor.id,
        created_at=utc_stamp(),
        config=config,
        snapshot=plan.snapshot(),
        status="collecting_first",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return plan_detail(session, record.id, actor)


def _load_record(session, plan_id: int) -> SamplingPlanRecord:
    record = session.get(SamplingPlanRecord, plan_id)
    if record is None:
        raise LookupError("找不到抽检计划")
    return record


def _restore(record: SamplingPlanRecord) -> SamplingPlan:
    return SamplingPlan.restore(record.snapshot)


def _persist(session, record: SamplingPlanRecord, plan: SamplingPlan) -> None:
    record.snapshot = plan.snapshot()
    if plan.supplement_status == "open":
        record.status = "collecting_supplement" if not plan.supplement_complete() else "ready_estimate"
    elif plan.supplement_status == "skipped":
        record.status = "stopped_first_wave"
    elif plan.supplement_status == "complete":
        record.status = "completed_census"
    elif plan.first_wave_complete():
        record.status = "allocating_supplement"
    else:
        record.status = "collecting_first"
    session.commit()


def plan_info(record: SamplingPlanRecord) -> dict:
    return {
        "id": record.id,
        "label": record.label,
        "run_id": record.run_id,
        "dataset_id": record.dataset_id,
        "created_at": record.created_at,
        "status": record.status,
        "config": record.config,
    }


def plan_detail(session, plan_id: int, actor) -> dict:
    record = _load_record(session, plan_id)
    plan = _restore(record)
    orders = [plan.blind_order_view(oid) for oid in plan.orders]
    audit = plan.audit_rows()
    estimates = plan.estimates()
    metrics = {
        "batch_count": len(plan.batch_ids),
        "first_wave_size": plan.first_wave_size,
        "first_wave_total": plan.first_wave_size * len(plan.batch_ids),
        "extra_quota": plan.extra_quota,
        "total_quota": plan.total_quota,
        "first_wave_complete": plan.first_wave_complete(),
        "supplement_status": plan.supplement_status,
        "supplement_complete": plan.supplement_complete(),
        "estimate_basis": estimates["basis"],
        "total_estimated_misses": estimates["total_estimate"],
    }
    return {
        **plan_info(record),
        "metrics": metrics,
        "first_wave_complete": plan.first_wave_complete(),
        "supplement_status": plan.supplement_status,
        "supplement_complete": plan.supplement_complete(),
        "estimates": estimates,
        "batches": audit,
        "orders": orders,
        # 与本项目其它增量接口一致的 tables 字典形态，便于按表核对抽检依据。
        "tables": {
            "batch_audit": audit,
            "sampling_basis": audit,
            "batch_sampling": audit,
            "orders": orders,
            "sampling_orders": orders,
            "estimate": [estimates],
        },
    }


def _receipt_verdict(data: dict):
    if data.get("audit_failed") is not None:
        return data["audit_failed"]
    if data.get("failed") is not None:
        return data["failed"]
    if data.get("verdict") is not None:
        return data["verdict"]
    if data.get("result") is not None:
        result = data["result"]
        if isinstance(result, str):
            return 1 if result.strip() in ("1", "fail", "failed", "失败", "不合格") else 0
        return result
    if data.get("missed") is not None:
        return 0 if data["missed"] is False else 1
    raise SamplingError("回执须包含 audit_failed（或 failed/verdict/result/missed）返检结论")


def submit_receipt(session, plan_id: int, payload, actor) -> dict:
    data = _pd(payload)
    item_id = _first(data, ("item_id", "id"))
    if not item_id:
        raise SamplingError("回执须提供 item_id（或 id）")
    has_verdict = any(data.get(name) is not None
                      for name in ("audit_failed", "failed", "verdict", "result"))
    verdict = _receipt_verdict(data) if has_verdict else None
    missed = data.get("missed")
    key = _first(data, ("idempotency_key", "key", "receipt_key"))
    with plan_lock(plan_id):
        record = _load_record(session, plan_id)
        plan = _restore(record)
        receipt, deduplicated = plan.receive_receipt(
            str(item_id), verdict, key=key, note=data.get("note") or "", missed=missed,
        )
        # 不在这里自动发补抽单：补抽由显式 /supplement 或 wave=2 的领单在齐套闸门后
        # 幂等开立（与“首拨没齐先别发补抽单、领单重试不重复发单”的口径一致）。
        _persist(session, record, plan)
        return {
            "deduplicated": deduplicated,
            "receipt": receipt.to_dict(),
            "first_wave_complete": plan.first_wave_complete(),
            "supplement_status": plan.supplement_status,
            "plan_id": plan_id,
        }


def open_supplement(session, plan_id: int, payload, actor) -> dict:
    with plan_lock(plan_id):
        record = _load_record(session, plan_id)
        plan = _restore(record)
        outcome = plan.open_supplement()
        _persist(session, record, plan)
        if outcome["status"] == "waiting_first_wave":
            raise HTTPException(409, {
                "message": "首拨回执尚未齐套，暂不发补抽单",
                "missing": outcome.get("missing", []),
            })
        return {"plan_id": plan_id, **outcome}


def claim_order(session, plan_id: int, payload, actor) -> dict:
    data = _pd(payload)
    key = _first(data, ("idempotency_key", "key", "request_id"))
    if not key:
        raise SamplingError("领单重试必须带 idempotency_key")
    with plan_lock(plan_id):
        record = _load_record(session, plan_id)
        plan = _restore(record)
        order_id = data.get("order_id")
        wave = data.get("wave")
        wants_supplement = bool(data.get("supplement")) or wave == 2
        if wants_supplement:
            outcome = plan.open_supplement()
            if outcome["status"] == "waiting_first_wave":
                _persist(session, record, plan)
                raise HTTPException(409, {
                    "message": "首拨回执尚未齐套，暂不发补抽单",
                    "missing": outcome.get("missing", []),
                })
            if not order_id and outcome.get("orders"):
                # 领到补抽单：按批顺序取第一张未领的补抽单交给该幂等键。
                claimed = set(plan._claim_keys.values())
                for order in plan.orders.values():
                    if order.wave == 2 and order.id not in claimed and order.claim_key is None:
                        order_id = order.id
                        break
        order, deduplicated = plan.claim(
            str(key), order_id=order_id,
            worker=getattr(actor, "email", None) or getattr(actor, "id", None),
        )
        _persist(session, record, plan)
        return {"deduplicated": deduplicated, "plan_id": plan_id,
                "order": plan.blind_order_view(order.id)}


def reconciliation(session, plan_id: int, actor) -> dict:
    record = _load_record(session, plan_id)
    plan = _restore(record)
    rows = []
    for item_id, item in plan._items.items():
        rows.append({
            "id": item.id,
            "batch_id": item.batch_id,
            "triage_action": "优先返测" if item.flagged else "常规末检",
            "score": item.score,
            "hidden_final_failed": item.final_failed,
            "wave": plan.sampled_wave.get(item_id, 0),
            "receipt_audit_failed": plan.receipts[item_id].audit_failed
            if item_id in plan.receipts else None,
            "routine_miss": plan.is_miss(item_id, basis="routine_failure")
            if item_id in plan.receipts else False,
            "final_pass_miss": plan.is_miss(item_id, basis="final_pass_audit_fail")
            if item_id in plan.receipts else False,
        })
    return {"plan_id": plan_id, "items": rows, "batches": plan.audit_rows(),
            "estimates": plan.estimates()}


def list_plans(session, actor):
    rows = session.scalars(
        select(SamplingPlanRecord).order_by(SamplingPlanRecord.id.desc()).limit(100)
    ).all()
    return [plan_info(row) for row in rows]


EXPORT_TABLES = {
    "sampling_basis": ("各批抽检依据", "batch"),
    "batch_audit": ("各批抽检依据", "batch"),
    "orders": ("盲样抽单", "order"),
    "sampling_orders": ("盲样抽单", "order"),
}


def export_sampling(session, plan_id: int, key: str, actor) -> dict:
    if key not in EXPORT_TABLES:
        raise LookupError("找不到导出表")
    record = _load_record(session, plan_id)
    plan = _restore(record)
    title, kind = EXPORT_TABLES[key]
    if kind == "batch":
        rows = []
        for row in plan.audit_rows():
            flat = dict(row)
            for list_key in ("first_wave_ids", "supplement_ids"):
                flat[list_key] = ";".join(row.get(list_key, []))
            rows.append(flat)
        columns = [
            "batch_id", "count", "first_wave_size", "first_wave_ids",
            "first_wave_receipts", "first_wave_misses", "never_sampled_count",
            "supplement_allocated", "supplement_ids", "supplement_receipts",
            "estimate_basis", "miss_rate", "estimated_misses", "allocation_reason",
        ]
    else:
        rows = []
        for order in plan.orders.values():
            rows.append({"order_id": order.id, "wave": order.wave, "batch_id": order.batch_id,
                         "item_ids": ";".join(order.item_ids), "claimed_by": order.claimed_by or "",
                         "reason": order.reason})
        columns = ["order_id", "wave", "batch_id", "item_ids", "claimed_by", "reason"]
    return {"key": key, "title": title, "columns": columns, "rows": rows}