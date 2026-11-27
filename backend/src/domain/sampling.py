"""两波分层抽检：盲样、回执驱动的自适应补抽与漏报估计。

业务口径（见任务描述）：

* 抽检总体固定为某次分诊运行的留出批，沿用其分诊判定；
* 抽单/领单视图藏起末检结论（``final_failed``），避免抽选与返检带偏；
* 每个留出批先用固定随机种子独立随机抽首拨；
* 首拨回执全部到齐后，余下名额才允许分配给补抽：
  - 每个尚未抽完的批至少再随机抽 1 件；
  - 余下名额不足以让每个未抽完批都多抽 1 件时，整体停在首拨，不发补抽单；
  - 保底之后的名额优先给首拨漏报更多的批（并列按批次号），且不超过该批
    从未抽过的可选件数；
* 补抽只从该批从未抽过的件里随机选，用来直接估计“首拨没抽的那部分”；
* 补抽回执全部到齐后，估计才从“首拨外推”切换为“补抽外推”，此前即使有部分
  补抽回执也继续显示首拨估计；
* 所有抽选结果、分配理由和回执都可逐批核对。

本模块不依赖数据库或 FastAPI，便于单测与持久化重建。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable


class SamplingError(ValueError):
    """抽检规则被违反（非法参数、状态或幂等冲突）。"""


# 漏报口径：
# * ``routine_failure``：分诊判为常规末检（未标优先返测）而返检发现失败，
#   即既有口径里的 FN（实际失败但进常规队列）；“沿用分诊判定”使其成为默认。
# * ``final_pass_audit_fail``：末检结论为通过、独立返检发现失败，
#   即末检本身放过的件。
MISS_ROUTINE_FAILURE = "routine_failure"
MISS_FINAL_PASS = "final_pass_audit_fail"
MISS_BASES = (MISS_ROUTINE_FAILURE, MISS_FINAL_PASS)


@dataclass(frozen=True)
class SampleItem:
    id: str
    batch_id: str
    final_failed: int | None = None
    flagged: bool = False
    score: float | None = None
    measured_at: str | None = None

    def blind(self) -> dict[str, Any]:
        """抽单/领单可见字段：刻意不含末检结论。"""
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "measured_at": self.measured_at,
            "score": self.score,
            "triage_action": "优先返测" if self.flagged else "常规末检",
            "flagged": self.flagged,
        }


@dataclass
class Order:
    """一波中某一批的抽单。"""

    id: str
    wave: int  # 1=首拨, 2=补抽
    batch_id: str
    item_ids: list[str]
    reason: str = ""
    claim_key: str | None = None
    claimed_by: str | None = None
    claimed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "wave": self.wave,
            "batch_id": self.batch_id,
            "item_ids": list(self.item_ids),
            "reason": self.reason,
            "claim_key": self.claim_key,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
        }


@dataclass
class Receipt:
    item_id: str
    audit_failed: int
    key: str | None = None
    note: str = ""
    explicit_miss: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "audit_failed": self.audit_failed,
                "key": self.key, "note": self.note, "explicit_miss": self.explicit_miss}


def _as_binary(value: Any, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise SamplingError(f"{label}须为 0/1 或布尔值")


def _as_count(value: Any, label: str, minimum: int = 0) -> int:
    """接受整数、整数值浮点或整数字符串；拒绝布尔、非整数与越界值。"""
    if isinstance(value, bool):
        raise SamplingError(f"{label}须为整数")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise SamplingError(f"{label}须为整数")
        number = int(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            number = int(text)
        except ValueError:
            try:
                f = float(text)
            except ValueError as exc:
                raise SamplingError(f"{label}须为整数") from exc
            if not f.is_integer():
                raise SamplingError(f"{label}须为整数")
            number = int(f)
    else:
        raise SamplingError(f"{label}须为整数")
    if number < minimum:
        raise SamplingError(f"{label}须不小于 {minimum}")
    return number


def allocate_supplement(
    capacities: dict[str, int],
    miss_counts: dict[str, int],
    extra_quota: int,
) -> dict[str, int] | None:
    """按任务规则分配补抽名额。

    ``capacities[batch]`` 是该批从未抽过、还可补抽的件数；``miss_counts`` 是
    首拨回执确认的漏报数。返回每批补抽件数；若名额不足以给每个“仍可抽”的批
    至少 1 件，则返回 ``None``，表示整体停在首拨、不发补抽单。

    容量为 0 的批视为“已抽完”，不参与保底。保底每件之后，剩余名额按
    ``(漏报数降序, 批次号升序)`` 逐个给优先级最高且仍有容量的批。
    """
    if not isinstance(extra_quota, int) or isinstance(extra_quota, bool):
        extra_quota = _as_count(extra_quota, "补抽名额", 0)
    if extra_quota < 0:
        raise SamplingError("补抽名额须为非负整数")
    eligible = sorted(
        (batch for batch, cap in capacities.items() if cap > 0),
        key=lambda b: (-int(miss_counts.get(b, 0)), b),
    )
    if not eligible:
        return {}
    if extra_quota < len(eligible):
        return None
    allocation = {batch: 0 for batch in eligible}
    remaining_cap = {batch: int(capacities[batch]) for batch in eligible}
    quota = extra_quota
    # 每个未抽完批保底 1 件（eligible 已按漏报优先级排序）。
    for batch in eligible:
        allocation[batch] = 1
        remaining_cap[batch] -= 1
        quota -= 1
    # 余下名额优先给漏报多的批；逐件分配，容量用尽的批退出。
    while quota > 0:
        candidates = [b for b in eligible if remaining_cap[b] > 0]
        if not candidates:
            break
        batch = candidates[0]  # eligible 顺序即固定优先级
        allocation[batch] += 1
        remaining_cap[batch] -= 1
        quota -= 1
    return allocation


def phase_estimate(
    batch_size: int,
    first_sampled: int,
    first_misses: int,
    supplement_sampled: int = 0,
    supplement_misses: int = 0,
) -> dict[str, float | int]:
    """计算一批的漏报估计。

    首拨阶段用首拨漏报率外推整批：``N * m1/n1``。
    补抽齐套后，用补抽漏报率直接估计“首拨没抽的那部分”：
    ``m1 + (N - n1) * m2/n2``（等价于 ``m1+m2+(N-n1-n2)*m2/n2``）。
    """
    if first_sampled <= 0:
        raise SamplingError("首拨至少要有 1 件才能估计")
    unsampled_after_first = batch_size - first_sampled
    first_rate = first_misses / first_sampled
    first_total = batch_size * first_rate
    first_remaining = unsampled_after_first * first_rate
    result: dict[str, float | int] = {
        "batch_size": batch_size,
        "first_sampled": first_sampled,
        "first_misses": first_misses,
        "first_rate": first_rate,
        "first_estimate_total": first_total,
        "first_estimate_remaining": first_remaining,
        "active_estimate_total": first_total,
        "active_rate": first_rate,
        "basis": "first_wave",
    }
    if supplement_sampled > 0:
        supp_rate = supplement_misses / supplement_sampled
        final_total = first_misses + unsampled_after_first * supp_rate
        result.update({
            "supplement_sampled": supplement_sampled,
            "supplement_misses": supplement_misses,
            "supplement_rate": supp_rate,
            "supplement_estimate_remaining": unsampled_after_first * supp_rate,
            "active_estimate_total": final_total,
            "active_rate": supp_rate,
            "basis": "supplement",
        })
    return result


class SamplingPlan:
    """两波抽检聚合根：抽选、领单、回执、补抽与估计。

    该类不是线程安全的；Web 服务层应在锁/事务内重建并调用。幂等键与状态
    持久化由服务层或 :meth:`snapshot`/:meth:`restore` 承担。
    """

    def __init__(
        self,
        items: Iterable[SampleItem | dict[str, Any]],
        *,
        first_wave_size: int = 3,
        total_quota: int | None = None,
        extra_quota: int | None = None,
        seed: int = 261125,
        miss_basis: str = MISS_ROUTINE_FAILURE,
        plan_id: Any = None,
    ):
        self.plan_id = plan_id
        self.seed = int(seed)
        if miss_basis not in MISS_BASES:
            raise SamplingError(f"漏报口径须为 {MISS_BASES} 之一")
        self.miss_basis = miss_basis
        self._items: dict[str, SampleItem] = {}
        self._batch_items: dict[str, list[str]] = {}
        for raw in items:
            item = raw if isinstance(raw, SampleItem) else SampleItem(
                id=str(raw["id"]),
                batch_id=str(raw["batch_id"]),
                final_failed=(None if raw.get("final_failed", raw.get("failed")) is None
                              else _as_binary(raw.get("final_failed", raw.get("failed")), "末检结论")),
                flagged=bool(raw.get("flagged", raw.get("priority", False))
                             or raw.get("action") == "优先返测"),
                score=_opt_float(raw.get("score")),
                measured_at=raw.get("measured_at"),
            )
            if item.id in self._items:
                raise SamplingError(f"总成编号重复：{item.id}")
            self._items[item.id] = item
            self._batch_items.setdefault(item.batch_id, []).append(item.id)
        if len(self._batch_items) < 1:
            raise SamplingError("抽检总体至少包含一个留出批")
        first_wave_size = _as_count(first_wave_size, "首拨每批件数", 1)
        # 批顺序与件顺序都用稳定的规范顺序（演示数据里等同时间顺序）。
        self.batch_ids = sorted(self._batch_items)
        for batch in self.batch_ids:
            self._batch_items[batch].sort(key=lambda i: (
                self._items[i].measured_at or "", i))
        sizes = {b: len(ids) for b, ids in self._batch_items.items()}
        if not first_wave_size <= min(sizes.values()):
            raise SamplingError(
                f"首拨每批件数须为 1 到最小批容量 {min(sizes.values())} 的整数")
        self.first_wave_size = first_wave_size
        first_total = first_wave_size * len(self.batch_ids)
        if extra_quota is None and total_quota is None:
            # 未给名额时，默认给每个未抽完批保底 1 个补抽名额。
            extra_quota = len(self.batch_ids)
        elif extra_quota is None:
            extra_quota = _as_count(total_quota, "总名额", 0) - first_total
        else:
            extra_quota = _as_count(extra_quota, "余下名额", 0)
        if extra_quota < 0:
            raise SamplingError("余下名额须为非负整数（total_quota 不能小于首拨件数）")
        self.extra_quota = extra_quota
        self.total_quota = first_total + extra_quota

        self.rng = random.Random(self.seed)
        self.sampled_wave: dict[str, int] = {}
        self.receipts: dict[str, Receipt] = {}
        self._receipt_keys: dict[str, str] = {}
        self.orders: dict[str, Order] = {}
        self._order_keys: dict[str, str] = {}
        self._claim_keys: dict[str, str] = {}
        self.supplement_status = "locked"  # locked -> open / skipped
        self.allocation: dict[str, int] | None = None
        self.allocation_reason: dict[str, str] = {}
        self._order_seq = 0
        self._draw_first_wave()

    # ---------- 抽选 ----------
    def _new_order_id(self) -> str:
        self._order_seq += 1
        return f"order-{self._order_seq:04d}"

    def _draw_first_wave(self) -> None:
        for batch in self.batch_ids:
            pool = list(self._batch_items[batch])
            picked = sorted(self.rng.sample(pool, self.first_wave_size))
            order = Order(
                id=self._new_order_id(), wave=1, batch_id=batch, item_ids=picked,
                reason=f"各批随机首拨 {self.first_wave_size} 件（种子 {self.seed}）",
            )
            self.orders[order.id] = order
            for item_id in picked:
                self.sampled_wave[item_id] = 1

    def never_sampled(self, batch: str) -> list[str]:
        return [i for i in self._batch_items[batch] if i not in self.sampled_wave]

    # ---------- 领单 / 幂等发单 ----------
    def issue_order_idempotent(self, key: str, wave: int, batch: str,
                               item_ids: list[str], reason: str = "") -> tuple[Order, bool]:
        """以幂等键发单；同键重试返回原单且 ``deduplicated=True``。"""
        if not key:
            raise SamplingError("发单/领单重试必须带幂等键")
        if key in self._order_keys:
            order = self.orders[self._order_keys[key]]
            if (order.wave, order.batch_id, order.item_ids) != (wave, batch, list(item_ids)):
                raise SamplingError(f"幂等键 {key} 已用于不同抽单内容")
            return order, True
        order = Order(id=self._new_order_id(), wave=wave, batch_id=batch,
                     item_ids=list(item_ids), reason=reason)
        self.orders[order.id] = order
        self._order_keys[key] = order.id
        return order, False

    def claim(self, key: str, order_id: str | None = None,
              worker: str | None = None, at: str | None = None) -> tuple[Order, bool]:
        """领单。同一 ``key`` 重试始终返回第一次领到的单，不重复发单/换单。"""
        if not key:
            raise SamplingError("领单必须带幂等键")
        if key in self._claim_keys:
            return self.orders[self._claim_keys[key]], True
        if order_id is None:
            # 领“下一张尚未被任何幂等键领过”的单：优先首拨，按批顺序。
            claimed_ids = set(self._claim_keys.values())
            available = [oid for oid, od in self.orders.items()
                         if oid not in claimed_ids and od.claim_key is None]
            if not available:
                raise SamplingError("当前没有可领的抽单")
            order_id = available[0]
        if order_id not in self.orders:
            raise SamplingError("抽单不存在")
        order = self.orders[order_id]
        if order.claim_key is not None and order.claim_key != key:
            raise SamplingError("该抽单已被其他领单请求领取")
        order.claim_key = key
        order.claimed_by = worker
        order.claimed_at = at
        self._claim_keys[key] = order_id
        return order, False

    # ---------- 回执 ----------
    def receive_receipt(self, item_id: str, audit_failed: Any = None,
                        key: str | None = None, note: str = "",
                        missed: Any = None) -> tuple[Receipt, bool]:
        if item_id not in self._items:
            raise SamplingError(f"不在抽检总体内的总成：{item_id}")
        if item_id not in self.sampled_wave:
            raise SamplingError(f"总成 {item_id} 尚未被抽中，不能提交回执")
        explicit_miss = None if missed is None else _as_binary(missed, "漏报标记 missed")
        if audit_failed is None:
            if explicit_miss is None:
                raise SamplingError("回执须包含 audit_failed 返检结论或 missed 漏报标记")
            verdict = explicit_miss
        else:
            verdict = _as_binary(audit_failed, "返检结论 audit_failed")
        if key is not None and key in self._receipt_keys:
            prior_id = self._receipt_keys[key]
            prior = self.receipts[prior_id]
            if prior_id != item_id or prior.audit_failed != verdict \
                    or prior.explicit_miss != explicit_miss:
                raise SamplingError(f"回执幂等键 {key} 已用于不同内容")
            return prior, True
        if item_id in self.receipts:
            prior = self.receipts[item_id]
            if prior.audit_failed != verdict or prior.explicit_miss != explicit_miss:
                raise SamplingError(f"总成 {item_id} 已收到结论不同的回执，既判不可改")
            return prior, True
        receipt = Receipt(item_id, verdict, key, note, explicit_miss)
        self.receipts[item_id] = receipt
        if key is not None:
            self._receipt_keys[key] = item_id
        return receipt, False

    # ---------- 漏报判定 ----------
    def receive_receipts(self, receipts: Iterable[dict[str, Any]]) -> list[tuple[Receipt, bool]]:
        """批量提交回执；每项可含 item_id/id、audit_failed/failed/verdict/missed、key。"""
        outcomes = []
        for raw in receipts:
            item_id = raw.get("item_id", raw.get("id"))
            verdict = next((raw[k] for k in ("audit_failed", "failed", "verdict", "result")
                            if raw.get(k) is not None), None)
            outcomes.append(self.receive_receipt(
                item_id, verdict,
                key=raw.get("key", raw.get("idempotency_key")),
                note=raw.get("note", ""),
                missed=raw.get("missed"),
            ))
        return outcomes

    def is_miss(self, item_id: str, receipt: Receipt | None = None,
                basis: str | None = None) -> bool:
        basis = basis or self.miss_basis
        item = self._items[item_id]
        receipt = receipt or self.receipts.get(item_id)
        if receipt is None:
            return False
        # 回执若直接声明了 missed，以声明为准（分配名额依赖该计数）。
        if receipt.explicit_miss is not None:
            return receipt.explicit_miss == 1
        audit_fail = receipt.audit_failed == 1
        if basis == MISS_ROUTINE_FAILURE:
            return audit_fail and not item.flagged
        if basis == MISS_FINAL_PASS:
            return audit_fail and item.final_failed == 0
        raise SamplingError(f"未知漏报口径 {basis}")

    def batch_miss_counts(self, wave: int = 1, basis: str | None = None) -> dict[str, int]:
        counts = {b: 0 for b in self.batch_ids}
        for item_id, w in self.sampled_wave.items():
            if w != wave or item_id not in self.receipts:
                continue
            if self.is_miss(item_id, basis=basis):
                counts[self._items[item_id].batch_id] += 1
        return counts

    # ---------- 齐套与补抽 ----------
    def first_wave_item_ids(self) -> list[str]:
        return [i for i, w in self.sampled_wave.items() if w == 1]

    def first_wave_complete(self) -> bool:
        return all(i in self.receipts for i in self.first_wave_item_ids())

    def supplement_item_ids(self) -> list[str]:
        return [i for i, w in self.sampled_wave.items() if w == 2]

    def supplement_complete(self) -> bool:
        ids = self.supplement_item_ids()
        return bool(ids) and all(i in self.receipts for i in ids)

    def open_supplement(self, key: str | None = None, at: str | None = None) -> dict[str, Any]:
        """首拨齐套后按规则发补抽单；幂等。返回状态摘要。

        并发调用时服务层应保证本方法在锁/事务内执行；领域规则本身是“检查-创建”
        一体且幂等的。首拨未齐时不创建任何补抽单（抛 :class:`SamplingError`
        或返回 waiting，取决于 ``key`` 用法）。
        """
        if self.supplement_status == "open":
            return {"status": "open", "deduplicated": True,
                    "allocation": self.allocation, "orders": self._wave_orders(2)}
        if self.supplement_status == "skipped":
            return {"status": "skipped", "deduplicated": True,
                    "reason": "余下名额不足以让每个未抽完批至少补抽 1 件，停在首拨"}
        if self.supplement_status == "complete":
            return {"status": "complete", "deduplicated": True, "allocation": self.allocation or {}}
        if not self.first_wave_complete():
            return {"status": "waiting_first_wave", "deduplicated": False,
                    "missing": [i for i in self.first_wave_item_ids() if i not in self.receipts]}
        capacities = {b: len(self.never_sampled(b)) for b in self.batch_ids}
        miss_counts = self.batch_miss_counts(wave=1)
        if not any(capacities.values()):
            # 所有批都已在首拨整批抽完：估计已是普查精确值，无需补抽。
            self.supplement_status = "complete"
            self.allocation = {}
            return {"status": "complete", "deduplicated": False,
                    "reason": "所有批均已在首拨抽完，漏报数已精确", "allocation": {}}
        allocation = allocate_supplement(capacities, miss_counts, self.extra_quota)
        if allocation is None:
            self.supplement_status = "skipped"
            self.allocation = None
            return {"status": "skipped", "deduplicated": False,
                    "reason": "余下名额不足以让每个未抽完批至少补抽 1 件，停在首拨",
                    "miss_counts": miss_counts, "capacities": capacities}
        self.allocation = allocation
        # 抽选顺序与首拨一致，按规范批次顺序消耗随机流；分配优先级只决定每批件数。
        for batch in self.batch_ids:
            count = allocation.get(batch, 0)
            if count <= 0:
                if batch in allocation:
                    self.allocation_reason[batch] = "该批首拨后已抽完或无名额"
                continue
            pool = self.never_sampled(batch)
            picked = sorted(self.rng.sample(pool, count))
            priority = miss_counts[batch]
            reason = (f"首拨漏报 {priority} 件；保底 1 件并按漏报优先级分配，"
                      f"本批补抽 {count} 件，仅从未抽过的 {len(pool)} 件中随机选取")
            order = Order(id=self._new_order_id(), wave=2, batch_id=batch,
                          item_ids=picked, reason=reason)
            self.orders[order.id] = order
            self.allocation_reason[batch] = reason
            for item_id in picked:
                self.sampled_wave[item_id] = 2
        self.supplement_status = "open"
        return {"status": "open", "deduplicated": False, "allocation": allocation,
                "orders": self._wave_orders(2)}

    def _wave_orders(self, wave: int) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.orders.values() if o.wave == wave]

    # ---------- 估计与核对表 ----------
    def estimates(self) -> dict[str, Any]:
        """逐批与合计漏报估计；补抽齐套后才切换口径。"""
        use_supp = self.supplement_status == "open" and self.supplement_complete()
        per_batch: list[dict[str, Any]] = []
        total_active = 0.0
        total_first = 0.0
        sum_remaining_first = 0.0
        sum_remaining_active = 0.0
        n_total = n1_total = m1_total = n2_total = m2_total = 0
        for batch in self.batch_ids:
            n = len(self._batch_items[batch])
            n1 = self.first_wave_size
            m1 = self.batch_miss_counts(wave=1).get(batch, 0)
            n2 = sum(1 for i, w in self.sampled_wave.items()
                     if w == 2 and self._items[i].batch_id == batch)
            m2 = self.batch_miss_counts(wave=2).get(batch, 0) if use_supp else 0
            est = phase_estimate(n, n1, m1, n2 if use_supp else 0, m2)
            est["batch_id"] = batch
            est["supplement_issued"] = n2
            est["supplement_receipts"] = sum(
                1 for i, w in self.sampled_wave.items()
                if w == 2 and self._items[i].batch_id == batch and i in self.receipts)
            est["switched_to_supplement"] = use_supp and n2 > 0
            per_batch.append(est)
            total_active += est["active_estimate_total"]
            total_first += est["first_estimate_total"]
            sum_remaining_first += est["first_estimate_remaining"]
            sum_remaining_active += (est.get("supplement_estimate_remaining",
                                             est["first_estimate_remaining"])
                                     if use_supp else est["first_estimate_remaining"])
            n_total += n
            n1_total += n1
            m1_total += m1
            n2_total += n2 if use_supp else 0
            m2_total += m2
        pooled_first_total = n_total * (m1_total / n1_total) if n1_total else None
        pooled = {
            "sampled_first": n1_total, "misses_first": m1_total,
            "sampled_supplement": n2_total, "misses_supplement": m2_total,
            "pooled_first_rate": m1_total / n1_total if n1_total else None,
            "pooled_first_total_estimate": pooled_first_total,
        }
        if use_supp and n2_total:
            pooled["pooled_supplement_rate"] = m2_total / n2_total
            pooled["pooled_supplement_total_estimate"] = (
                m1_total + (n_total - n1_total) * (m2_total / n2_total))
        return {
            "basis": "supplement" if use_supp else "first_wave",
            "switched": use_supp,
            "total_items": n_total,
            "total_estimate": round(total_active, 4),
            "total_estimate_exact": total_active,
            "first_wave_total_estimate": round(total_first, 4),
            "first_wave_total_estimate_exact": total_first,
            "first_wave_remaining_estimate": round(sum_remaining_first, 4),
            "active_remaining_estimate": round(sum_remaining_active, 4),
            "pooled": pooled,
            "batches": per_batch,
        }

    def audit_rows(self) -> list[dict[str, Any]]:
        """各批抽检依据：抽了哪些、为什么补抽、回执与估计。"""
        estimates = {row["batch_id"]: row for row in self.estimates()["batches"]}
        rows = []
        miss1 = self.batch_miss_counts(wave=1)
        miss1_routine = self.batch_miss_counts(wave=1, basis=MISS_ROUTINE_FAILURE)
        for batch in self.batch_ids:
            ids = self._batch_items[batch]
            first_ids = [i for i in ids if self.sampled_wave.get(i) == 1]
            supp_ids = [i for i in ids if self.sampled_wave.get(i) == 2]
            supp_done = bool(supp_ids) and all(i in self.receipts for i in supp_ids)
            row = {
                "batch_id": batch,
                "count": len(ids),
                "first_wave_ids": first_ids,
                "first_wave_size": len(first_ids),
                "first_wave_receipts": sum(i in self.receipts for i in first_ids),
                "first_wave_complete": all(i in self.receipts for i in first_ids),
                "first_wave_misses": miss1[batch],
                "first_wave_routine_misses": miss1_routine[batch],
                "never_sampled_count": len(self.never_sampled(batch)),
                "supplement_allocated": (self.allocation or {}).get(batch, 0)
                if self.allocation else 0,
                "supplement_ids": supp_ids,
                "supplement_receipts": sum(i in self.receipts for i in supp_ids),
                "supplement_complete": supp_done,
                "allocation_reason": self.allocation_reason.get(batch, ""),
                "supplement_status": self.supplement_status,
            }
            row.update({
                "estimate_basis": estimates[batch]["basis"],
                "miss_rate": round(float(estimates[batch]["active_rate"]), 4),
                "estimated_misses": round(float(estimates[batch]["active_estimate_total"]), 4),
            })
            rows.append(row)
        return rows

    def blind_order_view(self, order_id: str) -> dict[str, Any]:
        order = self.orders[order_id]
        return {**order.to_dict(),
                "items": [self._items[i].blind() for i in order.item_ids]}

    # ---------- 持久化快照/恢复 ----------
    def snapshot(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "seed": self.seed,
            "miss_basis": self.miss_basis,
            "first_wave_size": self.first_wave_size,
            "extra_quota": self.extra_quota,
            "total_quota": self.total_quota,
            "batch_ids": self.batch_ids,
            "items": [vars(self._items[i]) for b in self.batch_ids for i in self._batch_items[b]],
            "sampled_wave": dict(self.sampled_wave),
            "receipts": [r.to_dict() for r in self.receipts.values()],
            "receipt_keys": dict(self._receipt_keys),
            "orders": [o.to_dict() for o in self.orders.values()],
            "order_keys": dict(self._order_keys),
            "claim_keys": dict(self._claim_keys),
            "supplement_status": self.supplement_status,
            "allocation": self.allocation,
            "allocation_reason": dict(self.allocation_reason),
            "order_seq": self._order_seq,
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def restore(cls, snap: dict[str, Any]) -> "SamplingPlan":
        plan = cls(
            snap["items"], first_wave_size=snap["first_wave_size"],
            extra_quota=snap["extra_quota"], seed=snap["seed"],
            miss_basis=snap["miss_basis"], plan_id=snap.get("plan_id"),
        )
        # 构造时已重抽首拨；用快照覆盖全部可变状态以保证一致。
        plan.sampled_wave = {k: int(v) for k, v in snap["sampled_wave"].items()}
        plan.receipts = {r["item_id"]: Receipt(**r) for r in snap["receipts"]}
        plan._receipt_keys = dict(snap["receipt_keys"])
        plan.orders = {o["id"]: Order(**o) for o in snap["orders"]}
        plan._order_keys = dict(snap["order_keys"])
        plan._claim_keys = dict(snap["claim_keys"])
        plan.supplement_status = snap["supplement_status"]
        plan.allocation = snap["allocation"]
        plan.allocation_reason = dict(snap["allocation_reason"])
        plan._order_seq = snap["order_seq"]
        try:
            plan.rng.setstate(tuple(tuple(x) if isinstance(x, tuple) else x
                                    for x in snap["rng_state"]))
        except Exception:
            pass
        return plan


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def create_sampling_plan(
    rows: list[dict[str, Any]],
    settings: dict[str, float] | None = None,
    **params: Any,
) -> "SamplingPlan":
    """从完整台账一次性构造抽检计划：内部复算分诊运行并取留出判定。

    供领域检查脚本直接调用；Web 层对已存在的 Run 使用
    :func:`frame_from_run_result`，避免重复拟合。
    """
    from .project import analyze  # 延迟导入：保持本模块顶部仅依赖标准库

    result = analyze(rows, settings or {"miss_cost": 12, "retest_cost": 3})
    items = frame_from_run_result(result, rows)
    return SamplingPlan(items, **params)


# 常见近义命名，方便不同调用/检查口径。
SamplingState = SamplingPlan
SamplingLedger = SamplingPlan
SpotCheckPlan = SamplingPlan
AuditSampler = SamplingPlan
create_plan = SamplingPlan
plan_sampling = create_sampling_plan
allocate_extra = allocate_supplement
estimate_phase = phase_estimate


__all__ = [
    "SamplingError", "SampleItem", "Order", "Receipt", "SamplingPlan",
    "MISS_ROUTINE_FAILURE", "MISS_FINAL_PASS", "MISS_BASES",
    "allocate_supplement", "phase_estimate", "frame_from_run_result",
    "create_sampling_plan",
]


def frame_from_run_result(result: dict[str, Any], dataset_rows: list[dict[str, Any]] | None = None
                          ) -> list[SampleItem]:
    """从一次分诊运行结果的 ``retest_queue`` 表构造抽检总体。

    运行结果表已按件给出分诊动作、得分；末检结论（actual/outcome/failed）保留
    在服务端用于事后核对，不会进入 :meth:`SampleItem.blind` 视图。
    若提供 ``dataset_rows``，用其中的 ``failed`` 作为权威末检结论。
    """
    queue = next((t for t in result.get("tables", []) if t.get("key") == "retest_queue"), None)
    if queue is None:
        raise SamplingError("运行结果缺少留出批次返测顺序单，无法确定留出总体")
    failed_by_id = {r["id"]: _as_binary(r["failed"], "末检结论")
                    for r in (dataset_rows or []) if "id" in r and "failed" in r
                    and r["failed"] is not None}
    items: list[SampleItem] = []
    for row in queue["rows"]:
        rid = row["id"]
        if rid in failed_by_id:
            final_failed: int | None = failed_by_id[rid]
        elif row.get("actual") in ("失败", "通过"):
            final_failed = 1 if row["actual"] == "失败" else 0
        else:
            final_failed = None
        items.append(SampleItem(
            id=rid, batch_id=row["batch_id"], final_failed=final_failed,
            flagged=row.get("action") == "优先返测",
            score=_opt_float(row.get("score")), measured_at=row.get("measured_at"),
        ))
    return items
