from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Permissive(BaseModel):
    model_config = ConfigDict(extra='allow', str_strip_whitespace=True)

    def data(self) -> dict[str, Any]:
        dumped = self.model_dump(exclude_none=True)
        dumped.update(self.__pydantic_extra__ or {})
        return dumped


class SamplingPlanCreate(_Permissive):
    label: str | None = Field(default=None, min_length=2, max_length=140)
    run_id: int | None = Field(default=None, gt=0)
    dataset_id: int | None = Field(default=None, gt=0)
    settings: dict[str, float] = Field(default_factory=dict)

    seed: int | None = Field(default=None)
    first_wave_size: int | None = Field(default=None, ge=1, le=12)
    total_quota: int | None = Field(default=None, ge=0)
    extra_quota: int | None = Field(default=None, ge=0)
    miss_basis: str | None = None

    # 常见别名（其余写法经 extra='allow' 保留，由服务层解析）。
    initial_size: int | None = Field(default=None, ge=1, le=12)
    first_wave: int | None = Field(default=None, ge=1, le=12)
    per_batch: int | None = Field(default=None, ge=1, le=12)
    quota: int | None = Field(default=None, ge=0)
    budget: int | None = Field(default=None, ge=0)
    remaining_quota: int | None = Field(default=None, ge=0)


class ReceiptCreate(_Permissive):
    # 全部可选：调用方可能用 item_id/id、audit_failed/failed/verdict/missed，
    # 真正的必填与取值校验放在领域层，错误统一为 422。
    item_id: str | None = Field(default=None, max_length=80)
    id: str | None = Field(default=None, max_length=80)
    audit_failed: int | bool | str | None = None
    failed: int | bool | str | None = None
    verdict: int | bool | str | None = None
    result: int | bool | str | None = None
    missed: bool | int | None = None
    idempotency_key: str | None = Field(default=None, max_length=120)
    key: str | None = Field(default=None, max_length=120)
    note: str = Field(default="", max_length=500)


class ClaimCreate(_Permissive):
    idempotency_key: str | None = Field(default=None, max_length=120)
    key: str | None = Field(default=None, max_length=120)
    request_id: str | None = Field(default=None, max_length=120)
    order_id: str | None = None
    wave: int | None = Field(default=None, ge=1, le=2)
    supplement: bool = False
    worker: str | None = None


class SupplementOpen(_Permissive):
    idempotency_key: str | None = Field(default=None, max_length=120)
    key: str | None = Field(default=None, max_length=120)
