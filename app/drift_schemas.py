"""Pydantic 模型：检验批次漂移研究（CUSUM / EWMA）的请求校验。

校验内容：
* 研究至少包含 2 个冻结来料检验批次，批次 id 不重复，采样时刻严格递增
  （不得同时刻或逆序），且不混用带时区 / 不带时区时间串；
* 至少选择一个监控目标（关注尺寸或封闭环），尺寸目标不得重复或链外
  （链外在 API 层复核）；
* CUSUM 参数 k、h 与 EWMA 参数 λ、L 必须为正的有限数（0<λ≤1），
  最小样本量 ≥1；单侧方向支持仅上侧 / 仅下侧 / 双侧；
* 复制研究排除批次时必须逐批注明原因，排除对象必须是父研究来源批次。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class SidednessType(str, Enum):
    """控制图单侧 / 双侧方向。"""

    TWO_SIDED = "two_sided"          # 双侧：同时监控向上与向下漂移
    ONE_SIDED_UP = "one_sided_up"    # 单侧：仅监控均值增大
    ONE_SIDED_DOWN = "one_sided_down"  # 单侧：仅监控均值减小


class DriftMonitorDefaults(BaseModel):
    """监控参数缺省值（逐目标可用 DriftTargetOverrides 覆盖）。

    标准化制表 CUSUM：C+_t=max(0, C+_{t-1}+z_t−k)，
    C−_t=max(0, C−_{t-1}−z_t−k)，越过 h 报警；
    EWMA：q_t=λ·z_t+(1−λ)·q_{t-1}，q_0=0，
    时变控制限 ±L·sqrt(λ/(2−λ)·(1−(1−λ)^{2t}))。
    """

    sidedness: SidednessType = SidednessType.TWO_SIDED
    cusum_k: float = Field(
        0.5, gt=0,
        description="CUSUM 参考偏移 k（标准化单位，通常取目标漂移量的一半，如 0.5σ）",
    )
    cusum_h: float = Field(
        5.0, gt=0,
        description="CUSUM 报警阈值 h（决策区间，标准化单位，如 h=5 对应 h·σ0）",
    )
    ewma_lambda: float = Field(
        0.2, gt=0, le=1,
        description="EWMA 平滑系数 λ（0<λ≤1，越小越平滑、对小漂移越敏感）",
    )
    ewma_L: float = Field(
        3.0, gt=0,
        description="EWMA 控制限倍数 L（时变标准差倍数，常用 2.7~3）",
    )
    min_sample_size: int = Field(
        5, ge=1,
        description="每批次最小样本量；批次对该目标的有效样本不足则不参与作图",
    )


class DriftTargetOverrides(BaseModel):
    """逐目标监控参数覆盖（任一字段缺省即沿用研究级缺省值）。"""

    sidedness: SidednessType | None = None
    cusum_k: float | None = Field(None, gt=0)
    cusum_h: float | None = Field(None, gt=0)
    ewma_lambda: float | None = Field(None, gt=0, le=1)
    ewma_L: float | None = Field(None, gt=0)
    min_sample_size: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _finite(self) -> "DriftTargetOverrides":
        for f in ("cusum_k", "cusum_h", "ewma_lambda", "ewma_L"):
            v = getattr(self, f)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"监控参数 {f} 必须为有限数，收到 {v!r}")
        return self


class DriftBatchSpec(BaseModel):
    """研究中的一个来源批次及其采样时刻（按提交顺序即采样顺序）。"""

    batch_id: int = Field(..., ge=1, description="冻结来料检验批次 id")
    sampled_at: datetime = Field(
        ...,
        description="该批次采样时刻（ISO 8601；全研究统一带时区或统一不带时区）",
    )
    note: str = ""


class DriftDimensionTargetSpec(BaseModel):
    """一个关注尺寸目标及其可选参数覆盖。"""

    dimension_id: str = Field(..., min_length=1)
    overrides: DriftTargetOverrides | None = None


class DriftStudyCreate(BaseModel):
    """创建检验批次漂移研究：同一基线链的多个冻结批次按采样时刻成序。

    batches 必须按采样时间严格递增排列（采样顺序校验）；dimensions 列出
    关注尺寸，closure 字段出现（即使为 null）表示同时监控封闭环。
    研究创建后来源批次、算法参数与逐批结果立即固化，定稿（finalize）后
    研究本身也不可再派生排除版本。
    """

    name: str = Field(..., min_length=1)
    note: str = ""
    batches: list[DriftBatchSpec] = Field(..., min_length=2)
    dimensions: list[DriftDimensionTargetSpec] = Field(default_factory=list)
    monitor_closure: bool = Field(
        False,
        description="是否监控封闭环（true 时 closure_overrides 可给参数覆盖）",
    )
    closure_overrides: DriftTargetOverrides | None = Field(
        None, description="封闭环参数覆盖；缺省沿用 defaults"
    )
    defaults: DriftMonitorDefaults = Field(default_factory=DriftMonitorDefaults)

    @model_validator(mode="after")
    def _check_study(self) -> "DriftStudyCreate":
        # 批次 id 不重复
        ids = [b.batch_id for b in self.batches]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"来源批次重复列出: {dup}")

        # 采样时刻：时区口径一致 + 严格递增（唯一且顺序正确）
        aware = [b.sampled_at.tzinfo is not None for b in self.batches]
        if any(aware) and not all(aware):
            raise ValueError(
                "采样时刻不能混用带时区与不带时区时间：请统一使用 UTC 偏移"
                "（如 2026-09-01T08:00:00+00:00）或统一使用本地无时区时间"
            )

        def norm(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc) if dt.tzinfo is not None \
                else dt.replace(tzinfo=timezone.utc)

        prev = None
        for b in self.batches:
            t = norm(b.sampled_at)
            if prev is not None and t <= prev:
                raise ValueError(
                    f"采样时刻必须严格递增：批次 {b.batch_id} 的 {b.sampled_at.isoformat()}"
                    f" 不晚于前一批次时刻 {prev.isoformat()}（逆序或同时刻拒收）"
                )
            prev = t

        # 至少一个监控目标；尺寸目标不重复
        dim_ids = [d.dimension_id for d in self.dimensions]
        dup_dims = sorted({i for i in dim_ids if dim_ids.count(i) > 1})
        if dup_dims:
            raise ValueError(f"关注尺寸重复指定: {dup_dims}")
        if not dim_ids and not self.monitor_closure:
            raise ValueError(
                "漂移研究至少需要一个监控目标：请在 dimensions 中列出关注尺寸，"
                "或置 monitor_closure=true 以监控封闭环"
            )
        return self


class DriftExclusionSpec(BaseModel):
    """复制研究时排除的异常批次（必须注明原因）。"""

    batch_id: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1, description="排除原因（必填，随新版本冻结）")


class DriftStudyCopyRequest(BaseModel):
    """复制漂移研究：可注明原因排除异常批次，其余参数与父研究完全一致。"""

    name: str | None = Field(None, min_length=1)
    note: str = ""
    exclusions: list[DriftExclusionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "DriftStudyCopyRequest":
        ids = [e.batch_id for e in self.exclusions]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"排除批次重复列出: {dup}")
        return self


class DriftStudyFinalizeRequest(BaseModel):
    """定稿请求：定稿后研究（来源批次 / 算法参数 / 计算结果）不可再改。"""

    note: str = ""
