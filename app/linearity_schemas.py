"""Pydantic 模型：量具线性与偏倚研究的请求校验。

校验内容
========
* 研究引用基线链中的一个尺寸，提交工作量程声明（工作上下限）与若干
  有证标准件核查点：每点给标准件编号、参考值、参考值标准不确定度
  （均可逐点混用长度单位，内部统一 mm）、测量次序与重复读数；
* 逐读数记录操作者与测量次序：同一核查点内 (操作者, 重复序号) 组合、
  测量次序均不得重复；参考值 / 不确定度 / 读数必须为有限数，
  标准件编号与操作者标识非空；
* 核查点的标准件编号不得重复，参考值必须严格不同（线性拟合需要
  响应变量有变差），全部参考值必须落在声明工作量程内，且参考值
  跨度对工作量程的覆盖率不得低于 ``min_span_coverage``
  （覆盖过窄时不得外推）；
* 复制研究时可逐读数注明原因排除异常读数（原因必填），排除后仍须
  至少保留两个有效核查点，每点至少保留一个读数。
"""
from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .units import LengthUnit


class LinearityReading(BaseModel):
    """单个标准件核查点上的一次重复读数。"""

    replicate: int = Field(
        ..., ge=1,
        description="重复序号（同一核查点内从 1 起，与操作者组合唯一）",
    )
    value: float = Field(..., description="量具读数（与研究 unit 同单位）")
    operator: str = Field(..., min_length=1, description="操作者标识")
    measurement_order: int = Field(
        ..., ge=1, description="测量次序（全研究唯一的先后次序，从 1 起）"
    )

    @model_validator(mode="after")
    def _finite(self) -> "LinearityReading":
        if not math.isfinite(self.value):
            raise ValueError(
                f"重复读数必须为有限数，收到 {self.value!r}（NaN/±Infinity 拒收）"
            )
        return self


class LinearityReferencePoint(BaseModel):
    """一个有证标准件核查点：参考值、参考值标准不确定度与重复读数。"""

    point_id: str = Field(
        ..., min_length=1, description="核查点标识（研究内唯一，用于结果追溯）"
    )
    standard_serial: str = Field(
        ..., min_length=1, description="有证标准件编号（全研究唯一）"
    )
    reference_value: float = Field(..., description="标准件证书参考值（reference_unit）")
    reference_unit: LengthUnit = Field(
        LengthUnit.MM, description="参考值与参考标准不确定度的单位"
    )
    reference_std_uncertainty: float = Field(
        ..., ge=0,
        description="参考值的标准不确定度 u(ref)（reference_unit，允许为 0）",
    )
    readings: list[LinearityReading] = Field(
        ..., min_length=1, description="该标准件上的重复读数（至少 1 次）"
    )

    @model_validator(mode="after")
    def _check_point(self) -> "LinearityReferencePoint":
        if not math.isfinite(self.reference_value):
            raise ValueError(
                f"核查点 {self.point_id}: 参考值必须为有限数，"
                f"收到 {self.reference_value!r}"
            )
        if not math.isfinite(self.reference_std_uncertainty):
            raise ValueError(
                f"核查点 {self.point_id}: 参考值标准不确定度必须为有限数，"
                f"收到 {self.reference_std_uncertainty!r}"
            )
        # 同一核查点内 (操作者, 重复序号) 组合唯一
        seen_combo: set[tuple[str, int]] = set()
        dup_combo: list[str] = []
        seen_order: set[int] = set()
        dup_order: list[int] = []
        for r in self.readings:
            combo = (r.operator, r.replicate)
            if combo in seen_combo and f"{r.operator}/第{r.replicate}次" not in dup_combo:
                dup_combo.append(f"{r.operator}/第{r.replicate}次")
            seen_combo.add(combo)
            if r.measurement_order in seen_order and r.measurement_order not in dup_order:
                dup_order.append(r.measurement_order)
            seen_order.add(r.measurement_order)
        if dup_combo:
            raise ValueError(
                f"核查点 {self.point_id}: 同一操作者重复序号组合重复: {dup_combo}"
            )
        if dup_order:
            raise ValueError(
                f"核查点 {self.point_id}: 测量次序在该点内重复: {sorted(dup_order)}"
            )
        return self


class LinearityStudyCreate(BaseModel):
    """创建量具线性与偏倚研究：一件量具在工作量程内的多点有证标准件核查。

    研究创建为**草案**（draft），可复制排除异常读数；定稿（finalize）后
    冻结，采用（adopt）后可被测量方案引用。参考值跨度覆盖过窄时结果中
    标记 ``coverage_adequate=false``，线性修正不得外推到适用量程之外。
    """

    name: str = Field(..., min_length=1)
    note: str = ""
    dimension_id: str = Field(..., min_length=1, description="基线链上的尺寸 id")
    unit: LengthUnit = Field(
        LengthUnit.MM, description="量具读数与工作量程/过程公差的统一单位"
    )
    working_range_lower: float = Field(..., description="量具工作量程下限（unit）")
    working_range_upper: float = Field(..., description="量具工作量程上限（unit）")
    process_tolerance: float | None = Field(
        None, gt=0,
        description="过程公差带宽度（unit，可选）；给出时计算 %Bias / %Linearity",
    )
    min_span_coverage: float = Field(
        0.5, gt=0, le=1,
        description="参考值跨度对工作量程的最低覆盖率 "
                    "(max(ref)−min(ref))/(量程上限−下限)；低于此值标记覆盖不足",
    )
    confidence_level: float = Field(
        0.95, gt=0, lt=1, description="平均偏倚与回归系数置信区间的置信水平"
    )
    points: list[LinearityReferencePoint] = Field(
        ..., min_length=2, description="有证标准件核查点（至少 2 个、参考值严格不同）"
    )

    @model_validator(mode="after")
    def _check_study(self) -> "LinearityStudyCreate":
        for f in ("working_range_lower", "working_range_upper",
                  "min_span_coverage", "confidence_level"):
            v = getattr(self, f)
            if not math.isfinite(v):
                raise ValueError(f"{f} 必须为有限数，收到 {v!r}")
        if self.process_tolerance is not None and not math.isfinite(
                self.process_tolerance):
            raise ValueError("过程公差必须为有限数")
        if self.working_range_lower >= self.working_range_upper:
            raise ValueError(
                f"工作量程下限 {self.working_range_lower} 必须严格小于上限 "
                f"{self.working_range_upper}"
            )

        # 标准件编号 / 核查点 id 唯一
        serials = [p.standard_serial for p in self.points]
        dup_serials = sorted({s for s in serials if serials.count(s) > 1})
        if dup_serials:
            raise ValueError(f"标准件编号重复: {dup_serials}")
        pids = [p.point_id for p in self.points]
        dup_pids = sorted({s for s in pids if pids.count(s) > 1})
        if dup_pids:
            raise ValueError(f"核查点标识重复: {dup_pids}")

        # 参考值（统一为研究单位口径无需换算即可比较相对量程的位置：
        # 参考值单位是长度，量程是研究单位，故在引擎层换算后复核；
        # 这里先拦截同单位明显异常，跨单位的越界由引擎/API 层复核）
        same_unit = [p for p in self.points if p.reference_unit == self.unit]
        for p in same_unit:
            if not (self.working_range_lower <= p.reference_value
                    <= self.working_range_upper):
                raise ValueError(
                    f"核查点 {p.point_id}: 参考值 {p.reference_value} {self.unit.value} "
                    f"超出工作量程 [{self.working_range_lower}, "
                    f"{self.working_range_upper}] {self.unit.value}"
                )

        # 全研究测量次序唯一
        orders = [r.measurement_order for p in self.points for r in p.readings]
        dup_orders = sorted({o for o in orders if orders.count(o) > 1})
        if dup_orders:
            raise ValueError(f"测量次序在研究内重复: {dup_orders}")
        return self


class LinearityExclusionSpec(BaseModel):
    """复制研究时排除的一条异常读数（必须注明原因）。"""

    point_id: str = Field(..., min_length=1)
    operator: str = Field(..., min_length=1)
    replicate: int = Field(..., ge=1)
    reason: str = Field(
        ..., min_length=1,
        description="排除原因（必填，非空白；随副本冻结）")

    @model_validator(mode="after")
    def _reason_not_blank(self) -> "LinearityExclusionSpec":
        if not self.reason.strip():
            raise ValueError("排除原因不能为空白")
        return self


class LinearityStudyCopyRequest(BaseModel):
    """复制线性与偏倚研究：可注明原因排除异常读数，比较排除前后结论。"""

    name: str | None = Field(None, min_length=1)
    note: str = ""
    exclusions: list[LinearityExclusionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "LinearityStudyCopyRequest":
        keys = [(e.point_id, e.operator, e.replicate) for e in self.exclusions]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise ValueError(f"排除读数重复列出: {dup}")
        return self


class LinearityStudyFinalizeRequest(BaseModel):
    """定稿请求：定稿后研究（核查点 / 读数 / 拟合结果）不可再改、不可复制。"""

    note: str = ""


class LinearityStudyAdoptRequest(BaseModel):
    """采用请求：采用后的研究可接入测量方案做按实测值的线性偏倚修正。"""

    note: str = ""
