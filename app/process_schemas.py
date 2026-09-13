"""工序尺寸方案的 Pydantic 请求模型。

校验在第二阶段（结构校验在 app.process_plan.build_plan 中完成）：
* 名义值为正、下偏差 ≤ 上偏差、正态必须给 σ、成本非负；
* 毛坯面 / 工序面 / 定位基准 / 被加工面非空，工序 id 不重复；
* 能力档位系数与成本非负、标准步进为正、锁定值为正。
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .schemas import CorrelationSpec, DistributionType
from .units import LengthUnit


class BlankDimension(BaseModel):
    """毛坯尺寸：在加工前把一个新毛坯面挂到已形成表面上。"""

    id: str = Field(..., min_length=1)
    start_surface: str = Field(..., min_length=1)
    end_surface: str = Field(..., min_length=1)
    nominal: float = Field(..., gt=0)
    upper_deviation: float = 0.0
    lower_deviation: float = 0.0
    unit: LengthUnit = LengthUnit.MM
    distribution: DistributionType = DistributionType.UNIFORM
    std_dev: float | None = Field(None, ge=0)
    manufacturing_cost: float = Field(0.0, ge=0, description="制造成本（货币单位）")

    @model_validator(mode="after")
    def _check(self) -> "BlankDimension":
        if self.lower_deviation > self.upper_deviation:
            raise ValueError(
                f"毛坯尺寸 {self.id}: 下偏差大于上偏差")
        if self.distribution == DistributionType.NORMAL and self.std_dev is None:
            raise ValueError(f"毛坯尺寸 {self.id}: normal 分布必须提供 std_dev")
        return self


class OperationSpec(BaseModel):
    """一道工序：以 datum_surface 定位，加工出 machined_surface。"""

    id: str = Field(..., min_length=1)
    datum_surface: str = Field(..., min_length=1)
    machined_surface: str = Field(..., min_length=1)
    nominal: float = Field(..., gt=0)
    upper_deviation: float = Field(0.0)
    lower_deviation: float = Field(0.0)
    unit: LengthUnit = LengthUnit.MM
    distribution: DistributionType = DistributionType.NORMAL
    std_dev: float | None = Field(None, ge=0)
    manufacturing_cost: float = Field(0.0, ge=0)
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "OperationSpec":
        if self.datum_surface == self.machined_surface:
            raise ValueError(
                f"工序 {self.id}: 定位基准与被加工面不能相同")
        if self.lower_deviation > self.upper_deviation:
            raise ValueError(f"工序 {self.id}: 下偏差大于上偏差")
        if self.distribution == DistributionType.NORMAL and self.std_dev is None:
            raise ValueError(f"工序 {self.id}: normal 分布必须提供 std_dev")
        return self


class StockClosureSpec(BaseModel):
    """最小加工余量闭环（单边规格：sign·(x_machined−x_blank) ≥ min）。"""

    id: str = Field(..., min_length=1)
    blank_surface: str = Field(..., min_length=1)
    machined_surface: str = Field(..., min_length=1)
    min_allowance: float = Field(..., gt=0)
    sign: int = Field(1, description="测量方向投影 ±1")
    unit: LengthUnit = LengthUnit.MM
    note: str = ""


class InlineDesignDimension(BaseModel):
    """直接提交的不可变设计尺寸（设计闭环）：两端面 + 名义值 + 偏差。"""

    id: str = Field(..., min_length=1)
    start_surface: str = Field(..., min_length=1)
    end_surface: str = Field(..., min_length=1)
    nominal: float = Field(..., gt=0)
    upper_deviation: float = Field(0.0)
    lower_deviation: float = Field(0.0)
    unit: LengthUnit = LengthUnit.MM
    sign: int = Field(1, description="测量方向投影 ±1")
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> "InlineDesignDimension":
        if self.start_surface == self.end_surface:
            raise ValueError(f"设计尺寸 {self.id}: 两端面不能相同")
        if self.lower_deviation > self.upper_deviation:
            raise ValueError(f"设计尺寸 {self.id}: 下偏差大于上偏差")
        return self


class ProcessPlanCreate(BaseModel):
    """创建工序尺寸方案（一次零件加工路线独立成版，版本 1）。"""

    name: str = Field(..., min_length=1)
    note: str = ""
    origin_surface: str = Field(..., min_length=1, description="图纸基准面（坐标 0）")

    # 不可变设计尺寸链：二选一——从冻结基线链导入或直接提交设计尺寸。
    source_chain_id: int | None = Field(
        None, description="从冻结基线链导入设计尺寸（只读快照，基线不改写）")
    design_dimensions: list[InlineDesignDimension] = Field(default_factory=list)

    blank_dimensions: list[BlankDimension] = Field(default_factory=list)
    operations: list[OperationSpec] = Field(..., min_length=1)
    stock_closures: list[StockClosureSpec] = Field(default_factory=list)
    correlations: list[CorrelationSpec] = Field(default_factory=list)

    mc_samples: int = Field(200_000, ge=1_000, le=5_000_000)
    random_seed: int = Field(20260913, ge=0)

    @model_validator(mode="after")
    def _check_sources(self) -> "ProcessPlanCreate":
        if self.source_chain_id is None and not self.design_dimensions:
            raise ValueError(
                "必须提供不可变设计尺寸链：source_chain_id 导入冻结基线链，"
                "或用 design_dimensions 直接提交设计尺寸")
        if self.source_chain_id is not None and self.design_dimensions:
            raise ValueError(
                "source_chain_id 与 design_dimensions 只能二选一；"
                "需要在基线设计尺寸之外追加时，请把基线设计尺寸一并显式提交")
        # 毛坯尺寸 id 与工序 id 不冲突（它们同属方案边）
        ids = [b.id for b in self.blank_dimensions] + \
              [o.id for o in self.operations]
        dup = sorted({x for x in ids if ids.count(x) > 1})
        if dup:
            raise ValueError(f"工序/毛坯尺寸 id 重复: {dup}")
        stk = [s.id for s in self.stock_closures]
        dup_s = sorted({x for x in stk if stk.count(x) > 1})
        if dup_s:
            raise ValueError(f"余量闭环 id 重复: {dup_s}")
        return self


class ProcessPlanVersionCreate(BaseModel):
    """在同一方案下另建独立版本（完整新加工路线随版本冻结，历史不回改）。"""

    name: str = Field("", min_length=0)
    note: str = ""
    origin_surface: str = Field(..., min_length=1)
    source_chain_id: int | None = None
    design_dimensions: list[InlineDesignDimension] = Field(default_factory=list)
    blank_dimensions: list[BlankDimension] = Field(default_factory=list)
    operations: list[OperationSpec] = Field(..., min_length=1)
    stock_closures: list[StockClosureSpec] = Field(default_factory=list)
    correlations: list[CorrelationSpec] = Field(default_factory=list)
    mc_samples: int = Field(200_000, ge=1_000, le=5_000_000)
    random_seed: int | None = Field(None, ge=0)


# ------------------------------------------------------------- 反算求解

class CapabilityTier(BaseModel):
    """机床能力档位：公差半宽 T = K·∛N / 2（K=grade_factor），选用即附加成本。"""

    id: str = Field(..., min_length=1)
    grade_factor: float = Field(..., gt=0)
    setup_cost: float = Field(0.0, ge=0, description="每道选用该档位工序的附加成本")


class LockedDimension(BaseModel):
    edge_id: str = Field(..., min_length=1)
    nominal: float | None = Field(
        None, gt=0, description="锁定名义值（mm 口径的内部值）；null=沿用提交名义")


class ProcessSolveRequest(BaseModel):
    """锁定已定工序尺寸、反算其余，枚举档位×步进多解并排序。"""

    name: str = Field("solve", min_length=1)
    note: str = ""
    locked_dimensions: list[LockedDimension] = Field(default_factory=list)

    capability_tiers: list[CapabilityTier] = Field(..., min_length=1)
    default_grade: str | None = Field(
        None, description="记录用默认档位；枚举时各自由边默认在全部档位中选择")
    edge_grades: dict[str, str] = Field(
        default_factory=dict,
        description="限定某边只能用单个档位（edge_id -> tier id）")
    edge_grade_options: dict[str, list[str]] = Field(
        default_factory=dict,
        description="限定某边的可选档位子集（edge_id -> tier ids）")

    standard_step_sizes: list[float] = Field(
        default_factory=lambda: [1.0],
        description="标准尺寸步进候选（mm），枚举取首个为默认网格步进")
    edge_step_sizes: dict[str, float] = Field(default_factory=dict)

    mc_samples: int = Field(100_000, ge=1_000, le=5_000_000)
    random_seed: int | None = Field(None, ge=0)
    max_candidates: int = Field(20, ge=1, le=100)

    @model_validator(mode="after")
    def _check(self) -> "ProcessSolveRequest":
        ids = [t.id for t in self.capability_tiers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"能力档位 id 重复: {ids}")
        overlap = set(self.edge_grades) & set(self.edge_grade_options)
        if overlap:
            raise ValueError(
                f"同一工序不能同时给 edge_grades 与 edge_grade_options: "
                f"{sorted(overlap)}")
        known = set(ids)
        bad = [(e, g) for e, g in self.edge_grades.items() if g not in known]
        if bad:
            raise ValueError(f"edge_grades 引用了未声明档位: {bad}")
        for e, gs in self.edge_grade_options.items():
            if not gs:
                raise ValueError(f"工序 {e} 的 edge_grade_options 不能为空")
            unknown = [g for g in gs if g not in known]
            if unknown:
                raise ValueError(
                    f"工序 {e} 的可选档位未声明: {unknown}")
        for v in self.edge_step_sizes.values():
            if v <= 0:
                raise ValueError("边级标准步进必须为正")
        if self.default_grade is not None and self.default_grade not in known:
            raise ValueError(
                f"default_grade {self.default_grade!r} 未在 capability_tiers 声明")
        return self


class ProcessSelectRequest(BaseModel):
    """选定一个候选并冻结（方案快照已冻结，此处只固化反算结果）。"""

    rank: int = Field(..., ge=1)
    note: str = ""
