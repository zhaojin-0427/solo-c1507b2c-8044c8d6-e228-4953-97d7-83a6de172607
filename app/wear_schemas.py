"""Pydantic 模型：服役磨损研究与维护编排的请求校验。

校验内容：
* 磨损曲线分段衔接：首段从 (0, 0) 出发，相邻段的终点/起点循环数与累计
  磨损必须一致（分段线性累计曲线的连续性），段内循环数严格递增、
  累计磨损单调不减；
* 尺寸引用：磨损参数必须恰好覆盖基线链全部尺寸（不多不少）；
* 共用载荷相关矩阵：引用存在、不重复、半正定（与基线链同一口径）；
* 计算节点：严格递增、非负、不超出所有尺寸磨损曲线的覆盖范围；
* 维护编排：锁定件与更换成本不得重叠、维护节点必须属于研究计算节点。
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .schemas import ChainCreate, CorrelationSpec
from .units import LengthUnit


class WearDirection(str, Enum):
    """磨损后尺寸变化方向。"""

    INCREASE = "increase"   # 磨损后尺寸增大（如孔被磨大）
    DECREASE = "decrease"   # 磨损后尺寸减小（如轴被磨小）


class WearCurveSegment(BaseModel):
    """磨损曲线的一段：两个折点之间的线性段。

    累计磨损随循环数分段线性增长；段内 end_cycles 必须严格大于
    start_cycles，end_wear 不得小于 start_wear（累计磨损只增不减）。
    """

    start_cycles: float = Field(..., ge=0, description="段起点循环数")
    end_cycles: float = Field(..., description="段终点循环数")
    start_wear: float = Field(..., ge=0, description="段起点累计磨损（按 curve_unit 计）")
    end_wear: float = Field(..., ge=0, description="段终点累计磨损（按 curve_unit 计）")

    @model_validator(mode="after")
    def _check(self) -> "WearCurveSegment":
        for f in ("start_cycles", "end_cycles", "start_wear", "end_wear"):
            if not math.isfinite(getattr(self, f)):
                raise ValueError(
                    f"磨损曲线段的 {f} 必须为有限数，收到 {getattr(self, f)!r}"
                )
        if self.end_cycles <= self.start_cycles:
            raise ValueError(
                f"磨损曲线段终点循环数 {self.end_cycles} 必须严格大于起点 "
                f"{self.start_cycles}"
            )
        if self.end_wear < self.start_wear:
            raise ValueError(
                f"累计磨损不能随循环数减少：段终点 {self.end_wear} 小于段起点 "
                f"{self.start_wear}（累计磨损曲线单调不减）"
            )
        return self


class WearDimensionSpec(BaseModel):
    """单个组成尺寸的服役磨损参数。

    * wear_direction：磨损后尺寸增大 / 减小；
    * wear_curve：分段线性累计磨损曲线（折点按 curve_unit 计），
      首段必须从 (0, 0) 出发，相邻段终点与起点必须衔接；
    * rate_std_dev：磨损速率标准差（curve_unit / 循环），同一零件整个
      服役期速率偏差恒定（随机速率模型），N 循环后累计磨损标准差为
      rate_std_dev × N。
    """

    dimension_id: str = Field(..., min_length=1)
    wear_direction: WearDirection
    wear_curve: list[WearCurveSegment] = Field(
        ..., min_length=1, description="分段线性累计磨损曲线（至少一段）"
    )
    curve_unit: LengthUnit = LengthUnit.MM
    rate_std_dev: float = Field(
        ..., ge=0, description="磨损速率标准差（curve_unit / 循环）"
    )

    @model_validator(mode="after")
    def _check_curve(self) -> "WearDimensionSpec":
        if not math.isfinite(self.rate_std_dev):
            raise ValueError(
                f"尺寸 {self.dimension_id}: 磨损速率标准差必须为有限数，"
                f"收到 {self.rate_std_dev!r}"
            )
        first = self.wear_curve[0]
        if first.start_cycles != 0 or first.start_wear != 0:
            raise ValueError(
                f"尺寸 {self.dimension_id}: 磨损曲线首段必须从 (0, 0) 出发，"
                f"收到 ({first.start_cycles}, {first.start_wear})"
            )
        for prev, nxt in zip(self.wear_curve, self.wear_curve[1:]):
            if nxt.start_cycles != prev.end_cycles or not math.isclose(
                    nxt.start_wear, prev.end_wear,
                    rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"尺寸 {self.dimension_id}: 磨损曲线分段不衔接——前段终点 "
                    f"({prev.end_cycles}, {prev.end_wear}) 与后段起点 "
                    f"({nxt.start_cycles}, {nxt.start_wear}) 不一致；"
                    "相邻段必须共用同一折点"
                )
        return self

    @property
    def max_cycles(self) -> float:
        return self.wear_curve[-1].end_cycles


class WearStudyCreate(BaseModel):
    """在冻结基线链上创建服役磨损研究（独立成版，创建即冻结）。

    为链上每个组成尺寸填写磨损方向、分段线性累计磨损曲线与速率标准差；
    共用载荷导致的磨损速率相关用 wear_rate_correlations 声明（Pearson
    相关，与基线链相关矩阵同一口径）。计算节点为循环数序列，研究的
    封闭环限值独立设置（可与基线链规格不同）。
    """

    name: str = Field(..., min_length=1)
    note: str = ""
    dimensions: list[WearDimensionSpec] = Field(
        ..., min_length=1,
        description="逐尺寸磨损参数，必须恰好覆盖基线链全部尺寸（不多不少）",
    )
    wear_rate_correlations: list[CorrelationSpec] = Field(
        default_factory=list,
        description="共用载荷导致的磨损速率相关（对称，提交一次即可）",
    )
    evaluation_cycles: list[float] = Field(
        ..., min_length=2,
        description="计算节点（循环数，严格递增，至少 2 个；"
                    "不得超出任何尺寸磨损曲线的覆盖范围）",
    )
    closure_lower_limit: float | None = Field(
        None, description="研究封闭环允许下限（与 closure_unit 同单位）"
    )
    closure_upper_limit: float | None = Field(
        None, description="研究封闭环允许上限（与 closure_unit 同单位）"
    )
    closure_unit: LengthUnit = LengthUnit.MM
    mc_samples: int = Field(
        50_000, ge=1_000, le=1_000_000,
        description="固定种子蒙特卡洛样本数（制造散布与磨损速率两个子流）",
    )
    random_seed: int = Field(
        20260917, ge=0, description="磨损研究固定随机种子；同一研究重复计算不变"
    )

    @model_validator(mode="after")
    def _check(self) -> "WearStudyCreate":
        ids = [d.dimension_id for d in self.dimensions]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"磨损参数中尺寸重复填写: {dup}")
        id_set = set(ids)

        seen: set[frozenset[str]] = set()
        for c in self.wear_rate_correlations:
            if c.dim_a not in id_set or c.dim_b not in id_set:
                raise ValueError(
                    f"磨损速率相关项引用了未填写磨损参数的尺寸: "
                    f"{c.dim_a!r}, {c.dim_b!r}"
                )
            key = frozenset((c.dim_a, c.dim_b))
            if key in seen:
                raise ValueError(
                    f"尺寸 {c.dim_a} 与 {c.dim_b} 的磨损速率相关重复声明"
                )
            seen.add(key)
        ChainCreate._check_psd(ids, self.wear_rate_correlations)

        nodes = self.evaluation_cycles
        for n in nodes:
            if not math.isfinite(n) or n < 0:
                raise ValueError(
                    f"计算节点必须为非负有限数，收到 {n!r}"
                )
        for a, b in zip(nodes, nodes[1:]):
            if b <= a:
                raise ValueError(
                    f"计算节点必须严格递增：{a} 之后出现 {b}"
                )
        cover = min(d.max_cycles for d in self.dimensions)
        if nodes[-1] > cover:
            short = [d.dimension_id for d in self.dimensions
                     if d.max_cycles < nodes[-1]]
            raise ValueError(
                f"计算节点 {nodes[-1]} 超出磨损曲线覆盖范围（最短曲线止于 "
                f"{cover} 循环，尺寸 {short}）；请延长曲线或减少节点"
            )

        lo, hi = self.closure_lower_limit, self.closure_upper_limit
        if lo is None and hi is None:
            raise ValueError(
                "研究封闭环限值缺失：closure_lower_limit 与 "
                "closure_upper_limit 至少提供一个（否则无法定义超差与越界）"
            )
        for v in (lo, hi):
            if v is not None and not math.isfinite(v):
                raise ValueError(f"封闭环限值必须为有限数，收到 {v!r}")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(
                f"封闭环下限 {lo} 必须严格小于上限 {hi}"
            )
        return self


# ------------------------------------------------------------ 维护编排

class WearShimSpec(BaseModel):
    """候选垫片：在封闭环上增加 sign × 名义厚度的固定偏移（维护时加入）。

    垫片厚度不确定度（正态标准差）从加入节点起计入封闭环方差；
    同一规格在同一方案中可多次使用（垫片叠加以恢复间隙）。
    """

    shim_id: str = Field(..., min_length=1)
    name: str = ""
    thickness: float = Field(..., gt=0, description="垫片名义厚度（按 unit 计）")
    std_dev: float = Field(
        0.0, ge=0, description="垫片厚度标准差（正态，与 thickness 同单位）"
    )
    unit: LengthUnit = LengthUnit.MM
    closure_sign: Literal[1, -1] = Field(
        1, description="垫片在封闭环中的方向系数（+1 增隙 / -1 减隙）"
    )
    cost: float = Field(..., ge=0, description="单次加垫片成本（不含停机）")

    @model_validator(mode="after")
    def _check(self) -> "WearShimSpec":
        for f in ("thickness", "std_dev", "cost"):
            if not math.isfinite(getattr(self, f)):
                raise ValueError(
                    f"垫片 {self.shim_id}: {f} 必须为有限数，"
                    f"收到 {getattr(self, f)!r}"
                )
        return self


class MaintenanceSearchRequest(BaseModel):
    """维护编排搜索：继续使用 / 加垫片 / 更换零件的方案生成与排列。

    锁定件（locked_dimensions）不可更换；replacement_costs 列出的尺寸
    为可更换件（更换后该尺寸累计磨损清零、重新计循环）；每次维护动作
    （加垫片或更换）计一次停机成本 downtime_cost。编排判据 criterion
    决定「违规」口径：极值界（worst_case）或 RSS ±3σ 界（rss）越出
    研究封闭环限值即记一次违规。
    """

    name: str = Field(..., min_length=1)
    note: str = ""
    locked_dimensions: list[str] = Field(
        default_factory=list,
        description="不可更换件（锁定）；与 replacement_costs 不得重叠",
    )
    shim_candidates: list[WearShimSpec] = Field(
        default_factory=list, description="可选垫片规格"
    )
    maintainable_cycles: list[float] = Field(
        ..., min_length=1,
        description="允许维护的计算节点（循环数，必须是研究计算节点的子集）",
    )
    replacement_costs: dict[str, float] = Field(
        default_factory=dict,
        description="可更换件：尺寸 id -> 单次更换成本（不含停机）",
    )
    downtime_cost: float = Field(
        0.0, ge=0, description="每次维护动作的停机成本"
    )
    criterion: Literal["worst_case", "rss"] = Field(
        "worst_case", description="编排与违规统计口径：极值界或 RSS ±3σ 界"
    )
    mc_samples: int = Field(
        20_000, ge=1_000, le=1_000_000,
        description="方案复核固定种子蒙特卡洛样本数（研究种子派生子流）",
    )
    random_seed: int | None = Field(
        None, ge=0, description="方案复核种子；缺省沿用研究种子"
    )
    max_candidates: int = Field(10, ge=1, le=50)

    @model_validator(mode="after")
    def _check(self) -> "MaintenanceSearchRequest":
        locked = self.locked_dimensions
        if len(set(locked)) != len(locked):
            dup = sorted({i for i in locked if locked.count(i) > 1})
            raise ValueError(f"锁定尺寸重复列出: {dup}")
        shim_ids = [s.shim_id for s in self.shim_candidates]
        dup_s = sorted({i for i in shim_ids if shim_ids.count(i) > 1})
        if dup_s:
            raise ValueError(f"垫片 id 重复: {dup_s}")
        overlap = sorted(set(locked) & set(self.replacement_costs))
        if overlap:
            raise ValueError(
                f"尺寸既被锁定又给出更换成本（矛盾）: {overlap}；"
                "不可更换件请仅从 replacement_costs 中移除"
            )
        bad_cost = {k: v for k, v in self.replacement_costs.items()
                    if not math.isfinite(v) or v < 0}
        if bad_cost:
            raise ValueError(f"更换成本必须为非负有限数: {bad_cost}")
        if not math.isfinite(self.downtime_cost):
            raise ValueError("停机成本必须为有限数")
        nodes = self.maintainable_cycles
        for n in nodes:
            if not math.isfinite(n) or n < 0:
                raise ValueError(f"维护节点必须为非负有限数，收到 {n!r}")
        if len(set(nodes)) != len(nodes):
            dup = sorted({n for n in nodes if nodes.count(n) > 1})
            raise ValueError(f"维护节点重复列出: {dup}")
        return self


class MaintenanceSelectRequest(BaseModel):
    """选定一个维护方案并冻结（每个搜索结果只能选定一次）。"""

    rank: int = Field(..., ge=1, description="选定候选的 rank（1 起）")
    note: str = ""
