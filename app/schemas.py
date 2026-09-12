"""Pydantic 模型：API 请求 / 响应结构与输入校验。

校验内容：
* 尺寸名义值为正、偏差自洽（lower <= 0 <= upper）；
* 有向尺寸链构成单个闭合环：每条无向边唯一、方向闭合；
* 相关系数矩阵：对称、对角为 1、取值 [-1, 1]、正定（半正定容差）；
* 混合单位 / 单边公差允许提交，规范化时保留原始表示。
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

import math

from pydantic import BaseModel, Field, model_validator

from .units import LengthUnit


class DistributionType(str, Enum):
    NORMAL = "normal"          # 正态（按标准差截断由蒙特卡洛体现）
    UNIFORM = "uniform"        # 均匀分布，半带宽 = T
    TRIANGULAR = "triangular"  # 三角分布，半带宽 = T


class DimensionInput(BaseModel):
    """单个组成环（有向尺寸链的一条边）。"""

    id: str = Field(..., min_length=1, description="尺寸唯一标识")
    start: str = Field(..., min_length=1, description="起点节点")
    end: str = Field(..., min_length=1, description="终点节点")
    # 名义尺寸与测量方向：方向沿 start -> end 的投影符号（+1 / -1）。
    # 闭环方向由有向边自动求和，direction 用于测量方向非边方向的情形，
    # 常规尺寸链取 +1。
    direction: Literal[1, -1] = 1

    nominal: float = Field(..., gt=0, description="名义尺寸（>0）")
    upper_deviation: float = Field(..., description="上偏差 ES，可为 0 或负")
    lower_deviation: float = Field(..., description="下偏差 EI，可为 0 或正")
    unit: LengthUnit = LengthUnit.MM

    distribution: DistributionType = DistributionType.NORMAL
    std_dev: float | None = Field(
        None,
        ge=0,
        description="标准差（与 unit 同单位）；正态分布必填，其它分布缺省按理论标准差",
    )

    description: str = ""

    @model_validator(mode="after")
    def _check_deviations(self) -> "DimensionInput":
        if self.lower_deviation > self.upper_deviation:
            raise ValueError(
                f"尺寸 {self.id}: 下偏差 {self.lower_deviation} 大于上偏差 "
                f"{self.upper_deviation}"
            )
        return self

    @property
    def is_unilateral(self) -> bool:
        """单边公差：其中一个偏差恰为 0 且公差带宽度 > 0。"""
        width = self.upper_deviation - self.lower_deviation
        return width > 0 and (
            self.upper_deviation == 0.0 or self.lower_deviation == 0.0
        )


class CorrelationSpec(BaseModel):
    """两个尺寸之间的相关系数（对称，提交一次即可）。"""

    dim_a: str
    dim_b: str
    rho: float = Field(..., ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def _check_pair(self) -> "CorrelationSpec":
        if self.dim_a == self.dim_b:
            raise ValueError("相关系数必须声明在两个不同尺寸之间")
        return self


class ChainCreate(BaseModel):
    """创建一个有向尺寸链模型。"""

    name: str = Field(..., min_length=1)
    description: str = ""
    # 封闭环规格（与尺寸同一单位体系，内部换算为 mm）
    closure_lower_limit: float | None = Field(
        None, description="封闭环（间隙）允许下限，缺省 -∞"
    )
    closure_upper_limit: float | None = Field(
        None, description="封闭环（间隙）允许上限，缺省 +∞"
    )
    closure_unit: LengthUnit = LengthUnit.MM

    dimensions: list[DimensionInput] = Field(..., min_length=1)
    correlations: list[CorrelationSpec] = Field(default_factory=list)

    mc_samples: int = Field(200_000, ge=1_000, le=5_000_000)
    random_seed: int = Field(20260910, ge=0, description="蒙特卡洛固定随机种子")

    @model_validator(mode="after")
    def _validate_chain(self) -> "ChainCreate":
        dims = self.dimensions

        # 1) 尺寸 id 唯一
        ids = [d.id for d in dims]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"尺寸 id 重复: {dup}")

        # 2) 拒绝重复边：同一有序方向上出现两条尺寸即重复
        #    （a→b 与 b→a 是反向平行边，构成合法的 2 件配合闭合环）。
        seen_directed: set[tuple[str, str]] = set()
        for d in dims:
            key = (d.start, d.end)
            if key in seen_directed:
                raise ValueError(
                    f"重复边: 方向 {d.start} -> {d.end} 上已存在尺寸 {d.id}"
                )
            seen_directed.add(key)

        # 3) 方向闭合：每个节点的入度等于出度 => 有向图为若干环；
        #    再要求所有边同属一个连通分量 => 单个闭合尺寸链。
        balance: dict[str, int] = {}
        adjacency: dict[str, set[str]] = {}
        for d in dims:
            balance[d.start] = balance.get(d.start, 0) + 1
            balance[d.end] = balance.get(d.end, 0) - 1
            adjacency.setdefault(d.start, set()).add(d.end)
            adjacency.setdefault(d.end, set()).add(d.start)
        unbalanced = {n: b for n, b in balance.items() if b != 0}
        if unbalanced:
            detail = ", ".join(
                f"{n}(出-入={b:+d})" for n, b in sorted(unbalanced.items())
            )
            raise ValueError(f"有向尺寸链方向不闭合，节点度数不平衡: {detail}")

        if adjacency:
            start_node = next(iter(adjacency))
            stack = [start_node]
            reached = {start_node}
            while stack:
                node = stack.pop()
                for nxt in adjacency[node]:
                    if nxt not in reached:
                        reached.add(nxt)
                        stack.append(nxt)
            if reached != set(adjacency):
                missing = sorted(set(adjacency) - reached)
                raise ValueError(
                    f"尺寸链不是单一闭合环: 节点 {missing} 与主环不连通，"
                    "可能存在互不相连的多个环"
                )

        # 4) 相关系数引用存在、不重复
        id_set = set(ids)
        seen_pairs: set[frozenset[str]] = set()
        for c in self.correlations:
            if c.dim_a not in id_set or c.dim_b not in id_set:
                raise ValueError(
                    f"相关系数引用了不存在的尺寸: {c.dim_a!r}, {c.dim_b!r}"
                )
            key = frozenset((c.dim_a, c.dim_b))
            if key in seen_pairs:
                raise ValueError(
                    f"尺寸 {c.dim_a} 与 {c.dim_b} 的相关系数重复声明"
                )
            seen_pairs.add(key)

        # 5) 相关矩阵必须半正定
        self._check_psd(ids, self.correlations)

        # 6) 正态分布必须提供标准差
        for d in dims:
            if d.distribution == DistributionType.NORMAL and d.std_dev is None:
                raise ValueError(
                    f"尺寸 {d.id}: normal 分布必须提供 std_dev"
                )
        return self

    @staticmethod
    def _check_psd(ids: list[str], corrs: list[CorrelationSpec]) -> None:
        """构造相关矩阵并检查对称、对角、取值范围与半正定性。"""
        import numpy as np

        n = len(ids)
        idx = {name: i for i, name in enumerate(ids)}
        r = np.eye(n)
        for c in corrs:
            i, j = idx[c.dim_a], idx[c.dim_b]
            r[i, j] = c.rho
            r[j, i] = c.rho
        # 对称性与边界由构造保证（输入对称写入）；再做一次防御性检查
        if not np.allclose(r, r.T, atol=1e-10):
            raise ValueError("相关矩阵不对称")
        if np.any(r < -1.0 - 1e-9) or np.any(r > 1.0 + 1e-9):
            raise ValueError("相关矩阵存在超出 [-1, 1] 的元素")
        eig = np.linalg.eigvalsh(r)
        if eig.min() < -1e-8:
            raise ValueError(
                f"相关矩阵非半正定: 最小特征值 {eig.min():.3e}；"
                "请检查相关系数组合（如 A=B=0.9, C=-0.9 之类矛盾取值）"
            )


class ScenarioCreate(BaseModel):
    """基于基线创建方案分支（不覆盖基线）。"""

    name: str = Field(..., min_length=1)
    note: str = ""
    # 尺寸 id -> 新的上下偏差（与该尺寸原单位一致）
    tolerance_overrides: dict[str, "ToleranceOverride"] = Field(
        default_factory=dict
    )
    # 尺寸 id -> 新标准差（与该尺寸原单位一致，None 表示回退到理论值）
    std_dev_overrides: dict[str, float | None] = Field(default_factory=dict)
    random_seed: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _check_std_dev(self) -> "ScenarioCreate":
        negative = {k: v for k, v in self.std_dev_overrides.items()
                    if v is not None and v < 0}
        if negative:
            detail = ", ".join(f"{k}={v}" for k, v in negative.items())
            raise ValueError(
                f"标准差不能为负（请改用 0 表示零方差固定尺寸）: {detail}"
            )
        return self


class ToleranceOverride(BaseModel):
    upper_deviation: float
    lower_deviation: float

    @model_validator(mode="after")
    def _check(self) -> "ToleranceOverride":
        if self.lower_deviation > self.upper_deviation:
            raise ValueError("下偏差不能大于上偏差")
        return self


class BatchAdjustRequest(BaseModel):
    """批量调整：对一组（或全部）尺寸统一乘系数 / 加减公差，再对比基线。"""

    name: str = Field("batch", min_length=1)
    target_dimensions: list[str] | None = Field(
        None, description="要调整的尺寸 id；null 表示全部"
    )
    # 公差带半宽乘以该系数（0.8 即收紧 20%）
    tolerance_scale: float = Field(1.0, gt=0)
    # 标准差乘以该系数（缺省跟随公差带缩放）
    std_dev_scale: float | None = Field(None, gt=0)
    note: str = ""
    random_seed: int | None = None


class GapProbabilityRequest(BaseModel):
    """查询封闭环间隙落在 [lower, upper] 的概率。"""

    lower: float | None = None
    upper: float | None = None
    unit: LengthUnit = LengthUnit.MM

    @model_validator(mode="after")
    def _check(self) -> "GapProbabilityRequest":
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("区间下限不能大于上限")
        return self


class CostTargetRequest(BaseModel):
    """按单位收紧成本搜索达到目标超差率的候选组合。"""

    target_reject_rate: float = Field(..., gt=0, lt=1, description="目标超差率，如 0.01")
    # 尺寸 id -> 每将公差带半宽收紧 1mm（内部单位）对应的成本权重
    tightening_cost: dict[str, float] = Field(
        default_factory=dict,
        description="单位收紧成本（成本/每 mm 收紧量）；未列出尺寸使用 default_cost",
    )
    default_cost: float = Field(1.0, gt=0)
    # 搜索网格：相对当前半宽的可选比例（从紧到松给出也可，内部排序）
    scale_levels: list[float] = Field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        description="每个尺寸公差带半宽的可选比例",
    )
    max_evaluations: int = Field(4000, ge=1, le=50_000)
    random_seed: int | None = None
    scale_normal_sigma: bool = Field(
        False,
        description=(
            "正态分布的用户标准差是否随公差带等比收紧。"
            "默认 false：σ 代表来料/制程的实测波动，收紧公差带不改变 σ；"
            "true：假设采购更紧公差的同时制程能力同步提升，σ 按比例缩放"
        ),
    )

    @model_validator(mode="after")
    def _check_levels(self) -> "CostTargetRequest":
        levels = sorted(set(self.scale_levels))
        if not levels or any(v <= 0 for v in levels):
            raise ValueError("scale_levels 必须为正数且非空")
        self.scale_levels = levels
        return self


ScenarioCreate.model_rebuild()


# -------------------------------------------------------- 测量方案（量具误差）

class GaugeInput(BaseModel):
    """单个尺寸的量具误差声明（所有分量以 unit 计，内部换算为 mm）。

    * resolution：量具分辨率，标准不确定度按半宽均匀分布 resolution/(2√3)；
    * calibration_expanded_uncertainty + coverage_factor：校准证书扩展
      不确定度 U 与覆盖因子 k，**两者为每个量具声明的必填项**
      （无校准数据时须显式给 U=0 与对应 k），u_cal = U/k；
    * bias_correction：偏倚修正值（带符号，判定前加到实测值上）；
    * bias_std_uncertainty：该修正值的标准不确定度；
    * repeatability_std：重复性标准差。
    """

    dimension_id: str = Field(..., min_length=1)
    unit: LengthUnit = LengthUnit.MM
    resolution: float | None = Field(None, description="量具分辨率")
    calibration_expanded_uncertainty: float | None = Field(
        None, description="校准扩展不确定度 U"
    )
    coverage_factor: float | None = Field(
        None, description="校准扩展不确定度的覆盖因子 k（随 U 必填）"
    )
    bias_correction: float | None = Field(
        None, description="偏倚修正值（带符号， corrected = measured + 修正值）"
    )
    bias_std_uncertainty: float | None = Field(
        None, description="偏倚修正值的标准不确定度"
    )
    repeatability_std: float | None = Field(None, description="重复性标准差")

    @model_validator(mode="after")
    def _check_components(self) -> "GaugeInput":
        fields = (
            "resolution", "calibration_expanded_uncertainty", "coverage_factor",
            "bias_correction", "bias_std_uncertainty", "repeatability_std",
        )
        for f in fields:
            v = getattr(self, f)
            if v is not None and not math.isfinite(v):
                raise ValueError(
                    f"尺寸 {self.dimension_id}: 分量 {f} 必须为有限数，"
                    f"收到 {v!r}"
                )
        negative = [
            f for f in (
                "resolution", "calibration_expanded_uncertainty",
                "bias_std_uncertainty", "repeatability_std",
            )
            if getattr(self, f) is not None and getattr(self, f) < 0
        ]
        if negative:
            detail = ", ".join(f"{f}={getattr(self, f)}" for f in negative)
            raise ValueError(
                f"尺寸 {self.dimension_id}: 不确定度分量不能为负: {detail}"
            )
        if self.coverage_factor is not None and self.coverage_factor <= 0:
            raise ValueError(
                f"尺寸 {self.dimension_id}: 覆盖因子必须为正，"
                f"收到 {self.coverage_factor}"
            )
        u_cal = self.calibration_expanded_uncertainty
        k = self.coverage_factor
        if u_cal is None and k is None:
            raise ValueError(
                f"尺寸 {self.dimension_id}: 校准扩展不确定度与覆盖因子缺失"
                "（每个量具声明都必须给出校准扩展不确定度 U 及其覆盖因子 k；"
                "无校准数据时请显式给 U=0 与对应 k）"
            )
        if u_cal is not None and k is None:
            raise ValueError(
                f"尺寸 {self.dimension_id}: 已给校准扩展不确定度，"
                "但覆盖因子缺失（u_cal = U/k 需要 k）"
            )
        if k is not None and u_cal is None:
            raise ValueError(
                f"尺寸 {self.dimension_id}: 已给覆盖因子，"
                "但校准扩展不确定度缺失"
            )
        return self


class MeasurementPlanCreate(BaseModel):
    """针对基线链建立不可变测量方案：逐尺寸量具误差 + 共用量具相关项。"""

    name: str = Field(..., min_length=1)
    note: str = ""
    gauges: list[GaugeInput] = Field(
        ..., min_length=1, description="逐尺寸量具声明，必须恰好覆盖链上全部尺寸"
    )
    correlations: list[CorrelationSpec] = Field(
        default_factory=list,
        description="共用量具带来的测量误差相关系数（对称，提交一次即可）",
    )

    @model_validator(mode="after")
    def _check_plan(self) -> "MeasurementPlanCreate":
        ids = [g.dimension_id for g in self.gauges]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"测量方案中尺寸重复声明: {dup}")
        id_set = set(ids)
        seen_pairs: set[frozenset[str]] = set()
        for c in self.correlations:
            if c.dim_a not in id_set or c.dim_b not in id_set:
                raise ValueError(
                    f"量具相关项引用了方案外的尺寸: {c.dim_a!r}, {c.dim_b!r}"
                )
            key = frozenset((c.dim_a, c.dim_b))
            if key in seen_pairs:
                raise ValueError(
                    f"尺寸 {c.dim_a} 与 {c.dim_b} 的量具相关重复声明"
                )
            seen_pairs.add(key)
        # 相关矩阵必须半正定（与基线链同一校验口径）
        ChainCreate._check_psd(ids, self.correlations)
        return self


class GuardBandSpec(BaseModel):
    """保护带设置：按扩展不确定度倍数（multiple）或固定长度（fixed）。"""

    mode: Literal["multiple", "fixed"] = "multiple"
    multiple: float = Field(
        1.0, ge=0, description="mode=multiple 时 w = multiple × 扩展不确定度 U"
    )
    fixed: float | None = Field(
        None, ge=0, description="mode=fixed 时的固定保护带长度（按 unit 计）"
    )
    unit: LengthUnit = LengthUnit.MM

    @model_validator(mode="after")
    def _check(self) -> "GuardBandSpec":
        if not math.isfinite(self.multiple):
            raise ValueError("保护带倍数必须为有限数")
        if self.fixed is not None and not math.isfinite(self.fixed):
            raise ValueError("固定保护带长度必须为有限数")
        if self.mode == "fixed" and self.fixed is None:
            raise ValueError("保护带 mode=fixed 时必须给出固定长度 fixed")
        if self.mode == "multiple" and self.fixed is not None:
            raise ValueError(
                "保护带 mode=multiple 时不接受 fixed；如需固定长度请用 mode=fixed"
            )
        return self


# -------------------------------------------------------- 来料检验批次

class MeasurementInput(BaseModel):
    """单条工件行中某一个尺寸的实测值。

    value 必须为有限数（NaN / ±Infinity 一律拒收）；缺测用 null 表示，
    或直接在 measurements 中省略该尺寸。
    """

    dimension_id: str = Field(..., min_length=1)
    value: float | None = Field(
        ..., description="实测值；null 表示该尺寸缺测（可入库）"
    )
    unit: LengthUnit = Field(..., description="实测值单位，须为已知长度单位")

    @model_validator(mode="after")
    def _finite(self) -> "MeasurementInput":
        # Pydantic 接受 NaN/Infinity 为 float，这里显式拒绝
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(
                f"尺寸 {self.dimension_id} 实测值必须为有限数，"
                f"收到 {self.value!r}；缺测请用 null"
            )
        return self


class BatchRowInput(BaseModel):
    """一个工件序号及其各尺寸实测值（每行内同一尺寸只能出现一次）。"""

    serial: str = Field(..., min_length=1, description="工件序号")
    measurements: list[MeasurementInput] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _no_duplicate_dimension(self) -> "BatchRowInput":
        ids = [m.dimension_id for m in self.measurements]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(
                f"工件 {self.serial}: 同一尺寸重复测量 {dup}，每行每尺寸只允许一条"
            )
        return self


class InspectionBatchCreate(BaseModel):
    """创建来料检验批次：选定基线链，逐工件行提交实测值。

    可选引用测量方案（measurement_plan_id）：引用后先按方案修正偏倚，
    再对修正值做 GUM 线性传播与固定种子蒙特卡洛不确定度评定，
    按保护带给出接收 / 拒收 / 不确定判定。
    """

    name: str = Field(..., min_length=1)
    note: str = ""
    rows: list[BatchRowInput] = Field(..., min_length=1)
    # 封闭环 bootstrap 重采样（固定随机种子，结果可复现）
    bootstrap_samples: int = Field(
        10_000, ge=1_000, le=200_000,
        description="封闭环分布 bootstrap 重采样次数 B",
    )
    random_seed: int = Field(
        20260911, ge=0, description="bootstrap 固定随机种子"
    )
    # ---- 量具误差判定（可选，引用测量方案后生效）----
    measurement_plan_id: int | None = Field(
        None, ge=1, description="引用的测量方案 id（属于同一基线链）"
    )
    output_coverage_factor: float = Field(
        2.0, gt=0, description="输出扩展不确定度的覆盖因子 k_out（U=k_out·u_c）"
    )
    guard_band: GuardBandSpec | None = Field(
        None, description="保护带设置；缺省为 mode=multiple, multiple=1.0"
    )
    measurement_mc_samples: int = Field(
        100_000, ge=1_000, le=1_000_000,
        description="测量误差蒙特卡洛样本数（所有工件共用一组误差样本）",
    )
    measurement_mc_seed: int = Field(
        20260912, ge=0, description="测量误差蒙特卡洛固定随机种子"
    )


# -------------------------------------------------------- 选择性装配任务

class PoolSpec(BaseModel):
    """零件池：把链上一组尺寸映射到一个池，同池尺寸必须来自同一工件序号。"""

    name: str = Field(..., min_length=1, description="零件池唯一名称")
    dimensions: list[str] = Field(
        ..., min_length=1, description="映射到本池的链上尺寸 id（恰好划分链）"
    )


class TargetGapSpec(BaseModel):
    """目标装配间隙区间（封闭环合格区间，单位可混用，内部换算为 mm）。"""

    lower: float = Field(..., description="目标间隙下限")
    upper: float = Field(..., description="目标间隙上限")
    unit: LengthUnit = LengthUnit.MM

    @model_validator(mode="after")
    def _check(self) -> "TargetGapSpec":
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("目标间隙上下限必须为有限数")
        if self.lower >= self.upper:
            raise ValueError("目标间隙下限必须严格小于上限")
        return self


class SameBatchGroupSpec(BaseModel):
    """池间同批关系：列出的零件池在每个装配中必须取自同一来源批次。"""

    pools: list[str] = Field(..., min_length=2)


class ForbiddenMatchSpec(BaseModel):
    """池间禁配：两个池中指定实例禁止出现在同一装配。

    batch 省略时对该工件序号在所有来源批次中的实例生效。
    """

    pool_a: str = Field(..., min_length=1)
    batch_a: int | None = Field(None, ge=1)
    serial_a: str = Field(..., min_length=1)
    pool_b: str = Field(..., min_length=1)
    batch_b: int | None = Field(None, ge=1)
    serial_b: str = Field(..., min_length=1)


class AssemblyTaskCreate(BaseModel):
    """创建选择性装配任务（版本 1）：多批次取数、池映射、规则与求解。"""

    name: str = Field(..., min_length=1)
    note: str = ""
    batch_ids: list[int] = Field(
        ..., min_length=1,
        description="取数的冻结检验批次 id，必须同属该基线链",
    )
    pools: list[PoolSpec] = Field(..., min_length=1)
    assembly_count: int = Field(..., ge=1, description="要求装配数量")
    target_gap: TargetGapSpec
    cross_batch_limit: int | None = Field(
        None, ge=1,
        description="每个装配允许的最大不同来源批次数；缺省 = 零件池数（不限）",
    )
    same_batch_groups: list[SameBatchGroupSpec] = Field(default_factory=list)
    forbidden_matches: list[ForbiddenMatchSpec] = Field(default_factory=list)
    output_coverage_factor: float = Field(
        2.0, gt=0, description="组合扩展不确定度的覆盖因子 k_out"
    )
    guard_band: GuardBandSpec | None = Field(
        None, description="任务保护带；缺省 mode=multiple, multiple=1.0"
    )
    measurement_mc_samples: int = Field(
        50_000, ge=1_000, le=1_000_000,
        description="结果组合蒙特卡洛复核样本数（每实例独立误差列）",
    )
    random_seed: int = Field(
        20260913, ge=0, description="蒙特卡洛复核固定随机种子"
    )

    @model_validator(mode="after")
    def _basic_checks(self) -> "AssemblyTaskCreate":
        if len(set(self.batch_ids)) != len(self.batch_ids):
            dup = sorted({b for b in self.batch_ids
                          if self.batch_ids.count(b) > 1})
            raise ValueError(f"来源批次重复列出: {dup}")
        return self


class LockedAssemblySpec(BaseModel):
    """确认锁定的装配：每池一个 (batch_id, serial) 实例。"""

    members: list["LockedMemberSpec"] = Field(..., min_length=1)
    note: str = ""


class LockedMemberSpec(BaseModel):
    pool: str = Field(..., min_length=1)
    batch_id: int = Field(..., ge=1)
    serial: str = Field(..., min_length=1)


class AssemblyVersionCreate(BaseModel):
    """基于父版本另建版本：锁定确认组合，重排其余实例。"""

    note: str = ""
    locked_assemblies: list[LockedAssemblySpec] = Field(
        ..., min_length=1,
        description="本版本新增确认锁定的装配（在父版本已锁定集合之上）",
    )
    output_coverage_factor: float | None = Field(None, gt=0)
    guard_band: GuardBandSpec | None = None
    measurement_mc_samples: int | None = Field(None, ge=1_000, le=1_000_000)
    random_seed: int | None = Field(None, ge=0)


LockedAssemblySpec.model_rebuild()
