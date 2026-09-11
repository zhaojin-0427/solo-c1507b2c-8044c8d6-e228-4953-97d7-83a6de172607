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
    """创建来料检验批次：选定基线链，逐工件行提交实测值。"""

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
