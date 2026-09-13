"""孔系装配分析 Pydantic 模型：请求结构与创建期校验。

校验内容（对象级，问题一律指出对象并拒绝创建，端点转 422）：
* 匹配位两侧要素：名义坐标、孔径 / 销（或螺栓）直径极限、位置度与
  MMC / LMC / RFS 实体条件；尺寸极限倒置（lower > upper）拒绝；
* 要素类型组合：至多一侧为外要素（销 / 螺栓）；固定螺栓直径可取代
  B 侧孔要素；无任何内要素（无法容纳连接件）拒绝；
* 位置度基准引用必须有效：引用的基准标签必须存在于本侧基准框架、
  引用次序必须是基准优先次序的连续前缀（无跳级、无重复）；
* 基准框架：order 连续不重复、各基准约束的 2D 自由度合起来恰好覆盖
  tx/ty/rz（不缺不重）、两条平移基准边线平行（基准退化）拒绝；
* 匹配缺失（id 重复 / 空孔系 / 坐标非有限）拒绝。

长度单位复用全局 LengthUnit（内部统一 mm），角度一律以度提交。
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .schemas import DistributionType
from .units import LengthUnit

# 2D 基准框架可约束的刚体自由度
_DOF_ALL = {"tx", "ty", "rz"}


class MaterialCondition(str, Enum):
    """要素 / 基准的实体条件（GD&T 修饰符）。"""

    RFS = "RFS"    # Regardless of Feature Size，不要求实体状态（无补偿）
    MMC = "MMC"    # Maximum Material Condition（孔最小 / 销最大）
    LMC = "LMC"    # Least Material Condition（孔最大 / 销最小）


class FeatureKind(str, Enum):
    HOLE = "hole"   # 内要素（孔）
    PIN = "pin"    # 外要素（圆柱销 / 轴）


class DatumKind(str, Enum):
    SIZE = "size"  # 尺寸基准（孔 / 销等可漂移基准要素）
    EDGE = "edge"  # 边线基准（面贴合，平移基准 RFS 无漂移）


# 平移基准允许约束的自由度组合
_TRANSLATION_DOF = frozenset({frozenset({"tx"}), frozenset({"ty"}),
                              frozenset({"tx", "ty"})})


class MateFeature(BaseModel):
    """匹配位一侧的孔 / 销要素（B 侧为固定螺栓时可为 None）。"""

    kind: FeatureKind = Field(..., description="hole=内要素，pin=外要素")
    x: float = Field(..., description="基准框架下名义中心坐标 x（与 unit 同单位）")
    y: float = Field(..., description="基准框架下名义中心坐标 y（与 unit 同单位）")
    nominal_diameter: float = Field(..., gt=0, description="名义直径（>0）")
    diameter_upper_deviation: float = Field(
        ..., description="直径上偏差 ES（与 unit 同单位，孔通常 ≥0）")
    diameter_lower_deviation: float = Field(
        ..., description="直径下偏差 EI（与 unit 同单位，孔通常 ≤0）")
    position_tolerance: float = Field(
        ..., ge=0, description="位置度公差带直径 t（与 unit 同单位；0 = 无位置度）")
    material_condition: MaterialCondition = Field(
        MaterialCondition.MMC, description="位置度的实体条件修饰符")
    distribution: DistributionType = DistributionType.NORMAL
    diameter_std_dev: float | None = Field(
        None, ge=0, description="直径样本标准差（与 unit 同单位）；"
                                "normal 必填，其它分布缺省按公差带理论标准差")
    position_std_dev: float | None = Field(
        None, ge=0, description="位置误差径向标准差（与 unit 同单位）；"
                                "缺省按公差带理论标准差（normal 取 t/6）")
    position_datum_refs: list[str] = Field(
        default_factory=list,
        description="位置度基准引用（按优先次序，须为该侧框架的连续前缀）")

    @model_validator(mode="after")
    def _check(self) -> "MateFeature":
        if len(set(self.position_datum_refs)) != len(self.position_datum_refs):
            raise ValueError(
                f"{self.kind.value}@({self.x:g},{self.y:g}) "
                f"位置度基准引用重复: {self.position_datum_refs}")
        for name in ("x", "y", "nominal_diameter",
                     "diameter_upper_deviation", "diameter_lower_deviation",
                     "position_tolerance"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"要素坐标 / 尺寸必须为有限数值，字段 {name}={v!r}")
        if self.diameter_lower_deviation > self.diameter_upper_deviation:
            raise ValueError(
                f"{self.kind.value}@({self.x:g},{self.y:g}) 直径极限倒置："
                f"下偏差 {self.diameter_lower_deviation:g} > "
                f"上偏差 {self.diameter_upper_deviation:g}")
        if self.distribution == DistributionType.NORMAL and \
                self.diameter_std_dev is None:
            raise ValueError(
                f"{self.kind.value}@({self.x:g},{self.y:g}) normal 分布必须"
                "提供 diameter_std_dev")
        return self


class MateSpec(BaseModel):
    """一个匹配位：一对零件上的孔 / 销（或螺栓）。"""

    id: str = Field(..., min_length=1, description="匹配位唯一标识")
    feature_a: MateFeature = Field(..., description="零件 A（孔系）侧要素")
    # 零件 B（连接件）侧要素；bolt_diameter_limits 给出时允许为空（固定螺栓）
    feature_b: MateFeature | None = None
    bolt_diameter_upper: float | None = Field(
        None, description="固定螺栓直径上极限（与 unit 同单位）；给出时 "
                          "feature_b 必须为空且按无公差销处理")
    bolt_diameter_lower: float | None = Field(
        None, description="固定螺栓直径下极限（与 unit 同单位）")

    @model_validator(mode="after")
    def _check(self) -> "MateSpec":
        if self.bolt_diameter_lower is not None and \
                self.bolt_diameter_upper is not None:
            if not (math.isfinite(self.bolt_diameter_lower)
                    and math.isfinite(self.bolt_diameter_upper)):
                raise ValueError(f"匹配位 {self.id}: 螺栓直径极限必须为有限数值")
            if self.bolt_diameter_lower > self.bolt_diameter_upper:
                raise ValueError(
                    f"匹配位 {self.id}: 螺栓直径极限倒置：下限 "
                    f"{self.bolt_diameter_lower:g} > 上限 "
                    f"{self.bolt_diameter_upper:g}")
            if self.bolt_diameter_lower <= 0:
                raise ValueError(
                    f"匹配位 {self.id}: 螺栓直径下极限必须为正")
        bolt_given = self.bolt_diameter_upper is not None
        if bolt_given != (self.bolt_diameter_lower is not None):
            raise ValueError(
                f"匹配位 {self.id}: bolt_diameter_upper/lower 必须同时给出")
        if self.feature_b is None and not bolt_given:
            raise ValueError(
                f"匹配位 {self.id}: feature_b 缺失且未给螺栓直径极限"
                "（匹配缺失：B 侧既无孔要素也无连接件）")
        feats = [self.feature_a] + ([self.feature_b]
                                    if self.feature_b is not None else [])
        ext = [f for f in feats if f.kind == FeatureKind.PIN]
        if len(ext) > 1:
            raise ValueError(
                f"匹配位 {self.id}: 两侧同时为外要素（pin/pin 或 pin+螺栓），"
                "不存在可容纳连接件的内要素")
        # 浮动连接件（两个内要素穿过自由螺栓）必须同时给 feature_b 与螺栓极限；
        # 螺栓极限不允许与 B 侧外要素同时声明。
        both_holes = self.feature_a.kind == FeatureKind.HOLE and (
            self.feature_b is not None
            and self.feature_b.kind == FeatureKind.HOLE)
        if bolt_given and self.feature_b is not None and not both_holes:
            raise ValueError(
                f"匹配位 {self.id}: 螺栓直径极限只能与双孔（浮动连接件）一起"
                "声明；固定螺栓请省略 feature_b，固定销请省略螺栓极限")
        if both_holes and not bolt_given:
            raise ValueError(
                f"匹配位 {self.id}: 两侧均为孔但未给螺栓直径极限"
                "（浮动连接件必须声明 bolt_diameter_upper/lower）")
        return self


class DatumFeatureSpec(BaseModel):
    """基准框架中的单个基准要素。"""

    label: str = Field(..., min_length=1, description="基准标签（如 'A'、'B'）")
    order: int = Field(..., ge=1, le=3, description="基准优先次序（1=主基准）")
    kind: DatumKind = DatumKind.SIZE
    constrains: list[Literal["tx", "ty", "rz"]] = Field(
        ..., min_length=1, max_length=2, description="该基准在 2D 装配中约束的自由度")
    material_condition: MaterialCondition = Field(
        MaterialCondition.RFS,
        description="基准要素实体条件；MMB/LMB 产生基准偏移（基准漂移）")
    # size 基准要素尺寸（与 unit 同单位）
    nominal_diameter: float | None = Field(None, gt=0)
    diameter_upper_deviation: float | None = None
    diameter_lower_deviation: float | None = None
    distribution: DistributionType = DistributionType.UNIFORM
    diameter_std_dev: float | None = Field(None, ge=0)
    # rz 基准要素相对框架原点的力臂（与 unit 同单位）
    lever_radius: float | None = Field(
        None, gt=0, description="约束 rz 的尺寸基准力臂（>0）")
    # edge 基准：单位法向（指向零件外侧），基准线 n·p = offset
    normal_x: float | None = Field(None, description="边线单位法向 x 分量")
    normal_y: float | None = Field(None, description="边线单位法向 y 分量")
    line_offset: float = Field(0.0, description="边线有向距离 n·p = offset")
    # 边线基准自身的角度公差（度，单边半宽）；RFS 平移基准不产生漂移
    angular_half_range_deg: float | None = Field(
        None, ge=0, description="边线方向公差（度，单边半宽）；约束 rz 时必填")

    @model_validator(mode="after")
    def _check(self) -> "DatumFeatureSpec":
        dofs = list(self.constrains)
        if len(set(dofs)) != len(dofs):
            raise ValueError(f"基准 {self.label}: 约束自由度重复 {dofs}")
        dset = frozenset(dofs)
        if self.kind == DatumKind.EDGE:
            if self.material_condition != MaterialCondition.RFS:
                raise ValueError(
                    f"基准 {self.label}: 边线基准只支持 RFS（贴合面无尺寸漂移）")
            if self.normal_x is None or self.normal_y is None:
                raise ValueError(
                    f"基准 {self.label}: 边线基准必须给出 normal_x/normal_y")
            norm = math.hypot(self.normal_x, self.normal_y)
            if not math.isfinite(norm) or norm < 1e-9:
                raise ValueError(
                    f"基准 {self.label}: 边线法向为零向量或非有限")
            if "rz" in dset and (self.angular_half_range_deg is None):
                raise ValueError(
                    f"基准 {self.label}: 约束 rz 的边线基准必须给出 "
                    "angular_half_range_deg")
            if dset not in _TRANSLATION_DOF and dset != frozenset({"rz"}):
                raise ValueError(
                    f"基准 {self.label}: 边线基准只能约束平移自由度 "
                    f"(tx/ty/tx+ty) 或单独 rz，收到 {sorted(dset)}")
        else:  # size 基准
            if self.nominal_diameter is None:
                raise ValueError(
                    f"基准 {self.label}: 尺寸基准必须给出 nominal_diameter")
            if (self.diameter_upper_deviation is None
                    or self.diameter_lower_deviation is None):
                raise ValueError(
                    f"基准 {self.label}: 尺寸基准必须给出直径上下偏差")
            if self.diameter_lower_deviation > self.diameter_upper_deviation:
                raise ValueError(
                    f"基准 {self.label}: 基准要素直径极限倒置：下偏差 "
                    f"{self.diameter_lower_deviation:g} > 上偏差 "
                    f"{self.diameter_upper_deviation:g}")
            if dset not in _TRANSLATION_DOF and dset != frozenset({"rz"}):
                raise ValueError(
                    f"基准 {self.label}: 尺寸基准只能约束平移自由度 "
                    f"(tx/ty/tx+ty) 或单独 rz，收到 {sorted(dset)}")
            if "rz" in dset:
                if self.lever_radius is None:
                    raise ValueError(
                        f"基准 {self.label}: 约束 rz 的尺寸基准必须给出 "
                        "lever_radius（力臂）")
            if self.distribution == DistributionType.NORMAL and \
                    self.diameter_std_dev is None:
                raise ValueError(
                    f"基准 {self.label}: normal 分布必须提供 diameter_std_dev")
        return self


class DatumFrameSpec(BaseModel):
    """一个零件的基准框架（有序基准要素列表；允许为空 = 无基准约束）。"""

    datums: list[DatumFeatureSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "DatumFrameSpec":
        labels = [d.label for d in self.datums]
        if len(set(labels)) != len(labels):
            dup = sorted({x for x in labels if labels.count(x) > 1})
            raise ValueError(f"基准标签重复: {dup}")
        orders = sorted(d.order for d in self.datums)
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError(
                f"基准次序必须从 1 开始连续不重复，收到 {orders}")

        # 自由度必须恰好覆盖 tx/ty/rz：不缺、不重
        seen: set[str] = set()
        for d in self.datums:
            overlap = seen & set(d.constrains)
            if overlap:
                raise ValueError(
                    f"基准退化/重复约束：自由度 {sorted(overlap)} 已被先前"
                    f"基准约束，基准 {d.label} 再次约束")
            seen.update(d.constrains)
        missing = _DOF_ALL - seen
        if missing and self.datums:
            raise ValueError(
                f"基准框架欠约束：自由度 {sorted(missing)} 未被任何基准约束"
                "（基准退化；如允许自由平移/旋转请显式省略相应基准）")
        # 两条只约束单一平移方向的边线：法向平行 => 退化（重复约束同一方向）
        edges = [d for d in self.datums
                 if d.kind == DatumKind.EDGE
                 and frozenset(d.constrains) in
                 (frozenset({"tx"}), frozenset({"ty"}))]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                a, b = edges[i], edges[j]
                cross = abs(a.normal_x * b.normal_y - a.normal_y * b.normal_x)
                if cross < 1e-9:
                    raise ValueError(
                        f"基准退化：边线基准 {a.label} 与 {b.label} 法向平行，"
                        "不能同时约束两个平移方向")
        return self

    def labels(self) -> list[str]:
        return [d.label for d in sorted(self.datums, key=lambda d: d.order)]


class _HolePatternFields(BaseModel):
    """孔系版本的公共字段（首版 / 新版本共用）。"""

    name: str = Field(..., min_length=1)
    note: str = ""
    unit: LengthUnit = LengthUnit.MM
    mates: list[MateSpec] = Field(..., min_length=1)
    frame_a: DatumFrameSpec = Field(
        default_factory=DatumFrameSpec, description="零件 A 基准框架")
    frame_b: DatumFrameSpec = Field(
        default_factory=DatumFrameSpec, description="零件 B 基准框架")
    mc_samples: int = Field(20_000, ge=1_000, le=1_000_000)
    random_seed: int = Field(20260913, ge=0)
    theta_search_deg: float = Field(
        5.0, gt=0, le=45.0, description="最坏边界转角搜索半宽（度）")
    theta_grid_points: int = Field(
        121, ge=11, le=721, description="最坏边界转角网格点数（奇数）")
    mc_theta_points: int = Field(
        9, ge=1, le=21, description="蒙特卡洛转角候选点数（含 0）")

    @model_validator(mode="after")
    def _validate_pattern(self) -> "_HolePatternFields":
        ids = [m.id for m in self.mates]
        if len(set(ids)) != len(ids):
            dup = sorted({x for x in ids if ids.count(x) > 1})
            raise ValueError(f"匹配位 id 重复（匹配缺失）: {dup}")

        for side_name, frame, picker in (
            ("A", self.frame_a, lambda m: m.feature_a),
            ("B", self.frame_b, lambda m: m.feature_b),
        ):
            labels = set(frame.labels())
            for mate in self.mates:
                feat = picker(mate)
                if feat is None:
                    continue
                refs = feat.position_datum_refs  # 属性见下方模型扩展点
                if not refs:
                    continue
                if not frame.datums:
                    raise ValueError(
                        f"匹配位 {mate.id} 侧 {side_name} 的位置度引用了基准 "
                        f"{refs}，但该侧基准框架为空（位置度引用无效）")
                unknown = [r for r in refs if r not in labels]
                if unknown:
                    raise ValueError(
                        f"匹配位 {mate.id} 侧 {side_name} 的位置度引用了"
                        f"不存在的基准 {unknown}；该侧基准: {frame.labels()}"
                        "（位置度引用无效）")
                order = {d.label: d.order for d in frame.datums}
                ref_orders = [order[r] for r in refs]
                expected = list(range(1, len(refs) + 1))
                if sorted(ref_orders) != expected:
                    raise ValueError(
                        f"匹配位 {mate.id} 侧 {side_name} 的位置度基准引用 "
                        f"{refs} 不是基准优先次序的连续前缀（次序 "
                        f"{sorted(ref_orders)}，应为 {expected}；跳级或乱序）")
                if len(set(refs)) != len(refs):
                    raise ValueError(
                        f"匹配位 {mate.id} 侧 {side_name} 的位置度基准引用重复")
        return self


class HolePatternCreate(_HolePatternFields):
    """创建孔系装配分析（版本 1）。"""


class HolePatternVersionCreate(_HolePatternFields):
    """在同一孔系对象下另建独立版本（完整新定义随版本冻结）。"""

    parent_version_id: int | None = None


# ------------------------------------------------------- 整改方案搜索

class DrillOption(BaseModel):
    """候选钻孔直径（针对某个匹配位某侧孔要素）。"""

    mate_id: str
    side: Literal["a", "b"]
    nominal_diameter: float = Field(..., gt=0)
    diameter_upper_deviation: float = Field(..., )
    diameter_lower_deviation: float = Field(..., )

    @model_validator(mode="after")
    def _check(self) -> "DrillOption":
        if self.diameter_lower_deviation > self.diameter_upper_deviation:
            raise ValueError(
                f"匹配位 {self.mate_id} 侧 {self.side} 候选孔径极限倒置："
                f"{self.diameter_lower_deviation:g} > "
                f"{self.diameter_upper_deviation:g}")
        return self


class FastenerOption(BaseModel):
    """候选连接件（固定螺栓）直径规格。"""

    mate_id: str
    diameter_upper: float = Field(..., gt=0)
    diameter_lower: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _check(self) -> "FastenerOption":
        if self.diameter_lower > self.diameter_upper:
            raise ValueError(
                f"匹配位 {self.mate_id} 候选连接件直径极限倒置："
                f"{self.diameter_lower:g} > {self.diameter_upper:g}")
        return self


class HoleCorrectionOption(BaseModel):
    """允许的孔位修正（中心平移上界；针对 A 侧孔要素）。"""

    mate_id: str
    max_shift: float = Field(..., ge=0, description="允许孔位中心平移上界"
                                                    "（与孔系版本同一单位）")


class RemedySearchRequest(BaseModel):
    """从候选钻孔尺寸 / 连接件规格 / 孔位修正中搜索可采纳组合。"""

    name: str = Field(..., min_length=1)
    note: str = ""
    locked_mates: list[str] = Field(
        default_factory=list,
        description="锁定孔位的匹配位：不允许孔位修正（孔径 / 连接件仍可换）")
    locked_fasteners: list[str] = Field(
        default_factory=list,
        description="锁定连接件的匹配位：不允许更换连接件规格")
    drill_options: list[DrillOption] = Field(default_factory=list)
    fastener_options: list[FastenerOption] = Field(default_factory=list)
    correction_options: list[HoleCorrectionOption] = Field(default_factory=list)
    include_correction: bool = Field(
        True, description="是否允许使用声明的孔位修正")
    max_candidates: int = Field(200, ge=1, le=2000)
    mc_samples: int | None = Field(None, ge=1_000, le=1_000_000)
    random_seed: int | None = Field(None, ge=0)


class RemedySelectRequest(BaseModel):
    """采纳整改搜索结果中的某个候选（rank 从 1 开始）。"""

    rank: int = Field(..., ge=1)
    note: str = ""
