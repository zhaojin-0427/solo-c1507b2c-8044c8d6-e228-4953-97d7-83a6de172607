"""单位换算：所有内部计算统一为毫米 (mm)。

支持机械工程常见长度单位。单边/混合单位的尺寸在规范化时逐项换算，
原始数值与单位始终保留在 ``original`` 字段中。
"""
from __future__ import annotations

from enum import Enum


class LengthUnit(str, Enum):
    MM = "mm"
    CM = "cm"
    M = "m"
    IN = "in"
    MIL = "mil"        # 千分之一英寸
    UM = "um"          # 微米


# 各单位相对毫米的乘子：value_mm = value * TO_MM[unit]
TO_MM: dict[str, float] = {
    LengthUnit.MM.value: 1.0,
    LengthUnit.CM.value: 10.0,
    LengthUnit.M.value: 1000.0,
    LengthUnit.IN.value: 25.4,
    LengthUnit.MIL.value: 25.4e-3,
    LengthUnit.UM.value: 1e-3,
}


class UnitError(ValueError):
    """不支持或不一致的单位。"""


def to_mm(value: float, unit: str) -> float:
    try:
        factor = TO_MM[unit]
    except KeyError as exc:
        raise UnitError(f"不支持的长度单位: {unit!r}") from exc
    return value * factor


def from_mm(value_mm: float, unit: str) -> float:
    try:
        factor = TO_MM[unit]
    except KeyError as exc:
        raise UnitError(f"不支持的长度单位: {unit!r}") from exc
    return value_mm / factor
