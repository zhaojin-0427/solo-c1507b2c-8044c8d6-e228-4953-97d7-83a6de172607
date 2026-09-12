"""量具 R&R 研究（双因素随机效应 ANOVA，平衡交叉设计）。

模型与口径
==========
对基线链上某一尺寸，调用方提交 p 个零件 × o 名操作者 × r 轮重复的
**完整平衡交叉表**（每个零件由所有操作者等次数测量）：

    y_ijk = μ + P_i + O_j + (PO)_ij + ε_ijk      （全部随机效应）

均方与方差分量估计：

    σ²_重复性(EV) = MS_e
    σ²_交互       = (MS_PO − MS_e) / r
    σ²_操作者     = (MS_O − MS_PO) / (p·r)
    σ²_零件       = (MS_P − MS_PO) / (o·r)

负方差分量**截为零**（原估计保留在 raw_estimate 中）。再现性
AV = 操作者 + 交互（截零后合成），总量具 R&R = 重复性 + 再现性，
总变差 = GRR + 零件间。

指标（AIAG MSA 口径）
=====================
* 方差占比 %contribution = 100·σ²_c/σ²_总
* %Study Variation       = 100·σ_c/σ_总（研究变异倍数在比值中约去）
* %Tolerance             = 100·k·σ_c/过程公差（k 默认 5.15，对应正态 99% 散布）
* ndc = ⌊1.41·σ_零件/σ_GRR⌋（可区分类别数，≥5 视为可接受）

置信区间
========
固定种子非参数 bootstrap：对**零件整簇**有放回重采样（保留该零件完整的
操作者×重复 交叉表，操作者与重复结构不变），每次重算截零后的方差分量与
各项指标，取 2.5%/97.5% 分位为 95% CI；同时统计各分量成为主要变差来源
（最大方差分量）的频率。重采样中 σ_GRR=0 时 ndc 无界（记 NaN，不计入分位）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .measurement import dependency_versions
from .units import to_mm

NDC_FACTOR = 1.41  # AIAG：ndc = 1.41·(PV/GRR)，1.41 ≈ √2

# bootstrap 分块上限：B×p×o×r 超过该值时分块计算，避免一次性占满内存
_BOOT_CELL_CAP = 2_000_000

# 参与「主要变差来源」评比的四个基础分量（键序即 bootstrap 频次的键序）
BASE_COMPONENTS = ("repeatability", "operator", "interaction", "part_to_part")

COMPONENT_LABELS = {
    "repeatability": "设备重复性 (EV)",
    "operator": "操作者间（再现性分量）",
    "interaction": "零件×操作者交互（再现性分量）",
    "reproducibility": "再现性 (AV = 操作者 + 交互)",
    "total_gage_rr": "总量具 R&R (GRR = 重复性 + 再现性)",
    "part_to_part": "零件间变差 (PV)",
    "total": "总变差 (TV = GRR + PV)",
}

RR_FORMULAS = [
    "模型：y_ijk = μ + P_i + O_j + (PO)_ij + ε_ijk"
    "（双因素随机效应，含交互，p×o×r 平衡交叉设计）",
    "SS_零件 = o·r·Σ(x̄_i..−x̄...)²；SS_操作者 = p·r·Σ(x̄_.j.−x̄...)²",
    "SS_交互 = r·ΣΣ(x̄_ij.−x̄_i..−x̄_.j.+x̄...)²；"
    "SS_重复性 = ΣΣΣ(x_ijk−x̄_ij.)²",
    "自由度：零件 p−1，操作者 o−1，交互 (p−1)(o−1)，重复性 p·o·(r−1)",
    "σ²_重复性 = MS_e；σ²_交互 = (MS_PO−MS_e)/r；"
    "σ²_操作者 = (MS_O−MS_PO)/(p·r)；σ²_零件 = (MS_P−MS_PO)/(o·r)",
    "负方差分量截为零（原估计保留在 raw_estimate，truncated_to_zero 标记）",
    "σ²_再现性 = σ²_操作者 + σ²_交互（截零后合成）；"
    "σ²_GRR = σ²_重复性 + σ²_再现性；σ²_总 = σ²_GRR + σ²_零件",
    "方差占比 = 100·σ²_c/σ²_总；%Study Variation = 100·σ_c/σ_总"
    "（研究变异倍数在比值中约去）",
    "%Tolerance = 100·k·σ_c/过程公差（k 为研究变异倍数，默认 5.15）",
    "ndc = ⌊1.41·σ_零件/σ_GRR⌋（可区分类别数，≥5 视为可接受；"
    "σ_GRR=0 时 ndc 无界）",
    "bootstrap：固定种子对零件整簇有放回重采样（保留该零件完整 "
    "操作者×重复 交叉表），重算截零分量与指标，"
    "取 2.5%/97.5% 分位为 95% CI",
]


# ------------------------------------------------------------- 交叉表装配

def _build_cube(payload) -> tuple[np.ndarray, list[str], list[str]]:
    """把校验后的测量记录装配为 (p, o, r) 平衡立方体（单位换算为 mm）。

    零件 / 操作者按标识排序；单元内重复轮次的次序不影响 ANOVA。
    """
    parts = sorted({m.part for m in payload.measurements})
    operators = sorted({m.operator for m in payload.measurements})
    u = payload.unit.value if hasattr(payload.unit, "value") else str(payload.unit)
    cell: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for m in payload.measurements:
        cell.setdefault((m.part, m.operator), []).append((m.replicate, m.value))
    r = len(next(iter(cell.values())))
    cube = np.empty((len(parts), len(operators), r))
    for i, p in enumerate(parts):
        for j, o in enumerate(operators):
            reps = sorted(cell[(p, o)])
            cube[i, j, :] = [to_mm(v, u) for _, v in reps]
    return cube, parts, operators


# ------------------------------------------------------------- ANOVA 核心

def _anova(cb: np.ndarray) -> dict[str, Any]:
    """双因素随机效应 ANOVA（平衡设计）。cb 形状 (..., p, o, r)。

    返回自由度、SS、MS 与四个方差分量的**原始估计**（可负）；
    支持前导批维（bootstrap 向量化重算）。
    """
    p, o, r = cb.shape[-3:]
    grand = cb.mean(axis=(-3, -2, -1))
    part_m = cb.mean(axis=(-2, -1))            # (..., p)
    op_m = cb.mean(axis=(-3, -1))              # (..., o)
    cell_m = cb.mean(axis=-1)                  # (..., p, o)
    ss_part = o * r * ((part_m - grand[..., None]) ** 2).sum(axis=-1)
    ss_op = p * r * ((op_m - grand[..., None]) ** 2).sum(axis=-1)
    ss_inter = r * ((cell_m - part_m[..., None]
                     - op_m[..., None, :]
                     + grand[..., None, None]) ** 2).sum(axis=(-2, -1))
    ss_err = ((cb - cell_m[..., None]) ** 2).sum(axis=(-3, -2, -1))
    df = (p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1))
    ms_part = ss_part / df[0]
    ms_op = ss_op / df[1]
    ms_inter = ss_inter / df[2]
    ms_err = ss_err / df[3]
    return {
        "df": df,
        "ss": (ss_part, ss_op, ss_inter, ss_err),
        "ms": (ms_part, ms_op, ms_inter, ms_err),
        "raw": {
            "repeatability": ms_err,
            "interaction": (ms_inter - ms_err) / r,
            "operator": (ms_op - ms_inter) / (p * r),
            "part_to_part": (ms_part - ms_inter) / (o * r),
        },
    }


def _component_block(var: float, var_total: float, tol_mm: float,
                     k: float) -> dict[str, Any]:
    """单个分量的指标块：标准差、方差占比、%Study Variation、%Tolerance。"""
    std = math.sqrt(max(var, 0.0))
    return {
        "variance_mm2": var,
        "std_mm": std,
        "variance_contribution_pct": (
            100.0 * var / var_total if var_total > 0 else None
        ),
        "study_variation_pct": (
            100.0 * std / math.sqrt(var_total) if var_total > 0 else None
        ),
        "tolerance_pct": 100.0 * k * std / tol_mm,
    }


def _ci95(values: np.ndarray) -> list[float] | None:
    """2.5%/97.5% 分位（忽略 NaN 退化重采样）；全部退化时返回 None。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975))]


def _bootstrap(cube: np.ndarray, tol_mm: float, k: float,
               b_samples: int, seed: int) -> dict[str, Any]:
    """固定种子零件整簇 bootstrap：重算截零分量与指标，给出 95% CI。"""
    p, o, r = cube.shape
    rng = np.random.default_rng(seed)
    grr_std = np.empty(b_samples)
    sv_pct = np.empty(b_samples)
    tol_pct = np.empty(b_samples)
    ndc = np.empty(b_samples)
    dom = np.empty(b_samples, dtype=int)

    chunk = max(1, _BOOT_CELL_CAP // (p * o * r))
    done = 0
    while done < b_samples:
        b = min(chunk, b_samples - done)
        idx = rng.integers(0, p, size=(b, p))
        raw = _anova(cube[idx])["raw"]
        v_rep = raw["repeatability"]
        v_int = np.maximum(raw["interaction"], 0.0)
        v_op = np.maximum(raw["operator"], 0.0)
        v_part = np.maximum(raw["part_to_part"], 0.0)
        v_grr = v_rep + v_op + v_int
        v_total = v_grr + v_part
        s_grr = np.sqrt(v_grr)
        grr_std[done:done + b] = s_grr
        sv_pct[done:done + b] = np.where(
            v_total > 0,
            100.0 * s_grr / np.where(v_total > 0, np.sqrt(v_total), 1.0),
            np.nan)
        tol_pct[done:done + b] = 100.0 * k * s_grr / tol_mm
        ndc[done:done + b] = np.where(
            s_grr > 0,
            np.floor(NDC_FACTOR * np.sqrt(v_part)
                     / np.where(s_grr > 0, s_grr, 1.0)),
            np.nan)
        stacked = np.stack([v_rep, v_op, v_int, v_part], axis=1)
        dom[done:done + b] = stacked.argmax(axis=1)
        done += b

    return {
        "method": "非参数 bootstrap：固定种子对零件整簇有放回重采样"
                  "（保留该零件完整 操作者×重复 交叉表，"
                  "操作者与重复结构不变），每次重算截零后的方差分量与指标",
        "samples": b_samples,
        "random_seed": seed,
        "ci95": {
            "total_gage_std_mm": _ci95(grr_std),
            "study_variation_pct": _ci95(sv_pct),
            "tolerance_pct": _ci95(tol_pct),
            "ndc": _ci95(ndc),
        },
        "dominant_source_frequency": {
            key: float((dom == i).mean())
            for i, key in enumerate(BASE_COMPONENTS)
        },
        "ndc_unbounded_fraction": float(np.isnan(ndc).mean()),
        "zero_total_variance_fraction": float(np.isnan(sv_pct).mean()),
    }


# ------------------------------------------------------------- 研究总装

def run_study(payload) -> dict[str, Any]:
    """对已校验的研究请求做完整 ANOVA 与 bootstrap（数值单位均为 mm）。"""
    cube, parts, operators = _build_cube(payload)
    p, o, r = cube.shape
    u = payload.unit.value if hasattr(payload.unit, "value") else str(payload.unit)
    tol_mm = to_mm(payload.process_tolerance, u)
    k = float(payload.study_variation_multiplier)

    an = _anova(cube)
    df = an["df"]
    ss = [float(x) for x in an["ss"]]
    ms = [float(x) for x in an["ms"]]
    raw = {key: float(val) for key, val in an["raw"].items()}
    used = {key: max(0.0, val) for key, val in raw.items()}

    v_rep = used["repeatability"]
    v_op = used["operator"]
    v_int = used["interaction"]
    v_part = used["part_to_part"]
    v_av = v_op + v_int
    v_grr = v_rep + v_av
    v_total = v_grr + v_part

    variances = {
        "repeatability": v_rep,
        "operator": v_op,
        "interaction": v_int,
        "reproducibility": v_av,
        "total_gage_rr": v_grr,
        "part_to_part": v_part,
        "total": v_total,
    }
    components = {
        key: {**_component_block(var, v_total, tol_mm, k),
              "label": COMPONENT_LABELS[key]}
        for key, var in variances.items()
    }
    s_grr = components["total_gage_rr"]["std_mm"]
    s_part = components["part_to_part"]["std_mm"]

    if s_grr > 0:
        ndc_value: int | None = int(math.floor(NDC_FACTOR * s_part / s_grr))
        ndc_note = None
    else:
        ndc_value = None
        ndc_note = "σ_GRR = 0：量具变差为零，ndc 无界（可区分类别数任意大）"

    # 主要变差来源：四个基础分量中方差最大者（总变差为零时无意义）
    four = {key: used[key] for key in BASE_COMPONENTS}
    if v_total > 0:
        dom_key = max(four, key=lambda kk: four[kk])
        if dom_key == "part_to_part":
            dom_note = "最大变差来源为零件间差异：量具变差相对零件变差较小"
        else:
            dom_note = ("最大变差来源为量具（重复性/再现性）而非零件间差异："
                        "测量系统是当前主要瓶颈")
        dominant = {
            "component": dom_key,
            "label": COMPONENT_LABELS[dom_key],
            "variance_mm2": four[dom_key],
            "variance_share_of_total_pct": 100.0 * four[dom_key] / v_total,
            "note": dom_note,
        }
    else:
        dominant = {
            "component": None,
            "label": None,
            "variance_mm2": 0.0,
            "variance_share_of_total_pct": None,
            "note": "总变差为零：各分量均为零，无主要变差来源",
        }

    sv_grr = components["total_gage_rr"]["study_variation_pct"]
    if sv_grr is None:
        acceptance = "总变差为零，无法评估量具占比"
    elif sv_grr < 10.0:
        acceptance = "量具可接受（总量具 %Study Variation < 10%）"
    elif sv_grr <= 30.0:
        acceptance = ("条件可接受（10% ≤ 总量具 %Study Variation ≤ 30%），"
                      "视应用重要性与改进成本而定")
    else:
        acceptance = "量具不可接受（总量具 %Study Variation > 30%），需改进测量系统"
    if ndc_value is not None and ndc_value < 5:
        acceptance += f"；ndc={ndc_value} < 5，可区分类别数不足"

    boot = _bootstrap(cube, tol_mm, k, payload.bootstrap_samples,
                      payload.random_seed)

    variance_components: dict[str, Any] = {
        key: {
            "raw_estimate": raw[key],
            "used": used[key],
            "truncated_to_zero": raw[key] < 0.0,
        }
        for key in BASE_COMPONENTS
    }
    variance_components["reproducibility"] = {
        "used": v_av, "definition": "σ²_操作者 + σ²_交互（截零后合成）"}
    variance_components["total_gage_rr"] = {
        "used": v_grr, "definition": "σ²_重复性 + σ²_再现性"}
    variance_components["total"] = {
        "used": v_total, "definition": "σ²_GRR + σ²_零件"}

    return {
        "dimension_id": payload.dimension_id,
        "unit_submitted": u,
        "process_tolerance": {
            "submitted": payload.process_tolerance,
            "unit": u,
            "mm": tol_mm,
        },
        "study_variation_multiplier": k,
        "design": {
            "parts": parts,
            "operators": operators,
            "n_parts": p,
            "n_operators": o,
            "n_replicates": r,
            "cells": p * o,
            "total_measurements": p * o * r,
            "balanced": True,
        },
        "grand_mean_mm": float(cube.mean()),
        "anova": {
            "model": "y_ijk = μ + P_i + O_j + (PO)_ij + ε_ijk"
                     "（双因素随机效应，含交互，平衡交叉设计）",
            "degrees_of_freedom": {
                "part": df[0],
                "operator": df[1],
                "interaction": df[2],
                "repeatability": df[3],
                "total": p * o * r - 1,
            },
            "sum_of_squares_mm2": {
                "part": ss[0],
                "operator": ss[1],
                "interaction": ss[2],
                "repeatability": ss[3],
                "total": ss[0] + ss[1] + ss[2] + ss[3],
            },
            "mean_squares_mm2": {
                "part": ms[0],
                "operator": ms[1],
                "interaction": ms[2],
                "repeatability": ms[3],
            },
        },
        "variance_components_mm2": variance_components,
        "components": components,
        "total_gage_std_mm": s_grr,
        "ndc": ndc_value,
        "ndc_note": ndc_note,
        "dominant_source": dominant,
        "acceptance_note": acceptance,
        "bootstrap": boot,
        "formulas": RR_FORMULAS,
        "dependency_versions": dependency_versions(),
    }
