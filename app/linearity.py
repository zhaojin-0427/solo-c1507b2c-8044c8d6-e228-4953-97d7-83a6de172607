"""量具线性与偏倚研究引擎（有证标准件多点核查；内部统一 mm）。

数据模型
========
对基线链上的某一尺寸，用一件量具在工作量程内对 G 个**有证标准件**
（certified reference standards）各做 n_g 次重复读数。第 g 个核查点：

* x_g：标准件证书参考值（mm），标准不确定度 u_ref,g（Type B）；
* y_{gr}：第 r 次读数（mm），记录操作者与测量次序；
* 平均偏倚 b̄_g = mean_r(y_gr) − x_g。

逐点统计
========
* 重复性样本标准差 s_g = sqrt(Σ(y−ȳ)²/(n_g−1))（n_g ≥ 2）；
  n_g = 1 时用全研究合并（pooled-within）重复性 s_p：
      s_p² = Σ_g Σ_r (y_gr−ȳ)² / Σ_g (n_g−1)
  任何核查点都没有重复读数时重复性不可估计（记 0 并告警）。
* 读数均值的标准不确定度 u_rep,g = s_g/√n_g（单点时 s_p/√1）；
* 合成点标准不确定度（参考标准件与读数重复性独立，方和根）：
      u_g = sqrt(u_ref,g² + u_rep,g²)
* 平均偏倚 95% 置信区间 b̄_g ± t_{p,ν_g}·u_g，有效自由度用
  Welch–Satterthwaite：ν_g = u_g⁴·df_rep / u_rep,g⁴
  （u_ref 为 Type B，自由度 ∞；u_rep=0 时用正态 z）。
* 偏倚显著性：置信区间不含 0 即该参考点偏倚显著（α = 1−confidence_level）。

以标准不确定度加权的线性回归
============================
偏倚对参考值拟合（加权最小二乘，权重 w_g = 1/u_g²）：

    b_g = β0 + β1·x_g + ε_g,   β = (XᵀWX)⁻¹XᵀWb,   X_g = [1, x_g]

两套回归系数协方差并列返回：

* **GUM 协方差（含标准件不确定度）**：按已知点方差线性传播
  Cov_b = diag(u_g²)（各标准件独立），
      Cov_β = A·diag(u_g²)·Aᵀ,  A = (XᵀWX)⁻¹XᵀW
  权重恰为逆方差时 Cov_β = (XᵀWX)⁻¹；测量方案做偏倚修正时传播的
  「系数与标准件不确定度」即取此矩阵。
* **经典回归协方差（残差估计）**：s² = Σw_g e_g²/(G−2)，
  Cov_classic = s²·(XᵀWX)⁻¹，配合 t_{p,G−2} 做经典回归推断；
  G = 2 时残差自由度为 0，不给经典区间（拟合仍有效，是两点直线）。

系数显著性默认用 GUM 协方差的正态 z 检验（与不确定度口径一致）：
斜率显著表示偏倚随参考值线性变化（**存在线性**）；截距显著表示
零附近存在固定偏倚。另给加权失配检验 χ²_lof = Σ(e_g/u_g)²
（自由度 G−2）核查线性模型与所声明不确定度是否相容。

适用量程与禁止外推
==================
线性模型只在**被核查参考值覆盖的范围** [min x_g, max x_g] 内有效：
覆盖率 (x_max−x_min)/(工作量程上限−下限) 低于研究声明的
min_span_coverage 时标记 coverage_adequate=false；参考点不足 2 个、
参考值无变差或 (XᵀWX) 奇异时 regression_available=false 并给出
failure_reason。回归不可用或覆盖不足的研究不能被采用（adopt），
测量方案对落在适用量程之外的实测值不做偏倚修正（不外推）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .engine import normal_cdf
from .inspection import _gammp
from .units import LengthUnit, from_mm, to_mm

Z975 = 1.959963984540054  # 标准正态 97.5% 分位（大样本 / Type B 主导时）


def _dependency_versions() -> dict[str, str]:
    """响应追溯用：当前进程的关键依赖版本（与 measurement 模块同口径）。"""
    import sys

    import fastapi
    import pydantic
    import sqlalchemy

    return {
        "python": sys.version.split()[0],
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.__version__,
        "sqlalchemy": sqlalchemy.__version__,
        "numpy": np.__version__,
    }


class LinearityError(ValueError):
    """线性与偏倚研究输入错误（API 层映射为 422）。"""


# ------------------------------------------------------------- Student t 分位

def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b)（Numerical Recipes betacf）。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - front * _betacf(b, a, 1.0 - x)


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def t_critical(p: float, df: float) -> float:
    """Student t 的 p 分位（对称分布，二分反演 CDF）；df≤0 返回正态 z。"""
    if df is None or not math.isfinite(df) or df <= 0.0:
        # 正态分位（已知不确定度主导）
        return _norm_ppf(p)
    target = 2.0 * p - 1.0  # P(|T| ≤ t) = p → F(t)=p

    def f(t: float) -> float:
        # 学生 t CDF：F(t) = 1 − 0.5·I_{ν/(ν+t²)}(ν/2, 1/2)，t≥0
        x = df / (df + t * t)
        return 1.0 - 0.5 * _betai(df / 2.0, 0.5, x)

    lo, hi = 0.0, 1.0
    while f(hi) < p:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _norm_ppf(p: float) -> float:
    """标准正态分位（在 Φ 上二分；Φ 用 math.erf）。"""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    def phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- 数据装配

@dataclass
class _PointData:
    point_id: str
    standard_serial: str
    x_mm: float
    u_ref_mm: float
    ref_submitted: float
    ref_unit: str
    values_mm: list[float] = field(default_factory=list)
    reading_meta: list[dict[str, Any]] = field(default_factory=list)


def _build_points(payload, exclusions: set[tuple[str, str, int]] | None = None
                  ) -> list[_PointData]:
    """校验后的请求 → mm 单位的逐点数据；exclusions 中的读数被剔除。"""
    exclusions = exclusions or set()
    u_study = payload.unit.value if hasattr(payload.unit, "value") else str(payload.unit)
    lo = to_mm(payload.working_range_lower, u_study)
    hi = to_mm(payload.working_range_upper, u_study)
    points: list[_PointData] = []
    for p in payload.points:
        ru = p.reference_unit.value if hasattr(p.reference_unit, "value") \
            else str(p.reference_unit)
        x = to_mm(p.reference_value, ru)
        if not (lo - 1e-12 <= x <= hi + 1e-12):
            raise LinearityError(
                f"核查点 {p.point_id}（标准件 {p.standard_serial}）：参考值 "
                f"{p.reference_value} {ru}（{x:.6g} mm）超出量具工作量程 "
                f"[{payload.working_range_lower}, {payload.working_range_upper}] "
                f"{u_study}（[{lo:.6g}, {hi:.6g}] mm）"
            )
        pd = _PointData(
            point_id=p.point_id, standard_serial=p.standard_serial, x_mm=x,
            u_ref_mm=to_mm(p.reference_std_uncertainty, ru),
            ref_submitted=p.reference_value, ref_unit=ru)
        for r in p.readings:
            if (p.point_id, r.operator, r.replicate) in exclusions:
                continue
            pd.values_mm.append(to_mm(r.value, u_study))
            pd.reading_meta.append({
                "operator": r.operator, "replicate": r.replicate,
                "measurement_order": r.measurement_order,
                "value_submitted": r.value, "value_mm": to_mm(r.value, u_study),
            })
        points.append(pd)
    return [p for p in points if p.values_mm]


# ------------------------------------------------------------- 逐点统计

def _repeatability(points: list[_PointData]) -> tuple[float, int, list[float | None]]:
    """返回 (合并重复性 s_p mm, 合并自由度 Σ(n−1), 逐点 s_g 或 None)。"""
    ss = 0.0
    dfree = 0
    per_point: list[float | None] = []
    for p in points:
        if len(p.values_mm) >= 2:
            arr = np.asarray(p.values_mm, dtype=float)
            ss += float(((arr - arr.mean()) ** 2).sum())
            dfree += len(arr) - 1
            per_point.append(float(arr.std(ddof=1)))
        else:
            per_point.append(None)
    s_p = math.sqrt(ss / dfree) if dfree > 0 else 0.0
    return s_p, dfree, per_point


def _two_sided_p_normal(z: float) -> float:
    return 2.0 * (1.0 - float(normal_cdf(abs(z))))


def _chi2_sf(stat: float, df: int) -> float:
    """χ²(df) 生存函数 P(X>stat)（失配检验用）。"""
    if df <= 0 or stat <= 0.0:
        return 1.0
    return max(0.0, 1.0 - _gammp(df / 2.0, stat / 2.0))


# ------------------------------------------------------------- 研究总装

def run_study(payload, exclusions: list[dict[str, Any]] | None = None,
              copied_from: dict[str, Any] | None = None) -> dict[str, Any]:
    """对已通过 Pydantic 校验的研究请求执行逐点偏倚与加权线性回归。"""
    excl_set = {(e["point_id"], e["operator"], e["replicate"])
                for e in (exclusions or [])}
    points = _build_points(payload, excl_set)
    u_study = payload.unit.value if hasattr(payload.unit, "value") else str(payload.unit)
    lo = to_mm(payload.working_range_lower, u_study)
    hi = to_mm(payload.working_range_upper, u_study)
    tol_mm = (to_mm(payload.process_tolerance, u_study)
              if payload.process_tolerance is not None else None)
    conf = float(payload.confidence_level)
    alpha = 1.0 - conf
    z_alpha = t_critical(1.0 - alpha / 2.0, math.inf)

    points.sort(key=lambda p: p.x_mm)
    g = len(points)
    s_pool, df_pool, s_point = _repeatability(points)

    # ---- 逐点均值偏倚 / 合成标准不确定度 / 置信区间
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    single_read = [p for p in points if len(p.values_mm) == 1]
    if single_read and df_pool > 0:
        warnings.append(
            f"{len(single_read)} 个核查点只有单次读数：其重复性标准不确定度"
            f"借用全研究合并重复性 s_p={s_pool:.6g} mm（自由度 {df_pool}）"
        )
    if df_pool == 0:
        warnings.append(
            "所有核查点均无重复读数：重复性无法估计，点权重只反映标准件"
            "参考值不确定度 u(ref)，请补充重复测量以评定量具重复性"
        )

    u_g: list[float] = []
    b_g: list[float] = []
    w_g: list[float] = []
    for i, p in enumerate(points):
        n = len(p.values_mm)
        arr = np.asarray(p.values_mm, dtype=float)
        mean_y = float(arr.mean())
        bias = mean_y - p.x_mm
        if n >= 2:
            s_rep = float(arr.std(ddof=1))
            u_rep = s_rep / math.sqrt(n)
            df_rep = n - 1
            repeat_basis = "point_sample_std"
        elif df_pool > 0:
            s_rep = s_pool
            u_rep = s_pool / math.sqrt(n)
            df_rep = df_pool
            repeat_basis = "pooled_within_std"
        else:
            s_rep = 0.0
            u_rep = 0.0
            df_rep = math.inf
            repeat_basis = "unestimable"
        u_comb = math.sqrt(p.u_ref_mm ** 2 + u_rep ** 2)
        u_g.append(u_comb)
        b_g.append(bias)
        w_g.append(1.0 / u_comb ** 2 if u_comb > 0 else 0.0)

        # Welch–Satterthwaite 有效自由度（u_ref 为 Type B，df=∞）
        if u_rep > 0 and u_comb > 0:
            nu = u_comb ** 4 * df_rep / u_rep ** 4
            tcrit = t_critical(1.0 - alpha / 2.0, nu)
            df_label: Any = float(nu)
        else:
            tcrit = z_alpha
            df_label = None
        if u_comb > 0:
            half = tcrit * u_comb
            t_stat = bias / u_comb
            p_value = _two_sided_p_normal(t_stat)
        else:
            half = 0.0
            t_stat = None
            p_value = None if abs(bias) < 1e-15 else 0.0
        significant = abs(bias) > half if u_comb > 0 else abs(bias) >= 1e-15
        rows.append({
            "point_id": p.point_id,
            "standard_serial": p.standard_serial,
            "reference_value": {
                "mm": p.x_mm,
                "submitted": p.ref_submitted,
                "unit": p.ref_unit,
            },
            "reference_std_uncertainty_mm": p.u_ref_mm,
            "reading_count": n,
            "operators": sorted({m["operator"] for m in p.reading_meta}),
            "measurement_orders": sorted(m["measurement_order"]
                                         for m in p.reading_meta),
            "mean_reading_mm": mean_y,
            "mean_bias_mm": bias,
            "repeatability_std_mm": s_rep if repeat_basis != "unestimable" else None,
            "repeatability_basis": repeat_basis,
            "u_repeatability_of_mean_mm": u_rep if repeat_basis != "unestimable" else None,
            "combined_std_uncertainty_mm": u_comb,
            "effective_degrees_of_freedom": df_label,
            "critical_value": tcrit,
            "confidence_interval_mm": [bias - half, bias + half],
            "confidence_level": conf,
            "t_or_z_statistic": t_stat,
            "p_value": p_value,
            "bias_significant": bool(significant),
            "bias_pct_of_tolerance": (
                100.0 * abs(bias) / tol_mm if tol_mm else None),
            "readings": p.reading_meta,
        })

    # ---- 加权线性回归 b = β0 + β1·x
    x = np.array([p.x_mm for p in points], dtype=float)
    b = np.array(b_g, dtype=float)
    u = np.array(u_g, dtype=float)
    failure_reason: str | None = None
    regression = None

    equal_weights = bool((u <= 0).any())
    if equal_weights:
        w = np.ones(g)
        weight_policy = ("equal_weights（至少一个核查点 u=0：参考值不确定度"
                         "与重复性均为 0，逆方差权重无定义）")
        warnings.append("至少一个核查点合成标准不确定度为 0：拟合改用等权重，"
                        "GUM 系数协方差仍按各点已知方差精确传播")
    else:
        w = 1.0 / u ** 2
        weight_policy = "inverse_variance：w_g = 1/u_g²"

    if g < 2:
        failure_reason = f"有效核查点仅 {g} 个（至少需要 2 个参考值不同的点）"
    elif float(x.max() - x.min()) <= 0.0:
        failure_reason = "所有有效核查点参考值相同：参考值无变差，斜率不可识别"
    else:
        X = np.column_stack([np.ones(g), x])
        XtWX = X.T @ (w[:, None] * X)
        try:
            cov_factor = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            cov_factor = None
        if cov_factor is None or not np.isfinite(cov_factor).all():
            failure_reason = "加权设计矩阵 XᵀWX 奇异：线性拟合失效"
        else:
            beta = cov_factor @ (X.T @ (w * b))
            fitted = X @ beta
            resid = b - fitted
            # GUM 协方差：A·diag(u²)·Aᵀ，A=(XᵀWX)⁻¹XᵀW（含标准件不确定度）
            A = cov_factor @ (X.T * w)
            cov_gum = A @ np.diag(u ** 2) @ A.T
            cov_gum = 0.5 * (cov_gum + cov_gum.T)
            # 经典回归协方差：残差方差估计
            sse_w = float((w * resid ** 2).sum())
            df_res = g - 2
            s_w = math.sqrt(sse_w / df_res) if df_res > 0 else None
            cov_classic = (s_w ** 2 * cov_factor) if s_w is not None else None

            def _coef_block(label: str, tc: float | None,
                            se: float) -> dict[str, Any]:
                est = float(beta[idx])
                zstat = est / se if se > 0 else None
                half2 = (tc * se if tc is not None else z_alpha * se)
                return {
                    "estimate": est,
                    "std_error": se,
                    "covariance_basis": label,
                    "statistic": zstat,
                    "p_value_two_sided": (
                        _two_sided_p_normal(zstat)
                        if zstat is not None else None),
                    "confidence_interval": (
                        [est - half2, est + half2]
                        if se > 0 else [est, est]),
                    "significant": (bool(se > 0 and abs(est) > half2)
                                    if se > 0 else bool(abs(est) >= 1e-15)),
                }

            idx = 0
            se0_gum = math.sqrt(max(cov_gum[0, 0], 0.0))
            se1_gum = math.sqrt(max(cov_gum[1, 1], 0.0))
            t_res = (t_critical(1.0 - alpha / 2.0, df_res)
                     if df_res > 0 else None)
            intercept = _coef_block(
                "GUM：A·diag(u_g²)·Aᵀ（含标准件不确定度）",
                None, se0_gum)
            idx = 1
            slope = _coef_block(
                "GUM：A·diag(u_g²)·Aᵀ（含标准件不确定度）",
                None, se1_gum)
            intercept["unit"] = "mm"
            slope["unit"] = "mm/mm（无量纲）"
            # 经典推断块（残差自由度可用时）
            if cov_classic is not None:
                t0 = t_critical(1.0 - alpha / 2.0, df_res)
                idx = 0
                classic_intercept = _coef_block(
                    "classic：s²(XᵀWX)⁻¹，s²=加权残差平方和/(G−2)",
                    t0, math.sqrt(max(cov_classic[0, 0], 0.0)))
                classic_intercept["unit"] = "mm"
                idx = 1
                classic_slope = _coef_block(
                    "classic：s²(XᵀWX)⁻¹，s²=加权残差平方和/(G−2)",
                    t0, math.sqrt(max(cov_classic[1, 1], 0.0)))
                classic_slope["unit"] = "mm/mm（无量纲）"
                classic = {
                    "residual_std_mm": s_w,
                    "degrees_of_freedom": df_res,
                    "critical_t": t0,
                    "intercept": classic_intercept,
                    "slope": classic_slope,
                    "coefficient_covariance_mm2": cov_classic.tolist(),
                }
            else:
                classic = {
                    "residual_std_mm": None,
                    "degrees_of_freedom": 0,
                    "critical_t": None,
                    "note": "仅 2 个核查点：残差自由度为 0，不估计经典回归"
                            "协方差（两点确定一条直线），系数显著性改用"
                            "GUM 协方差的正态 z 检验",
                    "coefficient_covariance_mm2": None,
                }

            # 失配检验 χ²_lof = Σ(e_g/u_g)²（只用 u>0 的点）
            pos = u > 0
            lof_stat = float(((resid[pos] / u[pos]) ** 2).sum()) if pos.any() else 0.0
            lof_p = _chi2_sf(lof_stat, df_res) if df_res > 0 else None
            # 加权 R²（相对加权均值偏倚）
            bbar_w = float((w * b).sum() / w.sum())
            sst = float((w * (b - bbar_w) ** 2).sum())
            r2_w = 1.0 - sse_w / sst if sst > 0 else 1.0

            xmin, xmax = float(x.min()), float(x.max())
            regression = {
                "model": "b_g = β0 + β1·x_g（偏倚对参考值的加权线性回归）",
                "weight_policy": weight_policy,
                "intercept_mm": float(beta[0]),
                "slope": float(beta[1]),
                "intercept": intercept,
                "slope_block": slope,
                "coefficient_covariance": {
                    "gum_including_reference_mm2": cov_gum.tolist(),
                    "definition": ("Cov_β = A·diag(u_g²)·Aᵀ，"
                                   "A=(XᵀWX)⁻¹XᵀW；测量方案传播系数与标准件"
                                   "不确定度时使用此矩阵；逆方差权重下等于"
                                   "(XᵀWX)⁻¹"),
                    "matrix_order": ["intercept_mm", "slope"],
                },
                "classic": classic,
                "fitted_bias_mm": [float(v) for v in fitted],
                "residuals_mm": [float(v) for v in resid],
                "standardized_residuals": [
                    float(e / uu) if uu > 0 else None
                    for e, uu in zip(resid, u)],
                "weighted_residual_sum_of_squares": sse_w,
                "weighted_r_squared": float(r2_w),
                "lack_of_fit": {
                    "statistic": lof_stat,
                    "distribution": f"chi_square(df={df_res})" if df_res > 0 else None,
                    "degrees_of_freedom": df_res if df_res > 0 else None,
                    "p_value": lof_p,
                    "fits_declared_uncertainties": (
                        None if lof_p is None else bool(lof_p >= alpha)),
                    "note": "χ²_lof = Σ(e_g/u_g)²：失配检验在显著性水平 "
                            f"α={alpha:g} 下考察线性模型与所声明点不确定度"
                            "（含标准件 u(ref)）是否相容",
                },
                "applicable_range_mm": [xmin, xmax],
                "applicable_range_submitted": [
                    from_mm(xmin, u_study), from_mm(xmax, u_study)],
            }

    # ---- 总体平均偏倚（与拟合同口径的加权平均）
    weights = np.array(w_g, dtype=float)
    if equal_weights or not weights.any():
        weights = np.ones(g)
    bbar = float((weights @ np.array(b_g)) / weights.sum()) if g else None
    se_bar = float(1.0 / math.sqrt(weights.sum())) if weights.sum() > 0 else None
    if bbar is not None:
        half_bar = z_alpha * (se_bar or 0.0)
        mean_bias_block = {
            "weighted_mean_bias_mm": bbar,
            "simple_mean_bias_mm": (
                float(np.mean(b_g)) if g else None),
            "std_error_mm": se_bar,
            "weight_policy": weight_policy,
            "confidence_interval_mm": [
                bbar - half_bar, bbar + half_bar],
            "confidence_level": conf,
            "z_statistic": bbar / se_bar if se_bar and se_bar > 0 else None,
            "p_value_two_sided": (
                _two_sided_p_normal(bbar / se_bar)
                if se_bar and se_bar > 0 else None),
            "bias_significant": bool(
                se_bar and se_bar > 0 and abs(bbar) > half_bar),
            "bias_pct_of_tolerance": (
                100.0 * abs(bbar) / tol_mm if tol_mm else None),
        }
    else:
        mean_bias_block = None

    # ---- 覆盖范围与可采用性
    span = float(x.max() - x.min()) if g else 0.0
    working_span = hi - lo
    coverage = span / working_span if working_span > 0 else None
    coverage_ok = coverage is not None and coverage >= float(
        payload.min_span_coverage)
    if g < 5:
        warnings.append(
            f"有效核查点 {g} 个：AIAG MSA 建议线性研究至少使用 5 个标准件，"
            "点较少时失配检验与斜率识别能力有限")
    if coverage is not None and not coverage_ok:
        warnings.append(
            f"参考值跨度覆盖率 {coverage:.3f} < 最低要求 "
            f"{payload.min_span_coverage:g}：量程覆盖过窄，线性模型不得"
            "外推到未覆盖区间，研究不可采用接入测量方案")
    regression_available = regression is not None
    adoptable = bool(regression_available and coverage_ok)

    slope_sig = bool(regression and regression["slope_block"]["significant"])
    intercept_sig = bool(regression and regression["intercept"]["significant"])
    n_sig_points = sum(1 for r in rows if r["bias_significant"])
    linearity_block = None
    if regression is not None:
        swing = abs(regression["slope"]) * span
        linearity_block = {
            "slope_significant": slope_sig,
            "linearity_present": slope_sig,
            "bias_swing_over_reference_span_mm": swing,
            "linearity_pct_of_tolerance": (
                100.0 * abs(regression["slope"]) if tol_mm else None),
            "swing_pct_of_tolerance": (
                100.0 * swing / tol_mm if tol_mm else None),
            "slope_p_value": regression["slope_block"]["p_value_two_sided"],
            "note": ("斜率显著（α={:g}）表示偏倚随参考值线性变化，量具存在"
                     "线性；%Linearity = 100·|斜率|（AIAG 口径），"
                     "量程内偏倚摆动量 = |斜率|·(x_max−x_min)").format(alpha),
        }

    if regression_available and not slope_sig and not intercept_sig:
        conclusion = "斜率与截距均不显著：在评定不确定度下未发现显著线性或固定偏倚"
    elif regression_available:
        parts = []
        if slope_sig:
            parts.append("斜率显著（存在线性偏倚）")
        if intercept_sig:
            parts.append("截距显著（存在固定偏倚）")
        conclusion = "；".join(parts) + "：建议按研究结果修正或调整量具"
    else:
        conclusion = f"线性拟合不可用：{failure_reason}；不得外推或采用"
    if n_sig_points:
        conclusion += f"；{n_sig_points}/{g} 个参考点的偏倚单独显著"

    result: dict[str, Any] = {
        "dimension_id": payload.dimension_id,
        "unit_submitted": u_study,
        "working_range": {
            "submitted": [payload.working_range_lower,
                          payload.working_range_upper],
            "unit": u_study,
            "mm": [lo, hi],
        },
        "process_tolerance": (
            {"submitted": payload.process_tolerance, "unit": u_study,
             "mm": tol_mm} if tol_mm is not None else None),
        "confidence_level": conf,
        "min_span_coverage": float(payload.min_span_coverage),
        "design": {
            "reference_points": g,
            "total_readings": sum(len(p.values_mm) for p in points),
            "excluded_readings": len(exclusions or []),
            "operators": sorted({
                m["operator"] for p in points for m in p.reading_meta}),
            "pooled_repeatability_std_mm": s_pool if df_pool > 0 else None,
            "pooled_degrees_of_freedom": df_pool if df_pool > 0 else None,
            "measurement_order_policy": "全研究测量次序唯一，结果按参考值排序",
        },
        "points": rows,
        "regression": regression,
        "regression_available": regression_available,
        "fit_failure_reason": failure_reason,
        "coverage": {
            "reference_span_mm": span,
            "working_span_mm": working_span,
            "span_coverage_ratio": coverage,
            "min_span_coverage": float(payload.min_span_coverage),
            "coverage_adequate": bool(coverage_ok),
            "extrapolation_policy": "线性修正仅在适用量程 "
                                    "[min 参考值, max 参考值] 内使用，"
                                    "超出不修正、不外推",
        },
        "mean_bias": mean_bias_block,
        "linearity": linearity_block,
        "significant_point_count": n_sig_points,
        "adoptable": adoptable,
        "adoption_blockers": _adoption_blockers(
            regression_available, coverage_ok, failure_reason),
        "warnings": warnings,
        "conclusion": conclusion,
        "formulas": FORMULAS,
        "dependency_versions": _dependency_versions(),
    }
    if copied_from:
        result["copied_from"] = copied_from
    return result


def _adoption_blockers(regression_ok: bool, coverage_ok: bool,
                       failure_reason: str | None) -> list[str]:
    blockers: list[str] = []
    if not regression_ok:
        blockers.append(f"线性拟合不可用（{failure_reason}）")
    if not coverage_ok:
        blockers.append("参考值对工作量程的覆盖过窄（不得外推）")
    return blockers


# ------------------------------------------------------------- 复制比较

def compare_results(base_result: dict[str, Any],
                    new_result: dict[str, Any]) -> dict[str, Any]:
    """比较两份研究结果（通常为排除异常读数前后）的逐点偏倚与回归结论。"""
    base_by = {p["point_id"]: p for p in base_result["points"]}
    new_by = {p["point_id"]: p for p in new_result["points"]}
    points = []
    for pid, bp in base_by.items():
        np_ = new_by.get(pid)
        points.append({
            "point_id": pid,
            "standard_serial": bp["standard_serial"],
            "present_before": True,
            "present_after": np_ is not None,
            "mean_bias_mm_before": bp["mean_bias_mm"],
            "mean_bias_mm_after": np_["mean_bias_mm"] if np_ else None,
            "bias_significant_before": bp["bias_significant"],
            "bias_significant_after": (
                np_["bias_significant"] if np else None),
            "significance_changed": (
                np_ is not None
                and bp["bias_significant"] != np_["bias_significant"]),
            "reading_count_before": bp["reading_count"],
            "reading_count_after": np_["reading_count"] if np else 0,
            "dropped": np_ is None,
        })

    def reg_sig(res: dict[str, Any]) -> dict[str, Any] | None:
        reg = res.get("regression")
        if not reg:
            return None
        return {
            "intercept_mm": reg["intercept_mm"],
            "slope": reg["slope"],
            "slope_significant": reg["slope_block"]["significant"],
            "intercept_significant": reg["intercept"]["significant"],
            "applicable_range_mm": reg["applicable_range_mm"],
        }

    rb, rn = reg_sig(base_result), reg_sig(new_result)
    regression_changed = rb != rn
    return {
        "points": points,
        "regression_before": rb,
        "regression_after": rn,
        "regression_changed": regression_changed,
        "linearity_conclusion_before": (
            (base_result.get("linearity") or {}).get("linearity_present")),
        "linearity_conclusion_after": (
            (new_result.get("linearity") or {}).get("linearity_present")),
        "linearity_presence_changed": (
            (base_result.get("linearity") or {}).get("linearity_present")
            != (new_result.get("linearity") or {}).get("linearity_present")),
        "significant_point_count_before":
            base_result["significant_point_count"],
        "significant_point_count_after":
            new_result["significant_point_count"],
        "mean_bias_mm_before": (
            (base_result.get("mean_bias") or {}).get("weighted_mean_bias_mm")),
        "mean_bias_mm_after": (
            (new_result.get("mean_bias") or {}).get("weighted_mean_bias_mm")),
        "coverage_adequate_before":
            base_result["coverage"]["coverage_adequate"],
        "coverage_adequate_after":
            new_result["coverage"]["coverage_adequate"],
        "adoptable_before": base_result["adoptable"],
        "adoptable_after": new_result["adoptable"],
        "summary_note": (
            "比较排除异常读数前（父研究）后（复制研究）的逐点平均偏倚、"
            "显著性、回归截距 / 斜率及其显著性、适用量程与可采用性；"
            "结论改变说明被排除读数主导了原结论"
        ),
    }


# ------------------------------------------------------------- 测量方案应用

def bias_model_snapshot(study_row) -> dict[str, Any]:
    """从已采用的冻结研究抽取供测量方案快照使用的线性偏倚模型（mm）。"""
    res = study_row.result_json
    reg = res["regression"]
    return {
        "type": "linearity_study",
        "study_id": study_row.id,
        "study_name": study_row.name,
        "dimension_id": study_row.dimension_id,
        "intercept_mm": float(reg["intercept_mm"]),
        "slope": float(reg["slope"]),
        "coefficient_covariance_mm2": reg["coefficient_covariance"][
            "gum_including_reference_mm2"],
        "covariance_order": ["intercept_mm", "slope"],
        "applicable_range_mm": reg["applicable_range_mm"],
        "applicable_range_submitted": reg["applicable_range_submitted"],
        "reference_span_coverage_ratio": res["coverage"]["span_coverage_ratio"],
        "process_tolerance_mm": (
            (res.get("process_tolerance") or {}).get("mm")),
        "correction_model": (
            "测量偏倚 b(x)=a+b·x；逆回归修正 x_c=(x−a)/(1+b)，"
            "仅在适用量程 [min 参考值, max 参考值] 内使用，超出不修正"),
    }


def apply_linear_bias_model(model: dict[str, Any], x_mm: float) -> dict[str, Any]:
    """按线性偏倚模型评估单个实测读数（mm），供检验批次 / 装配 / 漂移复用。

    返回修正值、模型偏倚、是否落在适用量程内，以及修正值因回归系数与
    标准件不确定度产生的标准不确定度（GUM 一阶传播）。
    """
    a = float(model["intercept_mm"])
    b = float(model["slope"])
    cov = np.asarray(model["coefficient_covariance_mm2"], dtype=float)
    lo, hi = (float(v) for v in model["applicable_range_mm"])
    in_range = lo - 1e-12 <= x_mm <= hi + 1e-12
    denom = 1.0 + b
    if not in_range or abs(denom) < 1e-9:
        return {
            "measured_mm": x_mm,
            "in_applicable_range": bool(in_range),
            "applicable_range_mm": [lo, hi],
            "model_bias_mm": 0.0,
            "corrected_mm": x_mm,
            "bias_correction_mm": 0.0,
            "correction_applied": False,
            "correction_std_uncertainty_mm": 0.0,
            "predicted_bias_std_uncertainty_mm": 0.0,
            "note": ("实测值落在研究适用量程之外：不做线性偏倚修正、"
                     "不传播模型不确定度（禁止外推）"
                     if in_range is False else
                     "1+斜率 退化接近 0：逆回归修正无定义，不修正"),
        }
    bias = a + b * x_mm
    corrected = (x_mm - a) / denom
    # 预测偏倚 b(x)=[1,x]β 的标准不确定度
    h = np.array([1.0, x_mm])
    u_pred = math.sqrt(max(float(h @ cov @ h), 0.0))
    # 修正 x_c=(x−a)/(1+b) 对 (a,b) 的梯度
    grad = np.array([-1.0 / denom, -(x_mm - a) / denom ** 2])
    u_corr = math.sqrt(max(float(grad @ cov @ grad), 0.0))
    return {
        "measured_mm": x_mm,
        "in_applicable_range": True,
        "applicable_range_mm": [lo, hi],
        "model_bias_mm": bias,
        "corrected_mm": corrected,
        "bias_correction_mm": corrected - x_mm,
        "correction_applied": True,
        "correction_std_uncertainty_mm": u_corr,
        "predicted_bias_std_uncertainty_mm": u_pred,
    }


FORMULAS = [
    "平均偏倚：b̄_g = mean_r(y_gr) − x_g（逐核查点，mm）",
    "重复性：s_g = √(Σ(y−ȳ)²/(n_g−1))（n_g≥2）；单点借用合并重复性 "
    "s_p² = ΣΣ(y−ȳ)²/Σ(n_g−1)；无任何重复时重复性不可估计",
    "点合成标准不确定度：u_g = √(u_ref,g² + (s_g/√n_g)²)",
    "点偏倚置信区间：b̄_g ± t_{p,ν}·u_g，ν 用 Welch–Satterthwaite："
    "ν = u_g⁴·df_rep/u_rep⁴（u_ref 为 Type B，df=∞；u_rep=0 用正态 z）",
    "偏倚显著性：置信区间不含 0（α=1−置信水平）",
    "加权线性回归：b_g = β0 + β1·x_g，β=(XᵀWX)⁻¹XᵀWb，"
    "权重 w_g = 1/u_g²（任一点 u=0 时退化为等权重并告警）",
    "GUM 系数协方差（含标准件不确定度）：Cov_β = A·diag(u_g²)·Aᵀ，"
    "A=(XᵀWX)⁻¹XᵀW；逆方差权重下 Cov_β=(XᵀWX)⁻¹",
    "经典系数协方差：s²(XᵀWX)⁻¹，s²=Σw e²/(G−2)（G=2 时残差 df=0，"
    "不给经典区间）；系数显著性默认取 GUM 协方差正态 z 检验",
    "失配检验：χ²_lof = Σ(e_g/u_g)² ~ χ²(G−2)，考察线性模型与所声明"
    "不确定度（含 u(ref)）是否相容",
    "总体平均偏倚：b̄ = Σw b̄_g/Σw，SE(b̄)=1/√Σw（与拟合同权重）",
    "适用量程 = [min x_g, max x_g]；覆盖率 = (x_max−x_min)/工作量程宽度；"
    "覆盖不足 / 参考点不足 / 拟合失效时不外推、不可采用",
    "测量方案修正（逆回归）：b(x)=a+b·x，x_c=(x−a)/(1+b)；"
    "u²(x_c)=∇x_c·Cov_β·∇x_cᵀ（∂x_c/∂a=−1/(1+b)，"
    "∂x_c/∂b=−(x−a)/(1+b)²）；量程外不修正",
    "%Linearity = 100·|斜率|（AIAG）；量程内偏倚摆动 = |斜率|·参考值跨度；"
    "%Bias = 100·|偏倚|/过程公差",
]
