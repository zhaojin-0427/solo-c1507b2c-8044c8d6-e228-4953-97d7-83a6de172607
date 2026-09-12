"""量具误差测量方案与批次判定（GUM 线性传播 + 固定种子蒙特卡洛）。

测量误差模型（单位统一为 mm）
============================
每个尺寸的合成标准不确定度由四个分量合成：

* u_res  = resolution/(2√3)   分辨率：半宽 resolution/2 的均匀分布
* u_cal  = U_cal / k          校准扩展不确定度除以其覆盖因子
* u_bias                      偏倚修正值的标准不确定度
* u_rep                       重复性标准差
* u_c    = √(u_res² + u_cal² + u_bias² + u_rep²)

偏倚修正：x_c = x + b（b 为带符号修正值，修正后残余误差视为零均值）。

共用量具带来的相关项 ρ 是**测量误差**之间的 Pearson 相关，作用于
u_rest = √(u_cal² + u_bias² + u_rep²) 部分（公共误差源：同一台量具的
校准/偏倚/重复性）；分辨率量化误差相互独立。

封闭环 GUM 线性传播（s_i 为闭环方向系数）：
    u_C² = Σ_i s_i²·u_res,i² + Σ_iΣ_j s_i·s_j·ρ_ij·u_rest,i·u_rest,j

蒙特卡洛（固定种子，所有工件共用同一组误差样本）：
    ε_rest ~ N(0, DρD)（D = diag(u_rest)，Cholesky 分解）
    ε_res,i ~ U(−resolution_i/2, +resolution_i/2)，逐尺寸独立
    X_true = x_c + ε_rest + ε_res
逐尺寸与封闭环的扩展不确定度 U = k_out · u_c（k_out 为批次输出覆盖因子）。

判定（双边保护带，带宽 w）
=========================
w = h·U（h 为扩展不确定度倍数）或固定长度：

* 接收：  LSL + w ≤ x_c ≤ USL − w
* 拒收：  x_c < LSL − w 或 x_c > USL + w
* 其余：  不确定（indeterminate）

概率（GUM 正态近似解析式与蒙特卡洛经验比例并列）：
* 接收时真值越界概率 p_out = P(X_true ∉ [LSL, USL])
* 拒收时真值合格概率 p_in  = P(X_true ∈ [LSL, USL])
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .engine import NormalizedChain, _cholesky_psd, normal_cdf
from .units import to_mm

SQRT3 = math.sqrt(3.0)

PLAN_FORMULAS = [
    "u_res = resolution/(2√3)：分辨率半宽 resolution/2 的均匀分布标准不确定度",
    "u_cal = U_cal/k：校准证书扩展不确定度按其覆盖因子换算为标准不确定度",
    "u_bias：偏倚修正值的标准不确定度；u_rep：重复性标准差",
    "合成标准不确定度 u_c = √(u_res² + u_cal² + u_bias² + u_rep²)",
    "偏倚修正：x_c = x + b（b 为带符号修正值，修正后残余误差零均值）",
    "共用量具相关 ρ 为测量误差 Pearson 相关，作用于 "
    "u_rest = √(u_cal²+u_bias²+u_rep²)；分辨率分量相互独立",
    "封闭环 GUM 线性传播：u_C² = Σ_i s_i²u_res,i² "
    "+ Σ_iΣ_j s_i s_j ρ_ij u_rest,i u_rest,j",
    "输出扩展不确定度 U = k_out·u_c（k_out 为批次指定的输出覆盖因子）",
]

DECISION_FORMULAS = [
    "保护带 w：mode=multiple 时 w = h·U（h 为扩展不确定度倍数）；"
    "mode=fixed 时 w 为固定长度",
    "判定：LSL+w ≤ x_c ≤ USL−w → 接收；x_c < LSL−w 或 x_c > USL+w → 拒收；"
    "其余 → 不确定",
    "接收时真值越界概率 p_out = P(X_true∉[LSL,USL])；"
    "拒收时真值合格概率 p_in = P(X_true∈[LSL,USL])",
    "GUM 正态近似：p_out = Φ((LSL−x_c)/u_c) + 1 − Φ((USL−x_c)/u_c)，"
    "p_in = 1 − p_out",
    "蒙特卡洛：ε_rest ~ N(0, DρD)（D=diag(u_rest)，Cholesky），"
    "ε_res,i ~ U(−res_i/2, res_i/2) 独立，X_true = x_c + ε_rest + ε_res；"
    "p 为固定种子样本的经验比例",
    "所有工件共用同一组误差样本：判定之间相互一致，"
    "同一种子下整批结果可精确复现",
]


def dependency_versions() -> dict[str, str]:
    """响应追溯用：当前进程的关键依赖版本。"""
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


# ------------------------------------------------------------- 方案合成

def normalize_plan(nc: NormalizedChain, payload) -> dict[str, Any]:
    """校验测量方案对基线链的覆盖性，统一单位为 mm 并合成标准不确定度。

    分量符号 / 覆盖因子配对 / 相关矩阵半正定已由 Pydantic 拦截；
    这里复核链尺寸覆盖（缺失、链外）并组装规范化结果。
    """
    chain_ids = [d.id for d in nc.dimensions]
    gauge_by_id = {g.dimension_id: g for g in payload.gauges}
    missing = [i for i in chain_ids if i not in gauge_by_id]
    if missing:
        raise ValueError(f"测量方案缺少链上尺寸的量具声明: {missing}")
    external = sorted(set(gauge_by_id) - set(chain_ids))
    if external:
        raise ValueError(f"测量方案包含基线链之外的尺寸: {external}")

    # 相关矩阵（链尺寸顺序；schema 已校验半正定，这里防御性复核）
    idx = {name: i for i, name in enumerate(chain_ids)}
    k = len(chain_ids)
    corr = np.eye(k)
    for c in payload.correlations:
        i, j = idx[c.dim_a], idx[c.dim_b]
        corr[i, j] = corr[j, i] = c.rho
    eig_min = float(np.linalg.eigvalsh(corr).min())
    if eig_min < -1e-8:
        raise ValueError(
            f"量具相关矩阵非半正定: 最小特征值 {eig_min:.3e}"
        )

    dims_out: list[dict[str, Any]] = []
    for d in nc.dimensions:
        g = gauge_by_id[d.id]
        u = g.unit.value if hasattr(g.unit, "value") else str(g.unit)
        res_mm = to_mm(g.resolution, u) if g.resolution is not None else 0.0
        u_res = res_mm / (2.0 * SQRT3)
        u_cal = (
            to_mm(g.calibration_expanded_uncertainty, u) / g.coverage_factor
            if g.calibration_expanded_uncertainty is not None else 0.0
        )
        bias_mm = (
            to_mm(g.bias_correction, u) if g.bias_correction is not None else 0.0
        )
        u_bias = (
            to_mm(g.bias_std_uncertainty, u)
            if g.bias_std_uncertainty is not None else 0.0
        )
        u_rep = (
            to_mm(g.repeatability_std, u)
            if g.repeatability_std is not None else 0.0
        )
        u_rest = math.sqrt(u_cal ** 2 + u_bias ** 2 + u_rep ** 2)
        var = u_res ** 2 + u_rest ** 2
        u_comb = math.sqrt(var)
        share = (
            lambda v: v * v / var if var > 0 else 0.0  # noqa: E731
        )
        dims_out.append({
            "dimension_id": d.id,
            "unit": u,
            "submitted": {
                "resolution": g.resolution,
                "calibration_expanded_uncertainty":
                    g.calibration_expanded_uncertainty,
                "coverage_factor": g.coverage_factor,
                "bias_correction": g.bias_correction,
                "bias_std_uncertainty": g.bias_std_uncertainty,
                "repeatability_std": g.repeatability_std,
                "unit": u,
            },
            "resolution_mm": res_mm,
            "bias_correction_mm": bias_mm,
            "components_mm": {
                "u_resolution": u_res,
                "u_calibration": u_cal,
                "u_bias": u_bias,
                "u_repeatability": u_rep,
            },
            "u_rest_mm": u_rest,
            "combined_std_uncertainty_mm": u_comb,
            "variance_share": {
                "resolution": share(u_res),
                "calibration": share(u_cal),
                "bias": share(u_bias),
                "repeatability": share(u_rep),
            },
        })

    return {
        "unit_policy": "所有分量按声明单位换算为 mm 后合成；"
                       "原始数值与单位在 submitted 中原样保留",
        "dimensions": dims_out,
        "correlation": {
            "matrix": corr.tolist(),
            "dimension_order": chain_ids,
            "definition": "ρ 为测量误差（共用量具公共误差源）的 Pearson 相关，"
                          "作用于 u_rest=√(u_cal²+u_bias²+u_rep²) 部分；"
                          "分辨率量化误差相互独立",
        },
        "formulas": PLAN_FORMULAS,
        "dependency_versions": dependency_versions(),
    }


# ------------------------------------------------------------- 判定辅助

def _decide(x: float, lsl: float | None, usl: float | None, w: float) -> str:
    """双边保护带判定：接收 / 拒收 / 不确定。"""
    if lsl is not None and x < lsl - w:
        return "reject"
    if usl is not None and x > usl + w:
        return "reject"
    if lsl is not None and x < lsl + w:
        return "indeterminate"
    if usl is not None and x > usl - w:
        return "indeterminate"
    return "accept"


def _gum_probs(x: float, lsl: float | None, usl: float | None,
               u: float) -> tuple[float, float]:
    """GUM 正态近似：真值 ~ N(x, u²) 时越界 / 合格概率。"""
    if u <= 0.0:
        out = 0.0
        if lsl is not None and x < lsl:
            out = 1.0
        if usl is not None and x > usl:
            out = 1.0
        return out, 1.0 - out
    below = 0.0 if lsl is None else float(normal_cdf((lsl - x) / u))
    above = 0.0 if usl is None else 1.0 - float(normal_cdf((usl - x) / u))
    p_out = below + above
    return p_out, max(0.0, 1.0 - p_out)


def _mc_probs(sorted_eps: np.ndarray, x: float, lsl: float | None,
              usl: float | None) -> tuple[float, float]:
    """蒙特卡洛经验概率：X_true = x + ε，ε 为已排序误差样本列。"""
    n = sorted_eps.shape[0]
    below = 0
    if lsl is not None:
        below = int(np.searchsorted(sorted_eps, lsl - x, side="left"))
    above = 0
    if usl is not None:
        above = int(n - np.searchsorted(sorted_eps, usl - x, side="right"))
    p_out = (below + above) / n
    return p_out, 1.0 - p_out


# ------------------------------------------------------------- 批次判定

def evaluate_batch(nc: NormalizedChain, plan: dict[str, Any],
                   rows: list[dict[str, Any]], *,
                   plan_meta: dict[str, Any], k_out: float,
                   guard_band: dict[str, Any], mc_samples: int,
                   seed: int) -> dict[str, Any]:
    """对整批实测行做量具误差判定（数值单位均为 mm）。

    plan 为测量方案的规范化快照（normalize_plan 输出）；rows 为
    inspection.validate_rows 的输出。返回完整判定报告（随批次冻结入库）。
    """
    dims = nc.dimensions
    ids = [d.id for d in dims]
    k = len(ids)
    by_id = {p["dimension_id"]: p for p in plan["dimensions"]}
    u_c = np.array([by_id[i]["combined_std_uncertainty_mm"] for i in ids])
    u_res = np.array([by_id[i]["components_mm"]["u_resolution"] for i in ids])
    u_rest = np.array([by_id[i]["u_rest_mm"] for i in ids])
    bias = np.array([by_id[i]["bias_correction_mm"] for i in ids])
    corr = np.array(plan["correlation"]["matrix"], dtype=float)
    signs = np.array([d.sign for d in dims], dtype=float)

    # ---- 保护带宽度
    mode = guard_band["mode"]
    if mode == "multiple":
        h = float(guard_band["multiple"])
        w_dim = h * k_out * u_c
    else:
        w_dim = np.full(k, float(guard_band["fixed_mm"]))

    # ---- 封闭环 GUM 线性传播
    var_res = float(((signs * u_res) ** 2).sum())
    wr = signs * u_rest
    var_rest = float((corr * wr[:, None] * wr[None, :]).sum())
    var_c = var_res + var_rest
    u_closure = math.sqrt(max(var_c, 0.0))
    if mode == "multiple":
        w_closure = float(guard_band["multiple"]) * k_out * u_closure
    else:
        w_closure = float(guard_band["fixed_mm"])

    contributions = []
    for j, d in enumerate(dims):
        c_res = float(signs[j] ** 2 * u_res[j] ** 2)
        c_rest = float(wr[j] * float((corr[j] * wr).sum()))
        contributions.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "resolution_variance_mm2": c_res,
            "correlated_variance_contribution_mm2": c_rest,
            "variance_share": (c_res + c_rest) / var_c if var_c > 0 else 0.0,
        })

    # ---- 蒙特卡洛误差样本（所有工件共用一组，判定间一致且可复现）
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((mc_samples, k)) @ _cholesky_psd(corr).T
    eps *= u_rest[None, :]
    half_res = u_res * SQRT3
    eps += rng.uniform(-half_res, half_res, size=(mc_samples, k))
    eps_closure = eps @ signs
    sorted_eps = np.sort(eps, axis=0)
    sorted_eps_closure = np.sort(eps_closure)
    mc_std_closure = float(eps_closure.std(ddof=1))

    # ---- 逐尺寸判定
    dim_entries: list[dict[str, Any]] = []
    for j, d in enumerate(dims):
        lsl = d.nominal + d.lower_dev
        usl = d.nominal + d.upper_dev
        w = float(w_dim[j])
        counts = {"accept": 0, "reject": 0, "indeterminate": 0}
        items = []
        for row in rows:
            m = row["values"].get(d.id)
            if m is None:
                continue
            x = float(m["value_mm"])
            xc = x + float(bias[j])
            dec = _decide(xc, lsl, usl, w)
            counts[dec] += 1
            p_out_g, p_in_g = _gum_probs(xc, lsl, usl, float(u_c[j]))
            p_out_m, p_in_m = _mc_probs(sorted_eps[:, j], xc, lsl, usl)
            items.append({
                "serial": row["serial"],
                "measured_mm": x,
                "bias_correction_mm": float(bias[j]),
                "corrected_mm": xc,
                "decision": dec,
                "p_true_out_of_spec": {
                    "gum": p_out_g, "monte_carlo": p_out_m},
                "p_true_conforming": {
                    "gum": p_in_g, "monte_carlo": p_in_m},
            })
        plan_dim = by_id[d.id]
        dim_entries.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "spec_mm": {"lsl": lsl, "usl": usl},
            "components_mm": plan_dim["components_mm"],
            "variance_share": plan_dim["variance_share"],
            "combined_std_uncertainty_mm": float(u_c[j]),
            "expanded_uncertainty_mm": float(k_out * u_c[j]),
            "guard_band_mm": w,
            "accept_interval_mm": [lsl + w, usl - w],
            "items": items,
            "summary": {**counts, "measured": len(items)},
        })

    # ---- 封闭环判定（仅所有尺寸齐全的工件行）
    has_closure_spec = (nc.closure_lsl_mm is not None
                        or nc.closure_usl_mm is not None)
    closure_items = []
    excluded: list[str] = []
    c_counts = {"accept": 0, "reject": 0, "indeterminate": 0}
    for row in rows:
        vals = [row["values"].get(i) for i in ids]
        if any(v is None for v in vals):
            excluded.append(row["serial"])
            continue
        x = np.array([v["value_mm"] for v in vals], dtype=float)
        xc = x + bias
        c_val = float(xc @ signs)
        if has_closure_spec:
            dec = _decide(c_val, nc.closure_lsl_mm, nc.closure_usl_mm,
                          w_closure)
            c_counts[dec] += 1
            p_out_g, p_in_g = _gum_probs(
                c_val, nc.closure_lsl_mm, nc.closure_usl_mm, u_closure)
            p_out_m, p_in_m = _mc_probs(
                sorted_eps_closure, c_val, nc.closure_lsl_mm,
                nc.closure_usl_mm)
        else:
            dec = None
            p_out_g = p_in_g = p_out_m = p_in_m = None
        closure_items.append({
            "serial": row["serial"],
            "corrected_closure_mm": c_val,
            "decision": dec,
            "p_true_out_of_spec": (
                {"gum": p_out_g, "monte_carlo": p_out_m}
                if has_closure_spec else None
            ),
            "p_true_conforming": (
                {"gum": p_in_g, "monte_carlo": p_in_m}
                if has_closure_spec else None
            ),
        })

    closure_block = {
        "spec_mm": {"lsl": nc.closure_lsl_mm, "usl": nc.closure_usl_mm},
        "gum": {
            "combined_std_uncertainty_mm": u_closure,
            "expanded_uncertainty_mm": k_out * u_closure,
            "variance_from_resolution_mm2": var_res,
            "variance_from_correlated_rest_mm2": var_rest,
            "dimension_contributions": contributions,
        },
        "monte_carlo": {
            "std_mm": mc_std_closure,
            "samples": mc_samples,
            "seed": seed,
        },
        "guard_band_mm": w_closure if has_closure_spec else None,
        "accept_interval_mm": (
            [
                (nc.closure_lsl_mm + w_closure
                 if nc.closure_lsl_mm is not None else None),
                (nc.closure_usl_mm - w_closure
                 if nc.closure_usl_mm is not None else None),
            ] if has_closure_spec else None
        ),
        "items": closure_items,
        "excluded_serials": excluded,
        "exclusion_reason": (
            "以下工件行存在缺测尺寸，未参与封闭环判定" if excluded else None
        ),
        "summary": c_counts if has_closure_spec else None,
        "decision_unavailable_reason": (
            None if has_closure_spec else
            "基线链未声明封闭环上下限：只报告封闭环不确定度，不做接收判定"
        ),
    }

    return {
        "measurement_plan_id": plan_meta["plan_id"],
        "plan_snapshot": {
            "plan_id": plan_meta["plan_id"],
            "name": plan_meta["name"],
            "note": plan_meta["note"],
            "combined": plan,
        },
        "frozen": True,
        "policy": {
            "bias_correction": "x_c = x + b：先加偏倚修正值，"
                               "残余误差视为零均值，再做不确定度评定与判定",
            "output_coverage_factor": k_out,
            "guard_band": {
                "mode": mode,
                "multiple": (float(guard_band["multiple"])
                             if mode == "multiple" else None),
                "fixed_mm": (float(guard_band["fixed_mm"])
                             if mode == "fixed" else None),
                "submitted": guard_band["submitted"],
            },
            "decision_rule": "接收: LSL+w ≤ x_c ≤ USL−w；"
                             "拒收: x_c < LSL−w 或 x_c > USL+w；"
                             "其余: 不确定",
        },
        "dimensions": dim_entries,
        "closure": closure_block,
        "summary": {
            "rows_total": len(rows),
            "closure_evaluated": len(rows) - len(excluded),
            "closure_excluded": len(excluded),
        },
        "monte_carlo": {
            "samples": mc_samples,
            "seed": seed,
            "note": "所有工件共用同一组测量误差样本；"
                    "同一种子下整批判定可精确复现",
        },
        "formulas": PLAN_FORMULAS + DECISION_FORMULAS,
        "dependency_versions": dependency_versions(),
    }
