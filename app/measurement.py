"""量具误差测量方案与批次判定（GUM 线性传播 + 固定种子蒙特卡洛）。

测量误差模型（单位统一为 mm）
============================
每个尺寸的合成标准不确定度由四个分量合成：

* u_res  = resolution/(2√3)   分辨率：半宽 resolution/2 的均匀分布
* u_cal  = U_cal / k          校准扩展不确定度除以其覆盖因子
* u_bias                      偏倚修正值的标准不确定度
* u_rep                       重复性标准差（可引用冻结量具 R&R 研究，
                              以总量具标准差 σ_GRR 取代手填值）
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
扩展不确定度：逐尺寸 U_i = k_i·u_c,i（k_i 为方案中该量具的覆盖因子）；
封闭环 U_C = k_out·u_C（k_out 为批次输出覆盖因子，默认 2.0）。
GUM 与蒙特卡洛使用同一覆盖因子口径（U_MC = k·std(ε)）。

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
from .linearity import apply_linear_bias_model
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
    "逐尺寸扩展不确定度 U_i = k_i·u_c,i（k_i 为方案中该量具的覆盖因子）；"
    "封闭环 U_C = k_out·u_C（k_out 为批次输出覆盖因子，默认 2.0）",
]

DECISION_FORMULAS = [
    "保护带 w：mode=multiple 时 w = h·U（h 为扩展不确定度倍数，U 取对应层级："
    "尺寸用 k_i·u_c,i，封闭环用 k_out·u_C）；mode=fixed 时 w 为固定长度",
    "判定：LSL+w ≤ x_c ≤ USL−w → 接收；x_c < LSL−w 或 x_c > USL+w → 拒收；"
    "其余 → 不确定",
    "接收时真值越界概率 p_out = P(X_true∉[LSL,USL])；"
    "拒收时真值合格概率 p_in = P(X_true∈[LSL,USL])",
    "GUM 正态近似：p_out = Φ((LSL−x_c)/u_c) + 1 − Φ((USL−x_c)/u_c)，"
    "p_in = 1 − p_out",
    "蒙特卡洛：ε_rest ~ N(0, DρD)（D=diag(u_rest)，Cholesky），"
    "ε_res,i ~ U(−res_i/2, res_i/2) 独立，X_true = x_c + ε_rest + ε_res；"
    "p 为固定种子样本的经验比例",
    "蒙特卡洛扩展不确定度与 GUM 同一覆盖因子口径："
    "逐尺寸 U_MC,i = k_i·std(ε_i)，封闭环 U_MC,C = k_out·std(ε_C)",
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

def normalize_plan(nc: NormalizedChain, payload,
                   gage_rr: dict[str, dict[str, Any]] | None = None,
                   linearity: dict[str, dict[str, Any]] | None = None
                   ) -> dict[str, Any]:
    """校验测量方案对基线链的覆盖性，统一单位为 mm 并合成标准不确定度。

    分量符号 / 覆盖因子配对 / 相关矩阵半正定已由 Pydantic 拦截；
    这里复核链尺寸覆盖（缺失、链外）并组装规范化结果。

    gage_rr：尺寸 id -> {"study_id", "name", "total_gage_std_mm"}，
    被引用尺寸的重复性分量 u_rep 以冻结量具 R&R 研究的总量具标准差
    取代手填值（研究 id 随方案快照冻结）。

    linearity：尺寸 id -> 已采用线性与偏倚研究的偏倚模型快照
    （linearity.bias_model_snapshot 输出），被引用尺寸的偏倚修正改由
    线性模型按实测值给出（逆回归），回归系数与标准件不确定度在检验
    批次中传播；这里在适用量程中点给出代表性偏倚与其不确定度。
    """
    gage_rr = gage_rr or {}
    linearity = linearity or {}
    chain_ids = [d.id for d in nc.dimensions]
    gauge_by_id = {g.dimension_id: g for g in payload.gauges}
    missing = [i for i in chain_ids if i not in gauge_by_id]
    if missing:
        raise ValueError(f"测量方案缺少链上尺寸的量具声明: {missing}")
    external = sorted(set(gauge_by_id) - set(chain_ids))
    if external:
        raise ValueError(f"测量方案包含基线链之外的尺寸: {external}")
    external_rr = sorted(set(gage_rr) - set(chain_ids))
    if external_rr:
        raise ValueError(
            f"量具 R&R 研究引用了基线链之外的尺寸: {external_rr}"
        )
    external_lin = sorted(set(linearity) - set(chain_ids))
    if external_lin:
        raise ValueError(
            f"线性与偏倚研究引用了基线链之外的尺寸: {external_lin}"
        )

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
        u_rep_manual = (
            to_mm(g.repeatability_std, u)
            if g.repeatability_std is not None else 0.0
        )
        # 引用冻结量具 R&R 研究：重复性分量以研究的总量具标准差取代手填值
        rr = gage_rr.get(d.id)
        if rr is not None:
            u_rep = float(rr["total_gage_std_mm"])
            repeatability_source: dict[str, Any] = {
                "type": "gage_rr_study",
                "study_id": rr["study_id"],
                "study_name": rr["name"],
                "total_gage_std_mm": u_rep,
                "replaced_submitted_repeatability_std_mm": u_rep_manual,
                "note": "重复性分量由冻结量具 R&R 研究的总量具标准差"
                        "（含再现性）取代手填值；研究 id 随方案快照冻结",
            }
        else:
            u_rep = u_rep_manual
            repeatability_source = {"type": "manual"}

        # 引用已采用线性与偏倚研究：偏倚改由线性模型按实测值给出。
        # 方案合成块在适用量程中点给出代表性偏倚与预测偏倚标准不确定度；
        # 该 u_bias 与校准/重复性不同源（回归系数 + 标准件 u(ref)），不并入
        # 共用量具相关的 u_rest，在检验批次中逐实测值单独传播。
        lin = linearity.get(d.id)
        bias_model = None
        if lin is not None:
            lo_l, hi_l = (float(v) for v in lin["applicable_range_mm"])
            x_mid = 0.5 * (lo_l + hi_l)
            a, bb = float(lin["intercept_mm"]), float(lin["slope"])
            bias_mm = a + bb * x_mid
            cov_l = np.asarray(lin["coefficient_covariance_mm2"], dtype=float)
            hh = np.array([1.0, x_mid])
            u_lin = math.sqrt(max(float(hh @ cov_l @ hh), 0.0))
            u_bias = 0.0  # 模型项不进 u_rest（在批次中按实测值传播）
            bias_model = {
                **lin,
                "representative_reference_mm": x_mid,
                "representative_bias_mm": bias_mm,
                "representative_bias_std_uncertainty_mm": u_lin,
            }
        else:
            bias_mm = (
                to_mm(g.bias_correction, u) if g.bias_correction is not None else 0.0
            )
            u_bias = (
                to_mm(g.bias_std_uncertainty, u)
                if g.bias_std_uncertainty is not None else 0.0
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
            "bias_correction_source": (
                "linearity_study" if bias_model is not None else "manual"),
            "linear_bias_model": bias_model,
            "coverage_factor": float(g.coverage_factor),
            "components_mm": {
                "u_resolution": u_res,
                "u_calibration": u_cal,
                "u_bias": (
                    bias_model["representative_bias_std_uncertainty_mm"]
                    if bias_model is not None else u_bias),
                "u_repeatability": u_rep,
            },
            "repeatability_source": repeatability_source,
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
        "gage_rr_studies": [
            {
                "dimension_id": dim_id,
                "study_id": gage_rr[dim_id]["study_id"],
                "study_name": gage_rr[dim_id]["name"],
                "total_gage_std_mm": float(
                    gage_rr[dim_id]["total_gage_std_mm"]),
            }
            for dim_id in chain_ids if dim_id in gage_rr
        ],
        "linearity_studies": [
            {
                "dimension_id": dim_id,
                "study_id": linearity[dim_id]["study_id"],
                "study_name": linearity[dim_id]["study_name"],
                "intercept_mm": float(linearity[dim_id]["intercept_mm"]),
                "slope": float(linearity[dim_id]["slope"]),
                "applicable_range_mm":
                    linearity[dim_id]["applicable_range_mm"],
            }
            for dim_id in chain_ids if dim_id in linearity
        ],
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
    bias_const = np.array([
        0.0 if by_id[i].get("linear_bias_model") is not None
        else by_id[i]["bias_correction_mm"] for i in ids])
    lin_models = [by_id[i].get("linear_bias_model") for i in ids]
    has_linear = any(m is not None for m in lin_models)
    corr = np.array(plan["correlation"]["matrix"], dtype=float)
    signs = np.array([d.sign for d in dims], dtype=float)

    def _correct_one(j: int, x: float) -> dict[str, Any]:
        """单读数偏倚修正：线性模型（逆回归，量程内）或常量 b。"""
        model = lin_models[j]
        if model is not None:
            ev = apply_linear_bias_model(model, x)
            return ev
        b = float(bias_const[j])
        return {"measured_mm": x, "in_applicable_range": True,
                "applicable_range_mm": None, "model_bias_mm": -b,
                "corrected_mm": x + b, "bias_correction_mm": b,
                "correction_applied": True,
                "correction_std_uncertainty_mm": 0.0,
                "predicted_bias_std_uncertainty_mm": 0.0}

    # ---- 逐尺寸扩展因子：取方案中该量具的覆盖因子 k_i
    #      （历史快照缺字段时回退到批次 output_coverage_factor）
    def _plan_k(plan_dim: dict[str, Any]) -> float:
        k_i = plan_dim.get("coverage_factor")
        if k_i is None:
            k_i = plan_dim.get("submitted", {}).get("coverage_factor")
        return float(k_i) if k_i is not None else float(k_out)

    k_dim = np.array([_plan_k(by_id[i]) for i in ids])
    u_exp_dim = k_dim * u_c                       # U_i = k_i·u_c,i（GUM，常量偏倚）

    # ---- 保护带宽度（常量偏倚口径；线性模型项按实测值另加）
    mode = guard_band["mode"]
    if mode == "multiple":
        h = float(guard_band["multiple"])
        w_dim = h * u_exp_dim
    else:
        w_dim = np.full(k, float(guard_band["fixed_mm"]))

    # ---- 封闭环 GUM 线性传播（常量分量：分辨率 + ρ 相关 rest）
    var_res = float(((signs * u_res) ** 2).sum())
    wr = signs * u_rest
    var_rest = float((corr * wr[:, None] * wr[None, :]).sum())
    var_c_const = var_res + var_rest
    u_closure_const = math.sqrt(max(var_c_const, 0.0))

    contributions = []
    for j, d in enumerate(dims):
        c_res = float(signs[j] ** 2 * u_res[j] ** 2)
        c_rest = float(wr[j] * float((corr[j] * wr).sum()))
        contributions.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "resolution_variance_mm2": c_res,
            "correlated_variance_contribution_mm2": c_rest,
            "linear_bias_variance_mm2": 0.0,  # 随实测值变化，逐封闭环行给
            "variance_share": ((c_res + c_rest) / var_c_const
                               if var_c_const > 0 else 0.0),
        })

    # ---- 蒙特卡洛误差样本（所有工件共用一组，判定间一致且可复现）
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((mc_samples, k)) @ _cholesky_psd(corr).T
    eps *= u_rest[None, :]
    half_res = u_res * SQRT3
    eps += rng.uniform(-half_res, half_res, size=(mc_samples, k))
    # 线性偏倚模型：回归系数误差（含标准件 u(ref) 的 GUM 协方差）。
    # 各研究系数误差相互独立且对整批共用（系统误差，不随工件缩小）。
    lin_idx = [j for j, m in enumerate(lin_models) if m is not None]
    lin_seed_rng = np.random.default_rng(
        np.random.SeedSequence(seed).spawn(1)[0])
    for j in lin_idx:
        model = lin_models[j]
        covm = np.asarray(model["coefficient_covariance_mm2"], dtype=float)
        # 对称化 + 极小特征值保护
        covm = 0.5 * (covm + covm.T)
        try:
            chol = np.linalg.cholesky(covm + 1e-18 * np.eye(2))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(covm)
            eigvals = np.maximum(eigvals, 0.0)
            chol = eigvecs * np.sqrt(eigvals)
        delta = lin_seed_rng.standard_normal((mc_samples, 2)) @ chol.T
        lo_l, hi_l = (float(v) for v in model["applicable_range_mm"])
        # 预计算逆回归修正对 (a,b) 的梯度所需的系数误差列，逐读数按 x 组装
        lin_models[j] = {**model,
                         "_da": delta[:, 0], "_db": delta[:, 1],
                         "_range": (lo_l, hi_l)}
    eps_closure_const = eps @ signs
    sorted_eps = np.sort(eps, axis=0)
    sorted_eps_closure_const = np.sort(eps_closure_const)
    mc_std_dim_const = eps.std(axis=0, ddof=1)       # 常量分量逐尺寸 σ
    mc_std_closure_const = float(eps_closure_const.std(ddof=1))

    # ---- 逐尺寸判定
    dim_entries: list[dict[str, Any]] = []
    out_of_range_notes: dict[str, int] = {}
    for j, d in enumerate(dims):
        lsl = d.nominal + d.lower_dev
        usl = d.nominal + d.upper_dev
        w = float(w_dim[j])
        counts = {"accept": 0, "reject": 0, "indeterminate": 0}
        items = []
        mc_stds: list[float] = []
        for row in rows:
            m = row["values"].get(d.id)
            if m is None:
                continue
            x = float(m["value_mm"])
            ev = _correct_one(j, x)
            xc = float(ev["corrected_mm"])
            u_lin_item = float(ev["correction_std_uncertainty_mm"])
            u_item = math.sqrt(float(u_c[j]) ** 2 + u_lin_item ** 2)
            dec = _decide(xc, lsl, usl, w)
            counts[dec] += 1
            p_out_g, p_in_g = _gum_probs(xc, lsl, usl, u_item)
            # MC：常量误差列 + 该读数的线性系数误差列（系统误差批共用）
            model = lin_models[j]
            if model is not None and ev["in_applicable_range"] \
                    and ev["correction_applied"]:
                a = float(model["intercept_mm"])
                bsl = float(model["slope"])
                denom = 1.0 + bsl
                grad_a = -1.0 / denom
                grad_b = -(x - a) / denom ** 2
                lin_col = grad_a * model["_da"] + grad_b * model["_db"]
            else:
                lin_col = np.zeros(mc_samples)
                if model is not None and not ev["in_applicable_range"]:
                    out_of_range_notes[d.id] = out_of_range_notes.get(d.id, 0) + 1
            err_col = eps[:, j] + lin_col
            sorted_col = np.sort(err_col)
            mc_std_item = float(err_col.std(ddof=1))
            mc_stds.append(mc_std_item)
            p_out_m, p_in_m = _mc_probs(sorted_col, xc, lsl, usl)
            item_out = {
                "serial": row["serial"],
                "measured_mm": x,
                "bias_correction_mm": float(ev["bias_correction_mm"]),
                "corrected_mm": xc,
                "decision": dec,
                "in_applicable_range": ev["in_applicable_range"],
                "bias_model_applied": model is not None,
                "linear_bias_std_uncertainty_mm": u_lin_item,
                "combined_std_uncertainty_mm": u_item,
                "expanded_uncertainty_mm": {
                    "gum": float(k_dim[j] * u_item),
                    "monte_carlo": float(k_dim[j] * mc_std_item),
                },
                "guard_band_mm": w,
                "p_true_out_of_spec": {
                    "gum": p_out_g, "monte_carlo": p_out_m},
                "p_true_conforming": {
                    "gum": p_in_g, "monte_carlo": p_in_m},
            }
            if model is not None and not ev["in_applicable_range"]:
                item_out["no_correction_reason"] = (
                    "实测值超出线性研究适用量程：不做偏倚修正、不外推")
            items.append(item_out)
        plan_dim = by_id[d.id]
        lin_model = plan_dim.get("linear_bias_model")
        # 尺寸级代表性不确定度：线性模型取各读数 MC σ 均值，否则常量 σ
        mc_std_repr = (
            float(np.mean(mc_stds)) if mc_stds and lin_model is not None
            else float(mc_std_dim_const[j]))
        u_gum_repr = math.sqrt(
            float(u_c[j]) ** 2
            + ((plan_dim.get("linear_bias_model") or {})
               .get("representative_bias_std_uncertainty_mm", 0.0)) ** 2
        ) if lin_model is not None else float(u_c[j])
        dim_entries.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "spec_mm": {"lsl": lsl, "usl": usl},
            "components_mm": plan_dim["components_mm"],
            "variance_share": plan_dim["variance_share"],
            "bias_correction_source": plan_dim.get(
                "bias_correction_source", "manual"),
            "linear_bias_model": lin_model,
            "combined_std_uncertainty_mm": u_gum_repr,
            "coverage_factor": float(k_dim[j]),
            "expanded_uncertainty_mm": float(k_dim[j] * u_gum_repr),
            "monte_carlo": {
                "std_mm": mc_std_repr,
                "expanded_uncertainty_mm": float(k_dim[j] * mc_std_repr),
                "samples": mc_samples,
                "seed": seed,
                "item_dependent": lin_model is not None,
            },
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
    closure_var_examples: list[float] = []
    for row in rows:
        vals = [row["values"].get(i) for i in ids]
        if any(v is None for v in vals):
            excluded.append(row["serial"])
            continue
        x = np.array([v["value_mm"] for v in vals], dtype=float)
        evs = [_correct_one(j, float(x[j])) for j in range(k)]
        xc = np.array([e["corrected_mm"] for e in evs], dtype=float)
        c_val = float(xc @ signs)
        # 线性模型项的封闭环 GUM 方差：Σ s_j²·u_lin,j(x_j)²（各研究独立）
        u_lin_vec = np.array(
            [e["correction_std_uncertainty_mm"] for e in evs], dtype=float)
        var_lin = float((signs ** 2 * u_lin_vec ** 2).sum())
        u_row = math.sqrt(var_c_const + var_lin)
        closure_var_examples.append(u_row ** 2)
        # MC 封闭环误差列：常量公共列 + 本行各读数线性系数误差列
        lin_closure_col = np.zeros(mc_samples)
        for j in lin_idx:
            model = lin_models[j]
            ev = evs[j]
            if ev["in_applicable_range"] and ev["correction_applied"]:
                a = float(model["intercept_mm"])
                bsl = float(model["slope"])
                denom = 1.0 + bsl
                grad_a = -1.0 / denom
                grad_b = -(float(x[j]) - a) / denom ** 2
                lin_closure_col += signs[j] * (
                    grad_a * model["_da"] + grad_b * model["_db"])
        closure_err_col = eps_closure_const + lin_closure_col
        sorted_c = np.sort(closure_err_col)
        mc_std_row = float(closure_err_col.std(ddof=1))
        if mode == "multiple":
            w_row = h * k_out * u_row
            w_row_mc = h * k_out * mc_std_row
        else:
            w_row = w_row_mc = float(guard_band["fixed_mm"])
        if has_closure_spec:
            dec = _decide(c_val, nc.closure_lsl_mm, nc.closure_usl_mm, w_row)
            c_counts[dec] += 1
            p_out_g, p_in_g = _gum_probs(
                c_val, nc.closure_lsl_mm, nc.closure_usl_mm, u_row)
            p_out_m, p_in_m = _mc_probs(
                sorted_c, c_val, nc.closure_lsl_mm, nc.closure_usl_mm)
        else:
            dec = None
            p_out_g = p_in_g = p_out_m = p_in_m = None
        item = {
            "serial": row["serial"],
            "corrected_closure_mm": c_val,
            "decision": dec,
            "combined_std_uncertainty_mm": {
                "gum": u_row, "monte_carlo": mc_std_row},
            "linear_bias_variance_mm2": var_lin,
            "expanded_uncertainty_mm": {
                "gum": k_out * u_row,
                "monte_carlo": k_out * mc_std_row},
            "guard_band_mm": {
                "gum": w_row if has_closure_spec else None,
                "monte_carlo": w_row_mc if has_closure_spec else None},
            "p_true_out_of_spec": (
                {"gum": p_out_g, "monte_carlo": p_out_m}
                if has_closure_spec else None
            ),
            "p_true_conforming": (
                {"gum": p_in_g, "monte_carlo": p_in_m}
                if has_closure_spec else None
            ),
        }
        if has_linear:
            item["dimension_corrections"] = [
                {"dimension_id": ids[j], "measured_mm": float(x[j]),
                 "corrected_mm": float(xc[j]),
                 "in_applicable_range": evs[j]["in_applicable_range"],
                 "linear_bias_std_uncertainty_mm":
                     float(evs[j]["correction_std_uncertainty_mm"])}
                for j in range(k)]
        closure_items.append(item)

    # 封闭环代表性 GUM 不确定度（无模型时即常量值）
    u_closure = (math.sqrt(float(np.mean(closure_var_examples)))
                 if closure_var_examples and has_linear else u_closure_const)
    if mode == "multiple":
        w_closure = h * k_out * u_closure
    else:
        w_closure = float(guard_band["fixed_mm"])
    mc_std_closure = (
        float(np.mean([it["combined_std_uncertainty_mm"]["monte_carlo"]
                       for it in closure_items]))
        if closure_items and has_linear else mc_std_closure_const)
    var_lin_typical = (
        float(np.mean([it["linear_bias_variance_mm2"]
                       for it in closure_items]))
        if closure_items and has_linear else 0.0)

    linearity_policy = None
    if has_linear:
        linearity_policy = {
            "model": "引用已采用线性与偏倚研究：偏倚 b(x)=a+b·x，"
                     "逆回归修正 x_c=(x−a)/(1+b)",
            "uncertainty": "回归系数与标准件参考值不确定度（GUM 协方差）"
                           "对整批共用，按各实测值梯度传播；不同研究间相互独立",
            "no_extrapolation": "实测值超出研究适用量程 [min,max 参考值] 时"
                                "不做修正、不传播模型不确定度（禁止外推）",
            "out_of_range_item_counts": out_of_range_notes,
        }
        for c in contributions:
            c["note"] = "linear_bias_variance 随实测值变化，见 closure.items[]"

    closure_block = {
        "spec_mm": {"lsl": nc.closure_lsl_mm, "usl": nc.closure_usl_mm},
        "gum": {
            "combined_std_uncertainty_mm": u_closure,
            "expanded_uncertainty_mm": k_out * u_closure,
            "variance_from_resolution_mm2": var_res,
            "variance_from_correlated_rest_mm2": var_rest,
            "variance_from_linear_bias_mm2": var_lin_typical,
            "dimension_contributions": contributions,
            "item_dependent": has_linear,
        },
        "monte_carlo": {
            "std_mm": mc_std_closure,
            "expanded_uncertainty_mm": k_out * mc_std_closure,
            "samples": mc_samples,
            "seed": seed,
            "item_dependent": has_linear,
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
            "bias_correction": (
                "引用线性与偏倚研究的尺寸按逆回归 x_c=(x−a)/(1+b) 修正"
                "（仅适用量程内，量程外不外推）；其余尺寸 x_c=x+b 常量修正，"
                "残余误差视为零均值，再做不确定度评定与判定"
                if has_linear else
                "x_c = x + b：先加偏倚修正值，"
                "残余误差视为零均值，再做不确定度评定与判定"),
            "output_coverage_factor": k_out,
            "coverage_factor_policy": "逐尺寸扩展不确定度使用方案中各量具的"
                                      "覆盖因子 k_i；封闭环跨多台量具，"
                                      "使用批次 output_coverage_factor；"
                                      "GUM 与蒙特卡洛同口径",
            "linearity_policy": linearity_policy,
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
