"""来料检验（IQC）批次统计：实测行复核基线公差链。

数据口径
========
* 每行测量值按提交单位换算为 mm；规格限取基线尺寸的
  LSL_i = N_i + EI_i，USL_i = N_i + ES_i，带中心 C_i = N_i + m_i。
* 均值偏移 Δ_i = x̄_i − C_i（相对基线制程中心，与 RSS 的 μ_i 同口径）。
* 样本标准差 s_i = sqrt(Σ(x−x̄)²/(n−1))，n < 2 不给。
* 超差：x < LSL 或 x > USL（边界合格，与蒙特卡洛经验超差口径一致）。
* Cp  = (USL−LSL)/(6s)；Cpk = min((USL−x̄)/(3s), (x̄−LSL)/(3s))。
* 只有**所有尺寸齐全**的工件行（complete case）参与协方差与封闭环，
  缺测行列在 gaps / excluded_serials 中并注明剔除原因。

置信区间（95%）
===============
* Cp：(n−1)s²/σ² ~ χ²(n−1)，故
  [Cp·√(χ²_{0.025}/ν), Cp·√(χ²_{0.975}/ν)]，ν = n−1。
* Cpk：Bissell 正态近似
  SE = sqrt((1 + 9·Cpk²)/(9n) + 1/(2(n−1)))，区间 Cpk ± 1.96·SE。
* 封闭环：固定种子非参数 bootstrap（对完整工件行重采样），
  均值与超规格比例的 95% 区间取重采样统计量的 2.5%/97.5% 分位点；
  封闭环上下分位点取各重采样样本 0.5%/99.5% 分位点的 bootstrap 均值。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .engine import NormDimension, NormalizedChain
from .units import from_mm

Z975 = 1.959963984540054  # 标准正态 97.5% 分位（置信区间用）

# bootstrap 分块上限：B × n 超过该值时分块收集，避免一次性占满内存
_BOOT_POOL_CAP = 2_000_000


# ------------------------------------------------------- χ² 分位数（无 scipy）

def _gammp(a: float, x: float) -> float:
    """正则化下不完全伽马函数 P(a,x)（Numerical Recipes gser/gcf）。"""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        ap = a
        s = 1.0 / a
        term = 1.0 / a
        for _ in range(500):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_ppf(p: float, df: int) -> float:
    """χ²(df) 的 p 分位数：在正则化伽马上二分（Cp 置信区间用）。"""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.inf
    a = df / 2.0

    def f(x: float) -> float:
        return _gammp(a, x / 2.0) - p

    lo, hi = 0.0, max(1.0, float(df))
    while f(hi) < 0.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- 输入复核

def validate_rows(nc: NormalizedChain, payload) -> list[dict[str, Any]]:
    """把 API 行结构核对基线链并换算为内部表示。

    返回 [{serial, values: {dim_id: {"value_mm", "value", "unit"} | None}}]；
    链外尺寸、重复序号直接抛 ValueError（FastAPI 转 422）。
    单位合法性 / 非有限值 / 行内重复尺寸已由 Pydantic 拦截。
    """
    from .units import to_mm

    chain_ids = {d.id for d in nc.dimensions}
    problems: list[str] = []

    seen_serials: set[str] = set()
    dup_serials: list[str] = []
    for row in payload.rows:
        if row.serial in seen_serials and row.serial not in dup_serials:
            dup_serials.append(row.serial)
        seen_serials.add(row.serial)
    if dup_serials:
        problems.append(f"工件序号重复: {dup_serials}")

    rows: list[dict[str, Any]] = []
    external: list[str] = []
    for row in payload.rows:
        values: dict[str, dict[str, Any] | None] = {}
        for m in row.measurements:
            if m.dimension_id not in chain_ids:
                external.append(f"工件 {row.serial} 的链外尺寸 {m.dimension_id!r}")
                continue
            if m.value is None:
                values[m.dimension_id] = None
            else:
                u = m.unit.value if hasattr(m.unit, "value") else str(m.unit)
                values[m.dimension_id] = {
                    "value_mm": to_mm(m.value, u),
                    "value": m.value,
                    "unit": u,
                }
        rows.append({"serial": row.serial, "values": values})
    if external:
        problems.append("测量行包含基线链之外的尺寸（已拒收）: "
                        + "; ".join(sorted(set(external))))
    if problems:
        raise ValueError("；".join(problems))
    return rows


# ------------------------------------------------------------- 单尺寸统计

def _in_native_unit(d: NormDimension, value_mm: float) -> float:
    return from_mm(value_mm, d.unit)


def analyze_dimension(d: NormDimension,
                      points: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """points 为该尺寸有实测值的 (serial, {value_mm,value,unit}) 列表。"""
    n = len(points)
    lsl = d.nominal + d.lower_dev
    usl = d.nominal + d.upper_dev
    center = d.nominal + d.mid
    spec_width = usl - lsl

    out: dict[str, Any] = {
        "dimension_id": d.id,
        "edge": [d.start, d.end],
        "sign": d.sign,
        "sample_count": n,
        "spec_mm": {"lsl": lsl, "usl": usl, "center": center,
                    "nominal": d.nominal, "half_width": d.half_width},
        "spec_in_dimension_unit": {
            "unit": d.unit,
            "lsl": _in_native_unit(d, lsl),
            "usl": _in_native_unit(d, usl),
            "center": _in_native_unit(d, center),
            "nominal": _in_native_unit(d, d.nominal),
        },
        "warnings": [],
        "mean_mm": None,
        "mean_in_dimension_unit": None,
        "mean_shift_from_center_mm": None,
        "mean_shift_from_nominal_mm": None,
        "sample_std_mm": None,
        "sample_std_in_dimension_unit": None,
        "out_of_spec": False,
        "out_of_spec_count": 0,
        "out_of_spec_serials": [],
        "cp": None,
        "cp_ci95": None,
        "cpk": None,
        "cpk_ci95": None,
        "capability_note": None,
    }

    if n == 0:
        out["capability_note"] = "样本不足（n=0，该尺寸全部缺测）：" \
                                 "无法计算均值、标准差与能力指数"
        out["warnings"].append("n=0：无任何实测值")
        return out

    vals = np.array([p[1]["value_mm"] for p in points], dtype=float)
    mean = float(vals.mean())
    out["mean_mm"] = mean
    out["mean_in_dimension_unit"] = _in_native_unit(d, mean)
    out["mean_shift_from_center_mm"] = mean - center
    out["mean_shift_from_nominal_mm"] = mean - d.nominal

    # 超差明细（边界合格）
    for serial, m in points:
        v = m["value_mm"]
        if v < lsl or v > usl:
            out["out_of_spec_serials"].append({
                "serial": serial,
                "value_mm": v,
                "submitted": {"value": m["value"], "unit": m["unit"]},
                "side": "below_lsl" if v < lsl else "above_usl",
            })
    out["out_of_spec_count"] = len(out["out_of_spec_serials"])
    out["out_of_spec"] = out["out_of_spec_count"] > 0

    notes: list[str] = []
    if n < 2:
        s = None
        out["capability_note"] = (
            "样本不足（n=1）：仅给出均值，样本标准差 / Cp / Cpk 无法估计"
        )
        notes.append(out["capability_note"])
    else:
        s = float(vals.std(ddof=1))
        out["sample_std_mm"] = s
        out["sample_std_in_dimension_unit"] = _in_native_unit(d, s)
        nu = n - 1
        if s <= 0.0:
            notes.append("样本标准差为 0：Cp/Cpk 无定义（零散布，"
                         "按超差标记判定合格性）")
        else:
            if spec_width > 0.0:
                cp = spec_width / (6.0 * s)
                chi_lo = chi2_ppf(0.025, nu)
                chi_hi = chi2_ppf(0.975, nu)
                out["cp"] = cp
                cp_ci_hi = cp * math.sqrt(chi_hi / nu)
                out["cp_ci95"] = {
                    "lower": cp * math.sqrt(chi_lo / nu),
                    "upper": cp_ci_hi,
                    "confidence": 0.95,
                }
            else:
                notes.append("规格宽度为 0（上下偏差相等）：Cp 无定义")
            cpk = min((usl - mean) / (3.0 * s), (mean - lsl) / (3.0 * s))
            se = math.sqrt((1.0 + 9.0 * cpk * cpk) / (9.0 * n)
                           + 1.0 / (2.0 * nu))
            out["cpk"] = cpk
            out["cpk_ci95"] = {
                "lower": cpk - Z975 * se,
                "upper": cpk + Z975 * se,
                "confidence": 0.95,
            }
            if out["cpk_ci95"]["lower"] < 0.0:
                notes.append("Cpk 置信区间下限 < 0：制程中心偏移或样本不足，"
                             "能力判定不可靠")
        if n < 30:
            notes.append(
                f"样本量 n={n} < 30：Cp/Cpk 及其 95% 置信区间仅供参考"
            )
        out["capability_note"] = "；".join(notes) if notes else None
    if n < 30:
        out["warnings"].append(f"小样本 n={n}（<30），统计量不稳定")
    if out["out_of_spec"]:
        out["warnings"].append(
            f"{out['out_of_spec_count']}/{n} 个实测值超出规格限"
        )
    return out


# ------------------------------------------------------------- 协方差 / 封闭环

def _complete_matrix(nc: NormalizedChain, rows: list[dict[str, Any]]
                     ) -> tuple[np.ndarray | None, list[str], list[dict]]:
    """抽取所有尺寸齐全的工件行，返回 (n×k mm 矩阵, 完整序号列表, 缺口列表)。"""
    ids = [d.id for d in nc.dimensions]
    matrix_rows: list[list[float]] = []
    serials: list[str] = []
    gaps: list[dict] = []
    for row in rows:
        missing = [dim for dim in ids
                   if row["values"].get(dim) is None]
        if missing:
            gaps.append({"serial": row["serial"],
                         "missing_dimension_ids": missing})
        else:
            matrix_rows.append([row["values"][dim]["value_mm"] for dim in ids])
            serials.append(row["serial"])
    if not matrix_rows:
        return None, serials, gaps
    return np.array(matrix_rows, dtype=float), serials, gaps


def bootstrap_closure(closure: np.ndarray, lsl: float | None,
                      usl: float | None, b_samples: int, seed: int) -> dict:
    """固定种子非参数 bootstrap：对完整工件行的封闭环值重采样。

    返回均值（含 95% CI）、上下分位点（0.5/97.5% 体系与基线蒙特卡洛对齐）、
    超规格比例及其 95% CI；三者均为重采样统计量分布的经验分位点。
    """
    n = len(closure)
    rng = np.random.default_rng(seed)
    means = np.empty(b_samples)
    props = np.empty(b_samples)
    q_low = np.empty(b_samples)
    q_high = np.empty(b_samples)
    has_spec = lsl is not None or usl is not None

    # 分块：每块最多 _BOOT_POOL_CAP/n 次重采样，内存有界
    chunk = max(1, _BOOT_POOL_CAP // max(n, 1))
    done = 0
    while done < b_samples:
        b = min(chunk, b_samples - done)
        idx = rng.integers(0, n, size=(b, n))
        draws = closure[idx]                                  # b×n
        means[done:done + b] = draws.mean(axis=1)
        qs = np.quantile(draws, [0.005, 0.995], axis=1)
        q_low[done:done + b] = qs[0]
        q_high[done:done + b] = qs[1]
        if has_spec:
            bad = np.zeros((b, n), dtype=bool)
            if lsl is not None:
                bad |= draws < lsl
            if usl is not None:
                bad |= draws > usl
            props[done:done + b] = bad.mean(axis=1)
        else:
            props[done:done + b] = np.nan
        done += b

    obs_mean = float(closure.mean())
    obs_std = float(closure.std(ddof=1)) if n >= 2 else 0.0
    obs_q = np.quantile(closure, [0.005, 0.995])
    if has_spec:
        bad = np.zeros(n, dtype=bool)
        if lsl is not None:
            bad |= closure < lsl
        if usl is not None:
            bad |= closure > usl
        obs_prop = float(bad.mean())
        prop_ci = [float(np.nanquantile(props, 0.025)),
                   float(np.nanquantile(props, 0.975))]
    else:
        obs_prop = None
        prop_ci = None

    return {
        "method": "bootstrap",
        "random_seed": seed,
        "bootstrap_samples": b_samples,
        "complete_rows": int(n),
        "observed_mean_mm": obs_mean,
        "observed_std_mm": obs_std,
        "bootstrap_mean_mm": float(means.mean()),
        "mean_ci95_mm": [float(np.quantile(means, 0.025)),
                         float(np.quantile(means, 0.975))],
        "lower_quantile_mm": float(q_low.mean()),
        "upper_quantile_mm": float(q_high.mean()),
        "bounds_quantile": [0.005, 0.995],
        "observed_quantile_mm": [float(obs_q[0]), float(obs_q[1])],
        "out_of_spec_proportion": obs_prop,
        "out_of_spec_proportion_ci95": prop_ci,
        "formulas": [
            "非参数 bootstrap：固定种子 np.random.default_rng(seed)，"
            "对 n 个完整工件行的封闭环 C=Σs_i·X_i 有放回重采样 B 次",
            "均值估计 = 重采样均值的均值；均值 95% CI = 重采样均值的"
            " 2.5%/97.5% 分位点（percentile bootstrap）",
            "上下分位点 = 各重采样样本 0.5%/99.5% 分位点在 B 次上的均值"
            "（与基线蒙特卡洛的 0.5/99.5% 分位同口径）",
            "超规格比例 p̂ = #{C<LSL 或 C>USL}/n；其 95% CI = B 次重采样"
            "比例的 2.5%/97.5% 分位点",
        ],
    }


# ------------------------------------------------------------- 批次总装

FORMULAS_GLOBAL = [
    "规格限: LSL_i=N_i+EI_i, USL_i=N_i+ES_i；带中心 C_i=N_i+(ES_i+EI_i)/2",
    "均值偏移: Δ_i = x̄_i − C_i（与 RSS 制程中心假设同口径）",
    "样本标准差: s_i = sqrt(Σ(x−x̄)²/(n−1))，n<2 不估计",
    "Cp = (USL−LSL)/(6s)；Cpk = min((USL−x̄)/(3s), (x̄−LSL)/(3s))",
    "Cp 95% CI: [Cp·√(χ²_{0.025,ν}/ν), Cp·√(χ²_{0.975,ν}/ν)]，ν=n−1"
    "（(n−1)s²/σ²~χ²(ν)）",
    "Cpk 95% CI (Bissell): Cpk ± 1.96·sqrt((1+9Cpk²)/(9n)+1/(2(n−1)))",
    "协方差: 仅用所有尺寸齐全的工件行，Σ=(X−x̄)ᵀ(X−x̄)/(n−1)",
    "封闭环: C=Σs_i·X_i；固定种子 bootstrap 重采样完整工件行",
]


def analyze_batch(nc: NormalizedChain, rows: list[dict[str, Any]],
                  b_samples: int, seed: int) -> dict[str, Any]:
    """对已校验的测量行做全部统计（数值单位均为 mm，另附尺寸原单位）。"""
    # ---- 逐尺寸（缺测行不参与该尺寸）
    by_dim: dict[str, list[tuple[str, dict[str, Any]]]] = {
        d.id: [] for d in nc.dimensions}
    for row in rows:
        for dim, m in row["values"].items():
            if m is not None:
                by_dim[dim].append((row["serial"], m))
    dim_results = [analyze_dimension(d, by_dim[d.id]) for d in nc.dimensions]

    # ---- 完整工件行 -> 协方差 / 封闭环
    x, complete_serials, gaps = _complete_matrix(nc, rows)
    excluded = [g["serial"] for g in gaps]
    ids = [d.id for d in nc.dimensions]
    k = len(ids)
    warnings: list[str] = []

    covariance_block: dict[str, Any]
    closure_block: dict[str, Any] | None = None
    closure_reason: str | None = None

    n_complete = 0 if x is None else x.shape[0]
    if x is None or n_complete < 2:
        covariance_block = {
            "available": False,
            "dimension_order": ids,
            "complete_serials": complete_serials,
            "n_complete": n_complete,
            "excluded_serials": excluded,
            "exclusion_reason": (
                "至少 2 个所有尺寸齐全的工件行才能以 n−1 自由度估计协方差"
            ),
            "matrix_mm2": None,
            "correlation_matrix": None,
        }
        closure_reason = (
            f"完整配对不足：仅 {n_complete} 个工件行测齐全部 {k} 个尺寸"
            f"（需要 ≥2）。已剔除缺测工件行: "
            f"{excluded if excluded else '无'}；本次只返回单尺寸结果，"
            "不进行协方差估计与封闭环 bootstrap"
        )
        warnings.append(closure_reason)
    else:
        signs = np.array([d.sign for d in nc.dimensions], dtype=float)
        cov = np.cov(x, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
        sd = np.sqrt(np.diag(cov))
        denom = np.outer(sd, sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0),
                            0.0)
        # 零方差尺寸与自身的相关定义为 1（其与其它尺寸的相关记 0）
        np.fill_diagonal(corr, 1.0)
        closure_values = x @ signs
        closure_block = bootstrap_closure(
            closure_values, nc.closure_lsl_mm, nc.closure_usl_mm,
            b_samples, seed)
        closure_block["excluded_serials"] = excluded
        closure_block["exclusion_reason"] = (
            "以下工件行存在缺测尺寸，未参与协方差与封闭环 bootstrap"
        ) if excluded else None
        closure_block["spec_mm"] = {
            "lsl": nc.closure_lsl_mm,
            "usl": nc.closure_usl_mm,
        }
        covariance_block = {
            "available": True,
            "dimension_order": ids,
            "complete_serials": complete_serials,
            "n_complete": n_complete,
            "excluded_serials": excluded,
            "exclusion_reason": (
                "以下工件行存在缺测尺寸，未参与协方差估计"
            ) if excluded else None,
            "matrix_mm2": cov.tolist(),
            "correlation_matrix": corr.tolist(),
            "formula": "Σ_ij = Σ_p (X_pi−x̄_i)(X_pj−x̄_j)/(n_complete−1)，"
                       "仅用完整工件行",
        }
        if nc.closure_lsl_mm is None and nc.closure_usl_mm is None:
            warnings.append("基线链未声明封闭环上下限：超规格比例为 null，"
                            "仅报告分布统计")

    per_dim_n = {r["dimension_id"]: r["sample_count"] for r in dim_results}
    for r in dim_results:
        if r["sample_count"] < 30:
            warnings.append(
                f"尺寸 {r['dimension_id']} 样本 n={r['sample_count']} < 30"
            )
        if r["out_of_spec"]:
            warnings.append(
                f"尺寸 {r['dimension_id']} 有 {r['out_of_spec_count']} 个"
                "实测值超出规格限"
            )

    return {
        "sample_summary": {
            "rows_total": len(rows),
            "complete_rows": n_complete,
            "rows_with_gaps": len(gaps),
            "per_dimension_count": per_dim_n,
        },
        "gaps": gaps,
        "dimension_results": dim_results,
        "covariance": covariance_block,
        "closure_analysis": closure_block,
        "closure_unavailable_reason": closure_reason,
        "warnings": warnings,
        "formulas": FORMULAS_GLOBAL,
        "bootstrap_config": {"samples": b_samples, "seed": seed},
    }


# ------------------------------------------------------------- 与基线并列

def baseline_comparison(nc: NormalizedChain, chain_result: dict,
                        report: dict[str, Any]) -> dict[str, Any]:
    """实测结论与基线 RSS / 蒙特卡洛并列（注明样本数、剔除原因与公式）。"""
    rss = chain_result["results"]["rss"]
    mc = chain_result["results"]["monte_carlo"]
    wc = chain_result["results"]["worst_case"]
    closure = report["closure_analysis"]

    observed = None
    if closure is not None:
        observed = {
            "source": "incoming_measurement_bootstrap",
            "sample_count": closure["complete_rows"],
            "mean_mm": closure["bootstrap_mean_mm"],
            "std_mm": closure["observed_std_mm"],
            "lower_quantile_mm": closure["lower_quantile_mm"],
            "upper_quantile_mm": closure["upper_quantile_mm"],
            "reject_probability": closure["out_of_spec_proportion"],
            "reject_probability_ci95":
                closure["out_of_spec_proportion_ci95"],
            "mean_ci95_mm": closure["mean_ci95_mm"],
            "bootstrap_samples": closure["bootstrap_samples"],
            "random_seed": closure["random_seed"],
        }

    cov = report["covariance"]
    return {
        "closure_spec_mm": chain_result["closure_spec_mm"],
        "methods": {
            "baseline_worst_case": {
                "display_name": "基线极值法",
                "mean_mm": wc["mean_gap_mm"],
                "lower_bound_mm": wc["lower_bound_mm"],
                "upper_bound_mm": wc["upper_bound_mm"],
                "reject_probability": wc["reject_probability"],
            },
            "baseline_rss": {
                "display_name": "基线 RSS（正态近似，设计公差/理论或给定 σ）",
                "sample_count": None,
                "mean_mm": rss["mean_gap_mm"],
                "sigma_mm": rss["sigma_mm"],
                "lower_bound_mm": rss["lower_bound_mm"],
                "upper_bound_mm": rss["upper_bound_mm"],
                "reject_probability": rss["reject_probability"],
                "formulas": rss["formulas"],
            },
            "baseline_monte_carlo": {
                "display_name": "基线固定种子蒙特卡洛",
                "sample_count": mc["samples"],
                "random_seed": mc["random_seed"],
                "mean_mm": mc["mean_gap_mm"],
                "sigma_mm": mc["sigma_mm"],
                "lower_quantile_mm": mc["lower_bound_mm"],
                "upper_quantile_mm": mc["upper_bound_mm"],
                "reject_probability": mc["reject_probability"],
            },
            "observed_bootstrap": observed,
        },
        "observed_unavailable_reason": report["closure_unavailable_reason"],
        "excluded_serials": cov["excluded_serials"],
        "exclusion_reason": cov["exclusion_reason"],
        "sample_counts": {
            "baseline_monte_carlo": mc["samples"],
            "complete_workpiece_rows": cov["n_complete"],
            "rows_total": report["sample_summary"]["rows_total"],
            "per_dimension": report["sample_summary"]["per_dimension_count"],
        },
        "interpretation_note": (
            "基线 RSS/蒙特卡洛基于设计公差带与声明的分布/σ；"
            "实测 bootstrap 完全来自本批工件实测值，二者差异即来料相对"
            "基线的中心偏移与散布变化。实测超差为经验比例（含 95% CI），"
            "基线超差率为模型概率，口径不可直接等同"
        ),
    }
