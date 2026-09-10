"""按单位收紧成本搜索达到目标超差率的候选组合。

成本模型
========
对每个尺寸 i，公差带半宽可取 T_i · level（level ∈ scale_levels）。
收紧量 Δ_i = T_i·(1 - level)（mm），成本
    cost_i = c_i · Δ_i
（c_i 为每 mm 收紧成本，可逐尺寸给出，否则用 default_cost）。

搜索
====
先用解析 RSS 正态近似做快速筛选（beam search，按成本保留若干束），
再对达到目标的低成本候选用固定种子蒙特卡洛复核，返回若干个
成本递增的可行候选组合。
"""
from __future__ import annotations

import math

import numpy as np

from .engine import (
    NormalizedChain,
    _override_chain,
    _s_arr,
    _sigma_arr,
    mean_gap,
    monte_carlo,
    normal_cdf,
    worst_case,
)


def _rss_reject(nc_ov: NormalizedChain, sigmas: np.ndarray) -> float | None:
    lsl, usl = nc_ov.closure_lsl_mm, nc_ov.closure_usl_mm
    if lsl is None and usl is None:
        return None
    s = _s_arr(nc_ov.dimensions)
    w = s * sigmas
    var = float((nc_ov.corr * w[:, None] * w[None, :]).sum())
    sigma_c = math.sqrt(max(var, 0.0))
    mu = mean_gap(nc_ov)
    p = 0.0
    if lsl is not None:
        p += (
            1.0 if sigma_c == 0 and mu < lsl
            else float(normal_cdf((lsl - mu) / sigma_c)) if sigma_c > 0 else 0.0
        )
    if usl is not None:
        p += (
            1.0 if sigma_c == 0 and mu > usl
            else float(1.0 - normal_cdf((usl - mu) / sigma_c))
            if sigma_c > 0 else 0.0
        )
    return p


def search_cost_targets(nc: NormalizedChain, request) -> dict:
    dims = nc.dimensions
    k = len(dims)
    levels = sorted(set(request.scale_levels))
    costs_per_mm = np.array([
        request.tightening_cost.get(d.id, request.default_cost) for d in dims
    ])
    base_halfs = np.array([d.half_width for d in dims])
    base_sigmas = _sigma_arr(dims)
    mids = np.array([d.mid for d in dims])

    # 每个尺寸的候选: (level, half, sigma, cost)
    options: list[list[tuple]] = []
    for i, d in enumerate(dims):
        col = []
        for lv in levels:
            half = d.half_width * lv
            sigma = _sigma_for_level_level(
                d.distribution, half, d.sigma, d.half_width,
                scale_normal=request.scale_normal_sigma,
            )
            cost = costs_per_mm[i] * (d.half_width - half)
            col.append((lv, half, sigma, cost))
        options.append(col)

    target = request.target_reject_rate

    # ---------------- beam search（RSS 代理） ----------------
    # 每扩展一个尺寸保留 beam_width 个成本最低的部分解
    beam_width = max(20, min(400, request.max_evaluations // max(k, 1) + 10))
    beam = [
        {
            "halfs": [],
            "sigmas": [],
            "cost": 0.0,
            "levels": [],
        }
    ]
    for i in range(k):
        candidates = []
        for state in beam:
            for lv, half, sigma, cost in options[i]:
                candidates.append({
                    "halfs": state["halfs"] + [half],
                    "sigmas": state["sigmas"] + [sigma],
                    "cost": state["cost"] + cost,
                    "levels": state["levels"] + [lv],
                })
        candidates.sort(key=lambda st: st["cost"])
        beam = candidates[:beam_width]

    feasible = []
    seen = set()
    eval_count = 0
    for state in sorted(beam, key=lambda st: st["cost"]):
        if eval_count >= request.max_evaluations:
            break
        eval_count += 1
        halfs = np.array(state["halfs"])
        sigmas = np.array(state["sigmas"])
        ov = _override_chain(nc, sigmas, mids, halfs)
        rss_rej = _rss_reject(ov, sigmas)
        key = tuple(round(v, 12) for v in state["levels"])
        if rss_rej is None:
            continue
        # RSS 近似偏乐观，留 20% 安全裕量作为初筛
        if rss_rej <= target * 0.8:
            if key not in seen:
                seen.add(key)
                feasible.append((state, ov, sigmas, halfs, rss_rej))
        if len(feasible) >= 30:
            break

    # ---------------- MC 复核 ----------------
    seed = nc.seed if request.random_seed is None else request.random_seed
    verified = []
    for state, ov, sigmas, halfs, rss_rej in feasible:
        mc = monte_carlo(ov, seed=seed, sigmas=sigmas, mids=mids, halfs=halfs)
        mc_rej = mc["reject_probability"]
        if mc_rej is not None and mc_rej <= target:
            wc = worst_case(ov)
            verified.append({
                "total_cost": round(state["cost"], 9),
                "mc_reject_rate": mc_rej,
                "rss_reject_rate_estimate": rss_rej,
                "worst_case_reject": wc["reject_probability"],
                "mc_sigma_mm": mc["sigma_mm"],
                "mc_bounds_mm": [mc["lower_bound_mm"], mc["upper_bound_mm"]],
                "combination": [
                    {
                        "dimension_id": d.id,
                        "tolerance_scale": lv,
                        "new_half_width_mm": float(h),
                        "new_tolerance_band_mm": float(2.0 * h),
                        "tightening_mm": float(d.half_width - h),
                        "unit_cost_per_mm": float(costs_per_mm[i]),
                        "cost": round(
                            costs_per_mm[i] * (d.half_width - h), 9),
                    }
                    for i, (d, lv, h) in enumerate(
                        zip(dims, state["levels"], halfs))
                ],
                "verification": {
                    "method": "monte_carlo",
                    "samples": mc["samples"],
                    "random_seed": mc["random_seed"],
                    "criterion": f"经验超差率 <= {target}",
                },
            })
        if len(verified) >= 10:
            break

    # 基线现状
    base_ov = _override_chain(nc, base_sigmas, mids, base_halfs)
    base_rss = _rss_reject(base_ov, base_sigmas)
    base_mc = monte_carlo(base_ov, seed=seed)
    return {
        "target_reject_rate": target,
        "evaluations_rss": eval_count,
        "cost_model": {
            "formula": "cost_i = c_i · T_i · (1 - level_i)，"
                       "T_i 为当前公差带半宽(mm)",
            "unit_cost_per_mm": {
                d.id: float(costs_per_mm[i]) for i, d in enumerate(dims)
            },
            "scale_normal_sigma": request.scale_normal_sigma,
        },
        "baseline": {
            "rss_reject_rate_estimate": base_rss,
            "mc_reject_rate": base_mc["reject_probability"],
            "mc_sigma_mm": base_mc["sigma_mm"],
        },
        "candidates": verified,
        "note": (
            "候选按总成本升序；RSS 正态近似初筛（含 0.8 安全裕量）后"
            "以固定种子蒙特卡洛复核，仅保留经验超差率达标组合。"
            "正态分布默认仅收紧公差带、不缩放用户给定 σ；"
            "scale_normal_sigma=true 时正态 σ 随公差带等比缩放，"
            "均匀/三角分布的 σ 始终由公差带理论决定。"
        ),
    }


def _sigma_for_level_level(distribution: str, half: float,
                           base_sigma: float, base_half: float,
                           scale_normal: bool = False) -> float:
    if distribution == "uniform":
        return half / math.sqrt(3.0)
    if distribution == "triangular":
        return half / math.sqrt(6.0)
    if distribution == "normal" and scale_normal and base_half > 0:
        return base_sigma * (half / base_half)
    return base_sigma
