"""温度工况热分析引擎（内部统一 mm / °C / K^-1）。

热模型（每个尺寸 i）
====================
参考温度 T0_i 下的基线尺寸（与基线链同一口径）：

    L0_i = N_i + m_i + δ_i（δ_i 为制造偏差样本，分布同基线）
    名义热态长度  L_i(T) = L0_i · [1 + α_i·(T_i − T0_i)]

膨胀系数与温度本身带不确定度（GUM 一阶线性传播）：

    ∂L/∂α = L0·ΔT ，  ∂L/∂T = L0·α
    u²_热,i = (L0·ΔT)²·u(α)² + (L0·α)²·u(T)²

工况温度两种形式：

* **固定温度** mean + 标准不确定度 u(T)：RSS 取 u(T)，
  极值法（WC）按 3·u(T) 的扩展界；
* **温度区间** [lower, upper]：硬边界，WC 取半宽 h_T（含端点），
  RSS 按均匀分布取 h_T/√3，蒙特卡洛在区间内独立均匀抽样、不参与相关传播。

封闭环  C = Σ s_i·L_i + s_shim·t_shim（垫片在装配基准温度下为固定尺寸）。

三个不确定度来源（贡献拆分）
----------------------------
1. 制造偏差：沿用基线链的分布与相关矩阵，热态下按 1+αΔT 等比缩放；
2. 膨胀系数：u(α) 之间可用相关矩阵描述（多元正态抽样）；
3. 温度：固定温度 u(T) 的相关矩阵传播；区间温度为独立均匀硬边界。

    σ²_C = σ²_mfg + σ²_α + σ²_T

蒙特卡洛固定种子，四个独立子流（SeedSequence 派生）：组合样本按完整
非线性 L0(1+αΔT) 生成（制造偏差只乘一次热态比例）；另给三个分量的
零均值偏差样本。分量超差率在「该工况热态名义间隙 + 单项偏差」上对照
规格计算，而非把零均值偏差直接当作绝对间隙。

装配基准温度 T_a：把每个尺寸的参考温度重置为 T_a（重新基准化），
ΔT 以 T_a 为公共零点重算；垫片厚度定义在 T_a、不建模热膨胀。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import (
    NormalizedChain,
    _cholesky_psd,
    _sampling_half_widths,
    _s_arr,
    _triangular_inverse,
    latent_correlation_matrix,
    normal_cdf,
)
from .schemas import TO_PER_K, DistributionType
from .units import to_mm

Z_EXT = 3.0  # 标准不确定度在极值界中的扩展倍数（与 RSS ±3σ 同口径）

C_TO_K = 273.15
# 数值保护：1+αΔT 必须为正（负的或零热膨胀长度无物理意义）
MIN_SCALE = 1e-9
MAX_PLANS_ANALYTIC = 5_000     # 解析评估组合上限
MAX_PLANS_TOTAL = 20_000       # 笛卡尔积枚举上限
TOP_PLANS_MC = 40              # 解析排序后进入 MC 复核的方案数


class ThermalError(ValueError):
    """热分析输入 / 求解错误（API 层映射为 422）。"""


def _margin_key(m: float) -> float:
    """排序用最小余量：+∞（无规格）映射为有限大数，避免 -∞ 污染排序。"""
    return m if math.isfinite(m) else 1e18


# ------------------------------------------------------------------ 数据结构

@dataclass
class CTemp:
    """单个尺寸在某工况下的温度（统一 °C，温差 K/°C 数值相同）。"""

    kind: str                        # "fixed" | "range"
    mean: float                      # fixed: 均值；range: 区间中点
    half: float                      # fixed: 0；range: 区间半宽
    sigma: float                     # fixed: u(T)；range: h/√3
    raw: dict = field(default_factory=dict)


@dataclass
class ThermalModel:
    """归一化后的热分析模型（全部内部单位）。"""

    nc: NormalizedChain
    ids: list[str]
    t0_c: np.ndarray                 # 参考温度 °C
    alpha: np.ndarray                # K^-1
    u_alpha: np.ndarray              # K^-1
    alpha_unit: list[str]
    t0_unit: list[str]
    conditions: list[list[CTemp]]    # [工况][尺寸]
    condition_names: list[str]
    condition_notes: list[str]
    condition_temp_units: list[str]
    condition_raw: list[list[dict]]
    r_alpha: np.ndarray
    r_temp: np.ndarray
    mc_samples: int
    seed: int


# ------------------------------------------------------------------ 归一化

def _to_celsius(value: float, unit: str) -> float:
    return value - C_TO_K if unit == "K" else float(value)


def _one_ctemp(spec, cond_name: str, dim_id: str, temp_unit: str) -> CTemp:
    if spec.fixed is not None:
        f = spec.fixed
        return CTemp(
            kind="fixed",
            mean=_to_celsius(f.mean, temp_unit),
            half=0.0, sigma=float(f.std_uncertainty),
            raw={"mode": "fixed", "mean": f.mean,
                 "std_uncertainty": f.std_uncertainty, "unit": temp_unit})
    r = spec.range
    half = (r.upper - r.lower) / 2.0
    return CTemp(
        kind="range",
        mean=_to_celsius((r.lower + r.upper) / 2.0, temp_unit),
        half=float(half), sigma=float(half) / math.sqrt(3.0),
        raw={"mode": "range", "lower": r.lower, "upper": r.upper,
             "unit": temp_unit})


def build_model(nc: NormalizedChain, payload) -> ThermalModel:
    """校验覆盖关系并把 API 请求归一化为 ThermalModel。"""
    chain_ids = [d.id for d in nc.dimensions]
    chain_set = set(chain_ids)

    given = [d.dimension_id for d in payload.dimensions]
    given_set = set(given)
    missing = sorted(chain_set - given_set)
    extra = sorted(given_set - chain_set)
    if missing:
        raise ThermalError(f"漏填尺寸的热参数: {missing}")
    if extra:
        raise ThermalError(f"热参数引用了基线链外尺寸: {extra}")
    spec_by_id = {d.dimension_id: d for d in payload.dimensions}

    t0_c = np.zeros(len(chain_ids))
    alpha = np.zeros(len(chain_ids))
    u_alpha = np.zeros(len(chain_ids))
    alpha_unit, t0_unit = [], []
    for i, dim_id in enumerate(chain_ids):
        spec = spec_by_id[dim_id]
        t0_c[i] = _to_celsius(
            spec.reference_temperature, spec.reference_temperature_unit.value)
        factor = TO_PER_K[spec.alpha_unit.value]
        alpha[i] = spec.alpha * factor
        u_alpha[i] = spec.alpha_std_uncertainty * factor
        alpha_unit.append(spec.alpha_unit.value)
        t0_unit.append(spec.reference_temperature_unit.value)

    idx = {name: i for i, name in enumerate(chain_ids)}
    n = len(chain_ids)

    def corr_matrix(corrs):
        r = np.eye(n)
        for c in corrs:
            i, j = idx[c.dim_a], idx[c.dim_b]
            r[i, j] = r[j, i] = c.rho
        return r

    r_alpha = corr_matrix(payload.alpha_correlations)
    r_temp = corr_matrix(payload.temperature_correlations)

    conditions, names, notes, cunits, craw = [], [], [], [], []
    for cond in payload.conditions:
        temps, raws = [], []
        for dim_id in chain_ids:
            spec = cond.dimension_temperatures.get(dim_id,
                                                    cond.global_temperature)
            if spec is None:
                raise ThermalError(
                    f"工况 {cond.name}: 尺寸 {dim_id} 未提供温度"
                    "（既无逐尺寸温度也无全局温度）")
            ct = _one_ctemp(spec, cond.name, dim_id,
                            cond.temperature_unit.value)
            temps.append(ct)
            raws.append(ct.raw)
        conditions.append(temps)
        names.append(cond.name)
        notes.append(cond.note)
        cunits.append(cond.temperature_unit.value)
        craw.append(raws)

    return ThermalModel(
        nc=nc, ids=chain_ids, t0_c=t0_c, alpha=alpha, u_alpha=u_alpha,
        alpha_unit=alpha_unit, t0_unit=t0_unit,
        conditions=conditions, condition_names=names,
        condition_notes=notes, condition_temp_units=cunits,
        condition_raw=craw, r_alpha=r_alpha, r_temp=r_temp,
        mc_samples=payload.mc_samples, seed=payload.random_seed)


# ------------------------------------------------------- 解析（WC / RSS）

def _corr_variance(weight: np.ndarray, signs: np.ndarray,
                   corr: np.ndarray) -> tuple[float, np.ndarray]:
    """返回封闭环方差与逐尺寸贡献行：s_i·w_i·Σ_j ρ_ij s_j w_j。"""
    sw = signs * weight
    cov = corr * sw[:, None] * sw[None, :]
    return float(cov.sum()), cov.sum(axis=1)


def _scale_positivity(scale: np.ndarray, where) -> None:
    bad = scale <= 0
    if bad.any():
        ids = [where[i] for i in np.nonzero(bad)[0]]
        raise ThermalError(
            "热膨胀系数 × 温差使长度非正（1+αΔT ≤ 0），"
            f"请检查参考温度与工况温度: 尺寸 {ids}，scale={scale[bad].tolist()}")


def evaluate_condition(model: ThermalModel, ci: int,
                       t0: np.ndarray | None = None,
                       alpha: np.ndarray | None = None,
                       u_alpha: np.ndarray | None = None,
                       shim_nominal: float = 0.0,
                       shim_sigma: float = 0.0,
                       shim_sign: int = 1,
                       mc_samples: int | None = None,
                       mc_seed_base: int | None = None) -> dict:
    """单工况解析 WC/RSS + 固定种子蒙特卡洛（含分量拆分）。

    t0/alpha/u_alpha 可覆盖模型值（方案搜索：装配基准温度重基准化 / 换材料）。
    垫片以 s_shim·(nominal + N(0, σ²)) 叠加到封闭环，所有工况相同。
    """
    nc = model.nc
    nd = nc.dimensions
    k = len(nd)
    signs = _s_arr(nd)
    if t0 is None:
        t0 = model.t0_c
    if alpha is None:
        alpha = model.alpha
    if u_alpha is None:
        u_alpha = model.u_alpha

    temps = model.conditions[ci]
    t_mean = np.array([t.mean for t in temps])
    t_sigma = np.array([t.sigma for t in temps])
    t_half = np.array([t.half for t in temps])
    is_range = np.array([t.kind == "range" for t in temps])

    d_t = t_mean - t0                                   # ΔT_i (°C == K)
    scale = 1.0 + alpha * d_t
    _scale_positivity(scale, model.ids)
    l0_mean = np.array([d.nominal + d.mid for d in nd])  # 基线带中心
    l0_nom = np.array([d.nominal for d in nd])

    coeff_alpha = l0_mean * d_t                          # ∂L/∂α
    coeff_t = l0_mean * alpha                            # ∂L/∂T

    # ---------- 均值与名义值 ----------
    nominal = float(signs @ (l0_nom * scale)) + shim_sign * shim_nominal
    mean = float(signs @ (l0_mean * scale)) + shim_sign * shim_nominal

    # ---------- 制造偏差方差（热态等比缩放，沿用基线相关与分布）----------
    mfg_weight = np.array([d.sigma for d in nd]) * scale
    var_mfg, contrib_mfg = _corr_variance(
        mfg_weight, signs, nc.corr)
    var_mfg += shim_sigma ** 2

    # ---------- 膨胀系数不确定度 ----------
    w_alpha = coeff_alpha * u_alpha
    var_alpha, contrib_alpha = _corr_variance(
        w_alpha, signs, model.r_alpha)

    # ---------- 温度不确定度（区间独立均匀，固定温度走相关矩阵）----------
    t_weight = np.where(is_range, 0.0, coeff_t * t_sigma)
    var_temp_fixed, contrib_temp_fixed = _corr_variance(
        t_weight, signs, model.r_temp)
    # 区间温度独立：方差直接累加
    range_w = coeff_t * np.where(is_range, t_sigma, 0.0)
    var_temp_range = float((signs * range_w) ** 2 @ np.ones(k))
    contrib_temp = contrib_temp_fixed + (signs * range_w) ** 2
    var_temp = var_temp_fixed + var_temp_range

    var_total = var_mfg + var_alpha + var_temp
    sigma_total = math.sqrt(max(var_total, 0.0))
    sigma_mfg = math.sqrt(max(var_mfg, 0.0))
    sigma_alpha = math.sqrt(max(var_alpha, 0.0))
    sigma_temp = math.sqrt(max(var_temp, 0.0))

    # ---------- 极值法界 ----------
    # 制造公差带：热态半宽 = 基线半宽 × scale
    wc_mfg = float(np.abs(signs) @ (
        np.array([d.half_width for d in nd]) * scale)) + Z_EXT * shim_sigma
    wc_alpha = Z_EXT * math.sqrt(max(var_alpha, 0.0))
    # 温度：区间为硬半宽；固定温度用 3·u(T) 的相关传播
    wc_temp_range = float(np.abs(signs) @ (coeff_t * t_half))
    wc_temp_fixed = Z_EXT * math.sqrt(max(float(
        _corr_variance(np.where(is_range, 0.0, coeff_t * t_sigma),
                       signs, model.r_temp)[0]), 0.0))
    wc_temp = wc_temp_range + wc_temp_fixed
    wc_total = wc_mfg + wc_alpha + wc_temp
    lower, upper = mean - wc_total, mean + wc_total

    # ---------- 超差率 ----------
    lsl, usl = nc.closure_lsl_mm, nc.closure_usl_mm

    def reject_normal(sigma):
        if sigma <= 0:
            if (lsl is not None and mean < lsl) or (
                    usl is not None and mean > usl):
                return 1.0
            return 0.0
        p = 0.0
        if lsl is not None:
            p += float(normal_cdf((lsl - mean) / sigma))
        if usl is not None:
            p += float(1.0 - normal_cdf((usl - mean) / sigma))
        return p

    rss_reject = reject_normal(sigma_total)
    source_reject = {
        "manufacturing": reject_normal(sigma_mfg),
        "expansion_coefficient": reject_normal(sigma_alpha),
        "temperature": reject_normal(sigma_temp),
    }
    wc_reject = None
    if lsl is not None or usl is not None:
        wc_reject = 1.0 if ((lsl is not None and lower < lsl)
                           or (usl is not None and upper > usl)) else 0.0

    margin = None
    if lsl is not None and usl is not None:
        margin = min(lower - lsl, usl - upper)
    elif lsl is not None:
        margin = lower - lsl
    elif usl is not None:
        margin = usl - upper

    per_dimension = [
        {
            "dimension_id": nd[i].id,
            "sign": nd[i].sign,
            "temperature_c": {"mean": float(t_mean[i]),
                              "mode": temps[i].kind,
                              "half_width": float(t_half[i]),
                              "std_uncertainty": float(t_sigma[i])},
            "delta_t_c": float(d_t[i]),
            "scale_1_plus_alpha_dt": float(scale[i]),
            "hot_nominal_mm": float(l0_nom[i] * scale[i]),
            "hot_band_center_mm": float(l0_mean[i] * scale[i]),
            "hot_half_width_mm": float(nd[i].half_width * scale[i]),
            "sensitivity_alpha_mm_per_K_inv": float(coeff_alpha[i]),
            "sensitivity_temperature_mm_per_K": float(coeff_t[i]),
            "variance_contributions_mm2": {
                "manufacturing": float(contrib_mfg[i]),
                "expansion_coefficient": float(contrib_alpha[i]),
                "temperature": float(contrib_temp[i]),
            },
        }
        for i in range(k)
    ]

    analytic = {
        "name": model.condition_names[ci],
        "note": model.condition_notes[ci],
        "nominal_gap_mm": nominal,
        "mean_gap_mm": mean,
        "worst_case": {
            "lower_bound_mm": lower,
            "upper_bound_mm": upper,
            "total_extreme_half_width_mm": wc_total,
            "components_half_width_mm": {
                "manufacturing": wc_mfg,
                "expansion_coefficient": wc_alpha,
                "temperature": wc_temp,
                "temperature_range_hard_mm": wc_temp_range,
                "temperature_fixed_extended_mm": wc_temp_fixed,
            },
            "reject_probability": wc_reject,
            "convention": (
                f"制造半宽按 1+αΔT 缩放；标准不确定度（u(α)/固定温度 u(T)）"
                f"按 ±{Z_EXT:g}σ 扩展；温度区间取硬半宽 h_T"),
        },
        "rss": {
            "sigma_mm": sigma_total,
            "sigma_components_mm": {
                "manufacturing": sigma_mfg,
                "expansion_coefficient": sigma_alpha,
                "temperature": sigma_temp,
            },
            "variance_components_mm2": {
                "manufacturing": var_mfg,
                "expansion_coefficient": var_alpha,
                "temperature": var_temp,
            },
            "variance_share": {
                "manufacturing": var_mfg / var_total if var_total > 0 else 0.0,
                "expansion_coefficient":
                    var_alpha / var_total if var_total > 0 else 0.0,
                "temperature": var_temp / var_total if var_total > 0 else 0.0,
            },
            "lower_bound_mm": mean - Z_EXT * sigma_total,
            "upper_bound_mm": mean + Z_EXT * sigma_total,
            "z": Z_EXT,
            "reject_probability": rss_reject,
            "source_reject_probability": source_reject,
        },
        "spec_margin_mm": margin,
        "per_dimension": per_dimension,
    }

    mc = monte_carlo_condition(
        model, ci, t0=t0, alpha=alpha, u_alpha=u_alpha, scale=scale,
        l0_mean=l0_mean, d_t=d_t,
        n_samples=mc_samples, seed_base=mc_seed_base,
        shim_nominal=shim_nominal, shim_sigma=shim_sigma, shim_sign=shim_sign)
    return {"analytic": analytic, "monte_carlo": mc}


# ------------------------------------------------------------- 蒙特卡洛

def _gaussian_samples(rng: np.random.Generator, n: int, k: int,
                      corr: np.ndarray, nd) -> np.ndarray:
    """相关多元正态（单位方差）；独立时直接抽样。"""
    if np.allclose(corr, np.eye(k), atol=1e-12):
        return rng.standard_normal((n, k))
    return rng.standard_normal((n, k)) @ _cholesky_psd(corr).T


def _manufacturing_deviations(rng, nc: NormalizedChain, n: int) -> np.ndarray:
    """基线公差带口径的制造偏差样本 δ_i（mm），未做热态缩放。

    分布/相关口径与基线链蒙特卡洛完全一致（Gaussian copula）。
    热态制造分量为 δ_i·(1+αΔT)；缩放只允许发生一次——组合样本按
    (L0+δ)·[1+(α+Δα)(ΔT+δT)] 生成，绝不能先把 δ 乘上 scale 再乘一次。
    """
    nd = nc.dimensions
    k = len(nd)
    sigmas = np.array([d.sigma for d in nd])
    halfs = np.array([d.half_width for d in nd])
    explicit = [d.sigma_explicit for d in nd]
    sample_halfs = _sampling_half_widths(nd, sigmas, halfs, explicit)

    has_corr = not np.allclose(nc.corr, np.eye(k), atol=1e-12)
    if has_corr:
        latent = latent_correlation_matrix(nc)
        z = rng.standard_normal((n, k)) @ _cholesky_psd(latent).T
        erf_vec = np.vectorize(math.erf)
        uniforms = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))
    else:
        z = rng.standard_normal((n, k))
        uniforms = rng.random((n, k))

    delta = np.zeros((n, k))  # 相对带中心的基线制造偏差
    for j, d in enumerate(nd):
        a = sample_halfs[j]
        if d.distribution == DistributionType.NORMAL.value:
            delta[:, j] = sigmas[j] * z[:, j]
        elif d.distribution == DistributionType.UNIFORM.value:
            delta[:, j] = -a + 2.0 * a * uniforms[:, j]
        else:  # triangular
            delta[:, j] = _triangular_inverse(uniforms[:, j], a)
    return delta


def _closure_stats_vec(closure: np.ndarray, lsl, usl,
                       reject_reference: float = 0.0) -> dict:
    """封闭环样本向量的统计量与经验超差率。

    reject_reference 为超差率判定的叠加基准（热态名义间隙）：分量样本本身
    是零均值偏差，必须叠加到对应热态名义间隙上再与规格比较，不能直接拿
    零均值偏差对照绝对间隙规格。组合样本已是绝对间隙，传 0。
    """
    n = len(closure)
    sigma = float(closure.std(ddof=1)) if n > 1 else 0.0
    q005, q995 = np.quantile(closure, [0.005, 0.995])
    reject = None
    if lsl is not None or usl is not None:
        evaluated = closure + reject_reference
        mask = np.zeros(n, dtype=bool)
        if lsl is not None:
            mask |= evaluated < lsl
        if usl is not None:
            mask |= evaluated > usl
        reject = float(mask.mean())
    return {
        "mean_deviation_mm": float(closure.mean()),
        "sigma_mm": sigma,
        "lower_quantile_mm": float(q005),
        "upper_quantile_mm": float(q995),
        "min_sample_mm": float(closure.min()),
        "max_sample_mm": float(closure.max()),
        "reject_probability": reject,
        "reject_reference_gap_mm": (
            reject_reference if (lsl is not None or usl is not None) else None),
    }


def monte_carlo_condition(model: ThermalModel, ci: int, *,
                          t0: np.ndarray, alpha: np.ndarray,
                          u_alpha: np.ndarray, scale: np.ndarray,
                          l0_mean: np.ndarray, d_t: np.ndarray,
                          n_samples: int | None, seed_base: int | None,
                          shim_nominal: float, shim_sigma: float,
                          shim_sign: int) -> dict:
    """四子流固定种子蒙特卡洛：制造 / α / 温度分量 + 完整非线性组合。"""
    nc = model.nc
    nd = nc.dimensions
    k = len(nd)
    signs = _s_arr(nd)
    n = n_samples or model.mc_samples
    base = model.seed if seed_base is None else seed_base
    # 每个工况四个独立子流；工况序号参与派生，避免跨工况同流
    ss = np.random.SeedSequence([base, 0x74686572, ci])
    streams = [np.random.default_rng(c) for c in ss.spawn(4)]
    rng_mfg, rng_a, rng_t, rng_all = streams

    temps = model.conditions[ci]
    is_range = np.array([t.kind == "range" for t in temps])
    t_sigma = np.array([t.sigma for t in temps])

    # --- 制造偏差（基线公差带口径，未做热态缩放）---
    delta_mfg = _manufacturing_deviations(rng_mfg, nc, n)
    # 单来源热态制造偏差（热态缩放只发生这一次）：δ_i·(1+αΔT)
    mfg_component = (delta_mfg * scale[None, :]) @ signs

    # --- α 不确定度（多元正态，相关矩阵 r_alpha）---
    za = _gaussian_samples(rng_a, n, k, model.r_alpha, nd)
    da = u_alpha[None, :] * za                       # Δα_i
    alpha_component = (l0_mean[None, :] * d_t[None, :] * da) @ signs

    # --- 温度：固定 u(T) 走相关正态；区间独立均匀 ---
    dt_fluc = np.zeros((n, k))
    fixed_mask = ~is_range
    if fixed_mask.any():
        zt = _gaussian_samples(rng_t, n, k, model.r_temp, nd)
        dt_fluc[:, fixed_mask] = (
            t_sigma[None, fixed_mask] * zt[:, fixed_mask])
    if is_range.any():
        u_range = rng_t.random((n, int(is_range.sum())))
        halves = np.array([t.half for t in temps])[is_range]
        dt_fluc[:, is_range] = -halves[None, :] + 2.0 * halves[None, :] * u_range
    temp_component = (l0_mean[None, :] * alpha[None, :] * dt_fluc) @ signs

    # --- 完整非线性组合：L = (L0 + δ)·[1 + (α+Δα)·(ΔT + δT)] ---
    # 关键：L0+δ 用的是**未热态缩放**的基线偏差，热态比例只由括号内
    # [1+αΔT] 施加一次，避免制造波动被重复缩放（否则 scale=2 时组合 σ
    # 会变成制造分量的两倍）。
    l0_sample = l0_mean[None, :] + delta_mfg
    alpha_sample = alpha[None, :] + da
    dt_sample = d_t[None, :] + dt_fluc
    hot_scale = 1.0 + alpha_sample * dt_sample
    # 极端 u(α) 抽样可能使个别样本 scale 越界，截断到正下限保护
    hot_scale = np.clip(hot_scale, MIN_SCALE, None)

    # 垫片（固定尺寸 + 正态厚度不确定度）；用组合子流的独立一列，
    # 叠加在封闭环上，所有工况相同。
    shim_effect = np.zeros(n)
    if shim_nominal != 0.0 or shim_sigma > 0:
        shim_effect = shim_sign * (
            shim_nominal + shim_sigma * rng_all.standard_normal(n))

    lsl, usl = nc.closure_lsl_mm, nc.closure_usl_mm
    closure_all = (l0_sample * hot_scale) @ signs + shim_effect
    # 组合样本本身就是绝对间隙（含热态名义），超差判定基准为 0
    stats_all = _closure_stats_vec(closure_all, lsl, usl, reject_reference=0.0)
    stats_all["mean_gap_mm"] = stats_all["mean_deviation_mm"]

    # 分量样本是**零均值单项偏差**，必须叠加到对应热态名义间隙上，
    # 再与规格比较；不能直接拿零均值偏差对照绝对间隙规格（否则超差率≈1）。
    ref_gap = float(signs @ (l0_mean * scale)) + shim_sign * shim_nominal

    def comp_stats(component_closure):
        return _closure_stats_vec(component_closure, lsl, usl,
                                  reject_reference=ref_gap)

    stats_mfg = comp_stats(mfg_component)
    stats_alpha = comp_stats(alpha_component)
    stats_temp = comp_stats(temp_component)

    # 分量方差贡献（封闭环口径，热态），用于占比拆分
    def closure_var(closure):
        return float(closure.var(ddof=1)) if n > 1 else 0.0

    v_mfg = closure_var(mfg_component)
    v_alpha = closure_var(alpha_component)
    v_temp = closure_var(temp_component)
    v_all = v_mfg + v_alpha + v_temp
    v_shim = shim_sigma ** 2

    return {
        "samples": n,
        "random_seed": base,
        "condition_index": ci,
        "substreams": "SeedSequence([seed, 0x74686572, condition_index]).spawn(4)",
        "combined": {
            **stats_all,
            "nominal_model": "L=(L0+δ_mfg)[1+(α+Δα)(ΔT+δT)]，完整非线性",
        },
        "component_reject_convention": (
            "分量样本为零均值单项偏差，超差率在「热态名义间隙 "
            f"{ref_gap:.9g} mm + 单项偏差」上对照规格计算；"
            "sigma/分位/极值仍为偏差本身的统计量"),
        "components": {
            "manufacturing": stats_mfg,
            "expansion_coefficient": stats_alpha,
            "temperature": stats_temp,
        },
        "variance_share": {
            "manufacturing": v_mfg / v_all if v_all > 0 else 0.0,
            "expansion_coefficient": v_alpha / v_all if v_all > 0 else 0.0,
            "temperature": v_temp / v_all if v_all > 0 else 0.0,
        },
        "shim_sigma_contribution_mm2": v_shim,
    }


# ------------------------------------------------------------- 总装与排序

def analyze(model: ThermalModel, *, t0=None, alpha=None, u_alpha=None,
            shim_nominal=0.0, shim_sigma=0.0, shim_sign=1,
            mc_samples=None, mc_seed_base=None) -> dict:
    """逐工况评估并汇总（最差工况 / 最先越界工况）。"""
    lsl, usl = model.nc.closure_lsl_mm, model.nc.closure_usl_mm
    results = []
    for ci in range(len(model.conditions)):
        results.append(evaluate_condition(
            model, ci, t0=t0, alpha=alpha, u_alpha=u_alpha,
            shim_nominal=shim_nominal, shim_sigma=shim_sigma,
            shim_sign=shim_sign, mc_samples=mc_samples,
            mc_seed_base=mc_seed_base))

    # 最先越过规格的工况（按提交顺序）：
    # 1) WC 硬界实际穿过 LSL/USL；2) 否则 RSS ±3σ 界实际穿过规格；
    # 所有工况边界均在规格内（安全）时返回 None，不产生越界标记。
    # 注意：RSS 尾部概率 >0 在有限 σ 下几乎恒成立，不能仅凭它判定“越界”。
    first_breach = None
    for ci, r in enumerate(results):
        wc = r["analytic"]["worst_case"]
        if (lsl is not None and wc["lower_bound_mm"] < lsl) or (
                usl is not None and wc["upper_bound_mm"] > usl):
            first_breach = _breach_info(ci, r, "worst_case", lsl, usl)
            break
    if first_breach is None:
        for ci, r in enumerate(results):
            rr = r["analytic"]["rss"]
            if (lsl is not None and rr["lower_bound_mm"] < lsl) or (
                    usl is not None and rr["upper_bound_mm"] > usl):
                first_breach = _breach_info(ci, r, "rss", lsl, usl)
                break

    worst_wc = max(
        (r["analytic"]["worst_case"]["reject_probability"] or 0.0)
        for r in results)
    worst_rss = max(
        r["analytic"]["rss"]["reject_probability"] for r in results)
    worst_mc = max(
        (r["monte_carlo"]["combined"]["reject_probability"] or 0.0)
        for r in results)
    margins = [r["analytic"]["spec_margin_mm"] for r in results
               if r["analytic"]["spec_margin_mm"] is not None]
    return {
        "conditions": results,
        "summary": {
            "worst_reject_probability": {
                "worst_case": worst_wc,
                "rss": worst_rss,
                "monte_carlo": worst_mc,
            },
            "minimum_spec_margin_mm": min(margins) if margins else None,
            "first_spec_breach": first_breach,
        },
    }


def _breach_info(ci: int, result: dict, criterion: str,
                 lsl, usl) -> dict:
    """越界方向：WC/RSS 界实际穿过 LSL 记 lower，穿过 USL 记 upper。

    调用方只在边界确实越过规格时调用，因此 breach_side 至少有一个方向。
    """
    a = result["analytic"]
    side = []
    b = a[criterion]
    lo, hi = b["lower_bound_mm"], b["upper_bound_mm"]
    if lsl is not None and lo < lsl:
        side.append("lower")
    if usl is not None and hi > usl:
        side.append("upper")
    return {
        "condition_index": ci,
        "condition_name": a["name"],
        "criterion": criterion,
        "breach_side": side,
        "worst_case_bounds_mm": [
            a["worst_case"]["lower_bound_mm"],
            a["worst_case"]["upper_bound_mm"]],
        "rss_bounds_mm": [a["rss"]["lower_bound_mm"],
                          a["rss"]["upper_bound_mm"]],
        "spec_margin_mm": a["spec_margin_mm"],
    }


# ------------------------------------------------------------- 方案搜索

@dataclass
class _Plan:
    material_choice: dict[str, str | None]   # 尺寸 id -> material_id 或 None(现状)
    shim_id: str | None
    aref: float | None                        # 装配基准温度 °C
    aref_cost: float


def _enumerate_plans(model: ThermalModel, payload) -> list[_Plan]:
    """枚举 材料 × 垫片 × 装配基准温度 笛卡尔积（含零成本现状）。"""
    options: dict[str, list[tuple[str | None, float, float, float]]] = {
        # dim_id -> [(material_id, alpha, u_alpha, cost)]，首项为现状
        d: [(None, model.alpha[i], model.u_alpha[i], 0.0)]
        for i, d in enumerate(model.ids)
    }
    for cand in payload.candidates:
        for dim_id in cand.applies_to:
            if dim_id not in options:
                raise ThermalError(
                    f"候选材料 {cand.material_id} 引用了热版本外尺寸 {dim_id}")
        factor = TO_PER_K[cand.alpha_unit.value]
        for dim_id in cand.applies_to:
            locked = payload.locked_materials.get(dim_id)
            if locked is not None and locked != cand.material_id:
                continue  # 该尺寸锁定其它材料，本候选对它不可选
            options[dim_id].append((
                cand.material_id, cand.alpha * factor,
                cand.alpha_std_uncertainty * factor, cand.cost))

    # 锁定材料：删除未被锁定的候选项（只保留现状与锁定材料）
    for dim_id, mat_id in payload.locked_materials.items():
        options[dim_id] = [
            opt for opt in options[dim_id]
            if opt[0] is None or opt[0] == mat_id]
        if all(opt[0] is None for opt in options[dim_id]):
            raise ThermalError(
                f"尺寸 {dim_id} 锁定材料 {mat_id} 不适用（候选表中该材料"
                "未声明 applies_to 该尺寸）")

    shims: list[tuple[str | None, float, float, int, float]] = [
        (None, 0.0, 0.0, 1, 0.0)]
    for s in payload.shim_candidates:
        shims.append((
            s.shim_id, to_mm(s.thickness, s.unit.value),
            to_mm(s.std_dev, s.unit.value), s.closure_sign, s.cost))

    arefs: list[tuple[float | None, float]] = [(None, 0.0)]
    for a in payload.assembly_reference_temperatures:
        arefs.append((_to_celsius(a.temperature, a.unit.value), a.cost))

    total = 1
    for opts in options.values():
        total *= len(opts)
    total *= len(shims) * len(arefs)
    if total > MAX_PLANS_TOTAL:
        raise ThermalError(
            f"方案组合数 {total} 超过上限 {MAX_PLANS_TOTAL}；"
            "请减少每尺寸候选材料数 / 垫片数 / 装配基准温度数")

    plans: list[_Plan] = []
    dim_order = list(options.keys())
    cand_index = {c.material_id: c for c in payload.candidates}

    def recurse(pos, choice):
        if pos == len(dim_order):
            # 同一材料用于多个尺寸只计一次改动成本（按选中材料去重）
            selected = {m for m in choice.values() if m is not None}
            mat_cost = sum(cand_index[m].cost for m in selected)
            for shim_id, _, _, _, shim_cost in shims:
                for aref, aref_cost in arefs:
                    plans.append(_Plan(
                        dict(choice), shim_id, aref,
                        aref_cost if aref is not None else 0.0))
                    plan_costs.append(mat_cost + shim_cost + (
                        aref_cost if aref is not None else 0.0))
            return
        dim_id = dim_order[pos]
        for mat_id, _, _, _ in options[dim_id]:
            choice[dim_id] = mat_id
            recurse(pos + 1, choice)
        choice.pop(dim_id, None)

    plan_costs: list[float] = []
    recurse(0, {})
    if len(plans) > MAX_PLANS_ANALYTIC:
        raise ThermalError(
            f"待解析评估的方案数 {len(plans)} 超过上限 {MAX_PLANS_ANALYTIC}")
    return list(zip(plans, plan_costs))


def _plan_overrides(model: ThermalModel, payload, plan: _Plan):
    """从方案取出 alpha/u_alpha/t0 覆盖与垫片参数、成本。"""
    cand_by_id = {c.material_id: c for c in payload.candidates}
    alpha = model.alpha.copy()
    u_alpha = model.u_alpha.copy()
    for i, dim_id in enumerate(model.ids):
        mat_id = plan.material_choice[dim_id]
        if mat_id is not None:
            c = cand_by_id[mat_id]
            f = TO_PER_K[c.alpha_unit.value]
            alpha[i] = c.alpha * f
            u_alpha[i] = c.alpha_std_uncertainty * f
    t0 = model.t0_c if plan.aref is None else np.full(
        len(model.ids), plan.aref)
    shim_nom = shim_sigma = 0.0
    shim_sign = 1
    shim_cost = 0.0
    if plan.shim_id is not None:
        s = next(x for x in payload.shim_candidates
                 if x.shim_id == plan.shim_id)
        shim_nom = to_mm(s.thickness, s.unit.value)
        shim_sigma = to_mm(s.std_dev, s.unit.value)
        shim_sign = s.closure_sign
        shim_cost = s.cost
    # 同一材料用于多个尺寸只计一次改动成本
    chosen = {m for m in plan.material_choice.values() if m is not None}
    mat_cost = sum(cand_by_id[m].cost for m in chosen)
    cost = mat_cost + shim_cost + plan.aref_cost
    return (alpha, u_alpha, t0, shim_nom, shim_sigma, shim_sign, cost,
            mat_cost, shim_cost)


def evaluate_condition_fast(model: ThermalModel, ci: int, *,
                            t0, alpha, u_alpha,
                            shim_nominal, shim_sigma, shim_sign) -> dict:
    """方案粗筛用的纯解析评估（复制 evaluate_condition 的解析主体）。"""
    nc = model.nc
    nd = nc.dimensions
    k = len(nd)
    signs = _s_arr(nd)
    temps = model.conditions[ci]
    t_mean = np.array([t.mean for t in temps])
    t_sigma = np.array([t.sigma for t in temps])
    t_half = np.array([t.half for t in temps])
    is_range = np.array([t.kind == "range" for t in temps])

    d_t = t_mean - t0
    scale = 1.0 + alpha * d_t
    _scale_positivity(scale, model.ids)
    l0_mean = np.array([d.nominal + d.mid for d in nd])
    l0_nom = np.array([d.nominal for d in nd])
    coeff_alpha = l0_mean * d_t
    coeff_t = l0_mean * alpha

    nominal = float(signs @ (l0_nom * scale)) + shim_sign * shim_nominal
    mean = float(signs @ (l0_mean * scale)) + shim_sign * shim_nominal

    mfg_weight = np.array([d.sigma for d in nd]) * scale
    var_mfg, _ = _corr_variance(mfg_weight, signs, nc.corr)
    var_mfg += shim_sigma ** 2
    var_alpha, _ = _corr_variance(
        coeff_alpha * u_alpha, signs, model.r_alpha)
    t_fixed = np.where(is_range, 0.0, coeff_t * t_sigma)
    var_temp, _ = _corr_variance(t_fixed, signs, model.r_temp)
    range_w = coeff_t * np.where(is_range, t_sigma, 0.0)
    var_temp += float((signs * range_w) ** 2 @ np.ones(k))

    var_total = var_mfg + var_alpha + var_temp
    sig = math.sqrt(max(var_total, 0.0))
    wc_mfg = float(np.abs(signs) @ (
        np.array([d.half_width for d in nd]) * scale)) + Z_EXT * shim_sigma
    wc_alpha = Z_EXT * math.sqrt(max(var_alpha, 0.0))
    wc_temp_range = float(np.abs(signs) @ (coeff_t * t_half))
    wc_temp_fixed = Z_EXT * math.sqrt(max(float(
        _corr_variance(np.where(is_range, 0.0, coeff_t * t_sigma),
                       signs, model.r_temp)[0]), 0.0))
    wc = wc_mfg + wc_alpha + wc_temp_range + wc_temp_fixed
    lower, upper = mean - wc, mean + wc
    lsl, usl = nc.closure_lsl_mm, nc.closure_usl_mm
    wc_rej = None
    if lsl is not None or usl is not None:
        wc_rej = 1.0 if ((lsl is not None and lower < lsl)
                        or (usl is not None and upper > usl)) else 0.0
    p = 0.0
    if sig > 0:
        if lsl is not None:
            p += float(normal_cdf((lsl - mean) / sig))
        if usl is not None:
            p += float(1.0 - normal_cdf((usl - mean) / sig))
    else:
        p = 1.0 if ((lsl is not None and mean < lsl)
                    or (usl is not None and mean > usl)) else 0.0
    margin = None
    if lsl is not None and usl is not None:
        margin = min(lower - lsl, usl - upper)
    elif lsl is not None:
        margin = lower - lsl
    elif usl is not None:
        margin = usl - upper
    return {
        "name": model.condition_names[ci],
        "nominal_gap_mm": nominal,
        "mean_gap_mm": mean,
        "wc_lower": lower, "wc_upper": upper,
        "wc_reject": wc_rej,
        "rss_sigma": sig, "rss_reject": p,
        "rss_lower": mean - Z_EXT * sig,
        "rss_upper": mean + Z_EXT * sig,
        "spec_margin_mm": margin,
    }


def search_proposals(model: ThermalModel, payload) -> dict:
    """枚举材料 × 垫片 × 装配基准温度方案并按全工况指标排列。

    排序键（升序，越小越优）：
      1. 全工况最差 MC 超差率（解析粗筛阶段用 RSS 超差率）；
      2. 全工况最小 WC 规格余量（越小越差，升序即余量小者排后，
         故取 -margin 排序）；
      3. 改动总成本。
    流程：全组合解析粗筛 → 前 TOP_PLANS_MC（含零成本现状）固定种子
    MC 复核 → 按 MC 指标定序返回。
    """
    enumerated = _enumerate_plans(model, payload)
    fast_rows = []
    for plan, enum_cost in enumerated:
        (alpha, u_alpha, t0, shim_nom, shim_sigma, shim_sign, _cost,
         mat_cost, shim_cost) = _plan_overrides(model, payload, plan)
        cost = enum_cost  # 枚举时已按材料+垫片+基准温度汇总
        conds = [
            evaluate_condition_fast(
                model, ci, t0=t0, alpha=alpha, u_alpha=u_alpha,
                shim_nominal=shim_nom, shim_sigma=shim_sigma,
                shim_sign=shim_sign)
            for ci in range(len(model.conditions))
        ]
        worst_rss = max(c["rss_reject"] for c in conds)
        worst_wc = max((c["wc_reject"] or 0.0) for c in conds)
        margins = [c["spec_margin_mm"] for c in conds
                   if c["spec_margin_mm"] is not None]
        # 无规格限时余量按 +∞ 处理（不参与余量排序）
        min_margin = min(margins) if margins else float("inf")
        fast_rows.append({
            "plan": plan, "alpha": alpha, "u_alpha": u_alpha, "t0": t0,
            "shim_nom": shim_nom, "shim_sigma": shim_sigma,
            "shim_sign": shim_sign, "cost": cost,
            "mat_cost": mat_cost, "shim_cost": shim_cost,
            "worst_rss": worst_rss, "worst_wc": worst_wc,
            "min_margin": min_margin, "conds_fast": conds,
        })

    # 解析粗筛排序：最差 RSS 超差率升序、最小 WC 余量降序（余量大者优先）、
    # 成本升序；无规格限时余量按 +∞。
    fast_rows.sort(key=lambda r: (
        r["worst_rss"], -_margin_key(r["min_margin"]), r["cost"]))

    # 保证零成本现状（全 None + 无垫片 + 默认基准）进入 MC 复核
    baseline_idx = next(
        i for i, r in enumerate(fast_rows)
        if all(m is None for m in r["plan"].material_choice.values())
        and r["plan"].shim_id is None and r["plan"].aref is None)
    top = fast_rows[:TOP_PLANS_MC]
    if baseline_idx >= TOP_PLANS_MC:
        top.append(fast_rows[baseline_idx])

    # 方案复核样本数独立于分析版本（请求给定，固定版本种子保证可复现）
    mc_n = payload.mc_samples
    ranked = []
    for r in top:
        full = analyze(
            model, t0=r["t0"], alpha=r["alpha"], u_alpha=r["u_alpha"],
            shim_nominal=r["shim_nom"], shim_sigma=r["shim_sigma"],
            shim_sign=r["shim_sign"], mc_samples=mc_n,
            mc_seed_base=model.seed)
        worst_mc = full["summary"]["worst_reject_probability"]["monte_carlo"]
        margins = [c["analytic"]["spec_margin_mm"]
                   for c in full["conditions"]
                   if c["analytic"]["spec_margin_mm"] is not None]
        min_margin = min(margins) if margins else float("inf")
        ranked.append({
            "row": r, "full": full,
            "worst_mc": worst_mc,
            "min_margin": min_margin,
            "first_breach": full["summary"]["first_spec_breach"],
        })

    # MC 定序：最差 MC 超差率升序、最小 WC 余量降序、成本升序
    ranked.sort(key=lambda x: (
        x["worst_mc"], -_margin_key(x["min_margin"]),
        x["row"]["cost"]))

    proposals = []
    cand_by_id = {c.material_id: c for c in payload.candidates}
    shim_by_id = {s.shim_id: s for s in payload.shim_candidates}
    for rank, item in enumerate(ranked[:payload.max_candidates_returned], 1):
        r, plan = item["row"], item["row"]["plan"]
        materials_changed = []
        for dim_id, mat_id in plan.material_choice.items():
            if mat_id is not None:
                c = cand_by_id[mat_id]
                materials_changed.append({
                    "dimension_id": dim_id,
                    "material_id": mat_id,
                    "material_name": c.name,
                    "alpha": c.alpha,
                    "alpha_std_uncertainty": c.alpha_std_uncertainty,
                    "alpha_unit": c.alpha_unit.value,
                    "cost": c.cost,
                })
        shim_info = None
        if plan.shim_id is not None:
            s = shim_by_id[plan.shim_id]
            shim_info = {
                "shim_id": s.shim_id, "name": s.name,
                "thickness": s.thickness, "std_dev": s.std_dev,
                "unit": s.unit.value, "closure_sign": s.closure_sign,
                "thickness_mm": r["shim_nom"], "cost": s.cost,
            }
        proposals.append({
            "rank": rank,
            "materials": materials_changed,
            "shim": shim_info,
            "assembly_reference_temperature_c": plan.aref,
            "cost": {
                "total": r["cost"],
                "materials": r["mat_cost"],
                "shim": r["shim_cost"],
                "assembly_reference_temperature": plan.aref_cost,
            },
            "metrics": {
                "worst_reject_probability_mc": item["worst_mc"],
                "worst_reject_probability_rss": r["worst_rss"],
                "worst_case_breach": bool(r["worst_wc"] > 0),
                "minimum_spec_margin_mm": (
                    item["min_margin"]
                    if math.isfinite(item["min_margin"]) else None),
                "first_spec_breach": item["first_breach"],
            },
            "conditions": item["full"]["conditions"],
        })

    return {
        "ranking_rule": [
            "全工况最差固定种子蒙特卡洛超差率（升序）",
            "全工况最小极值法规格余量（降序）",
            "改动总成本（升序）",
        ],
        "evaluated_analytic": len(fast_rows),
        "evaluated_monte_carlo": len(ranked),
        "enumeration_cap_analytic": MAX_PLANS_ANALYTIC,
        "enumeration_cap_total": MAX_PLANS_TOTAL,
        "monte_carlo_samples_per_condition": mc_n,
        "monte_carlo_seed_base": model.seed,
        "proposals": proposals,
    }
