"""公差链计算引擎（全部内部计算使用 mm）。

约定与公式
==========
设尺寸链有 n 个组成环，沿闭环走向得到方向系数 s_i ∈ {+1,-1}
（再乘以测量方向投影 direction）：

* 名义封闭环间隙  C0 = Σ s_i · N_i
* 偏差带中点偏移  m_i = (ES_i + EI_i)/2，半带宽 T_i = (ES_i - EI_i)/2
* 制程中心假设位于公差带中点：μ_i = N_i + m_i
* 封闭环均值      μ_C = Σ s_i · μ_i

极值法 (Worst Case, WC)
    封闭环范围 [ μ_C - ΣT_i , μ_C + ΣT_i ]

统计法 (RSS)
    σ_C = sqrt( Σ_i Σ_j s_i s_j ρ_ij σ_i σ_j )
    默认统计界限 μ_C ± 3σ_C；超差率按正态近似
    P(超差) = Φ((LSL-μ_C)/σ_C) + 1 - Φ((USL-μ_C)/σ_C)
    组成环理论标准差：均匀 σ=T/√3，三角 σ=T/√6，正态由用户给定。

显式 σ 的统一含义（RSS 与蒙特卡洛必须同源）
    正态直接以 σ 抽样；均匀抽样半宽取 σ√3、三角取 σ√6，
    样本方差恰为 σ²，因此蒙特卡洛 σ_C 收敛到 RSS 解析值。
    未给显式 σ 时，均匀/三角按公差带抽样、σ 取理论值。
    σ=0 时封闭环退化为固定间隙，区间概率取 0/1 指示值。

蒙特卡洛 (MC)
    固定随机种子；独立维度直接抽样，存在相关系数时使用
    Gaussian copula（由相关矩阵 Cholesky 分解得到相关正态，
    再经概率积分变换得到相关均匀/三角变量）。
    输出样本均值、标准差、分位数、极值与经验超差率。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schemas import DistributionType

Z_STAT = 3.0  # RSS 统计界限倍数


@dataclass
class NormDimension:
    """规范化后的单个尺寸（数值均为 mm）。"""

    id: str
    sign: int                     # 闭环方向系数 s_i（含测量方向）
    nominal: float                # N_i mm
    upper_dev: float              # ES_i mm
    lower_dev: float              # EI_i mm
    mid: float                    # m_i
    half_width: float             # T_i
    sigma: float                  # σ_i mm
    distribution: str
    start: str
    end: str
    direction: int
    original: dict                # 原始单位表示（保留）
    unit: str
    # σ 是否由调用方显式给定：True 时蒙特卡洛按 σ 确定散布尺度，
    # False 时 σ 由公差带理论导出、抽样覆盖整个公差带。
    sigma_explicit: bool = False

    def band_bounds(self) -> tuple[float, float]:
        return self.nominal + self.lower_dev, self.nominal + self.upper_dev


@dataclass
class NormalizedChain:
    dimensions: list[NormDimension]
    closure_lsl_mm: float | None
    closure_usl_mm: float | None
    mc_samples: int
    seed: int
    corr: np.ndarray
    name: str = ""
    closure_unit: str = "mm"
    closure_original: dict | None = None


def chain_signs(dimensions) -> dict[str, int]:
    """沿单一闭合有向环遍历，得到每个尺寸相对封闭环的方向系数 s_i。"""
    first = dimensions[0]
    signs: dict[str, int] = {first.id: first.direction}
    order = [first.id]
    current = first.end
    while len(order) < len(dimensions):
        nxt = None
        for d in dimensions:
            if d.id in signs:
                continue
            if d.start == current or d.end == current:
                nxt = d
                break
        if nxt is None:
            raise ValueError(
                f"在节点 {current} 无法继续沿环遍历（输入校验本应拦截此情况）"
            )
        if nxt.start == current:
            signs[nxt.id] = nxt.direction          # 顺边方向 +1
            current = nxt.end
        else:
            signs[nxt.id] = -nxt.direction         # 逆边方向 -1
            current = nxt.start
        order.append(nxt.id)
    if current != first.start:
        raise ValueError(
            f"尺寸链不闭合: 遍历终止于 {current}，期望回到起点 {first.start}"
        )
    return signs


def _theoretical_sigma(distribution: str, half_width: float) -> float:
    if distribution == DistributionType.UNIFORM.value:
        return half_width / math.sqrt(3.0)
    if distribution == DistributionType.TRIANGULAR.value:
        return half_width / math.sqrt(6.0)
    raise ValueError(f"分布 {distribution} 缺少标准差")


def normalize_chain(chain_create) -> NormalizedChain:
    """把 API 输入换算为 mm 并展开理论标准差 / 相关矩阵。"""
    from .units import to_mm

    signs = chain_signs(chain_create.dimensions)
    norm_dims: list[NormDimension] = []
    for d in chain_create.dimensions:
        u = d.unit.value if hasattr(d.unit, "value") else str(d.unit)
        n_mm = to_mm(d.nominal, u)
        es_mm = to_mm(d.upper_deviation, u)
        ei_mm = to_mm(d.lower_deviation, u)
        t_mm = (es_mm - ei_mm) / 2.0
        sigma_explicit = d.std_dev is not None
        if sigma_explicit:
            sigma_mm = to_mm(d.std_dev, u)
        else:
            sigma_mm = _theoretical_sigma(d.distribution.value, t_mm)
        norm_dims.append(
            NormDimension(
                id=d.id,
                sign=signs[d.id],
                nominal=n_mm,
                upper_dev=es_mm,
                lower_dev=ei_mm,
                mid=(es_mm + ei_mm) / 2.0,
                half_width=t_mm,
                sigma=max(sigma_mm, 0.0),
                sigma_explicit=sigma_explicit,
                distribution=d.distribution.value,
                start=d.start,
                end=d.end,
                direction=d.direction,
                unit=u,
                original={
                    "nominal": d.nominal,
                    "upper_deviation": d.upper_deviation,
                    "lower_deviation": d.lower_deviation,
                    "std_dev": d.std_dev,
                    "unit": u,
                    "unilateral": d.is_unilateral,
                },
            )
        )

    cu = (
        chain_create.closure_unit.value
        if hasattr(chain_create.closure_unit, "value")
        else str(chain_create.closure_unit)
    )
    lsl = (
        to_mm(chain_create.closure_lower_limit, cu)
        if chain_create.closure_lower_limit is not None
        else None
    )
    usl = (
        to_mm(chain_create.closure_upper_limit, cu)
        if chain_create.closure_upper_limit is not None
        else None
    )

    ids = [d.id for d in chain_create.dimensions]
    idx = {name: i for i, name in enumerate(ids)}
    n = len(ids)
    corr = np.eye(n)
    for c in chain_create.correlations:
        i, j = idx[c.dim_a], idx[c.dim_b]
        corr[i, j] = c.rho
        corr[j, i] = c.rho

    return NormalizedChain(
        dimensions=norm_dims,
        closure_lsl_mm=lsl,
        closure_usl_mm=usl,
        mc_samples=chain_create.mc_samples,
        seed=chain_create.random_seed,
        corr=corr,
        name=chain_create.name,
        closure_unit=cu,
        closure_original={
            "lower_limit": chain_create.closure_lower_limit,
            "upper_limit": chain_create.closure_upper_limit,
            "unit": cu,
        },
    )


def _s_arr(nd: list[NormDimension]) -> np.ndarray:
    return np.array([d.sign for d in nd], dtype=float)


def _sigma_arr(nd: list[NormDimension]) -> np.ndarray:
    return np.array([d.sigma for d in nd], dtype=float)


def nominal_gap(nc: NormalizedChain) -> float:
    return float(sum(d.sign * d.nominal for d in nc.dimensions))


def mean_gap(nc: NormalizedChain) -> float:
    return float(sum(d.sign * (d.nominal + d.mid) for d in nc.dimensions))


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def _reject_normal(mean: float, sigma: float,
                   lsl: float | None, usl: float | None) -> float | None:
    if sigma <= 0:
        if lsl is not None and mean < lsl:
            return 1.0
        if usl is not None and mean > usl:
            return 1.0
        return 0.0
    p = 0.0
    if lsl is not None:
        p += float(normal_cdf((lsl - mean) / sigma))
    if usl is not None:
        p += float(1.0 - normal_cdf((usl - mean) / sigma))
    return p


def _empirical_reject(samples: np.ndarray,
                      lsl: float | None, usl: float | None) -> float | None:
    if lsl is None and usl is None:
        return None
    mask = np.zeros_like(samples, dtype=bool)
    if lsl is not None:
        mask |= samples < lsl
    if usl is not None:
        mask |= samples > usl
    return float(mask.mean())


# ---------------------------------------------------------------- 极值法

def worst_case(nc: NormalizedChain) -> dict:
    c0 = nominal_gap(nc)
    mu = mean_gap(nc)
    total_t = sum(d.half_width for d in nc.dimensions)
    lower, upper = mu - total_t, mu + total_t
    sens = []
    for d in nc.dimensions:
        sens.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "half_width_mm": d.half_width,
            "mean_shift_mm": d.sign * d.mid,
            "tolerance_share": (
                d.half_width / total_t if total_t > 0 else 0.0
            ),
        })
    reject = None
    if nc.closure_lsl_mm is not None and lower < nc.closure_lsl_mm:
        reject = 1.0
    if nc.closure_usl_mm is not None and upper > nc.closure_usl_mm:
        reject = 1.0
    if reject is None and (nc.closure_lsl_mm is not None
                           or nc.closure_usl_mm is not None):
        reject = 0.0
    return {
        "method": "worst_case",
        "display_name": "极值法 (Worst Case)",
        "nominal_gap_mm": c0,
        "mean_gap_mm": mu,
        "lower_bound_mm": lower,
        "upper_bound_mm": upper,
        "total_tolerance_band_mm": 2.0 * total_t,
        "reject_probability": reject,
        "sensitivity": sens,
        "formulas": [
            "C0 = Σ s_i·N_i （名义封闭环间隙）",
            "m_i=(ES_i+EI_i)/2, T_i=(ES_i-EI_i)/2",
            "μ_C = Σ s_i·(N_i+m_i)",
            "WC 界: [μ_C - ΣT_i, μ_C + ΣT_i]",
            "敏感度贡献: T_i / ΣT_j（对封闭环总带宽），s_i·m_i（对均值偏移）",
        ],
    }


# -------------------------------------------------------------------- RSS

def rss_analysis(nc: NormalizedChain, z: float = Z_STAT) -> dict:
    nd = nc.dimensions
    s = _s_arr(nd)
    sigma = _sigma_arr(nd)
    c0 = nominal_gap(nc)
    mu = mean_gap(nc)

    weighted = s * sigma                      # s_i σ_i
    cov_sum = nc.corr * weighted[:, None] * weighted[None, :]
    var = float(cov_sum.sum())
    sigma_c = math.sqrt(max(var, 0.0))
    lower, upper = mu - z * sigma_c, mu + z * sigma_c
    reject = _reject_normal(
        mu, sigma_c, nc.closure_lsl_mm, nc.closure_usl_mm
    )
    sens = []
    for i, d in enumerate(nd):
        contrib_var = float(cov_sum[i, :].sum())
        sens.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "sigma_mm": d.sigma,
            "std_contribution_mm": abs(sigma[i]),
            "variance_contribution_mm2": contrib_var,
            "variance_share": contrib_var / var if var > 0 else 0.0,
            "sensitivity_derivative": d.sign,
        })
    return {
        "method": "rss",
        "display_name": f"统计法 RSS (±{z:g}σ)",
        "nominal_gap_mm": c0,
        "mean_gap_mm": mu,
        "sigma_mm": sigma_c,
        "z": z,
        "lower_bound_mm": lower,
        "upper_bound_mm": upper,
        "reject_probability": reject,
        "sensitivity": sens,
        "formulas": [
            "σ_C = sqrt(Σ_i Σ_j s_i·s_j·ρ_ij·σ_i·σ_j)",
            f"统计界限: [μ_C - {z:g}σ_C, μ_C + {z:g}σ_C]",
            "超差率 P = Φ((LSL-μ_C)/σ_C) + 1 - Φ((USL-μ_C)/σ_C)（正态近似）",
            "组成环 σ: 均匀 T/√3，三角 T/√6，正态取用户输入；显式 σ 与"
            "蒙特卡洛同源（见蒙特卡洛公式）",
            "方差贡献: σ_i·s_i·Σ_j ρ_ij·s_j·σ_j；占比之和为 1",
        ],
    }


# ------------------------------------------------------------- 蒙特卡洛

def _cholesky_psd(corr: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # 半正定边界上的矩阵因舍入略偏负：加微小对角扰动
        jitter = 1e-10
        for _ in range(4):
            try:
                return np.linalg.cholesky(corr + jitter * np.eye(len(corr)))
            except np.linalg.LinAlgError:
                jitter *= 10
        # 退化兜底：特征值截断到 0
        w, v = np.linalg.eigh(corr)
        w = np.clip(w, 0, None)
        return v * np.sqrt(w)[None, :]


def _triangular_inverse(u: np.ndarray, half: float) -> np.ndarray:
    """[-T, T] 上关于 0 对称的三角分布分位数函数。"""
    out = np.empty_like(u)
    left = u < 0.5
    out[left] = -half + half * np.sqrt(2.0 * u[left])
    out[~left] = half - half * np.sqrt(2.0 * (1.0 - u[~left]))
    return out


def _sampling_half_widths(nd: list[NormDimension],
                          sigmas: np.ndarray,
                          halfs: np.ndarray,
                          explicit_flags) -> np.ndarray:
    """各维度蒙特卡洛抽样半宽 a_j（关于中心 m 对称）。

    显式 σ 对 RSS 与蒙特卡洛必须含义一致：抽样方差 = σ²。
      uniform    U(center-a, center+a): a = σ·√3
      triangular 峰在 center、端点 ±a:  a = σ·√6
    未显式给 σ 时，σ 是公差带的理论值，按公差带半宽 T 抽样。
    """
    a = halfs.copy()
    for j, d in enumerate(nd):
        is_explicit = (
            d.sigma_explicit if explicit_flags is None else bool(explicit_flags[j])
        )
        if not is_explicit:
            continue
        if d.distribution == DistributionType.UNIFORM.value:
            a[j] = sigmas[j] * math.sqrt(3.0)
        elif d.distribution == DistributionType.TRIANGULAR.value:
            a[j] = sigmas[j] * math.sqrt(6.0)
    return a


def sample_dimensions(nc: NormalizedChain, n_samples: int,
                      seed: int, sigmas: np.ndarray | None = None,
                      mids: np.ndarray | None = None,
                      halfs: np.ndarray | None = None,
                      explicit_flags=None) -> np.ndarray:
    """生成 n×k 样本矩阵（每个尺寸一列，单位 mm，中心在 N+m）。

    独立维度按各自分布直接抽样；存在相关系数时统一使用 Gaussian copula：
    由 Cholesky 得到相关标准正态 z，正态维度直接 center+σ·z，
    均匀/三角维度经 u=Φ(z) 再代入逆 CDF。

    显式给定 σ 的均匀/三角维度，其抽样半宽由 σ 反推（√3σ / √6σ），
    使抽样方差等于 σ²，与 RSS 口径一致；否则按公差带半宽抽样。
    """
    nd = nc.dimensions
    k = len(nd)
    rng = np.random.default_rng(seed)
    if sigmas is None:
        sigmas = _sigma_arr(nd)
    if mids is None:
        mids = np.array([d.mid for d in nd])
    if halfs is None:
        halfs = np.array([d.half_width for d in nd])
    if explicit_flags is None:
        explicit_flags = [d.sigma_explicit for d in nd]
    sample_halfs = _sampling_half_widths(nd, sigmas, halfs, explicit_flags)

    has_corr = not np.allclose(nc.corr, np.eye(k), atol=1e-12)
    if has_corr:
        z = rng.standard_normal((n_samples, k)) @ _cholesky_psd(nc.corr).T
        uniforms = 0.5 * (1.0 + np.vectorize(math.erf)(
            z / math.sqrt(2.0)))
    else:
        z = rng.standard_normal((n_samples, k))
        uniforms = rng.random((n_samples, k))

    samples = np.zeros((n_samples, k))
    for j, d in enumerate(nd):
        center = d.nominal + mids[j]
        a = sample_halfs[j]
        if d.distribution == DistributionType.NORMAL.value:
            samples[:, j] = center + sigmas[j] * z[:, j]
        elif d.distribution == DistributionType.UNIFORM.value:
            samples[:, j] = center - a + 2.0 * a * uniforms[:, j]
        elif d.distribution == DistributionType.TRIANGULAR.value:
            samples[:, j] = center + _triangular_inverse(
                uniforms[:, j], a)
        else:  # pragma: no cover - 枚举已固定
            raise ValueError(f"未知分布 {d.distribution}")
    return samples


def monte_carlo(nc: NormalizedChain, n_samples: int | None = None,
                seed: int | None = None,
                sigmas: np.ndarray | None = None,
                mids: np.ndarray | None = None,
                halfs: np.ndarray | None = None,
                explicit_flags=None) -> dict:
    n_samples = n_samples or nc.mc_samples
    seed = nc.seed if seed is None else seed
    nd = nc.dimensions
    x = sample_dimensions(
        nc, n_samples, seed, sigmas=sigmas, mids=mids, halfs=halfs,
        explicit_flags=explicit_flags,
    )
    s = _s_arr(nd)
    closure = x @ s
    mu = float(closure.mean())
    sigma_c = float(closure.std(ddof=1))
    q005, q995 = np.quantile(closure, [0.005, 0.995])
    reject = _empirical_reject(
        closure, nc.closure_lsl_mm, nc.closure_usl_mm
    )

    # 基于样本协方差的敏感度（线性回归斜率 / 方差贡献）
    var_c = float(np.var(closure, ddof=1))
    sens = []
    for j, d in enumerate(nd):
        cov_jc = float(np.cov(x[:, j], closure, ddof=1)[0, 1])
        slope = cov_jc / var_c if var_c > 0 else 0.0
        contrib = d.sign * cov_jc
        sens.append({
            "dimension_id": d.id,
            "sign": d.sign,
            "regression_slope": slope,
            "covariance_with_closure_mm2": cov_jc,
            "variance_contribution_mm2": contrib,
            "variance_share": contrib / var_c if var_c > 0 else 0.0,
            "sample_mean_mm": float(x[:, j].mean()),
            "sample_std_mm": float(x[:, j].std(ddof=1)),
        })

    return {
        "method": "monte_carlo",
        "display_name": "蒙特卡洛模拟 (Monte Carlo)",
        "nominal_gap_mm": nominal_gap(nc),
        "mean_gap_mm": mu,
        "sigma_mm": sigma_c,
        "lower_bound_mm": float(q005),
        "upper_bound_mm": float(q995),
        "bounds_quantile": [0.005, 0.995],
        "min_sample_mm": float(closure.min()),
        "max_sample_mm": float(closure.max()),
        "reject_probability": reject,
        "samples": n_samples,
        "random_seed": seed,
        "sensitivity": sens,
        "formulas": [
            "固定种子 np.random.default_rng(seed)，结果可精确复现",
            "C^(k) = Σ s_i·X_i^(k)，k=1..N",
            "独立维度按各自分布抽样；相关维度使用 Gaussian copula",
            "  z = L·ξ (LLᵀ=R, Cholesky)，u=Φ(z)，再代入各分布逆 CDF",
            "显式 σ：正态 center+σ·z；均匀抽样半宽 σ√3、三角 σ√6，"
            "样本方差=σ²，与 RSS 同源；未给 σ 时按公差带半宽抽样",
            "σ=0 时样本为固定间隙；超差率按固定值判定",
            "超差率 = #{C<LSL 或 C>USL} / N",
            "敏感度: 协方差法 slope_i=Cov(X_i,C)/Var(C)，方差贡献 s_i·Cov",
            "报告界为样本 0.5%/99.5% 分位数，另附样本最小值/最大值",
        ],
    }


# ----------------------------------------------------------- 结果总装

def _normalized_snapshot(nc: NormalizedChain) -> list[dict]:
    sigmas = _sigma_arr(nc.dimensions)
    halfs = np.array([d.half_width for d in nc.dimensions])
    sample_halfs = _sampling_half_widths(
        nc.dimensions, sigmas, halfs,
        [d.sigma_explicit for d in nc.dimensions])
    return [
        {
            "dimension_id": d.id,
            "edge": [d.start, d.end],
            "sign": d.sign,
            "measurement_direction": d.direction,
            "distribution": d.distribution,
            "sigma_explicit": d.sigma_explicit,
            "normalized_mm": {
                "nominal": d.nominal,
                "upper_deviation": d.upper_dev,
                "lower_deviation": d.lower_dev,
                "mid_shift": d.mid,
                "half_width": d.half_width,
                "sigma": d.sigma,
                "monte_carlo_sampling_half_width": float(sample_halfs[i]),
            },
            "original_representation": d.original,
        }
        for i, d in enumerate(nc.dimensions)
    ]


def compute_all(nc: NormalizedChain,
                sigmas: np.ndarray | None = None,
                mids: np.ndarray | None = None,
                halfs: np.ndarray | None = None,
                seed: int | None = None,
                explicit_flags=None) -> dict:
    """计算三种方法；sigmas/mids/halfs/explicit_flags 供方案分支覆盖。"""
    if sigmas is not None or mids is not None or halfs is not None:
        # 用覆盖参数做一个轻量副本（MC / RSS），WC 同步反映新公差带
        nd = nc.dimensions
        sigmas = _sigma_arr(nd) if sigmas is None else sigmas
        mids = np.array([d.mid for d in nd]) if mids is None else mids
        halfs = np.array([d.half_width for d in nd]) if halfs is None else halfs
        override_nc = _override_chain(
            nc, sigmas, mids, halfs, explicit_flags=explicit_flags)
        wc = worst_case(override_nc)
        rss = rss_analysis(override_nc)
        mc = monte_carlo(
            override_nc, seed=seed,
            sigmas=sigmas, mids=mids, halfs=halfs,
            explicit_flags=(
                [d.sigma_explicit for d in override_nc.dimensions]
                if explicit_flags is None else explicit_flags),
        )
    else:
        wc = worst_case(nc)
        rss = rss_analysis(nc)
        mc = monte_carlo(nc, seed=seed)
    return {
        "closure_spec_mm": {
            "lower_limit": nc.closure_lsl_mm,
            "upper_limit": nc.closure_usl_mm,
            "original": nc.closure_original,
        },
        "results": {"worst_case": wc, "rss": rss, "monte_carlo": mc},
        "normalized_inputs": _normalized_snapshot(nc),
        "traceability": {
            "unit_policy": "所有提交值按单位换算为 mm 参与计算；"
                          "原始名义值/偏差/标准差与单位原样保留",
            "mean_assumption": "制程中心假设位于公差带中点 N+m；"
                              "名义间隙 C0 仅由名义尺寸求和",
            "correlation": {
                "matrix": nc.corr.tolist(),
                "dimension_order": [d.id for d in nc.dimensions],
            },
            "monte_carlo": {"samples": nc.mc_samples, "seed": nc.seed},
        },
    }


def _override_chain(nc: NormalizedChain, sigmas: np.ndarray,
                    mids: np.ndarray, halfs: np.ndarray,
                    explicit_flags=None) -> NormalizedChain:
    new_dims: list[NormDimension] = []
    for j, (d, sigma, mid, half) in enumerate(
            zip(nc.dimensions, sigmas, mids, halfs)):
        flag = (
            d.sigma_explicit if explicit_flags is None
            else bool(explicit_flags[j])
        )
        new_dims.append(NormDimension(
            id=d.id, sign=d.sign, nominal=d.nominal,
            upper_dev=mid + half, lower_dev=mid - half,
            mid=float(mid), half_width=float(half),
            sigma=float(sigma), distribution=d.distribution,
            start=d.start, end=d.end, direction=d.direction,
            original=d.original, unit=d.unit,
            sigma_explicit=flag,
        ))
    return NormalizedChain(
        dimensions=new_dims,
        closure_lsl_mm=nc.closure_lsl_mm,
        closure_usl_mm=nc.closure_usl_mm,
        mc_samples=nc.mc_samples, seed=nc.seed, corr=nc.corr,
        name=nc.name, closure_unit=nc.closure_unit,
        closure_original=nc.closure_original,
    )


# --------------------------------------------------------- 区间概率

def closure_samples(nc: NormalizedChain, result: dict | None = None,
                    seed: int | None = None,
                    sigmas=None, mids=None, halfs=None,
                    explicit_flags=None) -> np.ndarray:
    """按 result 记录的种子/样本量（或指定种子）重放封闭环样本。"""
    if result is not None:
        mc = result["results"]["monte_carlo"]
        n_samples, used_seed = mc["samples"], mc["random_seed"]
    else:
        n_samples, used_seed = nc.mc_samples, nc.seed
    if seed is not None:
        used_seed = seed
    x = sample_dimensions(
        nc, n_samples, used_seed,
        sigmas=sigmas, mids=mids, halfs=halfs,
        explicit_flags=explicit_flags,
    )
    return x @ _s_arr(nc.dimensions)


def _interval_indicator(value: float, low: float | None,
                        high: float | None) -> float:
    scale = max(1.0, abs(value))
    if low is not None:
        scale = max(scale, abs(low))
    if high is not None:
        scale = max(scale, abs(high))
    eps = 1e-9 * scale  # 单位换算/求和浮点容差
    if low is not None and value < low - eps:
        return 0.0
    if high is not None and value > high + eps:
        return 0.0
    return 1.0


def gap_probability(nc: NormalizedChain, result: dict,
                    low_mm: float | None, high_mm: float | None,
                    closure: np.ndarray | None = None,
                    sigmas=None, mids=None, halfs=None,
                    explicit_flags=None) -> dict:
    """三种方法下封闭环落在 [low, high] 的概率（WC 为确定性 0/1）。

    零方差链（σ_C=0）的封闭环为固定间隙 μ_C：RSS 与蒙特卡洛均退化为
    指示函数——固定间隙落在查询闭区间内返回 1，否则 0。
    """
    mu = result["results"]["rss"]["mean_gap_mm"]
    sigma = result["results"]["rss"]["sigma_mm"]
    if sigma <= 0:
        rss_p = _interval_indicator(mu, low_mm, high_mm)
        deterministic = True
    else:
        upper_p = 1.0 if high_mm is None else float(
            normal_cdf((high_mm - mu) / sigma))
        lower_p = 0.0 if low_mm is None else float(
            normal_cdf((low_mm - mu) / sigma))
        rss_p = max(0.0, upper_p - lower_p)
        deterministic = False

    if closure is None:
        closure = closure_samples(
            nc, result, sigmas=sigmas, mids=mids, halfs=halfs,
            explicit_flags=explicit_flags)
    if float(np.asarray(closure).std(ddof=1)) <= 0:
        mc_p = _interval_indicator(float(np.asarray(closure)[0]),
                                   low_mm, high_mm)
    else:
        mask = np.ones_like(closure, dtype=bool)
        if low_mm is not None:
            mask &= closure >= low_mm
        if high_mm is not None:
            mask &= closure <= high_mm
        mc_p = float(mask.mean())

    wcl = result["results"]["worst_case"]["lower_bound_mm"]
    wcu = result["results"]["worst_case"]["upper_bound_mm"]
    eps = 1e-9 * max(1.0, abs(wcl), abs(wcu))
    contained = (low_mm is None or wcl >= low_mm - eps) and (
        high_mm is None or wcu <= high_mm + eps)
    return {
        "interval_mm": {"lower": low_mm, "upper": high_mm},
        "probabilities": {
            "rss": rss_p,
            "monte_carlo": mc_p,
            "worst_case": 1.0 if contained else 0.0,
        },
        "degenerate_fixed_gap_mm": mu if deterministic else None,
        "formulas": [
            "RSS: P = Φ((b-μ_C)/σ_C) - Φ((a-μ_C)/σ_C)；σ_C=0 时退化为"
            "固定间隙 μ_C 是否落在 [a,b] 的指示函数（0/1）",
            "MC: P = #{a ≤ C ≤ b} / N（固定种子经验频率）；样本无方差时"
            "同样按固定间隙判定 0/1",
            "WC: 确定性区间 [μ_C-ΣT, μ_C+ΣT] 完全落入请求区间记 1，否则 0",
        ],
        "worst_case_note": "极值法不含概率分布，0/1 表示确定性区间是否被覆盖",
    }
