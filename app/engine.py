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
    Gaussian copula。输入 ρ 统一为尺寸值的 Pearson 相关：
    先逐对把 ρ 反演为潜变量正态相关 ρ0（正态恒等；均匀-均匀
    ρ0=2sin(πρ/6)；均匀-正态 ρ0=ρ√(π/3)；三角配对固定样本标定），
    再由 Cholesky 生成相关正态、Φ 变换后代入各分布逆 CDF，
    因此抽样边缘 Pearson 相关与 RSS 公式中的 ρ 一致。
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
                sigma=sigma_mm,
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

# Pearson 相关 ρ 的统一语义：RSS 公式与蒙特卡洛抽样的**边缘 Pearson 相关**
# 都必须等于用户输入的 ρ。Gaussian copula 中需要校准的是潜变量正态相关
# ρ0，使 g(Φ(Z1))、h(Φ(Z2))（Z 相关 ρ0）的 Pearson 相关恰为 ρ：
#   uniform-uniform:  ρ = (6/π)·arcsin(ρ0/2)
#   uniform-normal:    Corr(2Φ(Z1)-1, Z2) = ρ0·√(3/π)
#   normal-normal:     ρ0 = ρ
# 含 triangular 的配对无简单闭式，用固定样本二分标定（结果缓存）。

_latent_cache: dict[tuple, float] = {}


def _triangular_inverse(u: np.ndarray, half: float) -> np.ndarray:
    """[-T, T] 上关于 0 对称的三角分布分位数函数。"""
    out = np.empty_like(np.asarray(u, dtype=float))
    u = np.asarray(u, dtype=float)
    left = u < 0.5
    out[left] = -half + half * np.sqrt(2.0 * u[left])
    out[~left] = half - half * np.sqrt(2.0 * (1.0 - u[~left]))
    return out


def norm_ppf(u: np.ndarray) -> np.ndarray:
    """标准正态分位数函数 Φ^{-1}（Peter Acklam 有理逼近，精度约 1e-9）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    u = np.asarray(u, dtype=float)
    x = np.zeros_like(u)
    plow, phigh = 0.02425, 1 - 0.02425

    low = u < plow
    if low.any():
        q = np.sqrt(-2.0 * np.log(u[low]))
        x[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                  * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    high = u > phigh
    if high.any():
        q = np.sqrt(-2.0 * np.log(1.0 - u[high]))
        x[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                    * q + c[5]) / \
                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    mid = ~(low | high)
    if mid.any():
        q = u[mid] - 0.5
        r = q * q
        x[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4])
                  * r + a[5]) * q / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4])
                  * r + 1.0)
    return x


def _standard_quantile(u: np.ndarray, distribution: str) -> np.ndarray:
    """单位尺度边缘分位数（normal: Φ⁻¹; uniform: 2U-1; triangular: 半宽1）。"""
    if distribution == DistributionType.NORMAL.value:
        return norm_ppf(u)
    if distribution == DistributionType.UNIFORM.value:
        return 2.0 * np.asarray(u) - 1.0
    return _triangular_inverse(u, 1.0)


_calib_bank: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_CALIB_SEED = 20260910


def _calib_bank_arrays(n: int = 50_000):
    """固定种子的标准正态两列及对应 Φ 值（标定用，进程缓存）。"""
    if n not in _calib_bank:
        rng = np.random.default_rng(_CALIB_SEED)
        z = rng.standard_normal((n, 2))
        u = 0.5 * (1.0 + np.vectorize(math.erf)(
            z / math.sqrt(2.0)))
        _calib_bank[n] = (z, u)
    return _calib_bank[n]


def _latent_rho(dist_a: str, dist_b: str, rho: float) -> float:
    """返回使两边缘 Pearson 相关等于 rho 的潜变量正态相关 rho0。"""
    if rho == 0.0:
        return 0.0
    N_, U_ = DistributionType.NORMAL.value, DistributionType.UNIFORM.value
    key = tuple(sorted((dist_a, dist_b)))
    cache_key = (key, round(rho, 12))
    if cache_key in _latent_cache:
        return _latent_cache[cache_key]

    if dist_a == dist_b == N_:
        rho0 = rho
    elif key == (U_, U_):
        # Corr(Φ(Z1), Φ(Z2)) = (6/π) arcsin(ρ0/2)
        rho0 = 2.0 * math.sin(math.pi * rho / 6.0)
    elif N_ in key and U_ in key:
        # 均匀单位边缘 X=2Φ(Z1)-1（方差 1/3）：
        # E[X·Z2]=2·E[Φ(Z1)Z2]=2ρ0·E[Z1Φ(Z1)]
        #        =2ρ0·E[φ(Z1)]=ρ0/√π（Stein 引理）
        # => Corr = ρ0·√(3/π)，反演 ρ0 = ρ·√(π/3)
        rho0 = rho * math.sqrt(math.pi / 3.0)
    else:
        rho0 = _calibrate_latent(dist_a, dist_b, rho)
    rho0 = min(0.999999, max(-0.999999, rho0))
    _latent_cache[cache_key] = rho0
    return rho0


def _calibrate_latent(dist_a: str, dist_b: str, rho: float) -> float:
    """含 triangular 配对：固定样本上二分求使经验 Pearson 相关=rho 的 rho0。"""
    z, _ = _calib_bank_arrays()
    z1col = z[:, 0]
    z2col = z[:, 1]
    erf_vec = np.vectorize(math.erf)

    def emp(rho0: float) -> float:
        w2 = rho0 * z1col + math.sqrt(max(0.0, 1.0 - rho0 * rho0)) * z2col
        q1 = 0.5 * (1.0 + erf_vec(z1col / math.sqrt(2.0)))
        q2 = 0.5 * (1.0 + erf_vec(w2 / math.sqrt(2.0)))
        # 直接由相关均匀值 q1、q2 生成两种边缘（保持 copula 结构）
        g1 = _standard_quantile(q1, dist_a)
        g2 = _standard_quantile(q2, dist_b)
        return float(np.corrcoef(g1, g2)[0, 1])

    lo, hi = -0.999999, 0.999999
    # 5 万固定样本的经验相关噪声约 4e-3，二分 18 次即低于该噪声底
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if emp(mid) < rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _nearest_psd(r0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """把相关矩阵投影到最近的正定相关矩阵（谱截断后重标定对角为 1）。"""
    r = (r0 + r0.T) / 2.0
    w, v = np.linalg.eigh(r)
    w = np.clip(w, eps, None)
    r = (v * w[None, :]) @ v.T
    d = np.sqrt(np.diag(r))
    r = r / d[:, None] / d[None, :]
    return (r + r.T) / 2.0


def latent_correlation_matrix(nc: NormalizedChain) -> np.ndarray:
    """把用户 Pearson 相关矩阵逐对换算为 copula 潜变量相关矩阵。

    逐对换算可能破坏整体正定性（高相关 + 混合分布），用最近正定相关
    矩阵投影兜底；输入矩阵本身已由 schema 校验为半正定。
    """
    k = len(nc.dimensions)
    r0 = np.eye(k)
    r = nc.corr
    for i in range(k):
        for j in range(i + 1, k):
            if abs(r[i, j]) > 1e-15:
                rho0 = _latent_rho(
                    nc.dimensions[i].distribution,
                    nc.dimensions[j].distribution,
                    float(r[i, j]))
                r0[i, j] = r0[j, i] = rho0
    eig_min = np.linalg.eigvalsh((r0 + r0.T) / 2.0).min()
    if eig_min < 1e-10:
        r0 = _nearest_psd(r0)
    return r0


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

    输入相关系数 ρ 统一解释为尺寸值之间的 Pearson 相关：相关维度先将
    用户相关矩阵逐对换算为 copula 潜变量相关（均匀/正态解析反演、
    三角数值标定），再 Cholesky 抽样，使样本经验 Pearson 相关与
    RSS 协方差公式中的 ρ 一致。
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
        latent = latent_correlation_matrix(nc)
        z = rng.standard_normal((n_samples, k)) @ _cholesky_psd(latent).T
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
            "  输入 ρ=尺寸值 Pearson 相关；先反演潜变量相关 ρ0（均匀-均匀"
            " 2sin(πρ/6)、均匀-正态 ρ√(π/3)、三角配对数值标定），"
            "再 z=L·ξ (LLᵀ=R0, Cholesky)，u=Φ(z)，代入各分布逆 CDF",
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


def _correlation_trace(nc: NormalizedChain) -> dict:
    corr_trace = {
        "matrix": nc.corr.tolist(),
        "dimension_order": [d.id for d in nc.dimensions],
        "definition": "输入 ρ 为尺寸值之间的 Pearson 相关系数；"
                      "RSS 协方差公式与蒙特卡洛抽样边缘采用同一定义",
    }
    if not np.allclose(nc.corr, np.eye(len(nc.dimensions)), atol=1e-12):
        corr_trace["monte_carlo_latent_matrix"] = \
            latent_correlation_matrix(nc).tolist()
        corr_trace["copula_calibration"] = (
            "Gaussian copula 潜变量正态相关已逐对反演，使抽样边缘 "
            "Pearson 相关等于输入 ρ：正态恒等；均匀-均匀 ρ0=2sin(πρ/6)；"
            "均匀-正态 ρ0=ρ√(π/3)；含三角分布的配对固定样本数值标定；"
            "整体矩阵在边界情形投影到最近正定相关矩阵"
        )
    return corr_trace


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
            "correlation": _correlation_trace(nc),
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
