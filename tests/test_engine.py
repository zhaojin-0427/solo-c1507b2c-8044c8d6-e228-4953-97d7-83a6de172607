"""计算正确性测试：手工核对 WC / RSS / MC 与相关系数影响。"""
import math

import numpy as np

import pytest

from app.engine import (
    closure_samples,
    compute_all,
    gap_probability,
    normalize_chain,
)
from app.schemas import ChainCreate, CorrelationSpec


def _make(dims, corrs=(), seed=11, lsl=0.05, usl=0.60, samples=120000):
    return ChainCreate.model_validate({
        "name": "calc",
        "closure_lower_limit": lsl,
        "closure_upper_limit": usl,
        "dimensions": dims,
        "correlations": list(corrs),
        "mc_samples": samples,
        "random_seed": seed,
    })


# A=50(+0.08/0) B=30(±0.05 均匀) C=80(0/-0.15, 测量方向 -1)
# C0 = 50+30-80 = 0；为得到 0.3 的名义间隙，把 B 名义设 30.3
BASE = [
    {"id": "A", "start": "a", "end": "b", "nominal": 50,
     "upper_deviation": 0.08, "lower_deviation": 0.0, "std_dev": 0.02},
    {"id": "B", "start": "b", "end": "c", "nominal": 30.3,
     "upper_deviation": 0.05, "lower_deviation": -0.05,
     "distribution": "uniform"},
    {"id": "C", "start": "c", "end": "a", "nominal": 80,
     "upper_deviation": 0.0, "lower_deviation": -0.15, "std_dev": 0.04,
     "direction": -1},
]


def test_nominal_and_mean():
    nc = normalize_chain(_make(BASE))
    res = compute_all(nc)
    assert res["results"]["worst_case"]["nominal_gap_mm"] == pytest.approx(
        0.3, abs=1e-12)
    # m_A=0.04, m_B=0, s_C=-1 且 m_C=-0.075
    # mean = 0.3 + 0.04 + 0 + (-1)(-0.075) = 0.415
    assert res["results"]["rss"]["mean_gap_mm"] == pytest.approx(
        0.415, abs=1e-12)


def test_worst_case_bounds():
    nc = normalize_chain(_make(BASE))
    res = compute_all(nc)["results"]["worst_case"]
    # T_A=0.04, T_B=0.05, T_C=0.075 -> ΣT=0.165
    assert res["lower_bound_mm"] == pytest.approx(0.25, abs=1e-12)
    assert res["upper_bound_mm"] == pytest.approx(0.58, abs=1e-12)
    assert res["total_tolerance_band_mm"] == pytest.approx(0.33, abs=1e-12)
    # [0.25, 0.58] 在规格 [0.05, 0.60] 内 -> 极值超差率 0
    assert res["reject_probability"] == 0.0


def test_rss_sigma_independent():
    nc = normalize_chain(_make(BASE))
    res = compute_all(nc)["results"]["rss"]
    var = 0.02 ** 2 + (0.05 / math.sqrt(3)) ** 2 + 0.04 ** 2
    assert res["sigma_mm"] == pytest.approx(math.sqrt(var), abs=1e-12)
    assert res["lower_bound_mm"] == pytest.approx(
        0.415 - 3 * math.sqrt(var), abs=1e-12)
    # 方差贡献占比之和为 1
    total = sum(s["variance_share"] for s in res["sensitivity"])
    assert abs(total - 1.0) < 1e-12
    # 方向系数
    signs = {s["dimension_id"]: s["sign"] for s in res["sensitivity"]}
    assert signs == {"A": 1, "B": 1, "C": -1}


def test_rss_correlation_increases_or_decreases_variance():
    # A(s=+1) 与 C(s=-1) 正相关 -> 协方差项为负 -> 方差减小
    pos = CorrelationSpec(dim_a="A", dim_b="C", rho=0.9)
    neg = CorrelationSpec(dim_a="A", dim_b="C", rho=-0.9)
    s_pos = compute_all(normalize_chain(_make(BASE, [pos])))["results"]["rss"]["sigma_mm"]
    s_neg = compute_all(normalize_chain(_make(BASE, [neg])))["results"]["rss"]["sigma_mm"]
    s_ind = compute_all(normalize_chain(_make(BASE)))["results"]["rss"]["sigma_mm"]
    assert s_pos < s_ind < s_neg


def test_monte_carlo_matches_rss_and_deterministic_seed():
    nc = normalize_chain(_make(BASE))
    r1 = compute_all(nc)["results"]["monte_carlo"]
    r2 = compute_all(nc)["results"]["monte_carlo"]
    assert r1["mean_gap_mm"] == r2["mean_gap_mm"]
    assert r1["random_seed"] == r2["random_seed"]
    rss = compute_all(nc)["results"]["rss"]
    assert abs(r1["mean_gap_mm"] - rss["mean_gap_mm"]) < 2e-3
    assert abs(r1["sigma_mm"] - rss["sigma_mm"]) < 1e-3
    assert 0.0 <= r1["reject_probability"] <= 0.01


def test_monte_carlo_correlation_via_copula():
    pos = CorrelationSpec(dim_a="A", dim_b="C", rho=0.8)
    all_normal = [
        dict(BASE[0]),
        {"id": "B", "start": "b", "end": "c", "nominal": 30.3,
         "upper_deviation": 0.05, "lower_deviation": -0.05,
         "distribution": "normal", "std_dev": 0.03},
        dict(BASE[2]),
    ]
    nc = normalize_chain(_make(all_normal, [pos]))
    mc = compute_all(nc)["results"]["monte_carlo"]
    rss = compute_all(nc)["results"]["rss"]
    assert abs(mc["sigma_mm"] - rss["sigma_mm"]) / rss["sigma_mm"] < 0.03


def test_opposed_uniform_rho_consistency():
    """用户缺陷：两个反向均匀尺寸 ρ=0.8、σ=0.05，MC σ 必须与 RSS 一致。

    未校准的 copula 潜变量相关会导致 3.5% 偏差；校准后应 <0.2%。
    """
    dims = [
        {"id": "A", "start": "a", "end": "b", "nominal": 10.0,
         "upper_deviation": 0.2, "lower_deviation": -0.2,
         "distribution": "uniform", "std_dev": 0.05},
        {"id": "B", "start": "b", "end": "a", "nominal": 10.0,
         "upper_deviation": 0.2, "lower_deviation": -0.2,
         "distribution": "uniform", "std_dev": 0.05, "direction": -1},
    ]
    nc = normalize_chain(
        _make(dims, [CorrelationSpec(dim_a="A", dim_b="B", rho=0.8)],
              lsl=-1, usl=1, samples=400_000))
    res = compute_all(nc)["results"]
    rss = res["rss"]["sigma_mm"]
    mc = res["monte_carlo"]["sigma_mm"]
    # 解析期望：σ_A²+σ_B²-2·0.8·σ_Aσ_B = 2·0.05²·0.2 = 0.001
    assert rss == pytest.approx(math.sqrt(0.001), rel=1e-12)
    assert abs(mc - rss) / rss < 0.002
    # 样本边缘 Pearson 相关必须就是输入的 0.8
    from app.engine import sample_dimensions
    x = sample_dimensions(nc, 200_000, 7)
    assert abs(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) - 0.8) < 0.01


@pytest.mark.parametrize("pair,target", [
    (("uniform", "uniform"), 0.8),
    (("uniform", "normal"), 0.8),
    (("triangular", "uniform"), 0.8),
    (("triangular", "triangular"), -0.6),
    (("triangular", "normal"), 0.3),
])
def test_latent_rho_realizes_target_pearson(pair, target):
    """潜变量相关校准后，边缘经验 Pearson 相关必须复现输入 ρ。"""
    from app.engine import _latent_rho, _standard_quantile
    rho0 = _latent_rho(pair[0], pair[1], target)
    rng = np.random.default_rng(4242)
    n = 200_000
    z = rng.standard_normal((n, 2))
    w2 = rho0 * z[:, 0] + math.sqrt(max(0.0, 1 - rho0 ** 2)) * z[:, 1]
    erf = np.vectorize(math.erf)
    q1 = 0.5 * (1 + erf(z[:, 0] / math.sqrt(2)))
    q2 = 0.5 * (1 + erf(w2 / math.sqrt(2)))
    g1 = _standard_quantile(q1, pair[0])
    g2 = _standard_quantile(q2, pair[1])
    got = float(np.corrcoef(g1, g2)[0, 1])
    assert abs(got - target) < 0.01


def test_gap_probability_methods():
    nc = normalize_chain(_make(BASE))
    result = compute_all(nc)
    closure = closure_samples(nc, result)
    # 请求区间恰为 WC 界 [0.25, 0.58]
    ans = gap_probability(nc, result, 0.25, 0.58, closure=closure)
    assert 0.9 < ans["probabilities"]["rss"] < 0.999
    assert abs(ans["probabilities"]["rss"]
               - ans["probabilities"]["monte_carlo"]) < 0.01
    assert ans["probabilities"]["worst_case"] == 1.0
    # 比 WC 更窄的区间不可能完全包含 WC
    ans2 = gap_probability(nc, result, 0.30, 0.50, closure=closure)
    assert ans2["probabilities"]["worst_case"] == 0.0
    assert ans2["probabilities"]["rss"] < ans["probabilities"]["rss"]


def test_explicit_sigma_uniform_sampling_variance():
    """显式 σ 的均匀/三角维度，蒙特卡洛抽样方差必须等于 σ²（与 RSS 同口径）。"""
    from app.engine import sample_dimensions

    chain = ChainCreate.model_validate({
        "name": "explicit-sigma",
        "dimensions": [
            # 显式 σ 远小于公差带理论值
            {"id": "U", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.2, "lower_deviation": -0.2,
             "distribution": "uniform", "std_dev": 0.05},
            {"id": "T", "start": "b", "end": "a", "nominal": 10,
             "upper_deviation": 0.2, "lower_deviation": -0.2,
             "distribution": "triangular", "std_dev": 0.03, "direction": -1},
        ],
        "mc_samples": 400000,
    })
    nc = normalize_chain(chain)
    assert nc.dimensions[0].sigma_explicit is True
    x = sample_dimensions(nc, 400_000, 99)
    # 样本标准差 ≈ 显式 σ，而非公差带理论 σ
    assert abs(float(x[:, 0].std(ddof=1)) - 0.05) < 2e-3
    assert abs(float(x[:, 1].std(ddof=1)) - 0.03) < 1.5e-3
    # 显式 σ=0.05 -> 抽样半宽 σ√3≈0.0866 < 公差带半宽 0.2：
    # 样本严格落在公差带内部，且散布由 σ（而非公差带）决定
    assert float(x[:, 0].max()) < 10.2 - 1e-9
    assert float(x[:, 0].min()) > 9.8 + 1e-9
    assert abs(float(x[:, 0].max()) - (10 + 0.05 * math.sqrt(3))) < 2e-3
    res = compute_all(nc)["results"]
    # 封闭环 σ 与 MC σ 一致
    assert abs(res["rss"]["sigma_mm"] - res["monte_carlo"]["sigma_mm"]) / \
        res["rss"]["sigma_mm"] < 0.02


def test_zero_variance_gap_probability_engine():
    nc = normalize_chain(_make(BASE, lsl=-10, usl=10))
    # 全部 σ 置零
    import numpy as np
    zero = np.zeros(len(nc.dimensions))
    mids = np.array([d.mid for d in nc.dimensions])
    halfs = np.array([d.half_width for d in nc.dimensions])
    from app.engine import _override_chain, gap_probability
    ov = _override_chain(nc, zero, mids, halfs,
                         explicit_flags=[True] * len(zero))
    result = compute_all(ov, sigmas=zero, mids=mids, halfs=halfs,
                         explicit_flags=[True] * len(zero))
    assert result["results"]["rss"]["sigma_mm"] == 0.0
    mu = result["results"]["rss"]["mean_gap_mm"]
    inside = gap_probability(ov, result, mu - 0.01, mu + 0.01)
    assert inside["probabilities"]["rss"] == 1.0
    assert inside["probabilities"]["monte_carlo"] == 1.0
    outside = gap_probability(ov, result, mu + 1.0, mu + 2.0)
    assert outside["probabilities"]["rss"] == 0.0
    assert outside["probabilities"]["monte_carlo"] == 0.0


def test_distribution_theoretical_sigmas():
    chain = ChainCreate.model_validate({
        "name": "t",
        "dimensions": [
            {"id": "X", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.06, "lower_deviation": -0.06,
             "distribution": "triangular"},
            {"id": "Y", "start": "b", "end": "a", "nominal": 10,
             "upper_deviation": 0.06, "lower_deviation": -0.06,
             "distribution": "uniform", "direction": -1},
        ],
        "mc_samples": 2000,
    })
    nc = normalize_chain(chain)
    assert nc.dimensions[0].sigma == pytest.approx(0.06 / math.sqrt(6))
    assert nc.dimensions[1].sigma == pytest.approx(0.06 / math.sqrt(3))


def test_unilateral_shift_flows_to_mean():
    """单边公差通过带中点 m 移动封闭环均值。"""
    dims = [
        {"id": "A", "start": "a", "end": "b", "nominal": 20,
         "upper_deviation": 0.1, "lower_deviation": 0.0,
         "std_dev": 0.02},
        {"id": "B", "start": "b", "end": "a", "nominal": 20,
         "upper_deviation": 0.0, "lower_deviation": -0.1,
         "std_dev": 0.02, "direction": -1},
    ]
    nc = normalize_chain(_make(dims, lsl=-1, usl=1))
    res = compute_all(nc)["results"]
    # C0=0；m_A=+0.05, m_B=-0.05, s_B=-1 -> mean = 0 + 0.05 + 0.05
    assert res["rss"]["mean_gap_mm"] == pytest.approx(0.10, abs=1e-12)
