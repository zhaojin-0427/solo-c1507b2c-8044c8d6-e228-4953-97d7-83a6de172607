"""服役磨损研究与维护编排测试。

覆盖：
* 创建期校验：漏填/链外尺寸、曲线不从 (0,0) 出发、分段不衔接、累计磨损
  下降、段循环数倒置、负速率σ、相关引用/重复/非半正定、节点不递增、
  节点超出曲线覆盖、封闭环限值缺失或倒置；
* 解析口径：节点 0 与基线链一致、分段线性插值、磨损方向符号、
  σ_wear=速率σ×N、共用载荷相关对消/放大、MC 与 RSS 同源；
* 首次越界循环分布与逐尺寸贡献；
* 维护编排：锁定/更换矛盾、维护节点子集、继续使用零成本、垫片恢复、
  更换清零重计、停机成本、排序键、选定冻结与不可改选；
* 冻结：研究重复读取不变、基线链不被改写。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engine import normalize_chain
from app.schemas import ChainCreate
from app.wear import analyze, build_model
from app.wear_schemas import WearStudyCreate


CHAIN = {
    "name": "wear-chain",
    "closure_lower_limit": 0.05,
    "closure_upper_limit": 0.60,
    "dimensions": [
        {"id": "L1", "start": "a", "end": "b", "nominal": 50,
         "upper_deviation": 0.08, "lower_deviation": 0.0,
         "std_dev": 0.02, "distribution": "normal"},
        {"id": "L2", "start": "b", "end": "c", "nominal": 30.3,
         "upper_deviation": 0.05, "lower_deviation": -0.05,
         "distribution": "uniform"},
        {"id": "L3", "start": "c", "end": "a", "nominal": 80,
         "upper_deviation": 0.0, "lower_deviation": -0.15,
         "std_dev": 0.04, "distribution": "normal", "direction": -1},
    ],
    "mc_samples": 20000,
    "random_seed": 7,
}

# 基线解析值：C0=0.3, μ=0.415, σ_C=sqrt(0.02²+(0.05/√3)²+0.04²)
BASE_MEAN = 0.415
BASE_SIGMA = math.sqrt(0.02 ** 2 + (0.05 / math.sqrt(3)) ** 2 + 0.04 ** 2)
BASE_HALF = 0.04 + 0.05 + 0.075

DIMS = [
    {"dimension_id": "L1", "wear_direction": "decrease",
     "wear_curve": [
         {"start_cycles": 0, "end_cycles": 10000,
          "start_wear": 0.0, "end_wear": 0.05},
         {"start_cycles": 10000, "end_cycles": 30000,
          "start_wear": 0.05, "end_wear": 0.20}],
     "rate_std_dev": 2e-6},
    {"dimension_id": "L2", "wear_direction": "increase",
     "wear_curve": [{"start_cycles": 0, "end_cycles": 30000,
                     "start_wear": 0.0, "end_wear": 0.06}],
     "rate_std_dev": 1e-6},
    {"dimension_id": "L3", "wear_direction": "increase",
     "wear_curve": [{"start_cycles": 0, "end_cycles": 30000,
                     "start_wear": 0.0, "end_wear": 0.03}],
     "rate_std_dev": 1e-6},
]

NODES = [0, 10000, 20000, 30000]


def study_payload(**over):
    base = {
        "name": "wear-v1",
        "dimensions": DIMS,
        "evaluation_cycles": NODES,
        "closure_lower_limit": 0.0,
        "closure_upper_limit": 0.8,
        "mc_samples": 20000,
        "random_seed": 99,
    }
    base.update(over)
    return base


def _model(**over):
    nc = normalize_chain(ChainCreate.model_validate(CHAIN))
    return build_model(nc, WearStudyCreate.model_validate(
        study_payload(**over)))


def _detail_text(r) -> str:
    d = r.json().get("detail")
    if isinstance(d, str):
        return d
    return " ".join(str(e.get("msg", "")) for e in d)


@pytest.fixture()
def chain_id(client):
    r = client.post("/chains", json=CHAIN)
    assert r.status_code == 201, r.text
    return r.json()["chain_id"]


@pytest.fixture()
def study_id(client, chain_id):
    r = client.post(f"/chains/{chain_id}/wear-studies", json=study_payload())
    assert r.status_code == 201, r.text
    return r.json()["study_id"]


# --------------------------------------------------------------- 解析口径

def test_node_zero_matches_baseline():
    """节点 0 无磨损：均值 / σ / 极值半宽与基线链完全一致。"""
    res = analyze(_model())
    n0 = res["nodes"][0]
    assert n0["cycles"] == 0.0
    assert n0["mean_gap_mm"] == pytest.approx(BASE_MEAN, abs=1e-12)
    assert n0["rss"]["sigma_mm"] == pytest.approx(BASE_SIGMA, abs=1e-12)
    assert n0["rss"]["sigma_wear_mm"] == 0.0
    assert n0["worst_case"]["total_extreme_half_width_mm"] == pytest.approx(
        BASE_HALF, abs=1e-12)
    assert res["baseline_manufacturing"]["mean_gap_mm"] == pytest.approx(
        BASE_MEAN, abs=1e-12)


def test_piecewise_linear_interpolation():
    """分段线性：L1 在 10000~30000 段内 20000 处插值为 0.125。"""
    model = _model()
    d = model.dims[0]
    assert d.id == "L1"
    assert float(np.interp(20000, d.curve_cycles, d.curve_wear_mm)) == \
        pytest.approx(0.125, abs=1e-12)
    assert float(np.interp(5000, d.curve_cycles, d.curve_wear_mm)) == \
        pytest.approx(0.025, abs=1e-12)


def test_wear_direction_signs():
    """磨损方向 × 闭环符号：L1 减小(s=+1)拉低均值，L2 增大抬高，L3 增大(s=-1)拉低。"""
    res = analyze(_model())
    # N=10000：shift = -0.05(L1) + 0.02(L2) - 0.01(L3) = -0.04
    n1 = res["nodes"][1]
    assert n1["cycles"] == 10000.0
    assert n1["wear_mean_shift_mm"] == pytest.approx(-0.04, abs=1e-12)
    assert n1["mean_gap_mm"] == pytest.approx(BASE_MEAN - 0.04, abs=1e-12)
    per = {p["dimension_id"]: p for p in n1["per_dimension"]}
    assert per["L1"]["closure_mean_shift_mm"] == pytest.approx(-0.05)
    assert per["L2"]["closure_mean_shift_mm"] == pytest.approx(0.02)
    assert per["L3"]["closure_mean_shift_mm"] == pytest.approx(-0.01)


def test_wear_std_grows_with_cycles():
    """随机速率模型：单尺寸磨损 σ(N)=速率σ×N，RSS 总 σ 为制造+磨损方差和。"""
    model = _model(
        dimensions=[DIMS[0]] + [
            {**DIMS[1], "rate_std_dev": 0.0},
            {**DIMS[2], "rate_std_dev": 0.0}],
        evaluation_cycles=[0, 20000])
    res = analyze(model)
    n1 = res["nodes"][1]
    sigma_wear = 2e-6 * 20000          # 0.04
    assert n1["rss"]["sigma_wear_mm"] == pytest.approx(sigma_wear, abs=1e-12)
    assert n1["rss"]["sigma_mm"] == pytest.approx(
        math.sqrt(BASE_SIGMA ** 2 + sigma_wear ** 2), abs=1e-12)
    # WC 半宽 = 制造 ΣT + 3σ_wear
    assert n1["worst_case"]["total_extreme_half_width_mm"] == pytest.approx(
        BASE_HALF + 3 * sigma_wear, abs=1e-12)


def test_shared_load_correlation_cancel_and_amplify():
    """共用载荷相关：c1·c2<0 时 ρ=+1 对消、ρ=-1 放大（方差 2r²N²(1-ρ)）。"""
    dims = [
        {**DIMS[0], "rate_std_dev": 3e-6},   # coef -1
        {**DIMS[1], "rate_std_dev": 3e-6},   # coef +1
        {**DIMS[2], "rate_std_dev": 0.0},
    ]

    def wear_var(rho):
        model = _model(
            dimensions=dims,
            wear_rate_correlations=(
                [] if rho is None else
                [{"dim_a": "L1", "dim_b": "L2", "rho": rho}]),
            evaluation_cycles=[0, 10000])
        return analyze(model)["nodes"][1]["rss"]["sigma_wear_mm"] ** 2

    r_n = 3e-6 * 10000
    assert wear_var(1.0) == pytest.approx(0.0, abs=1e-12)
    assert wear_var(0.0) == pytest.approx(2 * r_n ** 2, rel=1e-9)
    assert wear_var(-1.0) == pytest.approx(4 * r_n ** 2, rel=1e-9)


def test_monte_carlo_matches_rss_sigma():
    """固定种子 MC 与 RSS 同源：各节点 σ 与超差率接近解析值。"""
    res = analyze(_model())
    for node in res["nodes"]:
        rss = node["rss"]
        mc = node["monte_carlo"]
        assert mc["sigma_mm"] == pytest.approx(rss["sigma_mm"], rel=0.03)
        assert mc["mean_gap_mm"] == pytest.approx(
            node["mean_gap_mm"], abs=3e-3)


def test_first_exceedance_distribution():
    """首次越界分布：逐节点计数 + 从未越界 = 样本数；越界循环分位数存在。"""
    res = analyze(_model(
        closure_lower_limit=0.30, closure_upper_limit=0.60))
    dist = res["summary"]["first_exceedance_monte_carlo"]
    total = sum(p["count"] for p in dist["per_node"])
    total += dist["never_exceeded"]["count"]
    assert total == 20000
    frac = sum(p["fraction"] for p in dist["per_node"])
    frac += dist["never_exceeded"]["fraction"]
    assert frac == pytest.approx(1.0)
    assert dist["exceeded_count"] > 0
    q = dist["first_exceedance_cycle_quantiles"]
    assert q["min"] <= q["p50"] <= q["max"]
    # 越界只能发生在计算节点上
    node_set = set(NODES)
    assert all(p["cycles"] in node_set for p in dist["per_node"])


def test_dimension_contributions_grow_and_share_sums():
    """逐尺寸贡献：磨损方差贡献随循环增大；各节点贡献占比之和为 1。"""
    res = analyze(_model())
    n0, n3 = res["nodes"][0], res["nodes"][3]
    for p in n0["per_dimension"]:
        assert p["wear_variance_contribution_mm2"] == 0.0
        assert p["wear_std_at_node_mm"] == 0.0
    shares = sum(p["variance_share"] for p in n3["per_dimension"])
    assert shares == pytest.approx(1.0)
    per3 = {p["dimension_id"]: p for p in n3["per_dimension"]}
    assert per3["L1"]["wear_std_at_node_mm"] == pytest.approx(2e-6 * 30000)


def test_curve_unit_conversion():
    """磨损曲线可按 um 提交，内部换算为 mm。"""
    dims = [
        {"dimension_id": "L1", "wear_direction": "decrease",
         "wear_curve": [{"start_cycles": 0, "end_cycles": 10000,
                         "start_wear": 0.0, "end_wear": 50.0}],
         "curve_unit": "um", "rate_std_dev": 0.002},
        {**DIMS[1], "wear_curve": [{"start_cycles": 0, "end_cycles": 10000,
                                    "start_wear": 0.0, "end_wear": 0.0}],
         "rate_std_dev": 0.0},
        {**DIMS[2], "wear_curve": [{"start_cycles": 0, "end_cycles": 10000,
                                    "start_wear": 0.0, "end_wear": 0.0}],
         "rate_std_dev": 0.0},
    ]
    res = analyze(_model(dimensions=dims, evaluation_cycles=[0, 10000]))
    n1 = res["nodes"][1]
    # 50 um = 0.05 mm；0.002 um/循环 = 2e-6 mm/循环 → σ_wear=0.02 mm
    assert n1["wear_mean_shift_mm"] == pytest.approx(-0.05, abs=1e-12)
    assert n1["rss"]["sigma_wear_mm"] == pytest.approx(0.02, abs=1e-15)


# --------------------------------------------------------------- 创建校验

def test_missing_dimension_rejected(client, chain_id):
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=DIMS[:2]))
    assert r.status_code == 422
    assert "漏填" in _detail_text(r)


def test_extra_dimension_rejected(client, chain_id):
    dims = DIMS + [{"dimension_id": "LX", "wear_direction": "increase",
                    "wear_curve": [{"start_cycles": 0, "end_cycles": 30000,
                                    "start_wear": 0.0, "end_wear": 0.01}],
                    "rate_std_dev": 0.0}]
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=dims))
    assert r.status_code == 422
    assert "链外" in _detail_text(r)


def test_curve_must_start_at_origin(client, chain_id):
    dims = [dict(DIMS[0], wear_curve=[
        {"start_cycles": 100, "end_cycles": 30000,
         "start_wear": 0.0, "end_wear": 0.2}])] + DIMS[1:]
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=dims))
    assert r.status_code == 422
    assert "(0, 0)" in _detail_text(r)


def test_curve_segments_must_connect(client, chain_id):
    """分段不衔接（折点不一致）拒绝。"""
    dims = [dict(DIMS[0], wear_curve=[
        {"start_cycles": 0, "end_cycles": 10000,
         "start_wear": 0.0, "end_wear": 0.05},
        {"start_cycles": 10000, "end_cycles": 30000,
         "start_wear": 0.06, "end_wear": 0.20}])] + DIMS[1:]
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=dims))
    assert r.status_code == 422
    assert "不衔接" in _detail_text(r)


def test_cumulative_wear_cannot_decrease(client, chain_id):
    dims = [dict(DIMS[0], wear_curve=[
        {"start_cycles": 0, "end_cycles": 30000,
         "start_wear": 0.05, "end_wear": 0.01}])] + DIMS[1:]
    # 首段也必须从 0 出发；把首段改为合法再从第二段下降
    dims[0]["wear_curve"] = [
        {"start_cycles": 0, "end_cycles": 10000,
         "start_wear": 0.0, "end_wear": 0.10},
        {"start_cycles": 10000, "end_cycles": 30000,
         "start_wear": 0.10, "end_wear": 0.05}]
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=dims))
    assert r.status_code == 422
    assert "单调不减" in _detail_text(r) or "减少" in _detail_text(r)


def test_segment_cycles_inverted_rejected(client, chain_id):
    dims = [dict(DIMS[0], wear_curve=[
        {"start_cycles": 0, "end_cycles": 30000,
         "start_wear": 0.0, "end_wear": 0.2},
        {"start_cycles": 30000, "end_cycles": 10000,
         "start_wear": 0.2, "end_wear": 0.3}])] + DIMS[1:]
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=dims))
    assert r.status_code == 422


def test_negative_rate_std_rejected(client, chain_id):
    dims = [dict(DIMS[0], rate_std_dev=-1e-6)] + DIMS[1:]
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(dimensions=dims))
    assert r.status_code == 422


def test_correlation_reference_and_duplicate(client, chain_id):
    r = client.post(f"/chains/{chain_id}/wear-studies", json=study_payload(
        wear_rate_correlations=[{"dim_a": "L1", "dim_b": "LX", "rho": 0.5}]))
    assert r.status_code == 422
    assert "引用" in _detail_text(r)
    r = client.post(f"/chains/{chain_id}/wear-studies", json=study_payload(
        wear_rate_correlations=[
            {"dim_a": "L1", "dim_b": "L2", "rho": 0.5},
            {"dim_a": "L2", "dim_b": "L1", "rho": 0.4}]))
    assert r.status_code == 422
    assert "重复" in _detail_text(r)


def test_correlation_matrix_not_psd(client, chain_id):
    r = client.post(f"/chains/{chain_id}/wear-studies", json=study_payload(
        wear_rate_correlations=[
            {"dim_a": "L1", "dim_b": "L2", "rho": 0.9},
            {"dim_a": "L1", "dim_b": "L3", "rho": 0.9},
            {"dim_a": "L2", "dim_b": "L3", "rho": -0.9}]))
    assert r.status_code == 422
    assert "半正定" in _detail_text(r)


def test_nodes_must_increase(client, chain_id):
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(evaluation_cycles=[0, 30000, 10000]))
    assert r.status_code == 422
    assert "严格递增" in _detail_text(r)


def test_node_beyond_curve_rejected(client, chain_id):
    r = client.post(f"/chains/{chain_id}/wear-studies",
                    json=study_payload(evaluation_cycles=[0, 50000]))
    assert r.status_code == 422
    assert "覆盖范围" in _detail_text(r)


def test_closure_limits_required_and_ordered(client, chain_id):
    p = study_payload()
    del p["closure_lower_limit"]
    del p["closure_upper_limit"]
    r = client.post(f"/chains/{chain_id}/wear-studies", json=p)
    assert r.status_code == 422
    assert "限值" in _detail_text(r)
    r = client.post(f"/chains/{chain_id}/wear-studies", json=study_payload(
        closure_lower_limit=0.9, closure_upper_limit=0.1))
    assert r.status_code == 422


def test_unknown_chain_404(client):
    r = client.post("/chains/999/wear-studies", json=study_payload())
    assert r.status_code == 404


# --------------------------------------------------------------- 冻结与读取

def test_study_frozen_and_baseline_untouched(client, chain_id, study_id):
    r1 = client.get(f"/wear-studies/{study_id}")
    r2 = client.get(f"/wear-studies/{study_id}")
    assert r1.status_code == 200 and r1.json() == r2.json()
    body = r1.json()
    assert body["frozen"] is True
    assert body["baseline_preserved"] is True
    assert body["result"]["traceability"]["baseline_chain_id"] == chain_id
    # 基线链结果不被研究改写
    chain_after = client.get(f"/chains/{chain_id}").json()
    assert chain_after["result"]["results"]["rss"]["mean_gap_mm"] == \
        pytest.approx(BASE_MEAN)


def test_list_and_missing_study(client, chain_id, study_id):
    r = client.get(f"/chains/{chain_id}/wear-studies")
    assert r.status_code == 200
    studies = r.json()["studies"]
    assert len(studies) == 1 and studies[0]["id"] == study_id
    assert studies[0]["node_count"] == len(NODES)
    assert client.get("/wear-studies/999").status_code == 404


def test_first_spec_breach_analytic(client, chain_id, study_id):
    res = client.get(f"/wear-studies/{study_id}").json()["result"]
    breach = res["summary"]["first_spec_breach"]
    assert breach is not None
    assert breach["criterion"] == "worst_case"
    # 无相关时 σ_wear(20000)=sqrt(0.04²+0.02²+0.02²)，WC 下界首次跌破 0
    assert breach["cycles"] == 20000.0
    assert breach["breach_side"] == ["lower"]


# --------------------------------------------------------------- 维护编排

def maint_payload(**over):
    base = {
        "name": "maint-1",
        "locked_dimensions": ["L3"],
        "shim_candidates": [
            {"shim_id": "S1", "thickness": 0.1, "std_dev": 0.005,
             "closure_sign": 1, "cost": 50},
            {"shim_id": "S2", "thickness": 0.2, "std_dev": 0.005,
             "closure_sign": 1, "cost": 80}],
        "maintainable_cycles": [10000, 20000, 30000],
        "replacement_costs": {"L1": 300, "L2": 200},
        "downtime_cost": 100,
        "criterion": "worst_case",
        "mc_samples": 10000,
    }
    base.update(over)
    return base


@pytest.fixture()
def search_id(client, study_id):
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload())
    assert r.status_code == 201, r.text
    return r.json()["search_id"]


def test_maintenance_locked_and_cost_conflict(client, study_id):
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(replacement_costs={"L3": 100}))
    assert r.status_code == 422
    assert "锁定" in _detail_text(r)


def test_maintenance_node_subset(client, study_id):
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(maintainable_cycles=[15000]))
    assert r.status_code == 422
    assert "子集" in _detail_text(r)


def test_maintenance_unknown_dimension_refs(client, study_id):
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(locked_dimensions=["LX"]))
    assert r.status_code == 422
    assert "锁定尺寸" in _detail_text(r)
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(replacement_costs={"LX": 10}))
    assert r.status_code == 422
    assert "链外" in _detail_text(r)


def test_maintenance_ranking_and_costs(client, search_id):
    body = client.get(f"/wear-maintenance-searches/{search_id}").json()
    cands = body["result"]["candidates"]
    assert len(cands) >= 2
    # 排序键：违规数 → 最早越界点（晚者优先）→ 总成本
    keys = [(c["metrics"]["violations"],
             -(c["metrics"]["first_breach_cycles"] or float("inf")),
             c["cost"]["total"]) for c in cands]
    assert keys == sorted(keys)
    # 零动作现状方案始终在内且零成本
    cont = next(c for c in cands if c["strategy"] == "continue_only")
    assert cont["actions"] == [] and cont["cost"]["total"] == 0.0
    assert cont["metrics"]["violations"] >= 1
    # 最优方案零违规且成本含停机（一次维护事件 = 动作成本 + 停机）
    best = cands[0]
    assert best["metrics"]["violations"] == 0
    assert best["cost"]["total"] == pytest.approx(
        best["cost"]["shims"] + best["cost"]["replacements"]
        + best["cost"]["downtime"])
    assert best["cost"]["downtime"] == pytest.approx(
        100 * best["maintenance_events"])


def test_shim_restores_compliance(client, search_id):
    body = client.get(f"/wear-maintenance-searches/{search_id}").json()
    cands = body["result"]["candidates"]
    shim_plan = next(c for c in cands
                     if any(a["kind"] == "shim" for a in c["actions"]))
    act = shim_plan["actions"][0]
    node_row = next(n for n in shim_plan["nodes"]
                    if n["cycles"] == act["node_cycles"])
    assert node_row["violated"] is False
    # 垫片 +0.1 增隙：动作节点均值较现状方案同节点抬高
    cont = next(c for c in cands if c["strategy"] == "continue_only")
    cont_row = next(n for n in cont["nodes"]
                    if n["cycles"] == act["node_cycles"])
    assert node_row["mean_gap_mm"] == pytest.approx(
        cont_row["mean_gap_mm"] + 0.1, abs=1e-9)


def test_replacement_resets_wear(client, study_id):
    """更换零件：该尺寸累计磨损清零，自更换节点重新计循环。"""
    r = client.post(
        f"/wear-studies/{study_id}/maintenance-searches",
        json=maint_payload(name="replace-only", shim_candidates=[],
                           replacement_costs={"L1": 300}))
    assert r.status_code == 201, r.text
    cands = r.json()["result"]["candidates"]
    plan = next(c for c in cands
                if any(a["kind"] == "replace" for a in c["actions"]))
    act = next(a for a in plan["actions"] if a["kind"] == "replace")
    assert act["target"] == "L1"
    n_r = act["node_cycles"]
    # 更换前已累计的磨损 μ(N_r) 被清零：末节点均值较现状方案回升
    # coef·μ(N_r)（L1 为 decrease、闭环符号 +1 → coef=-1，回升 +μ(N_r)）
    cont = next(c for c in cands if c["strategy"] == "continue_only")
    last = plan["nodes"][-1]
    cont_last = next(n for n in cont["nodes"]
                     if n["cycles"] == last["cycles"])
    assert last["mean_gap_mm"] > cont_last["mean_gap_mm"]
    mu_nr = float(np.interp(n_r, [0, 10000, 30000], [0.0, 0.05, 0.20]))
    expected = cont_last["mean_gap_mm"] + mu_nr
    assert last["mean_gap_mm"] == pytest.approx(expected, abs=1e-9)
    # 更换节点本身按维护后状态评估（动作在该节点生效）
    node_r = next(n for n in plan["nodes"] if n["cycles"] == n_r)
    assert node_r["violated"] is False


def test_maintenance_criterion_rss(client, study_id):
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(name="rss-crit", criterion="rss"))
    assert r.status_code == 201, r.text
    assert r.json()["result"]["criterion"] == "rss"


def test_maintenance_frozen_readback(client, search_id):
    r1 = client.get(f"/wear-maintenance-searches/{search_id}")
    r2 = client.get(f"/wear-maintenance-searches/{search_id}")
    assert r1.json() == r2.json()
    lst = client.get(f"/wear-studies/{r1.json()['study_id']}"
                     "/maintenance-searches").json()
    assert any(s["id"] == search_id for s in lst["searches"])
    assert client.get("/wear-maintenance-searches/999").status_code == 404


def test_select_freezes_and_rejects_reselect(client, search_id):
    r = client.post(f"/wear-maintenance-searches/{search_id}/select",
                    json={"rank": 1, "note": "采纳"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frozen"] is True
    assert body["selected_rank"] == 1
    assert body["selected_candidate"]["rank"] == 1
    assert "冻结" in body["freeze_note"]
    # 重复选定 / 改选拒绝
    r = client.post(f"/wear-maintenance-searches/{search_id}/select",
                    json={"rank": 2})
    assert r.status_code == 422
    assert "不能改选" in _detail_text(r)
    # 结果内容在选定后不变
    again = client.get(f"/wear-maintenance-searches/{search_id}").json()
    assert again["selected_rank"] == 1


def test_select_bad_rank_and_missing(client, search_id):
    n_cands = len(client.get(f"/wear-maintenance-searches/{search_id}")
                   .json()["result"]["candidates"])
    r = client.post(f"/wear-maintenance-searches/{search_id}/select",
                    json={"rank": n_cands + 1})
    assert r.status_code == 422
    assert "超出候选数" in _detail_text(r)
    r = client.post("/wear-maintenance-searches/999/select",
                    json={"rank": 1})
    assert r.status_code == 404


def test_maintenance_seed_default_and_override(client, study_id):
    """缺省沿用研究种子；显式给种子时结果随请求冻结。"""
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(name="seed-default"))
    assert r.status_code == 201, r.text
    assert r.json()["result"]["monte_carlo"]["seed"] == 99  # 研究种子
    r = client.post(f"/wear-studies/{study_id}/maintenance-searches",
                    json=maint_payload(name="seed-override", random_seed=1234))
    assert r.status_code == 201, r.text
    assert r.json()["result"]["monte_carlo"]["seed"] == 1234
