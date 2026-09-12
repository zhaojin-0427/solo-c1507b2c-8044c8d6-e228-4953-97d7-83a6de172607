"""热分析（温度工况）API 与引擎测试。

覆盖：
* 创建期校验：漏填/链外/重复尺寸、温度上下界倒置、α 与温度相关矩阵
  引用外尺寸 / 非半正定、负 α、无封闭环规格拒绝；
* 热态换算手算：L(T)=L0[1+α(T−T0)] 的均值与公差带缩放；
* RSS 三来源（制造 / 膨胀系数 / 温度）方差分解，固定温度与温度区间口径；
* 固定种子蒙特卡洛与解析 σ 一致、四子流复现、版本冻结重复 GET 不变；
* 摄氏/开尔文温差等价、逐尺寸温度覆盖、相关 α / u(T) 传播方向性；
* 最先越界工况判定；
* 方案搜索：材料替换 / 垫片 / 装配基准温度、锁定材料、排序与成本、
  组合数上限、方案结果冻结。
"""
from __future__ import annotations

import math

import pytest

from app.scenarios import rebuild_normalized
from app.schemas import ThermalAnalysisCreate
from app import thermal as th


CHAIN = {
    "name": "thermal-chain",
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
    "mc_samples": 40000,
    "random_seed": 7,
}

DIMS = [
    {"dimension_id": "L1", "reference_temperature": 20,
     "alpha": 23e-6, "alpha_std_uncertainty": 1e-6},
    {"dimension_id": "L2", "reference_temperature": 20,
     "alpha": 12e-6, "alpha_std_uncertainty": 0.5e-6},
    {"dimension_id": "L3", "reference_temperature": 20,
     "alpha": 11e-6, "alpha_std_uncertainty": 0.5e-6},
]

CONDS = [
    {"name": "cold",
     "global_temperature": {"fixed": {"mean": -20, "std_uncertainty": 2}}},
    {"name": "hot",
     "global_temperature": {"range": {"lower": 60, "upper": 80}}},
]

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
def analysis_id(client, chain_id):
    payload = {"name": "thermal-v1", "dimensions": DIMS,
               "conditions": CONDS, "mc_samples": 20000,
               "random_seed": 42}
    r = client.post(f"/chains/{chain_id}/thermal-analyses", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["thermal_analysis_id"]


def thermal_payload(**over):
    base = {"name": "th", "dimensions": DIMS, "conditions": CONDS,
            "mc_samples": 10000, "random_seed": 42}
    base.update(over)
    return base


# --------------------------------------------------------------- 创建校验

def test_missing_dimension_rejected(client, chain_id):
    payload = thermal_payload(dimensions=DIMS[:2])
    r = client.post(f"/chains/{chain_id}/thermal-analyses", json=payload)
    assert r.status_code == 422
    assert "漏填" in r.json()["detail"]


def test_extra_dimension_rejected(client, chain_id):
    extra = DIMS + [{"dimension_id": "ZZ", "reference_temperature": 20,
                     "alpha": 1e-5, "alpha_std_uncertainty": 0.0}]
    r = client.post(f"/chains/{chain_id}/thermal-analyses",
                    json=thermal_payload(name="ex", dimensions=extra))
    assert r.status_code == 422 and "链外" in r.json()["detail"]


def test_duplicate_dimension_rejected(client, chain_id):
    r = client.post(
        f"/chains/{chain_id}/thermal-analyses",
        json=thermal_payload(name="dup", dimensions=DIMS + [DIMS[0]]))
    assert r.status_code == 422 and "重复" in _detail_text(r)


def test_inverted_temperature_range_rejected(client, chain_id):
    bad_conds = [{"name": "bad",
                  "global_temperature": {"range": {"lower": 90, "upper": 80}}}]
    r = client.post(f"/chains/{chain_id}/thermal-analyses",
                    json=thermal_payload(name="inv", conditions=bad_conds))
    assert r.status_code == 422 and "上界" in _detail_text(r)


def test_condition_without_temperature_rejected(client, chain_id):
    # 逐尺寸工况只覆盖 L1、无全局温度 => L2/L3 无温度
    conds = [{"name": "partial",
              "dimension_temperatures": {
                  "L1": {"fixed": {"mean": 40, "std_uncertainty": 1}}}}]
    r = client.post(f"/chains/{chain_id}/thermal-analyses",
                    json=thermal_payload(name="p", conditions=conds))
    assert r.status_code == 422 and "未提供温度" in r.json()["detail"]


def test_non_psd_alpha_correlation_rejected(client, chain_id):
    corrs = [{"dim_a": "L1", "dim_b": "L2", "rho": 0.9},
             {"dim_a": "L2", "dim_b": "L3", "rho": 0.9},
             {"dim_a": "L1", "dim_b": "L3", "rho": -0.9}]
    r = client.post(
        f"/chains/{chain_id}/thermal-analyses",
        json=thermal_payload(name="np1", alpha_correlations=corrs))
    assert r.status_code == 422 and "半正定" in _detail_text(r)


def test_alpha_correlation_unknown_dimension_rejected(client, chain_id):
    r = client.post(
        f"/chains/{chain_id}/thermal-analyses",
        json=thermal_payload(
            name="np2",
            temperature_correlations=[
                {"dim_a": "L1", "dim_b": "ZZ", "rho": 0.3}]))
    assert r.status_code == 422


def test_negative_alpha_rejected(client, chain_id):
    dims = [dict(d) for d in DIMS]
    dims[0] = {**dims[0], "alpha": -1e-6}
    r = client.post(f"/chains/{chain_id}/thermal-analyses",
                    json=thermal_payload(name="neg", dimensions=dims))
    assert r.status_code == 422 and "膨胀系数" in _detail_text(r)


def test_no_closure_spec_rejected(client):
    chain = dict(CHAIN, name="no-spec",
                 closure_lower_limit=None, closure_upper_limit=None)
    cid = client.post("/chains", json=chain).json()["chain_id"]
    r = client.post(f"/chains/{cid}/thermal-analyses",
                    json=thermal_payload(name="ns"))
    assert r.status_code == 422 and "封闭环" in r.json()["detail"]


def test_unknown_chain_404(client, analysis_id):
    r = client.post("/chains/9999/thermal-analyses", json=thermal_payload())
    assert r.status_code == 404


def test_condition_names_unique(client, chain_id):
    conds = CONDS + [{"name": "hot",
                      "global_temperature": {"fixed": {"mean": 30}}}]
    r = client.post(f"/chains/{chain_id}/thermal-analyses",
                    json=thermal_payload(name="dup-cond", conditions=conds))
    assert r.status_code == 422 and "工况名称重复" in _detail_text(r)


# --------------------------------------------------------- 手算与分解

def _model(seed=42, **over):
    nc = rebuild_normalized(CHAIN)
    payload = ThermalAnalysisCreate.model_validate(thermal_payload(**over))
    return th.build_model(nc, payload)


def test_hot_nominal_and_scale_manual():
    """hot 工况（区间中点 70°C，ΔT=50）名义/均值手算核对。"""
    model = _model()
    res = th.evaluate_condition(model, 1, mc_samples=20000)["analytic"]
    a = {p["dimension_id"]: p for p in res["per_dimension"]}
    # L1: 50*(1+23e-6*50)=50.0575；L2: 30.3*1.0006=30.31818；
    # L3(sign=-1): 80*(1+11e-6*50)=80.044
    assert a["L1"]["hot_nominal_mm"] == pytest.approx(50.0575, abs=1e-9)
    assert a["L2"]["hot_nominal_mm"] == pytest.approx(30.31818, abs=1e-9)
    assert a["L3"]["hot_nominal_mm"] == pytest.approx(80.044, abs=1e-9)
    # 名义封闭环 = +50.0575 + 30.31818 - 80.044 = 0.33168
    assert res["nominal_gap_mm"] == pytest.approx(0.33168, abs=1e-6)


def test_manufacturing_band_scales():
    """热态制造公差半宽 = 基线半宽 × (1+αΔT)，WC 制造分量随之缩放。"""
    model = _model()
    res = th.evaluate_condition(model, 0, mc_samples=5000)["analytic"]
    a = {p["dimension_id"]: p for p in res["per_dimension"]}
    # cold: ΔT=-40；L1 scale=1-23e-6*40=0.99908
    assert a["L1"]["scale_1_plus_alpha_dt"] == pytest.approx(0.99908, abs=1e-9)
    assert a["L1"]["hot_half_width_mm"] == pytest.approx(0.04 * 0.99908)
    wc_mfg = res["worst_case"]["components_half_width_mm"]["manufacturing"]
    expect = (0.04 * 0.99908 + 0.05 * (1 - 12e-6 * 40)
              + 0.075 * (1 - 11e-6 * 40))
    assert wc_mfg == pytest.approx(expect, rel=1e-9)


def test_rss_source_decomposition_adds_up():
    """σ²_total = σ²_mfg + σ²_α + σ²_T。"""
    model = _model()
    res = th.evaluate_condition(model, 0, mc_samples=5000)["analytic"]["rss"]
    v = res["variance_components_mm2"]
    assert res["sigma_mm"] ** 2 == pytest.approx(
        v["manufacturing"] + v["expansion_coefficient"]
        + v["temperature"], rel=1e-9)
    shares = res["variance_share"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    # 冷工况 u(T)=2°C，温度来源应为正贡献
    assert v["temperature"] > 0 and v["expansion_coefficient"] > 0


def test_temperature_range_sigma_is_half_over_sqrt3():
    """区间 [60,80]：h=10，u(T)=10/√3；固定 u(T)=0 时温度方差仅来自区间。"""
    model = _model()
    res = th.evaluate_condition(model, 1, mc_samples=5000)["analytic"]["rss"]
    v_t = res["variance_components_mm2"]["temperature"]
    # Σ s_i² (L0 α)² (h/√3)²，h=10
    nc = model.nc
    expect = sum(
        (d.sign * (d.nominal + d.mid) * model.alpha[i] * 10.0
         / math.sqrt(3.0)) ** 2
        for i, d in enumerate(nc.dimensions))
    assert v_t == pytest.approx(expect, rel=1e-9)


def test_monte_carlo_matches_analytic_sigma():
    """固定种子 MC σ 收敛到解析 RSS σ（非线性项为二阶小量）。"""
    model = _model()
    for ci in range(2):
        r = th.evaluate_condition(model, ci, mc_samples=120000)
        sig_a = r["analytic"]["rss"]["sigma_mm"]
        sig_mc = r["monte_carlo"]["combined"]["sigma_mm"]
        assert sig_mc == pytest.approx(sig_a, rel=2e-2)
        shares = r["monte_carlo"]["variance_share"]
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)


def test_monte_carlo_determinism_same_seed():
    model = _model()
    r1 = th.evaluate_condition(model, 0, mc_samples=10000)["monte_carlo"]
    r2 = th.evaluate_condition(model, 0, mc_samples=10000)["monte_carlo"]
    assert r1["combined"] == r2["combined"]
    assert r1["components"] == r2["components"]


def test_different_seed_changes_samples():
    m1 = _model(random_seed=1)
    m2 = _model(random_seed=2)
    r1 = th.evaluate_condition(m1, 0, mc_samples=10000)["monte_carlo"]
    r2 = th.evaluate_condition(m2, 0, mc_samples=10000)["monte_carlo"]
    assert r1["combined"]["min_sample_mm"] != r2["combined"]["min_sample_mm"]


def test_celsius_kelvin_delta_equivalent():
    """温差在 °C 与 K 下数值相同：T0=293.15K、T=253.15K 等价于 20→-20°C。"""
    dims_k = [{**d, "reference_temperature": 293.15,
               "reference_temperature_unit": "K"} for d in DIMS]
    cond_k = [{"name": "cold", "temperature_unit": "K",
               "global_temperature": {"fixed": {"mean": 253.15,
                                                "std_uncertainty": 2}}}]
    m_c = _model()
    m_k = _model(dimensions=dims_k, conditions=cond_k)
    r_c = th.evaluate_condition(m_c, 0, mc_samples=20000)["analytic"]
    r_k = th.evaluate_condition(m_k, 0, mc_samples=20000)["analytic"]
    assert r_k["mean_gap_mm"] == pytest.approx(r_c["mean_gap_mm"], abs=1e-9)
    assert r_k["rss"]["sigma_mm"] == pytest.approx(
        r_c["rss"]["sigma_mm"], rel=1e-9)


def test_ppm_alpha_unit():
    dims = [{**DIMS[0], "alpha": 23, "alpha_unit": "ppm/K",
             "alpha_std_uncertainty": 1,
             "_": None}]
    dims[0].pop("_")
    dims = [dims[0],
            {**DIMS[1], "alpha": 12, "alpha_unit": "ppm/K",
             "alpha_std_uncertainty": 0.5},
            {**DIMS[2], "alpha": 11, "alpha_unit": "ppm/K",
             "alpha_std_uncertainty": 0.5}]
    m_ppm = _model(dimensions=dims)
    m_perk = _model()
    r1 = th.evaluate_condition(m_ppm, 1, mc_samples=5000)["analytic"]
    r2 = th.evaluate_condition(m_perk, 1, mc_samples=5000)["analytic"]
    assert r1["mean_gap_mm"] == pytest.approx(r2["mean_gap_mm"], abs=1e-12)


def test_per_dimension_temperature_overrides_global(client, chain_id):
    conds = [{
        "name": "mixed",
        "global_temperature": {"fixed": {"mean": 40, "std_uncertainty": 1}},
        "dimension_temperatures": {
            "L3": {"range": {"lower": 50, "upper": 70}}},
    }]
    r = client.post(
        f"/chains/{chain_id}/thermal-analyses",
        json=thermal_payload(name="mix", conditions=conds, mc_samples=4000))
    assert r.status_code == 201, r.text
    cond = r.json()["result"]["conditions"][0]["analytic"]
    by = {p["dimension_id"]: p for p in cond["per_dimension"]}
    assert by["L1"]["temperature_c"]["mean"] == pytest.approx(40.0)
    assert by["L3"]["temperature_c"]["mean"] == pytest.approx(60.0)
    assert by["L3"]["temperature_c"]["half_width"] == pytest.approx(10.0)


def test_alpha_correlation_changes_sigma():
    """同号 s_i 的两尺寸 α 正相关增大 σ_α，负相关减小。"""
    base = _model(mc_samples=4000)
    pos = _model(mc_samples=4000, alpha_correlations=[
        {"dim_a": "L1", "dim_b": "L2", "rho": 0.9}])
    neg = _model(mc_samples=4000, alpha_correlations=[
        {"dim_a": "L1", "dim_b": "L2", "rho": -0.9}])
    vb = th.evaluate_condition(base, 0, mc_samples=2000)[
        "analytic"]["rss"]["variance_components_mm2"]["expansion_coefficient"]
    vp = th.evaluate_condition(pos, 0, mc_samples=2000)[
        "analytic"]["rss"]["variance_components_mm2"]["expansion_coefficient"]
    vn = th.evaluate_condition(neg, 0, mc_samples=2000)[
        "analytic"]["rss"]["variance_components_mm2"]["expansion_coefficient"]
    assert vp > vb > vn


def test_first_spec_breach_is_hot():
    model = _model()
    out = th.analyze(model, mc_samples=8000)
    breach = out["summary"]["first_spec_breach"]
    assert breach["condition_name"] == "hot"
    assert breach["criterion"] == "worst_case"
    assert "upper" in breach["breach_side"]
    assert out["summary"]["worst_reject_probability"]["monte_carlo"] > 0


# --------------------------------------------------------------- 冻结复现

def test_frozen_version_repeat_get_identical(client, analysis_id):
    first = client.get(f"/thermal-analyses/{analysis_id}").json()
    second = client.get(f"/thermal-analyses/{analysis_id}").json()
    assert first["result"] == second["result"]
    assert first["frozen"] is True


def test_baseline_chain_not_modified(client, chain_id, analysis_id):
    """热分析版本不覆盖基线：基线结果中不含热字段。"""
    chain = client.get(f"/chains/{chain_id}").json()
    assert "thermal" not in chain["result"]
    assert chain["result"]["results"]["rss"]["sigma_mm"] > 0


def test_listing_and_traceability(client, chain_id, analysis_id):
    r = client.get(f"/chains/{chain_id}/thermal-analyses")
    assert r.status_code == 200
    items = r.json()["thermal_analyses"]
    assert any(t["id"] == analysis_id for t in items)
    detail = client.get(f"/thermal-analyses/{analysis_id}").json()
    assert detail["model"]["formula"] == "L(T) = L0 [1 + alpha (T - T0)]"
    tr = detail["result"]["traceability"]
    assert tr["monte_carlo"]["seed"] == 42


# --------------------------------------------------------------- 方案搜索

PROP = {
    "name": "props-1",
    "candidates": [
        {"material_id": "inv", "name": "Invar 36", "applies_to": ["L1"],
         "alpha": 1.2e-6, "alpha_std_uncertainty": 0.3e-6, "cost": 50},
        {"material_id": "cert-steel", "name": "certified",
         "applies_to": ["L3"], "alpha": 11e-6,
         "alpha_std_uncertainty": 0.1e-6, "cost": 8},
    ],
    "shim_candidates": [
        {"shim_id": "s05", "name": "0.05 shim", "thickness": 0.05,
         "std_dev": 0.005, "closure_sign": 1, "cost": 2}],
    "assembly_reference_temperatures": [{"temperature": 25, "cost": 1}],
    "mc_samples": 8000,
}


def test_proposal_full_flow(client, analysis_id):
    r = client.post(f"/thermal-analyses/{analysis_id}/proposals", json=PROP)
    assert r.status_code == 201, r.text
    out = r.json()["result"]
    proposals = out["proposals"]
    assert proposals, "至少应返回零成本现状方案"
    # 排序：MC 超差率非降
    rates = [p["metrics"]["worst_reject_probability_mc"] for p in proposals]
    assert rates == sorted(rates)
    # 每个方案含材料/垫片/基准温度与成本拆分
    for p in proposals:
        assert set(p["cost"]) == {"total", "materials", "shim",
                                  "assembly_reference_temperature"}
        assert len(p["conditions"]) == 2
    # 零成本现状方案必须在候选集中
    assert any(
        not p["materials"] and p["shim"] is None
        and p["assembly_reference_temperature_c"] is None
        and p["cost"]["total"] == 0.0
        for p in proposals)
    # 最优方案成本拆分自洽
    top = proposals[0]
    assert top["cost"]["total"] == pytest.approx(
        top["cost"]["materials"] + top["cost"]["shim"]
        + top["cost"]["assembly_reference_temperature"])


def test_proposal_shim_moves_gap(client, analysis_id):
    """加 +0.05 垫片应把各工况均值整体抬高 0.05。"""
    req = {"name": "shim-only",
           "candidates": [
               {"material_id": "m", "applies_to": ["L1"], "alpha": 23e-6,
                "alpha_std_uncertainty": 1e-6, "cost": 100}],
           "shim_candidates": [
               {"shim_id": "s05", "thickness": 0.05, "std_dev": 0.0,
                "closure_sign": 1, "cost": 2}],
           "mc_samples": 4000}
    out = client.post(
        f"/thermal-analyses/{analysis_id}/proposals", json=req).json()
    shim_plan = next(p for p in out["result"]["proposals"]
                     if p["shim"] is not None)
    base_plan = next(p for p in out["result"]["proposals"]
                     if not p["materials"] and p["shim"] is None)
    for pc, pb in zip(shim_plan["conditions"], base_plan["conditions"]):
        assert pc["analytic"]["mean_gap_mm"] == pytest.approx(
            pb["analytic"]["mean_gap_mm"] + 0.05, abs=1e-9)


def test_locked_material_restricts_options(client, analysis_id):
    req = {**PROP, "name": "locked",
           "locked_materials": {"L1": "inv"}, "mc_samples": 4000}
    out = client.post(
        f"/thermal-analyses/{analysis_id}/proposals", json=req).json()
    for p in out["result"]["proposals"]:
        mats = {m["dimension_id"]: m["material_id"] for m in p["materials"]}
        # L1 若换材料只能是 inv；cert-steel 作用于 L3 不受影响
        assert mats.get("L1", "inv") == "inv"


def test_locked_material_must_exist(client, analysis_id):
    req = {**PROP, "name": "bad-lock",
           "locked_materials": {"L1": "ghost"}}
    r = client.post(f"/thermal-analyses/{analysis_id}/proposals", json=req)
    assert r.status_code == 422 and "候选表" in _detail_text(r)


def test_locked_material_not_applicable(client, analysis_id):
    """锁定材料的 applies_to 未覆盖该尺寸 => 422。"""
    req = {"name": "lock-na", "locked_materials": {"L2": "inv"},
           "candidates": PROP["candidates"], "mc_samples": 4000}
    r = client.post(f"/thermal-analyses/{analysis_id}/proposals", json=req)
    assert r.status_code == 422 and "不适用" in r.json()["detail"]


def test_proposal_candidate_unknown_dimension(client, analysis_id):
    req = {"name": "unk",
           "candidates": [{"material_id": "m", "applies_to": ["ZZ"],
                           "alpha": 1e-6, "alpha_std_uncertainty": 0.0,
                           "cost": 1}],
           "mc_samples": 2000}
    r = client.post(f"/thermal-analyses/{analysis_id}/proposals", json=req)
    assert r.status_code == 422 and "外尺寸" in r.json()["detail"]


def test_proposal_reproducible_and_frozen(client, analysis_id):
    r1 = client.post(f"/thermal-analyses/{analysis_id}/proposals",
                     json={**PROP, "name": "rep1"})
    pid = r1.json()["proposal_id"]
    stored = client.get(f"/thermal-proposals/{pid}").json()["result"]
    assert stored == r1.json()["result"]
    # 同输入再建一次，方案排序与指标相同（固定种子）
    r2 = client.post(f"/thermal-analyses/{analysis_id}/proposals",
                     json={**PROP, "name": "rep2"})
    p1 = r1.json()["result"]["proposals"]
    p2 = r2.json()["result"]["proposals"]
    assert [(p["metrics"]["worst_reject_probability_mc"],
             p["metrics"]["minimum_spec_margin_mm"], p["cost"]["total"])
            for p in p1] == [
        (p["metrics"]["worst_reject_probability_mc"],
         p["metrics"]["minimum_spec_margin_mm"], p["cost"]["total"])
        for p in p2]


def test_proposal_listing_and_404(client, analysis_id):
    client.post(f"/thermal-analyses/{analysis_id}/proposals", json=PROP)
    r = client.get(f"/thermal-analyses/{analysis_id}/proposals")
    assert r.status_code == 200 and len(r.json()["proposals"]) == 1
    assert client.get("/thermal-proposals/9999").status_code == 404
    assert client.post("/thermal-analyses/9999/proposals",
                       json=PROP).status_code == 404


def test_material_cost_charged_once_per_material(client, analysis_id):
    """同一材料应用到多个尺寸时改动成本只计一次。"""
    req = {"name": "once",
           "candidates": [
               {"material_id": "inv", "name": "Invar",
                "applies_to": ["L1", "L2", "L3"], "alpha": 1.2e-6,
                "alpha_std_uncertainty": 0.3e-6, "cost": 50}],
           "mc_samples": 3000}
    out = client.post(
        f"/thermal-analyses/{analysis_id}/proposals", json=req).json()
    plan = next(p for p in out["result"]["proposals"] if p["materials"])
    assert plan["cost"]["materials"] == pytest.approx(50.0)
    assert plan["cost"]["total"] == pytest.approx(50.0)
    assert len(plan["materials"]) == 3  # 三个尺寸都换料，但材料成本只算一次


def test_assembly_reference_temperature_rebases(client, analysis_id):
    """T_a=60°C 时热工况 ΔT 以 60 为零点：hot 区间中点偏移仅 10°C。"""
    req = {"name": "aref",
           "candidates": [
               {"material_id": "m", "applies_to": ["L1"], "alpha": 23e-6,
                "alpha_std_uncertainty": 1e-6, "cost": 100}],
           "assembly_reference_temperatures": [
               {"temperature": 60, "cost": 3}],
           "mc_samples": 4000}
    out = client.post(
        f"/thermal-analyses/{analysis_id}/proposals", json=req).json()
    plan = next(p for p in out["result"]["proposals"]
                if p["assembly_reference_temperature_c"] == 60.0
                and not p["materials"])
    hot = next(c for c in plan["conditions"] if c["analytic"]["name"] == "hot")
    by = {p["dimension_id"]: p for p in hot["analytic"]["per_dimension"]}
    assert by["L1"]["delta_t_c"] == pytest.approx(10.0)  # 70 - 60


# --------------------------------------------------------------- 退化情形

def test_zero_uncertainties_deterministic_sources():
    """u(α)=0 且温度精确（固定 uT=0）时，α/温度方差为 0，仅剩制造偏差。"""
    dims = [{**d, "alpha_std_uncertainty": 0.0} for d in DIMS]
    conds = [{"name": "iso",
              "global_temperature": {"fixed": {"mean": 40,
                                               "std_uncertainty": 0}}}]
    model = _model(dimensions=dims, conditions=conds)
    rss = th.evaluate_condition(model, 0, mc_samples=8000)[
        "analytic"]["rss"]
    v = rss["variance_components_mm2"]
    assert v["expansion_coefficient"] == 0.0
    assert v["temperature"] == 0.0
    assert v["manufacturing"] > 0
    assert rss["variance_share"]["manufacturing"] == pytest.approx(1.0)


def test_spec_margin_negative_when_band_crosses_limit():
    """hot 工况 WC 上界越过 USL=0.60 => 规格余量为负。"""
    model = _model()
    res = th.analyze(model, mc_samples=4000)
    hot = res["conditions"][1]["analytic"]
    assert hot["worst_case"]["upper_bound_mm"] > 0.60
    assert hot["spec_margin_mm"] < 0


# ------------------------------------------------- 统计逻辑回归（三项修正）

def test_manufacturing_fluctuation_not_double_scaled():
    """热伸缩比例 scale=2、仅有制造波动时：组合 σ 必须等于制造分量 σ。

    旧实现先把制造偏差乘一次 scale 再乘 [1+αΔT]，组合 σ 变成制造分量的
    两倍；修正后热态缩放只发生一次。
    """
    chain = {
        "name": "scale2",
        "closure_lower_limit": -1.0,
        "closure_upper_limit": 1.0,
        "dimensions": [
            {"id": "L1", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.02, "lower_deviation": -0.02,
             "std_dev": 0.01, "distribution": "normal"},
            {"id": "L2", "start": "b", "end": "a", "nominal": 10,
             "upper_deviation": 0.0, "lower_deviation": 0.0,
             "std_dev": 0.0, "distribution": "normal"}],
        "mc_samples": 100000, "random_seed": 1}
    nc = rebuild_normalized(chain)
    # α=1e-3, ΔT=1000°C => 1+αΔT=2；u(α)=0 且温度精确 => 仅制造波动
    payload = ThermalAnalysisCreate.model_validate({
        "name": "scale2-th",
        "dimensions": [
            {"dimension_id": "L1", "reference_temperature": 20,
             "alpha": 1e-3, "alpha_std_uncertainty": 0.0},
            {"dimension_id": "L2", "reference_temperature": 20,
             "alpha": 1e-3, "alpha_std_uncertainty": 0.0}],
        "conditions": [{"name": "double",
                        "global_temperature": {
                            "fixed": {"mean": 1020, "std_uncertainty": 0}}}],
        "mc_samples": 100000, "random_seed": 1})
    model = th.build_model(nc, payload)
    r = th.evaluate_condition(model, 0)
    mc = r["monte_carlo"]
    a = r["analytic"]["per_dimension"][0]
    assert a["scale_1_plus_alpha_dt"] == pytest.approx(2.0)
    sig_comb = mc["combined"]["sigma_mm"]
    sig_mfg = mc["components"]["manufacturing"]["sigma_mm"]
    # 制造分量理论 σ = 0.01 × 2 = 0.02
    assert sig_mfg == pytest.approx(0.02, rel=2e-2)
    # 组合 σ 不得变成制造分量的两倍（旧 bug），二者应一致
    assert sig_comb == pytest.approx(sig_mfg, rel=2e-2)
    assert r["analytic"]["rss"]["variance_components_mm2"][
        "expansion_coefficient"] == 0.0
    assert r["analytic"]["rss"]["variance_components_mm2"][
        "temperature"] == 0.0


def test_safe_condition_has_no_breach_marker():
    """所有 WC/RSS 边界均在规格内时，first_spec_breach 必须为 None。"""
    chain = dict(CHAIN, name="safe-chain",
                 closure_lower_limit=-2.0, closure_upper_limit=2.0)
    nc = rebuild_normalized(chain)
    payload = ThermalAnalysisCreate.model_validate({
        "name": "safe-th",
        "dimensions": DIMS,
        "conditions": [{"name": "mild",
                        "global_temperature": {
                            "fixed": {"mean": 40, "std_uncertainty": 1}}}],
        "mc_samples": 40000, "random_seed": 2})
    model = th.build_model(nc, payload)
    out = th.analyze(model, mc_samples=40000)
    a = out["conditions"][0]["analytic"]
    assert a["worst_case"]["lower_bound_mm"] >= -2.0
    assert a["worst_case"]["upper_bound_mm"] <= 2.0
    assert a["spec_margin_mm"] > 0
    assert out["summary"]["first_spec_breach"] is None


def test_component_reject_uses_hot_nominal_reference():
    """分量超差率须在「热态名义间隙 + 单项偏差」上对照规格，而非零均值偏差。"""
    chain = dict(CHAIN, name="ref-chain",
                 closure_lower_limit=-2.0, closure_upper_limit=2.0)
    nc = rebuild_normalized(chain)
    payload = ThermalAnalysisCreate.model_validate({
        "name": "ref-th",
        "dimensions": DIMS,
        "conditions": [{"name": "mild",
                        "global_temperature": {
                            "fixed": {"mean": 40, "std_uncertainty": 1}}}],
        "mc_samples": 40000, "random_seed": 3})
    model = th.build_model(nc, payload)
    r = th.evaluate_condition(model, 0, mc_samples=40000)
    hot_mean = r["analytic"]["mean_gap_mm"]
    mc = r["monte_carlo"]
    for name in ("manufacturing", "expansion_coefficient", "temperature"):
        comp = mc["components"][name]
        # 超差判定基准 = 热态名义间隙（而非 0）
        assert comp["reject_reference_gap_mm"] == pytest.approx(
            hot_mean, abs=1e-9)
        # 名义远在规格中部（±2），单项偏差很小 => 分量超差率应为 0，
        # 旧实现拿零均值偏差对照 ±2 时不会误判，但对照 0.05/0.60 规格会
        # ≈1.0；这里用真实规格再次核对
    chain2 = dict(CHAIN, name="ref-chain2")
    nc2 = rebuild_normalized(chain2)
    model2 = th.build_model(nc2, payload)
    r2 = th.evaluate_condition(model2, 0, mc_samples=40000)
    mc2 = r2["monte_carlo"]
    hot2 = r2["analytic"]["mean_gap_mm"]
    assert 0.05 < hot2 < 0.60  # 热态名义在规格内
    mfg = mc2["components"]["manufacturing"]
    assert mfg["reject_reference_gap_mm"] == pytest.approx(hot2, abs=1e-9)
    # 旧实现（基准 0）下该值 ≈1.0；修正后是真实的小概率尾部
    assert mfg["reject_probability"] < 0.5
