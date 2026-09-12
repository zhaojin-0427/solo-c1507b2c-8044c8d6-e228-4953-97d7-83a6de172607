"""量具 R&R 研究：ANOVA 手算核对、交叉表校验拒收、负分量截零、
bootstrap 种子复现、研究冻结、测量方案引用研究取代重复性分量。"""
import math

import pytest

SQRT3 = math.sqrt(3.0)

# 手算核对数据集（mm）：3 零件 × 2 操作者 × 2 轮，全部平衡
# 单元均值：P1A=10.01 P1B=10.02 / P2A=10.11 P2B=10.12 / P3A=9.91 P3B=9.92
# SS_part=0.08, SS_op=0.0003, SS_inter=0, SS_err=0.0012
# MS: 0.04 / 0.0003 / 0 / 0.0002
# 原始分量: rep=0.0002, inter=-0.0001(截零), op=0.00005, part=0.01
# σ_GRR=√0.00025, σ_total=√0.01025, ndc=⌊1.41·0.1/σ_GRR⌋=8
DATA = {
    "P1": {"A": [10.00, 10.02], "B": [10.01, 10.03]},
    "P2": {"A": [10.10, 10.12], "B": [10.11, 10.13]},
    "P3": {"A": [9.90, 9.92], "B": [9.91, 9.93]},
}
SIGMA_GRR = math.sqrt(0.00025)
SIGMA_TOTAL = math.sqrt(0.01025)

# 第二套数据（5 零件，单元内散布不均匀）：bootstrap 非退化用
DATA5 = {
    "P1": {"A": [10.00, 10.03], "B": [10.02, 10.01]},
    "P2": {"A": [10.10, 10.14], "B": [10.13, 10.10]},
    "P3": {"A": [9.90, 9.94], "B": [9.89, 9.93]},
    "P4": {"A": [10.20, 10.22], "B": [10.19, 10.24]},
    "P5": {"A": [9.80, 9.85], "B": [9.83, 9.79]},
}


def _measurements(data):
    return [
        {"part": p, "operator": o, "replicate": i + 1, "value": v}
        for p, by_op in data.items()
        for o, vals in by_op.items()
        for i, v in enumerate(vals)
    ]


def _study_payload(**over):
    body = {
        "name": "grr-L1",
        "note": "L1 卡尺 R&R",
        "dimension_id": "L1",
        "unit": "mm",
        "process_tolerance": 0.1,
        "measurements": _measurements(DATA),
        "bootstrap_samples": 2000,
        "random_seed": 11,
    }
    body.update(over)
    return body


def _make_chain(client, payload):
    r = client.post("/chains", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["chain_id"]


def _make_study(client, cid, **over):
    r = client.post(f"/chains/{cid}/gage-rr-studies",
                    json=_study_payload(**over))
    assert r.status_code == 201, r.text
    return r.json()


# ------------------------------------------------------------- ANOVA 手算

def test_study_anova_hand_computed(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    j = _make_study(client, cid)
    assert j["frozen"] is True and j["study_id"] == 1
    assert j["chain_id"] == cid and j["dimension_id"] == "L1"
    res = j["result"]

    # 设计结构
    d = res["design"]
    assert (d["n_parts"], d["n_operators"], d["n_replicates"]) == (3, 2, 2)
    assert d["cells"] == 6 and d["total_measurements"] == 12
    assert d["balanced"] is True
    assert res["grand_mean_mm"] == pytest.approx(10.015)
    assert res["process_tolerance"]["mm"] == pytest.approx(0.1)

    # ANOVA 表（手算值）
    anova = res["anova"]
    assert anova["degrees_of_freedom"] == {
        "part": 2, "operator": 1, "interaction": 2,
        "repeatability": 6, "total": 11}
    ss = anova["sum_of_squares_mm2"]
    assert ss["part"] == pytest.approx(0.08)
    assert ss["operator"] == pytest.approx(0.0003)
    assert ss["interaction"] == pytest.approx(0.0, abs=1e-12)
    assert ss["repeatability"] == pytest.approx(0.0012)
    assert ss["total"] == pytest.approx(0.0815)
    ms = anova["mean_squares_mm2"]
    assert ms["part"] == pytest.approx(0.04)
    assert ms["operator"] == pytest.approx(0.0003)
    assert ms["interaction"] == pytest.approx(0.0, abs=1e-12)
    assert ms["repeatability"] == pytest.approx(0.0002)

    # 方差分量：交互原始估计为负 → 截零，原估计保留
    vc = res["variance_components_mm2"]
    assert vc["repeatability"]["raw_estimate"] == pytest.approx(0.0002)
    assert vc["repeatability"]["truncated_to_zero"] is False
    assert vc["interaction"]["raw_estimate"] == pytest.approx(-0.0001)
    assert vc["interaction"]["used"] == 0.0
    assert vc["interaction"]["truncated_to_zero"] is True
    assert vc["operator"]["raw_estimate"] == pytest.approx(0.00005)
    assert vc["part_to_part"]["raw_estimate"] == pytest.approx(0.01)
    assert vc["reproducibility"]["used"] == pytest.approx(0.00005)
    assert vc["total_gage_rr"]["used"] == pytest.approx(0.00025)
    assert vc["total"]["used"] == pytest.approx(0.01025)

    # 指标：总量具 R&R / 方差占比 / %SV / %Tolerance / ndc
    comp = res["components"]
    assert res["total_gage_std_mm"] == pytest.approx(SIGMA_GRR)
    assert comp["total_gage_rr"]["std_mm"] == pytest.approx(SIGMA_GRR)
    assert comp["part_to_part"]["std_mm"] == pytest.approx(0.1)
    assert comp["total"]["std_mm"] == pytest.approx(SIGMA_TOTAL)
    base_sum = sum(comp[k]["variance_contribution_pct"] for k in
                   ("repeatability", "operator", "interaction",
                    "part_to_part"))
    assert base_sum == pytest.approx(100.0)
    assert comp["total_gage_rr"]["variance_contribution_pct"] == \
        pytest.approx(100 * 0.00025 / 0.01025)
    assert comp["part_to_part"]["variance_contribution_pct"] == \
        pytest.approx(100 * 0.01 / 0.01025)
    assert comp["total_gage_rr"]["study_variation_pct"] == pytest.approx(
        100 * SIGMA_GRR / SIGMA_TOTAL)
    assert comp["total_gage_rr"]["tolerance_pct"] == pytest.approx(
        100 * 5.15 * SIGMA_GRR / 0.1)
    assert comp["repeatability"]["tolerance_pct"] == pytest.approx(
        100 * 5.15 * math.sqrt(0.0002) / 0.1)
    assert res["ndc"] == 8
    assert res["ndc_note"] is None

    # 主要变差来源与验收提示
    dom = res["dominant_source"]
    assert dom["component"] == "part_to_part"
    assert dom["variance_share_of_total_pct"] == pytest.approx(
        100 * 0.01 / 0.01025)
    assert "零件间" in dom["label"]
    assert res["acceptance_note"]
    assert res["formulas"]
    assert set(res["dependency_versions"]) >= {
        "python", "numpy", "fastapi", "pydantic", "sqlalchemy"}


def test_study_unit_conversion(client, simple_chain_payload):
    """实测值与过程公差以 um 提交：内部换算 mm，结果与 mm 提交一致。"""
    cid = _make_chain(client, simple_chain_payload)
    um_data = {p: {o: [v * 1000 for v in vals]
                   for o, vals in by_op.items()}
               for p, by_op in DATA.items()}
    j = _make_study(client, cid, unit="um", process_tolerance=100.0,
                    measurements=_measurements(um_data))
    res = j["result"]
    assert res["unit_submitted"] == "um"
    assert res["process_tolerance"]["mm"] == pytest.approx(0.1)
    assert res["total_gage_std_mm"] == pytest.approx(SIGMA_GRR)
    assert res["components"]["total_gage_rr"]["tolerance_pct"] == \
        pytest.approx(100 * 5.15 * SIGMA_GRR / 0.1)
    assert res["ndc"] == 8


def test_study_multiplier_configurable(client, simple_chain_payload):
    """研究变异倍数 k 可配：%Tolerance 随之缩放，%SV 不变。"""
    cid = _make_chain(client, simple_chain_payload)
    j = _make_study(client, cid, study_variation_multiplier=6.0)
    comp = j["result"]["components"]
    assert comp["total_gage_rr"]["tolerance_pct"] == pytest.approx(
        100 * 6.0 * SIGMA_GRR / 0.1)
    assert comp["total_gage_rr"]["study_variation_pct"] == pytest.approx(
        100 * SIGMA_GRR / SIGMA_TOTAL)


# ------------------------------------------------------------- 校验拒收

def test_study_reject_unknown_dimension(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    r = client.post(f"/chains/{cid}/gage-rr-studies",
                    json=_study_payload(dimension_id="L9"))
    assert r.status_code == 422
    assert "未知尺寸" in r.text and "L9" in r.text
    # 未落库
    lst = client.get(f"/chains/{cid}/gage-rr-studies").json()["studies"]
    assert lst == []


def test_study_reject_duplicate_cell(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    ms = _measurements(DATA)
    ms.append({"part": "P1", "operator": "A", "replicate": 1,
               "value": 10.01})
    r = client.post(f"/chains/{cid}/gage-rr-studies",
                    json=_study_payload(measurements=ms))
    assert r.status_code == 422
    assert "重复单元" in r.text and "P1×A×第1轮" in r.text


def test_study_reject_incomplete_cross_table(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    # 删除 P2×B 整个单元 → 缺口应被指出
    ms = [m for m in _measurements(DATA)
          if not (m["part"] == "P2" and m["operator"] == "B")]
    r = client.post(f"/chains/{cid}/gage-rr-studies",
                    json=_study_payload(measurements=ms))
    assert r.status_code == 422
    assert "交叉表不完整" in r.text and "P2×B" in r.text


def test_study_reject_unbalanced_replicates(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    ms = _measurements(DATA)
    ms.append({"part": "P1", "operator": "A", "replicate": 3,
               "value": 10.01})
    r = client.post(f"/chains/{cid}/gage-rr-studies",
                    json=_study_payload(measurements=ms))
    assert r.status_code == 422
    assert "等次数" in r.text and "P1×A=3次" in r.text


def test_study_reject_insufficient_counts(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    # 仅 1 个零件
    r = client.post(f"/chains/{cid}/gage-rr-studies", json=_study_payload(
        measurements=_measurements({"P1": DATA["P1"]})))
    assert r.status_code == 422 and "零件数不足" in r.text
    # 仅 1 名操作者
    r = client.post(f"/chains/{cid}/gage-rr-studies", json=_study_payload(
        measurements=_measurements(
            {p: {"A": by_op["A"]} for p, by_op in DATA.items()})))
    assert r.status_code == 422 and "操作者数不足" in r.text
    # 每单元仅 1 轮重复
    one_rep = {p: {o: vals[:1] for o, vals in by_op.items()}
               for p, by_op in DATA.items()}
    r = client.post(f"/chains/{cid}/gage-rr-studies", json=_study_payload(
        measurements=_measurements(one_rep)))
    assert r.status_code == 422 and "至少需要 2 轮重复" in r.text


def test_study_reject_bad_inputs(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    # 过程公差非正
    for bad_tol in (0.0, -0.1):
        r = client.post(f"/chains/{cid}/gage-rr-studies",
                        json=_study_payload(process_tolerance=bad_tol))
        assert r.status_code == 422
    # 实测值非有限（NaN）
    body = (
        '{"name":"nan","dimension_id":"L1","process_tolerance":0.1,'
        '"measurements":['
        '{"part":"P1","operator":"A","replicate":1,"value":NaN},'
        '{"part":"P1","operator":"A","replicate":2,"value":10.0},'
        '{"part":"P1","operator":"B","replicate":1,"value":10.0},'
        '{"part":"P1","operator":"B","replicate":2,"value":10.0},'
        '{"part":"P2","operator":"A","replicate":1,"value":10.1},'
        '{"part":"P2","operator":"A","replicate":2,"value":10.1},'
        '{"part":"P2","operator":"B","replicate":1,"value":10.1},'
        '{"part":"P2","operator":"B","replicate":2,"value":10.1}]}'
    )
    r = client.post(f"/chains/{cid}/gage-rr-studies", content=body,
                    headers={"content-type": "application/json"})
    assert r.status_code == 422 and "有限数" in r.text
    # 链不存在
    r = client.post("/chains/9999/gage-rr-studies", json=_study_payload())
    assert r.status_code == 404


# ------------------------------------------------------------- 冻结与复现

def test_study_frozen_and_listed(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    j = _make_study(client, cid)
    sid = j["study_id"]
    g1 = client.get(f"/gage-rr-studies/{sid}").json()
    g2 = client.get(f"/gage-rr-studies/{sid}").json()
    assert g1 == g2
    assert g1["frozen"] is True
    # 提交快照原样保留（含交叉表与单位）
    assert g1["submitted_input"]["measurements"] == _measurements(DATA)
    assert g1["submitted_input"]["process_tolerance"] == 0.1
    # 无更新 / 删除端点
    assert client.put(f"/gage-rr-studies/{sid}", json={}).status_code == 405
    assert client.delete(f"/gage-rr-studies/{sid}").status_code == 405
    assert client.get("/gage-rr-studies/9999").status_code == 404
    # 列表
    lst = client.get(f"/chains/{cid}/gage-rr-studies").json()["studies"]
    assert len(lst) == 1
    assert lst[0]["dimension_id"] == "L1"
    assert lst[0]["total_gage_std_mm"] == pytest.approx(SIGMA_GRR)
    assert lst[0]["ndc"] == 8
    assert lst[0]["dominant_source"] == "part_to_part"
    assert lst[0]["frozen"] is True


def test_study_bootstrap_reproducible(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    a = _make_study(client, cid, name="boot-a",
                    measurements=_measurements(DATA5), random_seed=21)
    b = _make_study(client, cid, name="boot-b",
                    measurements=_measurements(DATA5), random_seed=21)
    # 同种子：结果逐字节一致
    assert a["result"] == b["result"]
    boot = a["result"]["bootstrap"]
    assert boot["samples"] == 2000 and boot["random_seed"] == 21
    ci = boot["ci95"]
    assert set(ci) == {"total_gage_std_mm", "study_variation_pct",
                       "tolerance_pct", "ndc"}
    for key in ("total_gage_std_mm", "study_variation_pct",
                "tolerance_pct"):
        lo, hi = ci[key]
        assert 0.0 <= lo <= hi
    # %Tolerance CI 与 σ_GRR CI 为同一单调变换（同一批重采样）
    k = a["result"]["study_variation_multiplier"]
    tol = a["result"]["process_tolerance"]["mm"]
    assert ci["tolerance_pct"][0] == pytest.approx(
        100 * k * ci["total_gage_std_mm"][0] / tol)
    assert ci["tolerance_pct"][1] == pytest.approx(
        100 * k * ci["total_gage_std_mm"][1] / tol)
    # 主要变差来源频次为概率分布
    freq = boot["dominant_source_frequency"]
    assert set(freq) == {"repeatability", "operator", "interaction",
                         "part_to_part"}
    assert sum(freq.values()) == pytest.approx(1.0)
    # 换种子：ANOVA 点估计不变，bootstrap 种子记录不同
    c = _make_study(client, cid, name="boot-c",
                    measurements=_measurements(DATA5), random_seed=22)
    assert c["result"]["anova"] == a["result"]["anova"]
    assert c["result"]["components"] == a["result"]["components"]
    assert c["result"]["bootstrap"]["random_seed"] == 22


def test_study_degenerate_zero_variance(client, simple_chain_payload):
    """全部实测值相同：总变差为零，占比/%SV 为 null，ndc 无界。"""
    cid = _make_chain(client, simple_chain_payload)
    flat = {p: {o: [10.0, 10.0] for o in ("A", "B")} for p in ("P1", "P2")}
    j = _make_study(client, cid, measurements=_measurements(flat))
    res = j["result"]
    assert res["total_gage_std_mm"] == 0.0
    assert res["ndc"] is None and res["ndc_note"]
    comp = res["components"]
    assert comp["total_gage_rr"]["variance_contribution_pct"] is None
    assert comp["total_gage_rr"]["study_variation_pct"] is None
    assert comp["total_gage_rr"]["tolerance_pct"] == 0.0
    assert res["dominant_source"]["component"] is None
    boot = res["bootstrap"]
    assert boot["zero_total_variance_fraction"] == 1.0
    assert boot["ndc_unbounded_fraction"] == 1.0
    assert boot["ci95"]["study_variation_pct"] is None
    assert boot["ci95"]["total_gage_std_mm"] == [0.0, 0.0]


# ------------------------------------------------------- 测量方案引用研究

PLAN_GAUGES = [
    {"dimension_id": "L1", "resolution": 0.01,
     "calibration_expanded_uncertainty": 0.004, "coverage_factor": 2.0,
     "bias_correction": 0.002, "bias_std_uncertainty": 0.001,
     "repeatability_std": 0.003},
    {"dimension_id": "L2", "resolution": 0.005,
     "calibration_expanded_uncertainty": 0.003, "coverage_factor": 2.0},
    {"dimension_id": "L3", "calibration_expanded_uncertainty": 0.006,
     "coverage_factor": 2.0, "repeatability_std": 0.004},
]


def _make_plan(client, cid, plan):
    r = client.post(f"/chains/{cid}/measurement-plans", json=plan)
    assert r.status_code == 201, r.text
    return r.json()


def test_plan_with_gage_rr_study(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    study = _make_study(client, cid)
    sid = study["study_id"]
    plan = {"name": "plan-with-rr", "gauges": PLAN_GAUGES,
            "gage_rr_studies": {"L1": sid}}
    j = _make_plan(client, cid, plan)
    dims = {d["dimension_id"]: d for d in j["plan"]["dimensions"]}
    l1 = dims["L1"]
    # 重复性分量被总量具标准差取代（手填 0.003 不再生效）
    assert l1["components_mm"]["u_repeatability"] == pytest.approx(SIGMA_GRR)
    src = l1["repeatability_source"]
    assert src["type"] == "gage_rr_study"
    assert src["study_id"] == sid
    assert src["total_gage_std_mm"] == pytest.approx(SIGMA_GRR)
    assert src["replaced_submitted_repeatability_std_mm"] == \
        pytest.approx(0.003)
    # 合成不确定度手算：√(u_res² + u_cal² + u_bias² + σ_GRR²)
    u_res = 0.01 / (2 * SQRT3)
    u_c = math.sqrt(u_res ** 2 + 0.002 ** 2 + 0.001 ** 2 + SIGMA_GRR ** 2)
    assert l1["combined_std_uncertainty_mm"] == pytest.approx(u_c)
    assert l1["u_rest_mm"] == pytest.approx(
        math.sqrt(0.002 ** 2 + 0.001 ** 2 + SIGMA_GRR ** 2))
    # 其余尺寸保持手填口径
    assert dims["L2"]["repeatability_source"]["type"] == "manual"
    assert dims["L3"]["components_mm"]["u_repeatability"] == \
        pytest.approx(0.004)
    # 方案快照记录研究 id
    rr = j["plan"]["gage_rr_studies"]
    assert len(rr) == 1
    assert rr[0]["dimension_id"] == "L1" and rr[0]["study_id"] == sid
    assert rr[0]["total_gage_std_mm"] == pytest.approx(SIGMA_GRR)


def test_plan_gage_rr_reference_errors(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    study = _make_study(client, cid)          # 研究针对 L1
    sid = study["study_id"]
    # 尺寸不匹配：研究针对 L1，映射到 L2
    r = client.post(f"/chains/{cid}/measurement-plans", json={
        "name": "bad", "gauges": PLAN_GAUGES,
        "gage_rr_studies": {"L2": sid}})
    assert r.status_code == 422 and "不能用于尺寸" in r.text
    # 研究不存在
    r = client.post(f"/chains/{cid}/measurement-plans", json={
        "name": "bad", "gauges": PLAN_GAUGES,
        "gage_rr_studies": {"L1": 9999}})
    assert r.status_code == 404
    # 研究属于另一条链
    cid2 = _make_chain(client, {**simple_chain_payload,
                                "name": "other-chain"})
    r = client.post(f"/chains/{cid2}/measurement-plans", json={
        "name": "bad", "gauges": PLAN_GAUGES,
        "gage_rr_studies": {"L1": sid}})
    assert r.status_code == 404
    # 其余参数仍须补齐：L1 量具缺覆盖因子 → 拒绝
    bad_gauges = [dict(g) for g in PLAN_GAUGES]
    del bad_gauges[0]["coverage_factor"]
    r = client.post(f"/chains/{cid}/measurement-plans", json={
        "name": "bad", "gauges": bad_gauges,
        "gage_rr_studies": {"L1": sid}})
    assert r.status_code == 422 and "覆盖因子缺失" in r.text
    # 研究引用链外尺寸
    r = client.post(f"/chains/{cid}/measurement-plans", json={
        "name": "bad", "gauges": PLAN_GAUGES,
        "gage_rr_studies": {"X9": sid}})
    assert r.status_code == 422 and "不能用于尺寸" in r.text
    # 全部拒绝：一个方案都不落库
    plans = client.get(f"/chains/{cid}/measurement-plans").json()["plans"]
    assert plans == []


def test_gage_rr_snapshot_frozen(client, simple_chain_payload):
    """历史研究、派生方案与检验批次均不改写。"""
    cid = _make_chain(client, simple_chain_payload)
    s1 = _make_study(client, cid, name="grr-v1")
    sid1 = s1["study_id"]
    plan = _make_plan(client, cid, {
        "name": "plan-rr-v1", "gauges": PLAN_GAUGES,
        "gage_rr_studies": {"L1": sid1}})
    pid = plan["plan_id"]
    # 批次引用该方案：L1 重复性分量 = 研究 v1 的 σ_GRR
    rows = [{"serial": "W1", "measurements": [
        {"dimension_id": "L1", "value": 50.04, "unit": "mm"},
        {"dimension_id": "L2", "value": 30.30, "unit": "mm"},
        {"dimension_id": "L3", "value": 80.00, "unit": "mm"}]}]
    r = client.post(f"/chains/{cid}/inspection-batches", json={
        "name": "lot-rr", "rows": rows, "measurement_plan_id": pid,
        "measurement_mc_samples": 20000, "measurement_mc_seed": 5})
    assert r.status_code == 201, r.text
    batch = r.json()
    l1_batch = next(d for d in batch["measurement"]["dimensions"]
                    if d["dimension_id"] == "L1")
    assert l1_batch["components_mm"]["u_repeatability"] == \
        pytest.approx(SIGMA_GRR)
    snap_rr = batch["measurement"]["plan_snapshot"]["combined"][
        "gage_rr_studies"]
    assert snap_rr[0]["study_id"] == sid1

    # 新研究 v2：单元内散布更大 → σ_GRR 不同
    data2 = {
        "P1": {"A": [10.00, 10.06], "B": [10.01, 10.07]},
        "P2": {"A": [10.10, 10.16], "B": [10.11, 10.17]},
        "P3": {"A": [9.90, 9.96], "B": [9.91, 9.97]},
    }
    s2 = _make_study(client, cid, name="grr-v2",
                     measurements=_measurements(data2))
    sigma2 = s2["result"]["total_gage_std_mm"]
    assert sigma2 > SIGMA_GRR

    # 历史研究不改写
    g1 = client.get(f"/gage-rr-studies/{sid1}").json()
    assert g1["result"]["total_gage_std_mm"] == pytest.approx(SIGMA_GRR)
    # 派生方案不改写
    p = client.get(f"/measurement-plans/{pid}").json()
    l1_plan = next(d for d in p["plan"]["dimensions"]
                   if d["dimension_id"] == "L1")
    assert l1_plan["components_mm"]["u_repeatability"] == \
        pytest.approx(SIGMA_GRR)
    assert l1_plan["repeatability_source"]["study_id"] == sid1
    # 检验批次不改写
    b = client.get(f"/inspection-batches/{batch['batch_id']}").json()
    assert b["measurement"] == batch["measurement"]

    # 新方案引用 v2：重复性分量更新，两方案并存
    plan2 = _make_plan(client, cid, {
        "name": "plan-rr-v2", "gauges": PLAN_GAUGES,
        "gage_rr_studies": {"L1": s2["study_id"]}})
    l1_v2 = next(d for d in plan2["plan"]["dimensions"]
                 if d["dimension_id"] == "L1")
    assert l1_v2["components_mm"]["u_repeatability"] == \
        pytest.approx(sigma2)
    p_again = client.get(f"/measurement-plans/{pid}").json()
    assert p_again["plan"] == p["plan"]
