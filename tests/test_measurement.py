"""测量方案（量具误差）与批次判定：合成不确定度手算核对、校验拒收、
保护带判定、GUM/蒙特卡洛概率、封闭环相关传播、种子复现、方案快照冻结。"""
import math

import numpy as np
import pytest

SQRT3 = math.sqrt(3.0)

# 与 simple_chain_payload 配套：L1 [50, 50.08]，L2 [30.25, 30.35]，
# L3 [79.85, 80.00]，封闭环 C = L1 + L2 − L3 ∈ [0.05, 0.60]
PLAN = {
    "name": "caliper-set-v1",
    "note": "L1/L3 共用一把数显卡尺（校准相关 0.6）",
    "gauges": [
        {"dimension_id": "L1", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.004, "coverage_factor": 2.0,
         "bias_correction": 0.002, "bias_std_uncertainty": 0.001,
         "repeatability_std": 0.003},
        {"dimension_id": "L2", "resolution": 0.005,
         "calibration_expanded_uncertainty": 0.003, "coverage_factor": 2.0,
         "bias_correction": 0.0, "bias_std_uncertainty": 0.0005,
         "repeatability_std": 0.002},
        {"dimension_id": "L3", "resolution": 0.01,
         "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2.0,
         "bias_correction": -0.001, "bias_std_uncertainty": 0.0015,
         "repeatability_std": 0.004},
    ],
    "correlations": [{"dim_a": "L1", "dim_b": "L3", "rho": 0.6}],
}

# 手算期望值（mm）
U_RES = [0.01 / (2 * SQRT3), 0.005 / (2 * SQRT3), 0.01 / (2 * SQRT3)]
U_CAL = [0.002, 0.0015, 0.003]
U_BIAS = [0.001, 0.0005, 0.0015]
U_REP = [0.003, 0.002, 0.004]
U_REST = [math.sqrt(a * a + b * b + c * c)
          for a, b, c in zip(U_CAL, U_BIAS, U_REP)]
U_C = [math.sqrt(r * r + t * t) for r, t in zip(U_RES, U_REST)]
BIAS = [0.002, 0.0, -0.001]
SIGNS = [1.0, 1.0, -1.0]
RHO13 = 0.6
VAR_RES_C = sum((s * r) ** 2 for s, r in zip(SIGNS, U_RES))
VAR_REST_C = (
    sum((s * t) ** 2 for s, t in zip(SIGNS, U_REST))
    + 2 * SIGNS[0] * SIGNS[2] * RHO13 * U_REST[0] * U_REST[2]
)
U_CLOSURE = math.sqrt(VAR_RES_C + VAR_REST_C)


def _make_chain(client, payload):
    r = client.post("/chains", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["chain_id"]


def _make_plan(client, cid, plan=None):
    r = client.post(f"/chains/{cid}/measurement-plans", json=plan or PLAN)
    assert r.status_code == 201, r.text
    return r.json()


def _rows(values_by_serial, dims=("L1", "L2", "L3")):
    rows = []
    for serial, vals in values_by_serial.items():
        ms = [{"dimension_id": d, "value": vals[d], "unit": "mm"}
              for d in dims if d in vals and vals[d] is not None]
        rows.append({"serial": serial, "measurements": ms})
    return rows


def _gauged_batch(client, cid, rows, plan_id=1, **over):
    body = {"name": over.pop("name", "gauged"), "rows": rows,
            "measurement_plan_id": plan_id,
            "measurement_mc_samples": 50000, "measurement_mc_seed": 5}
    body.update(over)
    r = client.post(f"/chains/{cid}/inspection-batches", json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ------------------------------------------------------------- 方案创建

def test_plan_combined_uncertainty_hand_computed(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    j = _make_plan(client, cid)
    assert j["immutable"] is True and j["plan_id"] == 1
    dims = {d["dimension_id"]: d for d in j["plan"]["dimensions"]}
    for i, dim_id in enumerate(("L1", "L2", "L3")):
        d = dims[dim_id]
        assert d["components_mm"]["u_resolution"] == pytest.approx(U_RES[i])
        assert d["components_mm"]["u_calibration"] == pytest.approx(U_CAL[i])
        assert d["components_mm"]["u_bias"] == pytest.approx(U_BIAS[i])
        assert d["components_mm"]["u_repeatability"] == pytest.approx(U_REP[i])
        assert d["u_rest_mm"] == pytest.approx(U_REST[i])
        assert d["combined_std_uncertainty_mm"] == pytest.approx(U_C[i])
        assert d["bias_correction_mm"] == pytest.approx(BIAS[i])
        assert sum(d["variance_share"].values()) == pytest.approx(1.0)
        # 原始提交值与单位保留
        assert d["submitted"]["resolution"] == PLAN["gauges"][i]["resolution"]
        assert d["submitted"]["unit"] == "mm"
    corr = j["plan"]["correlation"]
    assert corr["dimension_order"] == ["L1", "L2", "L3"]
    assert corr["matrix"][0][2] == pytest.approx(0.6)
    assert corr["matrix"][2][0] == pytest.approx(0.6)
    assert j["plan"]["formulas"]
    assert set(j["plan"]["dependency_versions"]) >= {
        "python", "numpy", "fastapi", "pydantic", "sqlalchemy"}


def test_plan_unit_conversion(client, simple_chain_payload):
    """混合单位分量统一换算为 mm 后合成。"""
    cid = _make_chain(client, simple_chain_payload)
    plan = {"name": "um-plan", "gauges": [
        {"dimension_id": "L1", "unit": "um", "resolution": 10.0,
         "calibration_expanded_uncertainty": 4.0, "coverage_factor": 2.0,
         "bias_correction": 2.0, "bias_std_uncertainty": 1.0,
         "repeatability_std": 3.0},
        {"dimension_id": "L2", "resolution": 0.005},
        {"dimension_id": "L3", "repeatability_std": 0.004},
    ]}
    j = _make_plan(client, cid, plan)
    dims = {d["dimension_id"]: d for d in j["plan"]["dimensions"]}
    l1 = dims["L1"]
    assert l1["components_mm"]["u_resolution"] == pytest.approx(
        0.01 / (2 * SQRT3))
    assert l1["components_mm"]["u_calibration"] == pytest.approx(0.002)
    assert l1["bias_correction_mm"] == pytest.approx(0.002)
    # 只给分辨率的 L2：u_c = u_res
    assert dims["L2"]["combined_std_uncertainty_mm"] == pytest.approx(
        0.005 / (2 * SQRT3))
    assert dims["L3"]["combined_std_uncertainty_mm"] == pytest.approx(0.004)


def test_plan_reject_missing_coverage_factor(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    base = [dict(g) for g in PLAN["gauges"]]
    # 给 U 缺 k
    bad1 = [dict(g) for g in base]
    del bad1[0]["coverage_factor"]
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json={"name": "bad", "gauges": bad1})
    assert r.status_code == 422 and "覆盖因子缺失" in r.text
    # 给 k 缺 U
    bad2 = [dict(g) for g in base]
    del bad2[1]["calibration_expanded_uncertainty"]
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json={"name": "bad", "gauges": bad2})
    assert r.status_code == 422 and "校准扩展不确定度缺失" in r.text


def test_plan_reject_negative_components(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    for field in ("resolution", "calibration_expanded_uncertainty",
                  "bias_std_uncertainty", "repeatability_std"):
        gauges = [dict(g) for g in PLAN["gauges"]]
        gauges[0][field] = -0.001
        r = client.post(f"/chains/{cid}/measurement-plans",
                        json={"name": "bad", "gauges": gauges})
        assert r.status_code == 422, field
        assert "不能为负" in r.text
    # 覆盖因子必须为正
    gauges = [dict(g) for g in PLAN["gauges"]]
    gauges[0]["coverage_factor"] = 0
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json={"name": "bad", "gauges": gauges})
    assert r.status_code == 422 and "覆盖因子必须为正" in r.text


def test_plan_reject_non_finite_components(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    body = (
        '{"name":"bad","gauges":['
        '{"dimension_id":"L1","resolution":NaN},'
        '{"dimension_id":"L2","resolution":0.005},'
        '{"dimension_id":"L3","resolution":0.01}]}'
    )
    r = client.post(f"/chains/{cid}/measurement-plans", content=body,
                    headers={"content-type": "application/json"})
    assert r.status_code == 422 and "有限数" in r.text


def test_plan_reject_non_psd_correlation(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    plan = {"name": "bad-corr", "gauges": PLAN["gauges"], "correlations": [
        {"dim_a": "L1", "dim_b": "L2", "rho": 0.9},
        {"dim_a": "L1", "dim_b": "L3", "rho": 0.9},
        {"dim_a": "L2", "dim_b": "L3", "rho": -0.9},
    ]}
    r = client.post(f"/chains/{cid}/measurement-plans", json=plan)
    assert r.status_code == 422 and "非半正定" in r.text


def test_plan_reject_coverage_problems(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    # 缺 L3
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json={"name": "bad", "gauges": PLAN["gauges"][:2]})
    assert r.status_code == 422 and "缺少链上尺寸" in r.text
    assert "L3" in r.text
    # 链外尺寸
    gauges = [dict(g) for g in PLAN["gauges"]] + [
        {"dimension_id": "X9", "resolution": 0.01}]
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json={"name": "bad", "gauges": gauges})
    assert r.status_code == 422 and "链之外" in r.text
    # 重复声明
    gauges = [dict(g) for g in PLAN["gauges"]] + [dict(PLAN["gauges"][0])]
    r = client.post(f"/chains/{cid}/measurement-plans",
                    json={"name": "bad", "gauges": gauges})
    assert r.status_code == 422 and "重复声明" in r.text
    # 相关项引用方案外尺寸
    r = client.post(f"/chains/{cid}/measurement-plans", json={
        "name": "bad", "gauges": PLAN["gauges"],
        "correlations": [{"dim_a": "L1", "dim_b": "ZZ", "rho": 0.5}]})
    assert r.status_code == 422 and "方案外的尺寸" in r.text
    # 链不存在
    r = client.post("/chains/9999/measurement-plans",
                    json={"name": "bad", "gauges": PLAN["gauges"]})
    assert r.status_code == 404


def test_plan_immutable_and_versioned(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    v1 = _make_plan(client, cid)
    plan2 = {"name": "caliper-set-v2", "gauges": [
        dict(g, repeatability_std=0.001) for g in PLAN["gauges"]]}
    v2 = _make_plan(client, cid, plan2)
    assert v2["plan_id"] == 2
    plans = client.get(f"/chains/{cid}/measurement-plans").json()["plans"]
    assert [p["name"] for p in plans] == ["caliper-set-v1", "caliper-set-v2"]
    # 无更新/删除端点
    assert client.put(f"/measurement-plans/{v1['plan_id']}", json={}).status_code == 405
    assert client.delete(f"/measurement-plans/{v1['plan_id']}").status_code == 405
    # v1 内容不受 v2 影响
    g1 = client.get(f"/measurement-plans/{v1['plan_id']}").json()
    assert g1["plan"]["dimensions"][0]["components_mm"][
        "u_repeatability"] == pytest.approx(0.003)
    assert client.get("/measurement-plans/9999").status_code == 404


# ------------------------------------------------------------- 批次判定

def test_batch_bias_correction_and_decisions(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    # L1 规格 [50, 50.08]，u_c=U_C[0]，k_out=2，multiple=1 → w = 2·u_c
    w1 = 2 * U_C[0]
    rows = _rows({
        "ACC":  {"L1": 50.04, "L2": 30.30, "L3": 80.00},
        # corrected = 50.007 ∈ [50−w, 50+w) → 低侧不确定
        "INDL": {"L1": 50.005, "L2": 30.30, "L3": 80.00},
        # corrected = 49.982 < 50−w → 拒收
        "REJL": {"L1": 49.98, "L2": 30.30, "L3": 80.00},
        # corrected = 50.092 > 50.08+w → 拒收
        "REJH": {"L1": 50.09, "L2": 30.30, "L3": 80.00},
        # corrected = 50.077 ∈ (50.08−w, 50.08+w] → 高侧不确定
        "INDH": {"L1": 50.075, "L2": 30.30, "L3": 80.00},
    })
    j = _gauged_batch(client, cid, rows)
    m = j["measurement"]
    assert m["measurement_plan_id"] == 1 and m["frozen"] is True
    l1 = next(d for d in m["dimensions"] if d["dimension_id"] == "L1")
    assert l1["combined_std_uncertainty_mm"] == pytest.approx(U_C[0])
    assert l1["expanded_uncertainty_mm"] == pytest.approx(2 * U_C[0])
    assert l1["guard_band_mm"] == pytest.approx(w1)
    assert l1["accept_interval_mm"] == pytest.approx(
        [50.0 + w1, 50.08 - w1])
    by_serial = {i["serial"]: i for i in l1["items"]}
    # 偏倚修正：corrected = measured + 0.002
    assert by_serial["ACC"]["corrected_mm"] == pytest.approx(50.042)
    assert by_serial["ACC"]["measured_mm"] == pytest.approx(50.04)
    assert by_serial["ACC"]["decision"] == "accept"
    assert by_serial["INDL"]["decision"] == "indeterminate"
    assert by_serial["REJL"]["decision"] == "reject"
    assert by_serial["REJH"]["decision"] == "reject"
    assert by_serial["INDH"]["decision"] == "indeterminate"
    assert l1["summary"] == {"accept": 1, "reject": 2, "indeterminate": 2,
                             "measured": 5}


def test_batch_probabilities_gum_vs_mc(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    # w = 2·u_c ≈ 0.00945：A corrected=50.008 落在低侧保护带（不确定）；
    # B corrected=49.987 < 50−w（拒收）
    rows = _rows({"A": {"L1": 50.006, "L2": 30.30, "L3": 80.00},
                  "B": {"L1": 49.985, "L2": 30.30, "L3": 80.00}})
    j = _gauged_batch(client, cid, rows)
    l1 = next(d for d in j["measurement"]["dimensions"]
              if d["dimension_id"] == "L1")
    items = {i["serial"]: i for i in l1["items"]}
    # GUM 解析：p_out = Φ((50−x_c)/u_c) + 1 − Φ((50.08−x_c)/u_c)
    xc_a = 50.006 + 0.002
    z = (50.0 - xc_a) / U_C[0]
    p_out_expect = 0.5 * (1 + math.erf(z / math.sqrt(2))) + (
        1 - 0.5 * (1 + math.erf((50.08 - xc_a) / U_C[0] / math.sqrt(2))))
    a = items["A"]
    assert a["decision"] == "indeterminate"
    assert 0.01 < p_out_expect < 0.1  # 边界附近非平凡
    assert a["p_true_out_of_spec"]["gum"] == pytest.approx(p_out_expect)
    assert a["p_true_conforming"]["gum"] == pytest.approx(1 - p_out_expect)
    # 蒙特卡洛与 GUM 同模型，大样本下应接近
    assert a["p_true_out_of_spec"]["monte_carlo"] == pytest.approx(
        p_out_expect, abs=0.01)
    # B 被判拒收：关注拒收时真值合格概率
    b = items["B"]
    assert b["decision"] == "reject"
    xc_b = 49.985 + 0.002
    p_in_expect = (
        0.5 * (1 + math.erf((50.08 - xc_b) / U_C[0] / math.sqrt(2)))
        - 0.5 * (1 + math.erf((50.0 - xc_b) / U_C[0] / math.sqrt(2))))
    assert b["p_true_conforming"]["gum"] == pytest.approx(p_in_expect)
    # 修正值已低于 LSL 2.75σ：拒收很可能正确，真值合格概率仅约 0.3%
    assert 0.0 < b["p_true_conforming"]["gum"] < 0.05
    assert b["p_true_conforming"]["monte_carlo"] == pytest.approx(
        p_in_expect, abs=0.02)


def test_batch_closure_gum_and_mc(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({
        "OK":   {"L1": 50.04, "L2": 30.30, "L3": 80.00},   # C=0.343 接收
        "LOW":  {"L1": 50.04, "L2": 30.30, "L3": 80.40},   # C=−0.057 拒收
        "EDGE": {"L1": 50.04, "L2": 30.30, "L3": 80.28},   # C=0.063 不确定
    })
    j = _gauged_batch(client, cid, rows)
    cl = j["measurement"]["closure"]
    # GUM 封闭环合成（含相关项 −2·0.6·u_rest1·u_rest3）
    assert cl["gum"]["combined_std_uncertainty_mm"] == pytest.approx(U_CLOSURE)
    assert cl["gum"]["expanded_uncertainty_mm"] == pytest.approx(2 * U_CLOSURE)
    assert cl["gum"]["variance_from_resolution_mm2"] == pytest.approx(VAR_RES_C)
    assert cl["gum"]["variance_from_correlated_rest_mm2"] == pytest.approx(
        VAR_REST_C)
    shares = [c["variance_share"]
              for c in cl["gum"]["dimension_contributions"]]
    assert sum(shares) == pytest.approx(1.0)
    # 蒙特卡洛封闭环标准差收敛到 GUM 解析值
    assert cl["monte_carlo"]["std_mm"] == pytest.approx(U_CLOSURE, rel=0.03)
    assert cl["monte_carlo"]["seed"] == 5
    # 保护带与判定（w = 2·u_C；偏倚合计 +0.003）
    w_c = 2 * U_CLOSURE
    assert cl["guard_band_mm"] == pytest.approx(w_c)
    items = {i["serial"]: i for i in cl["items"]}
    assert items["OK"]["corrected_closure_mm"] == pytest.approx(0.343)
    assert items["OK"]["decision"] == "accept"
    assert items["LOW"]["corrected_closure_mm"] == pytest.approx(-0.057)
    assert items["LOW"]["decision"] == "reject"
    assert items["EDGE"]["corrected_closure_mm"] == pytest.approx(0.063)
    assert items["EDGE"]["decision"] == "indeterminate"
    assert cl["summary"] == {"accept": 1, "reject": 1, "indeterminate": 1}
    # 接收时真值越界概率小；拒收（LOW 距 LSL 约 16σ）时真值合格概率≈0
    assert items["OK"]["p_true_out_of_spec"]["gum"] < 1e-6
    assert items["LOW"]["p_true_conforming"]["gum"] < 1e-6
    # EDGE 靠近 LSL：越界概率非平凡且 MC 与 GUM 接近
    p_edge = items["EDGE"]["p_true_out_of_spec"]
    z = (0.05 - 0.063) / U_CLOSURE
    p_expect = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert p_edge["gum"] == pytest.approx(p_expect)
    assert p_edge["monte_carlo"] == pytest.approx(p_expect, abs=0.02)


def test_batch_fixed_guard_band(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({"A": {"L1": 50.045, "L2": 30.30, "L3": 80.00}})
    j = _gauged_batch(client, cid, rows,
                      guard_band={"mode": "fixed", "fixed": 0.005})
    m = j["measurement"]
    for d in m["dimensions"]:
        assert d["guard_band_mm"] == pytest.approx(0.005)
    assert m["closure"]["guard_band_mm"] == pytest.approx(0.005)
    assert m["policy"]["guard_band"]["mode"] == "fixed"
    assert m["policy"]["guard_band"]["fixed_mm"] == pytest.approx(0.005)
    # corrected L1 = 50.047 ∈ [50.005, 50.075] → 接收
    l1 = next(d for d in m["dimensions"] if d["dimension_id"] == "L1")
    assert l1["items"][0]["decision"] == "accept"
    # 固定长度带单位换算：0.5 um = 0.0005 mm
    j2 = _gauged_batch(client, cid, rows, name="gb-um",
                       guard_band={"mode": "fixed", "fixed": 0.5,
                                   "unit": "um"})
    d1 = j2["measurement"]["dimensions"][0]
    assert d1["guard_band_mm"] == pytest.approx(0.0005)


def test_batch_guard_band_validation(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({"A": {"L1": 50.04, "L2": 30.30, "L3": 80.00}})
    base = {"name": "bad", "rows": rows, "measurement_plan_id": 1}
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={**base, "guard_band": {"mode": "fixed"}})
    assert r.status_code == 422 and "fixed" in r.text
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={**base, "guard_band": {"mode": "multiple",
                                                 "multiple": 1.0,
                                                 "fixed": 0.01}})
    assert r.status_code == 422
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={**base, "guard_band": {"mode": "multiple",
                                                 "multiple": -1.0}})
    assert r.status_code == 422


def test_batch_plan_reference_errors(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({"A": {"L1": 50.04, "L2": 30.30, "L3": 80.00}})
    # 方案不存在
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "x", "rows": rows,
                          "measurement_plan_id": 9999})
    assert r.status_code == 404
    # 方案属于另一条链
    cid2 = _make_chain(client, {**simple_chain_payload, "name": "other-chain"})
    r = client.post(f"/chains/{cid2}/inspection-batches",
                    json={"name": "x", "rows": rows,
                          "measurement_plan_id": 1})
    assert r.status_code == 404


def test_batch_missing_dimensions_closure_excluded(client,
                                                   simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({
        "FULL": {"L1": 50.04, "L2": 30.30, "L3": 80.00},
        "GAP":  {"L1": 50.05, "L3": 79.99},          # L2 缺测
    })
    j = _gauged_batch(client, cid, rows)
    m = j["measurement"]
    cl = m["closure"]
    assert [i["serial"] for i in cl["items"]] == ["FULL"]
    assert cl["excluded_serials"] == ["GAP"]
    assert cl["exclusion_reason"]
    assert m["summary"] == {"rows_total": 2, "closure_evaluated": 1,
                            "closure_excluded": 1}
    # 缺测尺寸不出现在该尺寸 items 中
    l2 = next(d for d in m["dimensions"] if d["dimension_id"] == "L2")
    assert [i["serial"] for i in l2["items"]] == ["FULL"]
    assert l2["summary"]["measured"] == 1


def test_batch_seed_reproducible(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({f"W{i}": {"L1": 50.04 + 0.001 * (i % 5),
                            "L2": 30.30, "L3": 80.00} for i in range(10)})
    a = _gauged_batch(client, cid, rows, name="s1")
    b = _gauged_batch(client, cid, rows, name="s2")
    assert a["measurement"] == b["measurement"]
    # 不同种子：GUM 解析值不变，蒙特卡洛经验比例允许抖动
    c = _gauged_batch(client, cid, rows, name="s3", measurement_mc_seed=6)
    la = next(d for d in a["measurement"]["dimensions"]
              if d["dimension_id"] == "L1")
    lc = next(d for d in c["measurement"]["dimensions"]
              if d["dimension_id"] == "L1")
    assert la["items"][0]["p_true_out_of_spec"]["gum"] == \
        lc["items"][0]["p_true_out_of_spec"]["gum"]
    assert a["measurement"]["monte_carlo"]["seed"] == 5
    assert c["measurement"]["monte_carlo"]["seed"] == 6


def test_batch_plan_snapshot_frozen(client, simple_chain_payload):
    """批次冻结方案快照：后续新版本方案不改变历史判定。"""
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({"A": {"L1": 50.04, "L2": 30.30, "L3": 80.00}})
    j1 = _gauged_batch(client, cid, rows, name="lot-v1")
    bid = j1["batch_id"]
    # 新版本方案：重复性大幅收紧
    plan2 = {"name": "caliper-set-v2", "gauges": [
        dict(g, repeatability_std=0.0001, resolution=0.001)
        for g in PLAN["gauges"]]}
    _make_plan(client, cid, plan2)
    # 历史批次多次读取内容不变，快照仍是 v1 的量具参数
    g1 = client.get(f"/inspection-batches/{bid}").json()
    g2 = client.get(f"/inspection-batches/{bid}").json()
    assert g1 == g2
    snap = g1["measurement"]["plan_snapshot"]
    assert snap["plan_id"] == 1 and snap["name"] == "caliper-set-v1"
    l1_snap = next(d for d in snap["combined"]["dimensions"]
                   if d["dimension_id"] == "L1")
    assert l1_snap["components_mm"]["u_repeatability"] == pytest.approx(0.003)
    l1_rep = next(d for d in g1["measurement"]["dimensions"]
                  if d["dimension_id"] == "L1")
    assert l1_rep["combined_std_uncertainty_mm"] == pytest.approx(U_C[0])
    # 新批次引用 v2：不确定度变小，两批判定报告不同
    j2 = _gauged_batch(client, cid, rows, name="lot-v2", plan_id=2)
    l1_v2 = next(d for d in j2["measurement"]["dimensions"]
                 if d["dimension_id"] == "L1")
    assert l1_v2["combined_std_uncertainty_mm"] < U_C[0]
    # 历史批次仍然不变
    g3 = client.get(f"/inspection-batches/{bid}").json()
    assert g3["measurement"] == g1["measurement"]


def test_batch_without_plan_measurement_null(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = _rows({"A": {"L1": 50.04, "L2": 30.30, "L3": 80.00}})
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "plain", "rows": rows})
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["measurement_plan_id"] is None
    assert j["measurement"] is None
    # 列表包含 measurement_plan_id 字段
    lst = client.get(f"/chains/{cid}/inspection-batches").json()["batches"]
    assert lst[0]["measurement_plan_id"] is None


def test_batch_response_traceability(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    _make_plan(client, cid)
    rows = _rows({"A": {"L1": 50.04, "L2": 30.30, "L3": 80.00}})
    j = _gauged_batch(client, cid, rows)
    m = j["measurement"]
    # 公式、随机种子、依赖版本、分量贡献齐全
    assert any("u_res" in f for f in m["formulas"])
    assert any("GUM" in f or "线性传播" in f for f in m["formulas"])
    assert m["monte_carlo"]["samples"] == 50000
    assert m["monte_carlo"]["seed"] == 5
    versions = m["dependency_versions"]
    assert set(versions) >= {"python", "numpy", "fastapi", "pydantic",
                             "sqlalchemy"}
    l1 = next(d for d in m["dimensions"] if d["dimension_id"] == "L1")
    assert set(l1["components_mm"]) == {
        "u_resolution", "u_calibration", "u_bias", "u_repeatability"}
    assert set(l1["variance_share"]) == {
        "resolution", "calibration", "bias", "repeatability"}
    assert m["policy"]["output_coverage_factor"] == 2.0
    assert m["policy"]["guard_band"]["mode"] == "multiple"
    assert m["policy"]["guard_band"]["multiple"] == 1.0


def test_closure_without_spec_no_decision(client):
    """封闭环无规格：只报告不确定度，不做接收判定。"""
    payload = {
        "name": "no-closure-spec-gauge",
        "dimensions": [
            {"id": "A", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02},
            {"id": "B", "start": "b", "end": "a", "nominal": 10,
             "upper_deviation": 0.1, "lower_deviation": -0.1, "std_dev": 0.02,
             "direction": -1},
        ],
        "mc_samples": 3000,
    }
    cid = _make_chain(client, payload)
    plan = {"name": "p", "gauges": [
        {"dimension_id": "A", "repeatability_std": 0.002},
        {"dimension_id": "B", "repeatability_std": 0.003},
    ]}
    _make_plan(client, cid, plan)
    rows = [{"serial": "W1", "measurements": [
        {"dimension_id": "A", "value": 10.01, "unit": "mm"},
        {"dimension_id": "B", "value": 10.02, "unit": "mm"}]}]
    j = _gauged_batch(client, cid, rows)
    cl = j["measurement"]["closure"]
    assert cl["gum"]["combined_std_uncertainty_mm"] == pytest.approx(
        math.sqrt(0.002 ** 2 + 0.003 ** 2))
    assert cl["items"][0]["decision"] is None
    assert cl["items"][0]["p_true_out_of_spec"] is None
    assert cl["summary"] is None
    assert cl["decision_unavailable_reason"]
    # 尺寸级判定照常（尺寸规格始终存在）
    da = next(d for d in j["measurement"]["dimensions"]
              if d["dimension_id"] == "A")
    assert da["items"][0]["decision"] == "accept"
