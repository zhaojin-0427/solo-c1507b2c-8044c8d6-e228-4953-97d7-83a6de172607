"""检验批次漂移研究：批次归属/采样顺序/参数校验，逐批 CUSUM/EWMA 统计、
阈值余量、漂移方向、首次报警批次与变点，量具不确定度传播与封闭环贡献，
复制研究排除异常批次并比较，定稿冻结。"""
import math

import numpy as np
import pytest


def _make_chain(client, payload=None):
    if payload is None:
        payload = {
            "name": "drift-chain",
            "closure_lower_limit": 0.05,
            "closure_upper_limit": 0.60,
            "dimensions": [
                {"id": "L1", "start": "a", "end": "b", "nominal": 50,
                 "upper_deviation": 0.08, "lower_deviation": 0.0,
                 "std_dev": 0.02},
                {"id": "L2", "start": "b", "end": "c", "nominal": 30.3,
                 "upper_deviation": 0.05, "lower_deviation": -0.05,
                 "std_dev": 0.02},
                {"id": "L3", "start": "c", "end": "a", "nominal": 80,
                 "upper_deviation": 0.0, "lower_deviation": -0.15,
                 "std_dev": 0.04, "direction": -1},
            ],
            "mc_samples": 5000, "random_seed": 7,
        }
    r = client.post("/chains", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["chain_id"]


def _batch(client, cid, name, n, means, *, seed=0, plan_id=None,
           stds=(0.015, 0.015, 0.02)):
    """means=(μL1,μL2,μL3)，生成 n 行齐全测量并创建冻结批次。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({"serial": f"{name}-{i:02d}", "measurements": [
            {"dimension_id": "L1",
             "value": float(means[0] + rng.normal(0, stds[0])), "unit": "mm"},
            {"dimension_id": "L2",
             "value": float(means[1] + rng.normal(0, stds[1])), "unit": "mm"},
            {"dimension_id": "L3",
             "value": float(means[2] + rng.normal(0, stds[2])), "unit": "mm"},
        ]})
    body = {"name": name, "rows": rows, "bootstrap_samples": 1000,
            "random_seed": 11}
    if plan_id is not None:
        body["measurement_plan_id"] = plan_id
    r = client.post(f"/chains/{cid}/inspection-batches", json=body)
    assert r.status_code == 201, r.text
    return r.json()["batch_id"]


def _specs(batch_ids, times=None):
    if times is None:
        times = [f"2026-0{m+1}-01T08:00:00+00:00"
                 for m in range(len(batch_ids))]
    return [{"batch_id": b, "sampled_at": t}
            for b, t in zip(batch_ids, times)]


def _target(result, key):
    return next(t for t in result["targets"] if t["target"] == key)


# ------------------------------------------------------------- 正常流程

def test_drift_study_cusum_ewma_statistics(client):
    cid = _make_chain(client)
    # L1 从在控中心逐批上漂；L2/L3 保持在控
    bids = [
        _batch(client, cid, f"lot-{t}", 20,
               (50.04 + 0.012 * t, 30.30, 80.00), seed=t)
        for t in range(6)
    ]
    body = {"name": "d1", "batches": _specs(bids),
            "dimensions": [{"dimension_id": "L1"}],
            "monitor_closure": True}
    r = client.post(f"/chains/{cid}/drift-studies", json=body)
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["frozen"] is False and j["version_no"] == 1
    res = j["result"]
    t1 = _target(res, "dim:L1")

    # 逐批统计：均值/标准误/z，z 随漂移走高
    assert [b["sequence_no"] for b in t1["batch_results"]] == [1, 2, 3, 4, 5, 6]
    assert [b["batch_id"] for b in t1["batch_results"]] == bids
    z = [b["z"] for b in t1["batch_results"]]
    assert z[-1] > z[0] and z[-1] > 5
    # SE = sqrt(σ0²/n + u_g²)，无测量方案时 u_g=0 => σ0/√n = 0.02/√20
    assert t1["batch_results"][0]["standard_error_mm"] == pytest.approx(
        0.02 / math.sqrt(20))
    assert t1["batch_results"][0]["u_gage_batch_mm"] == 0.0
    assert t1["in_control"]["mean_mm"] == pytest.approx(50.04)
    assert t1["in_control"]["sigma0_mm"] == pytest.approx(0.02)

    cu, ew = t1["cusum"], t1["ewma"]
    assert cu["stat_minus"] is not None and cu["stat_plus"] is not None
    # 阈值余量 = h - C+；报警后为负
    margins = cu["threshold_margin_up"]
    assert margins[0] > 0
    assert any(m < 0 for m in margins)
    assert cu["first_alarm_direction"] == "up"
    assert ew["first_alarm_direction"] == "up"
    # 时变控制限从宽到严趋稳
    lim = ew["control_limit"]
    assert lim[0] < lim[-1]
    # 余量在报警批转负
    assert ew["threshold_margin_up"][-1] < 0
    # 变点：报警轮累计起点，给出批次序号与 σ 单位偏移
    cp = cu["change_point"]
    assert cp["batch_index"] >= 2 and cp["estimated_shift_sigma"] > 0
    assert res["conclusion"]["any_alarm"] is True
    fa = res["conclusion"]["first_alarm"]
    assert fa["target"] in ("dim:L1", "closure")
    assert fa["batch"]["batch_id"] in bids


def test_drift_no_alarm_when_in_control(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 30, (50.04, 30.30, 80.00), seed=t)
            for t in range(5)]
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "stable", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1"}]})
    assert r.status_code == 201, r.text
    t1 = _target(r.json()["result"], "dim:L1")
    assert t1["conclusion_cusum"]["alarmed"] is False
    assert t1["conclusion_cusum"]["first_alarm_batch"] is None
    assert t1["conclusion_ewma"]["alarmed"] is False
    # 阈值余量始终为正
    assert all(m > 0 for m in t1["cusum"]["threshold_margin_up"])
    assert all(m > 0 for m in t1["cusum"]["threshold_margin_down"])
    assert r.json()["result"]["conclusion"]["any_alarm"] is False


def test_one_sided_chart_and_param_overrides(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 20,
                   (50.04 + 0.01 * t, 30.30, 80.00), seed=t)
           for t in range(6)]
    body = {"name": "one-sided", "batches": _specs(bids),
            "dimensions": [{"dimension_id": "L1", "overrides": {
                "sidedness": "one_sided_up",
                "cusum_k": 0.5, "cusum_h": 4.0,
                "ewma_lambda": 0.4, "ewma_L": 2.7}}],
            "defaults": {"min_sample_size": 10, "cusum_h": 6.0}}
    r = client.post(f"/chains/{cid}/drift-studies", json=body)
    assert r.status_code == 201, r.text
    t1 = _target(r.json()["result"], "dim:L1")
    assert t1["params"]["sidedness"] == "one_sided_up"
    assert t1["params"]["cusum_h"] == 4.0          # 覆盖生效
    assert t1["params"]["cusum_k"] == 0.5
    # 单侧：下侧统计量与余量不维护
    assert all(v is None for v in t1["cusum"]["stat_minus"])
    assert all(v is None for v in t1["cusum"]["threshold_margin_down"])
    assert t1["cusum"]["stat_plus"] is not None
    # 研究级缺省未被覆盖的字段透传
    assert t1["params"]["min_sample_size"] == 10


def test_one_sided_down_drift_detected_deterministic(client):
    """批内同值的确定性下漂序列：z=0,-2.68,-5.37,...；单侧下向图必报 down。"""
    cid = _make_chain(client)

    def flat(name, v1):
        rows = [{"serial": f"{name}-{i}", "measurements": [
            {"dimension_id": "L1", "value": v1, "unit": "mm"},
            {"dimension_id": "L2", "value": 30.30, "unit": "mm"},
            {"dimension_id": "L3", "value": 80.00, "unit": "mm"}]}
            for i in range(20)]
        r = client.post(f"/chains/{cid}/inspection-batches",
                        json={"name": name, "rows": rows,
                              "bootstrap_samples": 1000})
        return r.json()["batch_id"]

    bids = [flat(f"flat-{t}", 50.04 - 0.012 * t) for t in range(6)]
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "down", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1",
                        "overrides": {"sidedness": "one_sided_down"}}]})
    assert r.status_code == 201, r.text
    t1 = _target(r.json()["result"], "dim:L1")
    assert t1["cusum"]["alarmed"] is True
    assert t1["cusum"]["first_alarm_direction"] == "down"
    assert t1["ewma"]["alarmed"] is True
    assert t1["ewma"]["first_alarm_direction"] == "down"
    # 单侧下：上侧统计量不维护（None），下侧余量在报警批转负
    assert all(v is None for v in t1["cusum"]["stat_plus"])
    assert any(m is not None and m < 0
               for m in t1["cusum"]["threshold_margin_down"])
    # z1=0（首批恰在 μ0），标准误 σ0/√n = 0.02/√20
    assert t1["batch_results"][0]["z"] == pytest.approx(0.0, abs=1e-9)
    assert t1["batch_results"][1]["z"] == pytest.approx(
        -0.012 / (0.02 / math.sqrt(20)), abs=1e-6)
    # 变点位于开始下漂的第 2 批，变点后平均约 -8.05 σ
    cp = t1["cusum"]["change_point"]
    assert cp["batch_index"] == 2
    assert cp["estimated_shift_sigma"] == pytest.approx(-8.05, abs=0.02)


def test_closure_drift_contributions(client):
    cid = _make_chain(client)
    # 仅 L1 系统性上漂；封闭环 C=L1+L2−L3，故 L1 贡献为正
    bids = [_batch(client, cid, f"lot-{t}", 30,
                   (50.04 + 0.01 * t, 30.30, 80.00), seed=100 + t)
           for t in range(5)]
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "contrib", "batches": _specs(bids),
        "monitor_closure": True})
    assert r.status_code == 201, r.text
    cc = r.json()["result"]["closure_drift_contributions"]
    by = {d["dimension_id"]: d for d in cc["dimensions"]}
    # L1 符号 +1，漂移贡献为正且占主导；份额之和（绝对值口径）≈ 1
    assert by["L1"]["sign"] == 1 and by["L3"]["sign"] == -1
    assert by["L1"]["closure_drift_contribution_mm"] == pytest.approx(
        by["L1"]["mean_shift_mm"])
    total = sum(abs(d["closure_drift_contribution_mm"])
                for d in cc["dimensions"])
    assert abs(by["L1"]["closure_drift_contribution_mm"]) > 0.5 * total
    # 分量带符号合计 ≈ 实测封闭环首末批漂移
    assert cc["signed_contributions_total_mm"] == pytest.approx(
        cc["observed_closure_drift_mm"], abs=1e-9)


# ------------------------------------------------------------- 量具传播

def test_gage_uncertainty_propagation(client):
    cid = _make_chain(client)
    plan = {"name": "p", "gauges": [
        {"dimension_id": "L1", "resolution": 0.002,
         "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2,
         "bias_correction": 0.01, "bias_std_uncertainty": 0.003,
         "repeatability_std": 0.004},
        {"dimension_id": "L2", "resolution": 0.002,
         "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2,
         "repeatability_std": 0.004},
        {"dimension_id": "L3", "resolution": 0.002,
         "calibration_expanded_uncertainty": 0.006, "coverage_factor": 2,
         "repeatability_std": 0.004}]}
    pid = client.post(f"/chains/{cid}/measurement-plans",
                      json=plan).json()["plan_id"]
    bids = [_batch(client, cid, f"lot-{t}", 20,
                   (50.04, 30.30, 80.00), seed=t, plan_id=pid)
            for t in range(4)]
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "gauged", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1"}],
        "monitor_closure": True})
    assert r.status_code == 201, r.text
    res = r.json()["result"]
    t1 = _target(res, "dim:L1")
    b0 = t1["batch_results"][0]
    # u_rest = sqrt(0.003² 校准 + 0.003² 偏倚 + 0.004² 重复)
    u_rest = math.sqrt(0.003 ** 2 + 0.003 ** 2 + 0.004 ** 2)
    assert b0["u_gage_batch_mm"] == pytest.approx(u_rest)
    assert b0["gage_uncertainty"]["measurement_plan_id"] == pid
    # 偏倚 +0.01 已在批次均值中修正（x_c = x + b）
    assert b0["mean_mm"] > 50.03
    # SE 含 u_g：sqrt(σ0²/n + u_g²)，严格大于 σ0/√n
    assert b0["standard_error_mm"] == pytest.approx(
        math.sqrt(0.02 ** 2 / 20 + u_rest ** 2))
    # 封闭环 GUM 传播：L1 u_rest=0.00583，L2/L3=0.005；s=(+1,+1,-1)
    sb = res["source_batches"][0]["gage_uncertainty"]
    assert sb["closure_u_g_mm"] == pytest.approx(
        math.sqrt(u_rest ** 2 + 0.005 ** 2 + 0.005 ** 2))
    assert sb["closure_expanded_uncertainty_mm"] == pytest.approx(
        2 * sb["closure_u_g_mm"])


# ------------------------------------------------------------- 校验拒收

def test_reject_wrong_chain_and_missing_batch(client, simple_chain_payload):
    cid1 = _make_chain(client, simple_chain_payload | {"name": "c1"})
    cid2 = _make_chain(client, simple_chain_payload | {"name": "c2"})
    rows = [{"serial": f"W{i}", "measurements": [
        {"dimension_id": "L1", "value": 50.04, "unit": "mm"},
        {"dimension_id": "L2", "value": 30.3, "unit": "mm"},
        {"dimension_id": "L3", "value": 80.0, "unit": "mm"}]}
        for i in range(3)]
    b1 = client.post(f"/chains/{cid1}/inspection-batches",
                     json={"name": "b1", "rows": rows}).json()["batch_id"]
    b2 = client.post(f"/chains/{cid2}/inspection-batches",
                     json={"name": "b2", "rows": rows}).json()["batch_id"]
    specs = [
        {"batch_id": b1, "sampled_at": "2026-01-01T08:00:00+00:00"},
        {"batch_id": b2, "sampled_at": "2026-02-01T08:00:00+00:00"}]
    r = client.post(f"/chains/{cid1}/drift-studies", json={
        "name": "wrong", "batches": specs,
        "dimensions": [{"dimension_id": "L1"}]})
    assert r.status_code == 422 and "不属于当前链" in r.text
    # 不存在批次 404
    r = client.post(f"/chains/{cid1}/drift-studies", json={
        "name": "ghost", "batches": [
            {"batch_id": b1, "sampled_at": "2026-01-01T08:00:00+00:00"},
            {"batch_id": 99999, "sampled_at": "2026-02-01T08:00:00+00:00"}],
        "dimensions": [{"dimension_id": "L1"}]})
    assert r.status_code == 404


def test_reject_sampling_order_and_duplicate_and_tz(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 10, (50.04, 30.30, 80.00), seed=t)
            for t in range(3)]
    base = {"dimensions": [{"dimension_id": "L1"}]}
    # 逆序
    r = client.post(f"/chains/{cid}/drift-studies", json={
        **base, "name": "rev",
        "batches": list(reversed(_specs(bids)))})
    assert r.status_code == 422 and "严格递增" in r.text
    # 同时刻
    same = _specs(bids, ["2026-01-01T08:00:00+00:00"] * 3)
    r = client.post(f"/chains/{cid}/drift-studies",
                    json={**base, "name": "same", "batches": same})
    assert r.status_code == 422 and "严格递增" in r.text
    # 混用时区
    mixed = _specs(bids)
    mixed[1]["sampled_at"] = "2026-02-01T08:00:00"
    r = client.post(f"/chains/{cid}/drift-studies",
                    json={**base, "name": "tz", "batches": mixed})
    assert r.status_code == 422 and "时区" in r.text
    # 批次重复
    dup = _specs([bids[0], bids[0]])
    r = client.post(f"/chains/{cid}/drift-studies",
                    json={**base, "name": "dup", "batches": dup})
    assert r.status_code == 422 and "重复" in r.text


def test_reject_target_and_param_validation(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 10, (50.04, 30.30, 80.00), seed=t)
            for t in range(3)]
    specs = _specs(bids)
    # 无监控目标
    r = client.post(f"/chains/{cid}/drift-studies",
                    json={"name": "no-target", "batches": specs})
    assert r.status_code == 422 and "监控目标" in r.text
    # 链外尺寸
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "ext", "batches": specs,
        "dimensions": [{"dimension_id": "ZZ"}]})
    assert r.status_code == 422 and "未知尺寸" in r.text
    # 关注尺寸重复
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "dup-dim", "batches": specs,
        "dimensions": [{"dimension_id": "L1"}, {"dimension_id": "L1"}]})
    assert r.status_code == 422 and "关注尺寸重复" in r.text
    # 参数越界（λ>1、h≤0、最小样本量 0）
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "bad-lambda", "batches": specs,
        "dimensions": [{"dimension_id": "L1"}],
        "defaults": {"ewma_lambda": 1.5}})
    assert r.status_code == 422
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "bad-h", "batches": specs,
        "dimensions": [{"dimension_id": "L1"}],
        "defaults": {"cusum_h": 0}})
    assert r.status_code == 422
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "bad-n", "batches": specs,
        "dimensions": [{"dimension_id": "L1"}],
        "defaults": {"min_sample_size": 0}})
    assert r.status_code == 422


def test_min_sample_size_excludes_batches(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 30,
                   (50.04 + 0.012 * t, 30.30, 80.00), seed=t)
           for t in range(4)]
    # 最小样本量 25：全部可用；改为 100：全部不可用 => 不作图
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "hi-min", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1",
                        "overrides": {"min_sample_size": 100}}]})
    assert r.status_code == 201, r.text
    t1 = _target(r.json()["result"], "dim:L1")
    assert t1["batches_used"] == 0
    assert t1["cusum"] is None and t1["ewma"] is None
    assert "至少需要 2 个" in t1["charts_unavailable_reason"]
    assert len(t1["insufficient_batches"]) == 4


# ------------------------------------------------------------- 复制 / 定稿

def test_copy_with_exclusion_and_comparison(client):
    cid = _make_chain(client)
    # 末两批强烈上漂触发报警
    bids = [_batch(client, cid, f"lot-{t}", 25,
                   (50.04 + (0.02 * max(0, t - 3)), 30.30, 80.00), seed=t)
           for t in range(6)]
    r = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "orig", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1"}]})
    assert r.status_code == 201, r.text
    sid = r.json()["study_id"]
    parent_result = r.json()["result"]
    assert parent_result["conclusion"]["any_alarm"] is True

    r2 = client.post(f"/drift-studies/{sid}/copy", json={
        "name": "clean", "exclusions": [
            {"batch_id": bids[4], "reason": "换模首批，工艺未稳定"},
            {"batch_id": bids[5], "reason": "设备异常停线恢复批"}]})
    assert r2.status_code == 201, r2.text
    cj = r2.json()
    assert cj["parent_study_id"] == sid
    assert cj["version_no"] == 2
    assert {e["batch_id"] for e in cj["exclusions"]} == {bids[4], bids[5]}
    assert [b["batch_id"] for b in
            cj["result"]["source_batches"]] == bids[:4]
    cmp_ = cj["comparison_with_parent"]
    assert cmp_["overall"]["any_alarm_before"] is True
    l1cmp = next(t for t in cmp_["targets"] if t["target"] == "dim:L1")
    assert l1cmp["batches_used_before"] == 6
    assert l1cmp["batches_used_after"] == 4
    assert l1cmp["alarm_disappeared"] is True
    assert l1cmp["changed"] is True
    # 父研究不被复制改写
    g = client.get(f"/drift-studies/{sid}").json()
    assert g["result"] == parent_result and g["frozen"] is False

    # 排除非父研究来源批次 / 空原因 / 剩 <2 批 拒收
    r = client.post(f"/drift-studies/{sid}/copy", json={"exclusions": [
        {"batch_id": 99999, "reason": "x"}]})
    assert r.status_code == 422 and "不在父研究" in r.text
    r = client.post(f"/drift-studies/{sid}/copy", json={"exclusions": [
        {"batch_id": bids[0], "reason": ""}]})
    assert r.status_code == 422
    r = client.post(f"/drift-studies/{sid}/copy", json={"exclusions": [
        {"batch_id": b, "reason": "x"} for b in bids[:5]]})
    assert r.status_code == 422 and "至少需要 2 个" in r.text


def test_finalize_freezes_study(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 20, (50.04, 30.30, 80.00), seed=t)
            for t in range(3)]
    sid = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "fin", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1"}]}).json()["study_id"]
    r = client.post(f"/drift-studies/{sid}/finalize",
                    json={"note": "季度来料评审定稿"})
    assert r.status_code == 200
    assert r.json()["frozen"] is True
    assert r.json()["finalized_note"] == "季度来料评审定稿"
    assert r.json()["finalized_at"] is not None
    # 重复定稿 422
    assert client.post(f"/drift-studies/{sid}/finalize",
                       json={}).status_code == 422
    # 定稿后不能再复制排除
    r = client.post(f"/drift-studies/{sid}/copy", json={"exclusions": [
        {"batch_id": bids[0], "reason": "x"}]})
    assert r.status_code == 422 and "已定稿" in r.text
    # 读取内容不变
    g = client.get(f"/drift-studies/{sid}").json()
    assert g["frozen"] is True and g["result"]["status"] == "completed"
    # 列表带定稿与报警摘要
    lst = client.get(f"/chains/{cid}/drift-studies").json()["studies"]
    assert lst[0]["name"] == "fin" and lst[0]["frozen"] is True
    assert lst[0]["batch_count"] == 3


def test_drift_study_immutable_and_listing(client):
    cid = _make_chain(client)
    bids = [_batch(client, cid, f"lot-{t}", 20, (50.04, 30.30, 80.00), seed=t)
            for t in range(3)]
    sid = client.post(f"/chains/{cid}/drift-studies", json={
        "name": "freeze-check", "batches": _specs(bids),
        "dimensions": [{"dimension_id": "L1"}],
        "monitor_closure": True}).json()["study_id"]
    g1 = client.get(f"/drift-studies/{sid}").json()
    g2 = client.get(f"/drift-studies/{sid}").json()
    assert g1["result"] == g2["result"]
    assert client.get("/drift-studies/9999").status_code == 404
    # 研究无更新/删除端点；快照含提交输入与公式追溯
    assert g1["submitted_input"]["batches"][0]["batch_id"] == bids[0]
    assert any("CUSUM" in f for f in g1["result"]["formulas"])
    assert any("EWMA" in f for f in g1["result"]["formulas"])
