"""来料检验批次：校验拒收、缺测、逐尺寸统计、Cp/Cpk 置信区间、
协方差、固定种子封闭环 bootstrap、冻结与多批次、与基线并列。"""
import math

import numpy as np
import pytest


def _make_chain(client, payload):
    r = client.post("/chains", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["chain_id"]


def _rows(values_by_serial, dims=("L1", "L2", "L3"), unit="mm"):
    """{serial: {dim: value_or_None}} -> API rows（None=缺测，直接省略）。"""
    rows = []
    for serial, vals in values_by_serial.items():
        ms = [{"dimension_id": d, "value": vals[d], "unit": unit}
              for d in dims if d in vals and vals[d] is not None]
        rows.append({"serial": serial, "measurements": ms})
    return rows


# ------------------------------------------------------------- 逐尺寸统计

def test_batch_dimension_stats_and_gaps(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    data = {}
    for i in range(30):
        data[f"W{i:02d}"] = {
            "L1": 50.05 + 0.001 * i,
            "L2": 30.30 - 0.001 * i,
            "L3": 80.00 + 0.002 * i,
        }
    # W03 缺 L2；W10 整条只有 L3（L1/L2 缺测）
    del data["W03"]["L2"]
    data["W10"] = {"L3": 79.95}
    # 重新构造 L1 期望样本（W10 无 L1；L3 始终 30 个）
    l1_expected = np.array([v["L1"] for k, v in data.items()
                            if k != "W10"])
    rows = _rows(data)
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "lot1", "rows": rows,
                          "bootstrap_samples": 4000, "random_seed": 11})
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["frozen"] is True
    rep = j["report"]

    assert rep["sample_summary"] == {
        "rows_total": 30, "complete_rows": 28, "rows_with_gaps": 2,
        "per_dimension_count": {"L1": 29, "L2": 28, "L3": 30},
    }
    gaps = {g["serial"]: g["missing_dimension_ids"] for g in rep["gaps"]}
    assert gaps == {"W03": ["L2"], "W10": ["L1", "L2"]}

    by_id = {d["dimension_id"]: d for d in rep["dimension_results"]}
    l1 = by_id["L1"]
    l1_vals = l1_expected
    assert l1["sample_count"] == 29
    assert l1["mean_mm"] == pytest.approx(l1_vals.mean())
    # 中心 C = 50 + (0.08+0)/2 = 50.04
    assert l1["mean_shift_from_center_mm"] == pytest.approx(
        l1_vals.mean() - 50.04)
    assert l1["mean_shift_from_nominal_mm"] == pytest.approx(
        l1_vals.mean() - 50.0)
    assert l1["sample_std_mm"] == pytest.approx(l1_vals.std(ddof=1))
    assert l1["spec_mm"] == pytest.approx(
        {"lsl": 50.0, "usl": 50.08, "center": 50.04,
         "nominal": 50.0, "half_width": 0.04})
    assert l1["out_of_spec"] is False
    # Cp/Cpk 与 95% CI
    cp_expect = 0.08 / (6 * l1_vals.std(ddof=1))
    assert l1["cp"] == pytest.approx(cp_expect)
    cpk_expect = min((50.08 - l1_vals.mean()) / (3 * l1_vals.std(ddof=1)),
                     (l1_vals.mean() - 50.0) / (3 * l1_vals.std(ddof=1)))
    assert l1["cpk"] == pytest.approx(cpk_expect)
    assert l1["cp_ci95"]["confidence"] == 0.95
    assert l1["cp_ci95"]["lower"] < l1["cp"] < l1["cp_ci95"]["upper"]
    assert l1["cpk_ci95"]["lower"] < l1["cpk"] < l1["cpk_ci95"]["upper"]
    # n=29 < 30 的小样本提示
    assert "n=29" in l1["capability_note"]
    assert any("30" in w for w in l1["warnings"])
    assert any("L1" in w and "30" in w for w in rep["warnings"])


def test_batch_hand_computed_cp_cpk(client, simple_chain_payload):
    """4 个手算点核对 s / Cp / Cpk / 边界合格；4 行齐全 => 封闭环可用。"""
    cid = _make_chain(client, simple_chain_payload)
    vals = [50.02, 50.04, 50.06, 50.08]  # 50.08 恰在 USL 边界，合格
    rows = _rows({f"P{i}": {"L1": v, "L2": 30.3, "L3": 80.0}
                  for i, v in enumerate(vals)})
    rep = client.post(f"/chains/{cid}/inspection-batches",
                      json={"name": "h", "rows": rows,
                            "bootstrap_samples": 2000}).json()["report"]
    l1 = rep["dimension_results"][0]
    assert l1["out_of_spec"] is False
    assert l1["sample_std_mm"] == pytest.approx(math.sqrt(0.002 / 3))
    assert l1["cp"] == pytest.approx(0.08 / (6 * math.sqrt(0.002 / 3)))
    assert l1["cpk"] == pytest.approx(0.03 / (3 * math.sqrt(0.002 / 3)))
    assert rep["closure_analysis"] is not None
    assert rep["closure_analysis"]["complete_rows"] == 4


def test_out_of_spec_flag_and_detail(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = _rows({
        "A": {"L1": 50.09, "L2": 30.3, "L3": 80.0},   # L1 超上差
        "B": {"L1": 49.99, "L2": 30.3, "L3": 80.0},   # L1 低于 LSL=50
        "C": {"L1": 50.04, "L2": 30.4, "L3": 79.80},  # L3 低于 79.85
    })
    rep = client.post(f"/chains/{cid}/inspection-batches",
                      json={"name": "o", "rows": rows,
                            "bootstrap_samples": 2000}).json()["report"]
    by_id = {d["dimension_id"]: d for d in rep["dimension_results"]}
    l1, l3 = by_id["L1"], by_id["L3"]
    assert l1["out_of_spec_count"] == 2
    assert {x["serial"] for x in l1["out_of_spec_serials"]} == {"A", "B"}
    sides = {x["serial"]: x["side"] for x in l1["out_of_spec_serials"]}
    assert sides == {"A": "above_usl", "B": "below_lsl"}
    assert l3["out_of_spec_serials"][0]["serial"] == "C"
    # 原始值+单位在超差明细中保留
    assert l3["out_of_spec_serials"][0]["submitted"] == {
        "value": 79.8, "unit": "mm"}
    assert any("超出规格限" in w for w in rep["warnings"])


# ------------------------------------------------------------- 协方差/封闭环

def test_covariance_and_closure_bootstrap(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rng = np.random.default_rng(0)
    n = 60
    x1 = 50.04 + rng.normal(0, 0.02, n)
    x2 = 30.30 + rng.normal(0, 0.02, n)
    x3 = 80.00 + rng.normal(0, 0.04, n)
    rows = [{"serial": f"W{i}", "measurements": [
        {"dimension_id": "L1", "value": float(x1[i]), "unit": "mm"},
        {"dimension_id": "L2", "value": float(x2[i]), "unit": "mm"},
        {"dimension_id": "L3", "value": float(x3[i]), "unit": "mm"}]}
        for i in range(n)]
    j = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "mc", "rows": rows,
                          "bootstrap_samples": 8000,
                          "random_seed": 42}).json()
    rep = j["report"]
    x = np.column_stack([x1, x2, x3])
    cov = np.array(rep["covariance"]["matrix_mm2"])
    assert rep["covariance"]["available"] is True
    assert rep["covariance"]["n_complete"] == 60
    assert np.allclose(cov, np.cov(x, rowvar=False, ddof=1))
    corr = np.array(rep["covariance"]["correlation_matrix"])
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)

    closure = x1 + x2 - x3
    cl = rep["closure_analysis"]
    assert cl["complete_rows"] == 60
    assert cl["random_seed"] == 42 and cl["bootstrap_samples"] == 8000
    assert cl["observed_mean_mm"] == pytest.approx(closure.mean())
    assert cl["observed_std_mm"] == pytest.approx(closure.std(ddof=1))
    lo, hi = cl["mean_ci95_mm"]
    assert lo < closure.mean() < hi
    assert cl["lower_quantile_mm"] < cl["bootstrap_mean_mm"] < cl["upper_quantile_mm"]
    # 规格 0.05~0.60，本数据封闭环全在界内 => 比例 0，CI 退化为 [0,0]
    assert cl["out_of_spec_proportion"] == 0.0
    assert cl["out_of_spec_proportion_ci95"] == [0.0, 0.0]


def test_closure_out_of_spec_proportion_ci(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    # L3 整体偏大 => 封闭环 x1+x2-x3 偏小，部分工件低于 LSL=0.05
    n = 80
    rows = [{"serial": f"W{i}", "measurements": [
        {"dimension_id": "L1", "value": 50.05, "unit": "mm"},
        {"dimension_id": "L2", "value": 30.30, "unit": "mm"},
        {"dimension_id": "L3", "value": 80.28 + 0.02 * (i % 5), "unit": "mm"}]}
        for i in range(n)]
    rep = client.post(f"/chains/{cid}/inspection-batches",
                      json={"name": "shift", "rows": rows,
                            "bootstrap_samples": 5000}).json()["report"]
    cl = rep["closure_analysis"]
    assert 0.0 < cl["out_of_spec_proportion"] < 1.0
    lo, hi = cl["out_of_spec_proportion_ci95"]
    assert 0.0 <= lo <= cl["out_of_spec_proportion"] <= hi <= 1.0


def test_no_closure_limits_proportion_null(client):
    payload = {
        "name": "no-limits-iqc",
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
    rows = _rows({f"W{i}": {"A": 10.0, "B": 10.0 + 0.001 * i}
                  for i in range(20)}, dims=("A", "B"))
    rr = client.post(f"/chains/{cid}/inspection-batches",
                     json={"name": "x", "rows": rows})
    assert rr.status_code == 201, rr.text
    rep = rr.json()["report"]
    cl = rep["closure_analysis"]
    assert cl is not None
    assert cl["out_of_spec_proportion"] is None
    assert cl["out_of_spec_proportion_ci95"] is None
    assert any("封闭环上下限" in w for w in rep["warnings"])


# ------------------------------------------------- 完整配对不足 / 样本不足

def test_incomplete_pairs_only_dimension_results(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    # 只有 1 个行测齐全部尺寸 => 无法估协方差 / bootstrap
    rows = [
        {"serial": "A", "measurements": [
            {"dimension_id": "L1", "value": 50.04, "unit": "mm"},
            {"dimension_id": "L2", "value": 30.30, "unit": "mm"},
            {"dimension_id": "L3", "value": 80.00, "unit": "mm"}]},
        {"serial": "B", "measurements": [
            {"dimension_id": "L1", "value": 50.05, "unit": "mm"},
            {"dimension_id": "L2", "value": 30.31, "unit": "mm"}]},
        {"serial": "C", "measurements": [
            {"dimension_id": "L3", "value": 79.95, "unit": "mm"}]},
    ]
    j = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "sparse", "rows": rows})
    assert j.status_code == 201, j.text
    rep = j.json()["report"]
    assert rep["closure_analysis"] is None
    assert rep["covariance"]["available"] is False
    assert rep["covariance"]["matrix_mm2"] is None
    reason = rep["closure_unavailable_reason"]
    assert "完整配对不足" in reason and "B" in reason and "C" in reason
    # 单尺寸结果照常给出
    counts = {d["dimension_id"]: d["sample_count"]
              for d in rep["dimension_results"]}
    assert counts == {"L1": 2, "L2": 2, "L3": 2}
    # 基线 RSS / MC 仍然并列，实测栏为 null 并解释
    cmp_ = j.json()["baseline_comparison"]
    assert cmp_["methods"]["observed_bootstrap"] is None
    assert "完整配对不足" in cmp_["observed_unavailable_reason"]
    assert cmp_["methods"]["baseline_rss"]["sigma_mm"] is not None
    assert cmp_["methods"]["baseline_monte_carlo"]["sample_count"] == 50000


def test_n1_and_n0_dimension_insufficient(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = [
        {"serial": "A", "measurements": [
            {"dimension_id": "L1", "value": 50.04, "unit": "mm"}]},
        {"serial": "B", "measurements": [
            {"dimension_id": "L1", "value": 50.05, "unit": "um"}]},
    ]
    rep = client.post(f"/chains/{cid}/inspection-batches",
                      json={"name": "tiny", "rows": rows}).json()["report"]
    by_id = {d["dimension_id"]: d for d in rep["dimension_results"]}
    # 50.05 um = 0.05005 mm => 均值 (50.04+50.05005)/2
    assert by_id["L1"]["mean_mm"] == pytest.approx((50.04 + 0.05005) / 2)
    assert by_id["L2"]["sample_count"] == 0
    assert by_id["L2"]["mean_mm"] is None
    assert "n=0" in by_id["L2"]["capability_note"]
    assert rep["closure_analysis"] is None


# ------------------------------------------------------------- 拒收 422

def test_reject_external_dimension(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = [{"serial": "A", "measurements": [
        {"dimension_id": "L1", "value": 50.0, "unit": "mm"},
        {"dimension_id": "L2", "value": 30.3, "unit": "mm"},
        {"dimension_id": "L3", "value": 80.0, "unit": "mm"},
        {"dimension_id": "X9", "value": 1.0, "unit": "mm"}]}]
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "bad", "rows": rows})
    assert r.status_code == 422
    assert "链外尺寸" in r.text and "X9" in r.text


def test_reject_duplicate_serial(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = _rows({"A": {"L1": 50.0}, "B": {"L1": 50.01}})
    rows[1]["serial"] = "A"
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "bad", "rows": rows})
    assert r.status_code == 422 and "序号重复" in r.text


def test_reject_duplicate_dimension_in_row(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    body = {"name": "bad", "rows": [{"serial": "A", "measurements": [
        {"dimension_id": "L1", "value": 50.0, "unit": "mm"},
        {"dimension_id": "L1", "value": 50.1, "unit": "mm"}]}]}
    r = client.post(f"/chains/{cid}/inspection-batches", json=body)
    assert r.status_code == 422 and "重复测量" in r.text


def test_reject_non_finite_and_unknown_unit(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    for token in ("NaN", "Infinity", "-Infinity"):
        body = (
            '{"name":"bad","rows":[{"serial":"A","measurements":['
            f'{{"dimension_id":"L1","value":{token},"unit":"mm"}}'
            ']}]}'
        )
        r = client.post(f"/chains/{cid}/inspection-batches", content=body,
                        headers={"content-type": "application/json"})
        assert r.status_code == 422, token
        assert "有限数" in r.text
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "bad", "rows": [{"serial": "A", "measurements": [
                        {"dimension_id": "L1", "value": 1, "unit": "foot"}]}]})
    assert r.status_code == 422


def test_reject_empty_batch(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "bad", "rows": []})
    assert r.status_code == 422


# ------------------------------------------------------------- 冻结 / 并列

def test_batch_frozen_list_and_baseline_unchanged(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    before = client.get(f"/chains/{cid}").json()["result"]
    rows = _rows({f"W{i}": {"L1": 50.04, "L2": 30.3, "L3": 80.0 + 0.001 * i}
                  for i in range(25)})
    r = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "lot1", "rows": rows, "random_seed": 99})
    bid = r.json()["batch_id"]
    # 多次读取内容不变
    g1 = client.get(f"/inspection-batches/{bid}").json()
    g2 = client.get(f"/inspection-batches/{bid}").json()
    assert g1["report"] == g2["report"]
    assert g1["rows"][0]["measurements"][0]["unit"] == "mm"
    # 缺测值以 null 原样存回
    rows2 = _rows({"A": {"L1": 50.0, "L2": 30.3},
                   "B": {"L1": 50.0, "L2": 30.3, "L3": None}})
    # 显式 null
    rows2[1]["measurements"].append(
        {"dimension_id": "L3", "value": None, "unit": "mm"})
    r2 = client.post(f"/chains/{cid}/inspection-batches",
                     json={"name": "lot2", "rows": rows2})
    assert r2.status_code == 201, r2.text
    stored = client.get(
        f"/inspection-batches/{r2.json()['batch_id']}").json()["rows"]
    assert stored[1]["measurements"][-1] == {
        "dimension_id": "L3", "value": None, "unit": "mm"}
    # 列表
    lst = client.get(f"/chains/{cid}/inspection-batches").json()["batches"]
    assert [b["name"] for b in lst] == ["lot1", "lot2"]
    assert all(b["frozen"] is True for b in lst)
    # 基线未被检验批次改动
    after = client.get(f"/chains/{cid}").json()["result"]
    assert after == before
    assert client.get("/inspection-batches/9999").status_code == 404


def test_bootstrap_seed_reproducible(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = _rows({f"W{i}": {
        "L1": 50.04 + 0.001 * (i % 7),
        "L2": 30.30 + 0.001 * (i % 5),
        "L3": 80.00 + 0.002 * (i % 3)} for i in range(40)})
    body = {"name": "seed-a", "rows": rows, "bootstrap_samples": 3000,
            "random_seed": 777}
    a = client.post(f"/chains/{cid}/inspection-batches", json=body).json()
    body["name"] = "seed-b"
    b = client.post(f"/chains/{cid}/inspection-batches", json=body).json()
    ca, cb = a["report"]["closure_analysis"], b["report"]["closure_analysis"]
    assert ca["bootstrap_mean_mm"] == cb["bootstrap_mean_mm"]
    assert ca["mean_ci95_mm"] == cb["mean_ci95_mm"]
    assert ca["out_of_spec_proportion_ci95"] == cb["out_of_spec_proportion_ci95"]


def test_baseline_comparison_lists_three_baselines_and_observed(
        client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = _rows({f"W{i}": {"L1": 50.05, "L2": 30.30, "L3": 80.02}
                  for i in range(30)})
    j = client.post(f"/chains/{cid}/inspection-batches",
                    json={"name": "cmp", "rows": rows,
                          "bootstrap_samples": 2000}).json()
    cmp_ = j["baseline_comparison"]
    assert set(cmp_["methods"]) == {
        "baseline_worst_case", "baseline_rss",
        "baseline_monte_carlo", "observed_bootstrap"}
    obs = cmp_["methods"]["observed_bootstrap"]
    assert obs["sample_count"] == 30
    assert obs["bootstrap_samples"] == 2000
    # 样本数 / 剔除原因 / 公式齐备
    assert cmp_["sample_counts"]["baseline_monte_carlo"] == 50000
    assert cmp_["sample_counts"]["complete_workpiece_rows"] == 30
    assert any("ρ" in f or "σ_C" in f
               for f in cmp_["methods"]["baseline_rss"]["formulas"])
    assert any("bootstrap" in f for f in
               j["report"]["closure_analysis"]["formulas"])
    assert "口径不可直接等同" in cmp_["interpretation_note"]
    # 报告级公式列表
    assert any("Cp =" in f for f in j["report"]["formulas"])
    assert any("Bissell" in f for f in j["report"]["formulas"])


def test_mixed_units_dimension_values(client, simple_chain_payload):
    cid = _make_chain(client, simple_chain_payload)
    rows = [{
        "serial": f"W{i}",
        "measurements": [
            {"dimension_id": "L1", "value": 5.004 + 0.0001 * i,
             "unit": "cm"},                       # cm -> ×10
            {"dimension_id": "L2", "value": 30.30, "unit": "mm"},
            {"dimension_id": "L3", "value": 80.00, "unit": "mm"},
        ]} for i in range(20)]
    rep = client.post(f"/chains/{cid}/inspection-batches",
                      json={"name": "u", "rows": rows}).json()["report"]
    # 5.004 cm = 50.04 mm；步长 0.0001 cm = 0.001 mm
    l1 = rep["dimension_results"][0]
    assert l1["mean_mm"] == pytest.approx(50.04 + 0.0095)
    # 尺寸原单位为 mm，规格原单位字段同样给出
    assert l1["spec_in_dimension_unit"]["unit"] == "mm"
    assert l1["spec_in_dimension_unit"]["usl"] == pytest.approx(50.08)
