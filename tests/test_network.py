"""多闭环公差网络 API 与引擎测试。

覆盖：
* 创建期校验：路径断开 / 方向不衔接 / 未知或重复尺寸 / 不闭合、
  闭环数量 2~20、闭环 id 重复、上下限倒置、相关矩阵引用未知尺寸 /
  重复声明 / 非半正定、正态缺 σ、导入尺寸与追加尺寸 id 冲突；
* 手算核对：闭环系数（路径方向 × 测量方向）、WC 界、RSS σ 与超差率、
  闭环间解析协方差、尺寸×闭环敏感度占比；
* 蒙特卡洛共享抽样：共享尺寸每轮只抽一次，MC 协方差 / 相关与解析一致，
  联合合格率与同时失效组合自洽，固定种子可复现；
* 版本线：从冻结基线链导入（基线不改写）、派生版本独立冻结、
  父版本血缘、重复 GET 内容不变；
* 方案分支搜索：锁定尺寸不调整、σ 策略、按全部闭环达标 / 最高优先级
  余量 / 收紧成本排序、结果冻结。
"""
from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------- 测试数据

# 共享尺寸池（全部 direction=+1，除 C 为 -1 以检验系数换算）
POOL = [
    {"id": "A", "start": "a", "end": "b", "nominal": 10,
     "upper_deviation": 0.04, "lower_deviation": -0.04,
     "distribution": "uniform"},                                  # T=0.04 σ=0.04/√3
    {"id": "B", "start": "b", "end": "c", "nominal": 20,
     "upper_deviation": 0.06, "lower_deviation": -0.06,
     "distribution": "uniform"},                                  # T=0.06 σ=0.06/√3
    {"id": "C", "start": "c", "end": "a", "nominal": 30,
     "upper_deviation": 0.0, "lower_deviation": -0.10,
     "std_dev": 0.02, "distribution": "normal", "direction": -1},  # T=0.05 m=-0.05
    {"id": "D", "start": "c", "end": "d", "nominal": 5,
     "upper_deviation": 0.02, "lower_deviation": 0.0,
     "std_dev": 0.005, "distribution": "normal"},                  # T=0.01 m=+0.01
    {"id": "E", "start": "d", "end": "a", "nominal": 8,
     "upper_deviation": 0.03, "lower_deviation": -0.03,
     "distribution": "triangular"},                                # T=0.03 σ=0.03/√6
]

# GAP1: a→b→c→a，系数 A=+1, B=+1, C=(+1)×(-1)=-1
#   μ1 = 10 + 20 - 29.95 = 0.05，带宽 0.04+0.06+0.05 = 0.15
# GAP2: a→c→d→a（C 反向通过），系数 C=(-1)×(-1)=+1, D=+1, E=+1
#   μ2 = 29.95 + 5.01 + 8 = 42.96，带宽 0.05+0.01+0.03 = 0.09
LOOPS = [
    {"id": "GAP1", "priority": 2, "lower_limit": -0.20, "upper_limit": 0.30,
     "path": [{"dimension_id": "A", "sign": 1},
              {"dimension_id": "B", "sign": 1},
              {"dimension_id": "C", "sign": 1}]},
    {"id": "GAP2", "priority": 1, "lower_limit": 42.50, "upper_limit": 43.50,
     "path": [{"dimension_id": "C", "sign": -1},
              {"dimension_id": "D", "sign": 1},
              {"dimension_id": "E", "sign": 1}]},
]

SIGMA_A2 = 0.04 ** 2 / 3.0
SIGMA_B2 = 0.06 ** 2 / 3.0
SIGMA_C2 = 0.02 ** 2
SIGMA_D2 = 0.005 ** 2
SIGMA_E2 = 0.03 ** 2 / 6.0


def _payload(**over):
    p = {
        "name": "net-test",
        "note": "手算核对网络",
        "dimensions": POOL,
        "loops": LOOPS,
        "correlations": [{"dim_a": "A", "dim_b": "B", "rho": 0.3}],
        "mc_samples": 50000,
        "random_seed": 42,
    }
    p.update(over)
    return p


def _create(client, **over):
    r = client.post("/networks", json=_payload(**over))
    assert r.status_code == 201, r.text
    return r.json()


def _create_chain(client, **over):
    payload = {
        "name": "net-source-chain",
        "closure_lower_limit": -0.2, "closure_upper_limit": 0.3,
        "dimensions": [
            {"id": "A", "start": "a", "end": "b", "nominal": 10,
             "upper_deviation": 0.04, "lower_deviation": -0.04,
             "distribution": "uniform"},
            {"id": "B", "start": "b", "end": "c", "nominal": 20,
             "upper_deviation": 0.06, "lower_deviation": -0.06,
             "distribution": "uniform"},
            {"id": "C", "start": "c", "end": "a", "nominal": 30,
             "upper_deviation": 0.0, "lower_deviation": -0.10,
             "std_dev": 0.02, "distribution": "normal", "direction": -1},
        ],
        "mc_samples": 20000, "random_seed": 7,
    }
    payload.update(over)
    r = client.post("/chains", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------- 手算核对

def test_loop_coefficients_and_worst_case(client):
    j = _create(client)
    loops = {l["loop_id"]: l for l in j["result"]["loops"]}
    g1, g2 = loops["GAP1"], loops["GAP2"]

    coef1 = {p["dimension_id"]: p["coefficient"] for p in g1["path"]}
    coef2 = {p["dimension_id"]: p["coefficient"] for p in g2["path"]}
    assert coef1 == {"A": 1, "B": 1, "C": -1}     # C 测量方向 -1
    assert coef2 == {"C": 1, "D": 1, "E": 1}      # C 反向通过 × 测量方向 -1

    assert g1["worst_case"]["mean_gap_mm"] == pytest.approx(0.05)
    assert g1["worst_case"]["lower_bound_mm"] == pytest.approx(-0.10)
    assert g1["worst_case"]["upper_bound_mm"] == pytest.approx(0.20)
    assert g1["worst_case"]["in_spec"] is True
    assert g2["worst_case"]["mean_gap_mm"] == pytest.approx(42.96)
    assert g2["worst_case"]["lower_bound_mm"] == pytest.approx(42.87)
    assert g2["worst_case"]["upper_bound_mm"] == pytest.approx(43.05)


def test_rss_sigma_and_covariance_hand_calc(client):
    j = _create(client)
    loops = {l["loop_id"]: l for l in j["result"]["loops"]}
    var1 = SIGMA_A2 + SIGMA_B2 + SIGMA_C2 + 2 * 0.3 * math.sqrt(
        SIGMA_A2 * SIGMA_B2)
    var2 = SIGMA_C2 + SIGMA_D2 + SIGMA_E2
    assert loops["GAP1"]["rss"]["sigma_mm"] == pytest.approx(math.sqrt(var1))
    assert loops["GAP2"]["rss"]["sigma_mm"] == pytest.approx(math.sqrt(var2))

    joint = j["result"]["joint"]
    cov = joint["covariance_mm2"]["analytic"]
    assert joint["loop_order"] == ["GAP1", "GAP2"]
    assert cov[0][0] == pytest.approx(var1)
    assert cov[1][1] == pytest.approx(var2)
    # 共享尺寸 C：系数 (-1)×(+1)，协方差 = -σ_C²
    assert cov[0][1] == pytest.approx(-SIGMA_C2)
    assert cov[1][0] == pytest.approx(-SIGMA_C2)
    corr = joint["correlation"]["analytic"]
    assert corr[0][1] == pytest.approx(
        -SIGMA_C2 / math.sqrt(var1 * var2))


def test_monte_carlo_shared_sampling_consistency(client):
    """共享抽样：MC 的闭环 σ / 协方差 / 相关与解析值一致（同一份样本）。"""
    j = _create(client)
    res = j["result"]
    joint = res["joint"]
    var1 = SIGMA_A2 + SIGMA_B2 + SIGMA_C2 + 2 * 0.3 * math.sqrt(
        SIGMA_A2 * SIGMA_B2)
    var2 = SIGMA_C2 + SIGMA_D2 + SIGMA_E2

    mc_cov = joint["covariance_mm2"]["monte_carlo"]
    assert mc_cov[0][0] == pytest.approx(var1, rel=0.03)
    assert mc_cov[1][1] == pytest.approx(var2, rel=0.03)
    assert mc_cov[0][1] == pytest.approx(-SIGMA_C2, abs=2e-5)

    mc_corr = joint["correlation"]["monte_carlo"]
    an_corr = joint["correlation"]["analytic"]
    assert mc_corr[0][1] == pytest.approx(an_corr[0][1], abs=0.02)
    # 共享 C 且系数反号 -> 闭环间负相关
    assert mc_corr[0][1] < 0

    loops = {l["loop_id"]: l for l in res["loops"]}
    assert loops["GAP1"]["monte_carlo"]["sigma_mm"] == pytest.approx(
        math.sqrt(var1), rel=0.03)
    # 联合合格率与同时失效组合自洽
    combos = joint["failure_combinations"]
    combo_rate = sum(c["rate"] for c in combos)
    assert combo_rate == pytest.approx(
        joint["joint_reject_rate_monte_carlo"], abs=1e-9)
    assert (joint["joint_pass_rate_monte_carlo"]
            == pytest.approx(1.0 - joint["joint_reject_rate_monte_carlo"]))
    pairwise = joint["pairwise_joint_reject_monte_carlo"]
    assert pairwise[0][0] == pytest.approx(
        loops["GAP1"]["monte_carlo"]["reject_probability"])
    assert pairwise[1][1] == pytest.approx(
        loops["GAP2"]["monte_carlo"]["reject_probability"])


def test_sensitivity_matrix(client):
    j = _create(client)
    sens = j["result"]["sensitivity_matrix"]
    assert sens["dimension_order"] == ["A", "B", "C", "D", "E"]
    assert sens["loop_order"] == ["GAP1", "GAP2"]
    assert len(sens["entries"]) == 5 * 2
    by_key = {(e["dimension_id"], e["loop_id"]): e for e in sens["entries"]}

    a1 = by_key[("A", "GAP1")]
    assert a1["in_path"] is True and a1["coefficient"] == 1.0
    assert a1["wc_tolerance_share"] == pytest.approx(0.04 / 0.15)
    var1 = SIGMA_A2 + SIGMA_B2 + SIGMA_C2 + 2 * 0.3 * math.sqrt(
        SIGMA_A2 * SIGMA_B2)
    contrib_a = SIGMA_A2 + 0.3 * math.sqrt(SIGMA_A2 * SIGMA_B2)
    assert a1["rss_variance_share"] == pytest.approx(contrib_a / var1)

    # 非路径尺寸恒为 0
    a2 = by_key[("A", "GAP2")]
    assert a2["in_path"] is False
    assert a2["coefficient"] == 0.0
    assert a2["rss_variance_share"] == 0.0
    assert a2["wc_tolerance_share"] == 0.0

    # 每个闭环的 RSS 方差占比之和为 1
    for loop_id in ("GAP1", "GAP2"):
        total = sum(by_key[(d, loop_id)]["rss_variance_share"]
                    for d in ("A", "B", "C", "D", "E"))
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------- 校验拒绝

def test_reject_broken_path(client):
    loops = [
        LOOPS[0],
        {"id": "BAD", "priority": 1, "lower_limit": 0, "upper_limit": 1,
         "path": [{"dimension_id": "A", "sign": 1},
                  {"dimension_id": "D", "sign": 1}]},   # b 与 c/d 不相连
    ]
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422
    assert "路径断开" in r.text


def test_reject_direction_mismatch(client):
    loops = [
        LOOPS[0],
        # E(-1): a->d；D(-1): d->c；此时位于节点 c，
        # C(-1) 声明从节点 a 出发（C 边为 c->a），与当前节点 c 不衔接
        {"id": "BAD", "priority": 1, "lower_limit": 0, "upper_limit": 100,
         "path": [{"dimension_id": "E", "sign": -1},
                  {"dimension_id": "D", "sign": -1},
                  {"dimension_id": "C", "sign": -1}]},
    ]
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422
    assert "方向不衔接" in r.text


def test_reject_unknown_dimension(client):
    loops = [LOOPS[0],
             {"id": "BAD", "priority": 1, "lower_limit": 0, "upper_limit": 1,
              "path": [{"dimension_id": "NOPE", "sign": 1},
                       {"dimension_id": "A", "sign": 1}]}]
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422
    assert "未知尺寸" in r.text


def test_reject_duplicate_dimension_in_path(client):
    loops = [LOOPS[0],
             {"id": "BAD", "priority": 1, "lower_limit": 0, "upper_limit": 1,
              "path": [{"dimension_id": "A", "sign": 1},
                       {"dimension_id": "B", "sign": 1},
                       {"dimension_id": "A", "sign": -1}]}]
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422
    assert "重复" in r.text


def test_reject_not_closed(client):
    loops = [LOOPS[0],
             {"id": "OPEN", "priority": 1, "lower_limit": 0, "upper_limit": 100,
              "path": [{"dimension_id": "A", "sign": 1},
                       {"dimension_id": "B", "sign": 1}]}]  # 终止于 c 未回 a
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422
    assert "不闭合" in r.text


def test_reject_loop_count_bounds(client):
    r1 = client.post("/networks", json=_payload(loops=[LOOPS[0]]))
    assert r1.status_code == 422
    many = [
        {"id": f"L{i}", "priority": 1, "lower_limit": -100, "upper_limit": 100,
         "path": [{"dimension_id": "A", "sign": 1},
                  {"dimension_id": "B", "sign": 1},
                  {"dimension_id": "C", "sign": 1}]}
        for i in range(21)
    ]
    r21 = client.post("/networks", json=_payload(loops=many))
    assert r21.status_code == 422
    ok20 = [
        {"id": f"L{i}", "priority": 1, "lower_limit": -100, "upper_limit": 100,
         "path": [{"dimension_id": "A", "sign": 1},
                  {"dimension_id": "B", "sign": 1},
                  {"dimension_id": "C", "sign": 1}]}
        for i in range(20)
    ]
    r20 = client.post("/networks", json=_payload(loops=ok20))
    assert r20.status_code == 201, r20.text


def test_reject_duplicate_loop_id(client):
    loops = [LOOPS[0], dict(LOOPS[1], id="GAP1")]
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422
    assert "闭环 id 重复" in r.text


def test_reject_inverted_limits(client):
    loops = [LOOPS[0], dict(LOOPS[1], lower_limit=43.5, upper_limit=42.5)]
    r = client.post("/networks", json=_payload(loops=loops))
    assert r.status_code == 422


def test_reject_correlation_problems(client):
    r = client.post("/networks", json=_payload(
        correlations=[{"dim_a": "A", "dim_b": "NOPE", "rho": 0.5}]))
    assert r.status_code == 422 and "未知尺寸" in r.text

    r = client.post("/networks", json=_payload(
        correlations=[{"dim_a": "A", "dim_b": "B", "rho": 0.3},
                      {"dim_a": "B", "dim_b": "A", "rho": 0.3}]))
    assert r.status_code == 422 and "重复" in r.text

    # 非半正定：A-B=0.9, A-D=0.9, B-D=-0.9
    r = client.post("/networks", json=_payload(
        correlations=[{"dim_a": "A", "dim_b": "B", "rho": 0.9},
                      {"dim_a": "A", "dim_b": "D", "rho": 0.9},
                      {"dim_a": "B", "dim_b": "D", "rho": -0.9}]))
    assert r.status_code == 422 and "半正定" in r.text


def test_reject_normal_without_std_dev(client):
    dims = [dict(POOL[0], distribution="normal", std_dev=None)] + POOL[1:]
    r = client.post("/networks", json=_payload(dimensions=dims))
    assert r.status_code == 422
    assert "std_dev" in r.text


def test_reject_no_source_and_no_dimensions(client):
    r = client.post("/networks", json=_payload(dimensions=[]))
    assert r.status_code == 422


# ---------------------------------------------------------------- 基线导入

def test_import_from_baseline_chain(client):
    chain = _create_chain(client)
    cid = chain["chain_id"]
    before = client.get(f"/chains/{cid}").json()["result"]

    r = client.post("/networks", json=_payload(
        name="net-imported",
        source_chain_id=cid,
        dimensions=[
            {"id": "D", "start": "c", "end": "d", "nominal": 5,
             "upper_deviation": 0.02, "lower_deviation": 0.0,
             "std_dev": 0.005, "distribution": "normal"},
            {"id": "E", "start": "d", "end": "a", "nominal": 8,
             "upper_deviation": 0.03, "lower_deviation": -0.03,
             "distribution": "triangular"},
        ],
    ))
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["source"]["chain_id"] == cid
    assert j["source"]["imported_dimension_ids"] == ["A", "B", "C"]
    assert j["source"]["pool_dimension_ids"] == ["A", "B", "C", "D", "E"]
    assert j["baseline_preserved"] is True
    # 导入尺寸的规范化值与基线链一致（A 的半宽 0.04）
    pool = {d["dimension_id"]: d for d in j["snapshot"]["dimensions"]}
    assert pool["A"]["normalized_mm"]["half_width"] == pytest.approx(0.04)
    assert pool["C"]["measurement_direction"] == -1
    # 来源基线不被改写
    after = client.get(f"/chains/{cid}").json()["result"]
    assert after == before


def test_import_missing_chain_404(client):
    r = client.post("/networks", json=_payload(source_chain_id=9999))
    assert r.status_code == 404


def test_import_id_conflict_rejected(client):
    chain = _create_chain(client)
    r = client.post("/networks", json=_payload(
        name="net-conflict",
        source_chain_id=chain["chain_id"],
        dimensions=[dict(POOL[0])],   # A 与导入尺寸冲突
    ))
    assert r.status_code == 422
    assert "冲突" in r.text


# ---------------------------------------------------------------- 版本线

def test_versions_independent_and_frozen(client):
    j1 = _create(client)
    nid, v1 = j1["network_id"], j1["version_id"]

    v2_def = _payload(note="第二版", loops=[
        LOOPS[0],
        dict(LOOPS[1], upper_limit=43.4),
    ])
    v2_def.pop("name")
    r2 = client.post(f"/networks/{nid}/versions", json=v2_def)
    assert r2.status_code == 201, r2.text
    j2 = r2.json()
    assert j2["version_no"] == 2
    assert j2["parent_version_id"] == v1

    # 版本 1 内容不变（冻结）
    g1 = client.get(f"/network-versions/{v1}").json()
    assert g1["result"] == j1["result"]
    assert g1["snapshot"]["loops"][1]["spec_mm"]["upper_limit"] == 43.5
    g2 = client.get(f"/network-versions/{j2['version_id']}").json()
    assert g2["snapshot"]["loops"][1]["spec_mm"]["upper_limit"] == 43.4
    # 重复读取不变
    assert client.get(f"/network-versions/{v1}").json() == g1

    # 列表与详情
    nets = client.get("/networks").json()["networks"]
    mine = [n for n in nets if n["network_id"] == nid][0]
    assert [v["version_no"] for v in mine["versions"]] == [1, 2]
    detail = client.get(f"/networks/{nid}").json()
    assert len(detail["versions"]) == 2


def test_version_parent_must_belong_to_network(client):
    j1 = _create(client)
    other = _create(client, name="net-other")
    r = client.post(
        f"/networks/{j1['network_id']}/versions",
        json={**_payload(), "parent_version_id": other["version_id"]})
    assert r.status_code == 404


def test_network_404s(client):
    assert client.get("/networks/9999").status_code == 404
    assert client.get("/network-versions/9999").status_code == 404
    assert client.get("/network-scenarios/9999").status_code == 404


def test_mixed_units_normalized(client):
    dims = [
        dict(POOL[0], nominal=10000, upper_deviation=4, lower_deviation=-4,
             unit="um"),          # 10 mm ± 0.004 mm
        dict(POOL[1], nominal=2.0, upper_deviation=0.06,
             lower_deviation=-0.06, unit="cm"),   # 20 mm ± 0.6 mm
    ] + POOL[2:]
    j = _create(client, name="net-units", dimensions=dims)
    pool = {d["dimension_id"]: d for d in j["snapshot"]["dimensions"]}
    assert pool["A"]["normalized_mm"]["nominal"] == pytest.approx(10.0)
    assert pool["A"]["normalized_mm"]["half_width"] == pytest.approx(0.004)
    assert pool["A"]["original_representation"]["unit"] == "um"
    assert pool["B"]["normalized_mm"]["nominal"] == pytest.approx(20.0)
    assert pool["B"]["normalized_mm"]["half_width"] == pytest.approx(0.6)
    loops = {l["loop_id"]: l for l in j["result"]["loops"]}
    assert loops["GAP1"]["worst_case"]["mean_gap_mm"] == pytest.approx(0.05)


# ---------------------------------------------------------------- 方案分支

def _failing_network(client):
    """基线联合超差的网络：GAP1 规格偏紧（μ=0.05，σ≈0.051，界 ±0.13），
    收紧到 0.5 级（σ 同步缩放）后 z≈5，20k 样本几乎必然零失效。"""
    tight = dict(LOOPS[0], lower_limit=-0.08, upper_limit=0.18)
    return _create(client, name="net-tight", loops=[tight, LOOPS[1]])


def test_scenario_search_ranking_and_lock(client):
    j = _failing_network(client)
    vid = j["version_id"]
    base_reject = j["result"]["joint"]["joint_reject_rate_monte_carlo"]
    assert base_reject > 0.0

    r = client.post(f"/network-versions/{vid}/scenarios", json={
        "name": "tighten",
        "locked_dimensions": ["E"],
        "scale_levels": [0.5, 0.7, 1.0],
        "scale_normal_sigma": True,
        "tightening_cost": {"A": 2.0},
        "default_cost": 1.0,
        "mc_samples": 20000,
        "random_seed": 11,
    })
    assert r.status_code == 201, r.text
    sj = r.json()
    res = sj["result"]
    assert res["highest_priority_loops"] == ["GAP1"]
    assert res["locked_dimensions"] == ["E"]
    cands = res["candidates"]
    assert len(cands) >= 2

    # 锁定尺寸在所有候选中保持原值
    for cand in cands:
        assert cand["levels"]["E"] == 1.0
        e_row = [b for b in cand["cost_breakdown"]
                 if b["dimension_id"] == "E"][0]
        assert e_row["locked"] is True and e_row["cost"] == 0.0

    # 排序键：达标优先，其次最高优先级余量降序，再次成本升序
    keys = [
        (0 if c["all_loops_pass"] else 1,
         -c["priority_margin_mm"], c["total_cost"])
        for c in cands
    ]
    assert keys == sorted(keys)
    # 存在达标候选且排在未达标之前
    assert cands[0]["all_loops_pass"] is True
    # 收紧降低成本非负且按半宽差计算
    a_row = [b for b in cands[0]["cost_breakdown"]
             if b["dimension_id"] == "A"][0]
    assert a_row["cost"] == pytest.approx(
        2.0 * (a_row["old_half_width_mm"] - a_row["new_half_width_mm"]))
    # 基线（现状）候选始终在列且成本为 0
    base = [c for c in cands if c["is_current_baseline"]]
    assert len(base) == 1 and base[0]["total_cost"] == 0.0
    assert res["baseline"]["joint_reject_rate_monte_carlo"] == pytest.approx(
        base_reject)

    # 结果冻结：重复读取一致
    sid = sj["scenario_id"]
    got = client.get(f"/network-scenarios/{sid}").json()
    assert got["result"] == res
    lst = client.get(f"/network-versions/{vid}/scenarios").json()
    assert any(s["id"] == sid for s in lst["scenarios"])
    # 网络版本结果不被方案改写
    again = client.get(f"/network-versions/{vid}").json()
    assert again["result"]["joint"]["joint_reject_rate_monte_carlo"] == \
        pytest.approx(base_reject)


def test_scenario_search_baseline_kept_beyond_candidate_limit(client):
    """零成本现状候选即使排在截断窗口之外也必须出现在候选列表中。"""
    j = _failing_network(client)
    vid = j["version_id"]
    r = client.post(f"/network-versions/{vid}/scenarios", json={
        "name": "tighten-limited",
        "locked_dimensions": ["E"],
        "scale_levels": [0.5, 0.7, 1.0],
        "scale_normal_sigma": True,
        "tightening_cost": {"A": 2.0},
        "default_cost": 1.0,
        "mc_samples": 20000,
        "random_seed": 11,
        "max_candidates": 1,
    })
    assert r.status_code == 201, r.text
    cands = r.json()["result"]["candidates"]
    base = [c for c in cands if c["is_current_baseline"]]
    assert len(base) == 1 and base[0]["total_cost"] == 0.0
    assert base[0]["included_beyond_limit"] is True


def test_scenario_std_dev_policies(client):
    j = _failing_network(client)
    vid = j["version_id"]
    # std_dev_scale 统一缩放未锁定尺寸 σ 并视为显式
    r = client.post(f"/network-versions/{vid}/scenarios", json={
        "name": "sigma-scale",
        "scale_levels": [1.0],
        "std_dev_scale": 0.5,
        "mc_samples": 10000,
        "random_seed": 3,
    })
    assert r.status_code == 201, r.text
    cands = r.json()["result"]["candidates"]
    non_base = [c for c in cands if not c["is_current_baseline"]]
    assert non_base, "应存在 σ 缩放候选"
    for row in non_base[0]["cost_breakdown"]:
        assert row["sigma_explicit"] is True
    c_row = [b for b in non_base[0]["cost_breakdown"]
             if b["dimension_id"] == "C"][0]
    assert c_row["new_sigma_mm"] == pytest.approx(0.02 * 0.5)

    # scale_normal_sigma：正态 σ 随公差带等比缩放
    r2 = client.post(f"/network-versions/{vid}/scenarios", json={
        "name": "normal-sigma-follow",
        "scale_levels": [0.5, 1.0],
        "scale_normal_sigma": True,
        "mc_samples": 10000,
        "random_seed": 3,
    })
    assert r2.status_code == 201, r2.text
    cand05 = None
    for c in r2.json()["result"]["candidates"]:
        rows = {b["dimension_id"]: b for b in c["cost_breakdown"]}
        if rows["C"]["tolerance_scale"] == 0.5:
            cand05 = rows
            break
    assert cand05 is not None
    assert cand05["C"]["new_sigma_mm"] == pytest.approx(0.02 * 0.5)
    # 均匀分布未显式 σ：随公差带理论重算 σ = T' / √3
    assert cand05["A"]["new_sigma_mm"] == pytest.approx(
        0.04 * 0.5 / math.sqrt(3.0))


def test_scenario_unknown_locked_or_cost_rejected(client):
    j = _failing_network(client)
    vid = j["version_id"]
    r = client.post(f"/network-versions/{vid}/scenarios", json={
        "name": "bad-lock", "locked_dimensions": ["NOPE"]})
    assert r.status_code == 422
    r = client.post(f"/network-versions/{vid}/scenarios", json={
        "name": "bad-cost", "tightening_cost": {"NOPE": 1.0}})
    assert r.status_code == 422


def test_scenario_seed_reproducible(client):
    j = _failing_network(client)
    vid = j["version_id"]
    req = {"name": "rep", "scale_levels": [0.6, 1.0],
           "mc_samples": 10000, "random_seed": 99}
    r1 = client.post(f"/network-versions/{vid}/scenarios", json=req)
    r2 = client.post(f"/network-versions/{vid}/scenarios",
                     json={**req, "name": "rep2"})
    assert r1.status_code == r2.status_code == 201
    c1 = r1.json()["result"]["candidates"]
    c2 = r2.json()["result"]["candidates"]
    assert [c["joint_reject_rate_monte_carlo"] for c in c1] == \
           [c["joint_reject_rate_monte_carlo"] for c in c2]


def test_frozen_version_result_reproducible(client):
    """同一版本重复 GET：联合指标与失效组合完全一致（创建时固化）。"""
    j = _create(client)
    vid = j["version_id"]
    g1 = client.get(f"/network-versions/{vid}").json()
    g2 = client.get(f"/network-versions/{vid}").json()
    assert g1["result"]["joint"]["failure_combinations"] == \
        g2["result"]["joint"]["failure_combinations"]
    assert g1["result"]["joint"]["covariance_mm2"] == \
        g2["result"]["joint"]["covariance_mm2"]
    assert g1["created_at"] == g2["created_at"]
