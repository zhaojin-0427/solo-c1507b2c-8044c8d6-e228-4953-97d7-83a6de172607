"""工序尺寸方案：创建校验、传递矩阵、反算 LP/消元、多解排序与冻结。"""
import numpy as np
import pytest

from app.process_plan import (
    ProcessPlanError,
    build_plan,
    linprog_bounds,
    plan_from_snapshot,
    solve_plan,
    SolveRequest,
)
from app.process_schemas import ProcessPlanCreate


# ------------------------------------------------------------- 单纯形 LP

def test_simplex_bounded_interval():
    # 1 <= x1 + x2 <= 3，x>=1e-9：x1 ∈ [1e-9, 3−1e-9]
    A = np.array([[-1, -1], [1, 1]])
    b = np.array([-1.0, 3.0])
    lb = np.full(2, 1e-9)
    lo = linprog_bounds(2, [1, 0], A, b, lower_bounds=lb)[0][0]
    hi = linprog_bounds(2, [-1, 0], A, b, lower_bounds=lb)[0][0]
    assert lo == pytest.approx(1e-9, abs=1e-6)
    assert hi == pytest.approx(3.0, abs=1e-6)


def test_simplex_infeasible_and_unbounded():
    A = np.array([[-1, -1], [1, 1]])
    assert linprog_bounds(
        2, [1, 0], A, np.array([-5.0, 3.0]),
        lower_bounds=np.full(2, 1e-9))[2] == "infeasible"
    # 仅 x1+x2>=1：x1 无上界
    assert linprog_bounds(
        2, [-1, 0], np.array([[-1, -1]]), np.array([-1.0]),
        lower_bounds=np.full(2, 1e-9))[2] == "unbounded"
    assert linprog_bounds(
        2, [1, 0], np.array([[-1, -1]]), np.array([-1.0]),
        lower_bounds=np.full(2, 1e-9))[2] == "optimal"


def test_simplex_known_optimum():
    # max 3x+4y s.t. x+2y<=14, 3x-y>=0, x-y<=2 -> (6,4), val 34
    A = np.array([[1, 2], [-3, 1], [1, -1]])
    b = np.array([14.0, 0.0, 2.0])
    x, val, st = linprog_bounds(2, [-3, -4], A, b)
    assert st == "optimal"
    assert x == pytest.approx([6.0, 4.0], abs=1e-6)
    assert -val == pytest.approx(34.0, abs=1e-6)


# ------------------------------------------------------------- 方案构建

def _plan_payload(**over):
    p = {
        "name": "shaft-route",
        "origin_surface": "O",
        "design_dimensions": [
            {"id": "D1", "start_surface": "O", "end_surface": "S2",
             "nominal": 50, "upper_deviation": 0.3, "lower_deviation": -0.3}],
        "blank_dimensions": [
            {"id": "B1", "start_surface": "O", "end_surface": "B0",
             "nominal": 60, "upper_deviation": 0.5, "lower_deviation": -0.5}],
        "operations": [
            {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
             "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform", "manufacturing_cost": 10},
            {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
             "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform", "manufacturing_cost": 12}],
        "mc_samples": 20000, "random_seed": 7,
    }
    p.update(over)
    return p


def _design():
    return [{"id": "D1", "start": "O", "end": "S2", "nominal": 50.0,
             "upper_deviation": 0.3, "lower_deviation": -0.3,
             "unit": "mm", "sign": 1, "source": "inline"}]


def _build(**over):
    payload = ProcessPlanCreate(**_plan_payload(**over))
    return build_plan(payload, _design())


def test_transfer_matrix_chain_summation():
    plan = _build()
    p = plan.transfer_matrix()
    # D1 = x_S2 - x_O = OP1 + OP2；毛坯 B1 不相关
    assert p.shape == (1, 3)
    assert p[0].tolist() == [0.0, 1.0, 1.0]


def test_unformed_datum_rejected():
    p = _plan_payload(operations=[
        {"id": "OP1", "datum_surface": "SX", "machined_surface": "S1",
         "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
         "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"}])
    with pytest.raises(ProcessPlanError, match="尚未形成"):
        build_plan(ProcessPlanCreate(**p), _design())


def test_blank_path_broken_rejected():
    p = _plan_payload(blank_dimensions=[
        {"id": "B1", "start_surface": "B9", "end_surface": "B8",
         "nominal": 60}])
    with pytest.raises(ProcessPlanError, match="基准路径断开"):
        build_plan(ProcessPlanCreate(**p), _design())


def test_duplicate_machined_surface_rejected():
    p = _plan_payload(operations=[
        {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
         "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP2", "datum_surface": "O", "machined_surface": "S1",
         "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"}])
    with pytest.raises(ProcessPlanError, match="重复"):
        build_plan(ProcessPlanCreate(**p), _design())


def test_rank_deficient_operation_rejected_with_dof():
    p = _plan_payload(operations=[
        {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
         "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
         "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP3", "datum_surface": "S2", "machined_surface": "S3",
         "nominal": 5, "upper_deviation": 0.05, "lower_deviation": -0.05,
         "distribution": "uniform"}])
    with pytest.raises(ProcessPlanError, match="秩不足") as exc:
        build_plan(ProcessPlanCreate(**p), _design())
    assert "OP3" in str(exc.value) and "1 个自由度" in str(exc.value)


def test_stock_closure_covers_extra_operation():
    plan = _build(
        operations=[
            {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
             "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform"},
            {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
             "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform"},
            {"id": "OP3", "datum_surface": "S2", "machined_surface": "S3",
             "nominal": 5, "upper_deviation": 0.05, "lower_deviation": -0.05,
             "distribution": "uniform"}],
        stock_closures=[
            {"id": "ST1", "blank_surface": "O", "machined_surface": "S3",
             "min_allowance": 1.0}])
    # ST1 = OP1+OP2+OP3 覆盖 OP3
    rows = {cl.id: cl.p_edge for cl in plan.closures if cl.id == "ST1"}
    assert set(rows["ST1"]) == {"OP1", "OP2", "OP3"}


# ------------------------------------------------------------- 反算求解

def _solve_req(locked=None, grades=("fine", 0.06)):
    gid, f = grades
    return SolveRequest(
        locked=locked or {}, grades={gid: f},
        grade_costs={gid: 0.0}, default_grade=gid,
        edge_grades={}, edge_grade_list={}, step_sizes=[1.0],
        edge_steps={}, mc_samples=20000, seed=7, max_candidates=10)


def test_nominal_ranges_bounded():
    plan = _build()
    res = solve_plan(plan, _solve_req())
    assert res["status"] == "optimal"
    rng = {r["operation_id"]: r for r in res["nominal_ranges"]}
    # 49.7 <= OP1+OP2 <= 50.3
    assert rng["OP1"]["nominal_lower_mm"] == pytest.approx(0.0, abs=1e-6)
    assert rng["OP1"]["nominal_upper_mm"] == pytest.approx(50.3, abs=1e-6)
    assert all(r["bounded"] for r in res["nominal_ranges"])


def test_elimination_free_and_pivot_edges():
    plan = _build()
    res = solve_plan(plan, _solve_req())
    elim = res["elimination"]
    # 1 个设计闭环 -> 1 主元（OP1/OP2 之一），另一为自由度，毛坯为已知
    assert elim["rank"] == 1
    assert len(elim["pivot_edges"]) == 1
    assert set(elim["free_edges"]) <= {"OP1", "OP2"}
    assert "B1" in elim["blank_edges"]
    assert elim["expressions"]


def test_locked_makes_other_determined():
    plan = _build()
    # 锁定 OP1=30，OP2 由 D1=50 居中唯一确定（~20）
    res = solve_plan(plan, _solve_req(locked={"OP1": 30.0}))
    assert res["status"] == "optimal"
    op2 = [o for o in res["candidates"][0]["operations"]
           if o["operation_id"] == "OP2"][0]
    assert op2["nominal_mm"] == pytest.approx(20.0, abs=1.0)


def test_underconstrained_after_lock_rejected_with_dof():
    # OP3 只被单边余量闭环覆盖（x_S3>=1），名义无上界 -> 秩不足
    plan = _build(
        operations=[
            {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
             "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform"},
            {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
             "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform"},
            {"id": "OP3", "datum_surface": "S2", "machined_surface": "S3",
             "nominal": 5, "upper_deviation": 0.05, "lower_deviation": -0.05,
             "distribution": "uniform"}],
        stock_closures=[
            {"id": "ST1", "blank_surface": "O", "machined_surface": "S3",
             "min_allowance": 1.0}])
    with pytest.raises(ProcessPlanError, match="秩不足") as exc:
        solve_plan(plan, _solve_req(locked={"OP1": 30.0}))
    assert "OP3" in str(exc.value)


def test_candidates_sorted_and_closures_in_spec():
    plan = _build()
    req = SolveRequest(
        locked={}, grades={"fine": 0.06, "coarse": 0.12},
        grade_costs={"fine": 8.0, "coarse": 2.0}, default_grade="coarse",
        edge_grades={}, edge_grade_list={}, step_sizes=[1.0],
        edge_steps={}, mc_samples=20000, seed=7, max_candidates=10)
    res = solve_plan(plan, req)
    cands = res["candidates"]
    assert cands
    # 排序键单调：达标数降序、最差余量降序、改动量升序、成本升序
    keys = [
        (-c["design_closures_in_spec_wc"], -c["worst_margin_mm"],
         c["nominal_change_total_mm"], c["manufacturing_cost_total"])
        for c in cands]
    assert keys == sorted(keys)
    assert all(c["design_closures_in_spec_wc"]
               == c["design_closures_total"] for c in cands)


def test_rss_matches_monte_carlo_sigma():
    plan = _build()
    res = solve_plan(plan, _solve_req())
    detail = res["candidates"][0]["closure_detail"][0]
    assert detail["rss"]["sigma_mm"] == pytest.approx(
        detail["monte_carlo"]["sigma_mm"], rel=0.08)


def test_all_locked_single_candidate():
    plan = _build()
    res = solve_plan(plan, _solve_req(locked={"OP1": 30.0, "OP2": 20.0}))
    assert len(res["candidates"]) == 1
    assert res["candidates"][0]["all_locked"] is True


def test_snapshot_roundtrip_preserves_matrix():
    plan = _build()
    p0 = plan.transfer_matrix()
    from app.process_plan import plan_snapshot
    snap = plan_snapshot(plan, {"type": "inline"})
    rebuilt = plan_from_snapshot(snap)
    assert np.allclose(rebuilt.transfer_matrix(), p0)


# ------------------------------------------------------------- API

def test_api_create_and_get_frozen(client):
    r = client.post("/process-plans", json=_plan_payload())
    assert r.status_code == 201, r.text
    vid = r.json()["version_id"]
    assert r.json()["transfer_matrix"] == [[0.0, 1.0, 1.0]]
    got = client.get(f"/process-plan-versions/{vid}")
    assert got.status_code == 200
    # 快照冻结：重复读取一致
    assert got.json()["snapshot"] == r.json()["snapshot"]


def test_api_solve_select_freeze(client):
    r = client.post("/process-plans", json=_plan_payload())
    vid = r.json()["version_id"]
    solve = {
        "name": "s1",
        "capability_tiers": [
            {"id": "fine", "grade_factor": 0.06, "setup_cost": 8},
            {"id": "coarse", "grade_factor": 0.12, "setup_cost": 2}],
        "standard_step_sizes": [1.0], "mc_samples": 20000,
        "random_seed": 7}
    r = client.post(f"/process-plan-versions/{vid}/solve", json=solve)
    assert r.status_code == 201, r.text
    sid = r.json()["solution_id"]
    res = r.json()["result"]
    assert res["candidates"]

    sel = client.post(f"/process-solutions/{sid}/select", json={"rank": 1})
    assert sel.status_code == 200, sel.text
    assert sel.json()["frozen"] is True
    # 重复选定 / 改选拒绝
    again = client.post(f"/process-solutions/{sid}/select", json={"rank": 2})
    assert again.status_code == 422
    # 冻结读取不变
    got = client.get(f"/process-solutions/{sid}").json()
    assert got["frozen"] is True and got["selected_rank"] == 1


def test_api_structural_rejections(client):
    # 工序引用尚未形成的表面
    bad = _plan_payload(name="bad-unformed", operations=[
        {"id": "OP1", "datum_surface": "SX", "machined_surface": "S1",
         "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
         "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"}])
    assert client.post("/process-plans", json=bad).status_code == 422

    # 秩不足：OP3 不被任何闭环覆盖
    bad = _plan_payload(name="bad-rank", operations=[
        {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
         "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
         "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
         "distribution": "uniform"},
        {"id": "OP3", "datum_surface": "S2", "machined_surface": "S3",
         "nominal": 5, "upper_deviation": 0.05, "lower_deviation": -0.05,
         "distribution": "uniform"}])
    r = client.post("/process-plans", json=bad)
    assert r.status_code == 422 and "自由度" in str(r.json()["detail"])


def test_api_source_chain_update_keeps_history(client):
    """来源基线链后续创建不影响已冻结工序方案的设计快照。"""
    chain = {
        "name": "dsn-pp", "closure_lower_limit": -0.3,
        "closure_upper_limit": 0.3,
        "dimensions": [
            {"id": "L1", "start": "O", "end": "S1", "nominal": 30,
             "upper_deviation": 0.15, "lower_deviation": -0.15,
             "distribution": "uniform"},
            {"id": "L2", "start": "S1", "end": "S2", "nominal": 20,
             "upper_deviation": 0.15, "lower_deviation": -0.15,
             "distribution": "uniform"},
            {"id": "L3", "start": "S2", "end": "O", "nominal": 50,
             "upper_deviation": 0.3, "lower_deviation": 0.0,
             "distribution": "uniform", "direction": -1}],
        "mc_samples": 5000, "random_seed": 1}
    cid = client.post("/chains", json=chain).json()["chain_id"]
    plan = {
        "name": "from-chain-pp", "origin_surface": "O",
        "source_chain_id": cid,
        "operations": [
            {"id": "OP1", "datum_surface": "O", "machined_surface": "S1",
             "nominal": 30, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform", "manufacturing_cost": 10},
            {"id": "OP2", "datum_surface": "S1", "machined_surface": "S2",
             "nominal": 20, "upper_deviation": 0.1, "lower_deviation": -0.1,
             "distribution": "uniform", "manufacturing_cost": 12}],
        "mc_samples": 5000}
    r = client.post("/process-plans", json=plan)
    assert r.status_code == 201, r.text
    snap_ids = [d["id"] for d in
                r.json()["snapshot"]["design_chain_snapshot"]["dimensions"]]
    assert set(snap_ids) == {"L1", "L2"}
    assert r.json()["snapshot"]["source"]["dropped_closure_edges"] == ["L3"]
