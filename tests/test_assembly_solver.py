"""装配求解器内部：枚举 + 分支定界字典序最优性（穷举对照）。"""
import random

from app.assembly import enumerate_combos, solve


def _pools(n_per_pool, seed=0, spread=(0.10, 0.20)):
    """合成零件池：每池 n 个实例，全局工件编号 = 100*池+序号。"""
    rng = random.Random(seed)
    pools = []
    for p, n in enumerate(n_per_pool):
        pools.append({
            "name": f"P{p}",
            "dimensions": [f"L{p}"],
            "instances": [
                {"pool": f"P{p}", "batch_id": 1 + (k % 2),
                 "serial": f"{p}-{k}", "key_id": 100 * p + k,
                 "gap_part": rng.uniform(*spread), "u2": 0.0}
                for k in range(n)],
            "excluded": [],
            "batch_ids": [1, 2],
        })
    return pools


def _backfill_key_masks(combos, pools):
    for c in combos:
        mask = 0
        for p, i in enumerate(c["indices"]):
            mask |= 1 << pools[p]["instances"][i]["key_id"]
        c["key_mask"] = mask


def _enum(pools):
    P = len(pools)
    combos, _, truncated = enumerate_combos(
        pools, class_of={q: q for q in range(P)},
        cross_limit=P, forbidden=set(),
        lsl=0.30, usl=0.45, center=0.375,
        k_out=2.0, guard={"mode": "multiple", "multiple": 0.0})
    assert not truncated
    _backfill_key_masks(combos, pools)
    return combos


def _brute_force_optimum(combos, sizes, target):
    """穷举实例 / 工件互不相交合格组合集合的字典序最优值。"""
    accept = [c for c in combos if c["decision"] == "accept"]
    P = len(sizes)
    best = None

    def rec(k, masks, key_mask, picked):
        nonlocal best
        if picked:
            val = (-len(picked),
                   sum(c["center_deviation_mm"] for c in picked),
                   -min(c["guard_margin_mm"] for c in picked),
                   sum(c["cross_transitions"] for c in picked))
            if best is None or val < best:
                best = val
        if len(picked) >= target:
            return
        for j in range(k, len(accept)):
            c = accept[j]
            if all(masks[p] >> c["indices"][p] & 1 for p in range(P)) \
                    and (c["key_mask"] & key_mask) == 0:
                nm = list(masks)
                for p in range(P):
                    nm[p] &= ~(1 << c["indices"][p])
                rec(j + 1, tuple(nm), key_mask | c["key_mask"], picked + [c])

    rec(0, tuple((1 << n) - 1 for n in sizes), 0, [])
    return best if best is not None else (0, 0.0, 0.0, 0)


def _value(picked):
    return (-len(picked),
            sum(c["center_deviation_mm"] for c in picked),
            -min(c["guard_margin_mm"] for c in picked) if picked else 0.0,
            sum(c["cross_transitions"] for c in picked))


def test_solve_matches_brute_force():
    for seed, sizes in enumerate([(3, 3, 3), (4, 3, 2), (5, 4), (4, 4, 3)]):
        pools = _pools(sizes, seed=seed)
        combos = _enum(pools)
        target = min(sizes)
        picked, info = solve(combos, list(sizes), target)
        assert info["exact"]
        assert len(picked) <= target
        assert _value(picked) == _brute_force_optimum(combos, list(sizes), target)


def test_solve_count_outranks_center_deviation():
    """偏差最小的组合与其余组合全冲突时，仍应优先保证装配数量。"""
    pools = _pools((3, 3, 3), seed=1)

    def mk(idx, gap):
        return {"indices": idx, "key_mask": 0, "batches": (1,),
                "gap_mm": gap, "u_mm": 0.0, "guard_band_mm": 0.0,
                "decision": "accept",
                "center_deviation_mm": abs(gap - 0.375),
                "guard_margin_mm": 0.05, "cross_transitions": 0}

    combos = [
        mk((0, 0, 0), 0.375),  # A：偏差 0，与 B/C 各共享一个实例
        mk((0, 1, 1), 0.360),  # B：与 C 实例 / 工件均不交
        mk((1, 0, 2), 0.390),  # C
    ]
    _backfill_key_masks(combos, pools)
    picked, info = solve(combos, [3, 3, 3], 3)
    assert info["exact"]
    assert len(picked) == 2
    assert {picked[0]["indices"], picked[1]["indices"]} == {(0, 1, 1), (1, 0, 2)}


def test_cross_batch_limit_and_same_batch_prune():
    """cross_batch_limit=1 只保留单批组合；同批类强制逐池同批。"""
    pools = _pools((3, 3), seed=2)
    P = 2
    combos, prune, _ = enumerate_combos(
        pools, class_of={0: 0, 1: 0},  # 两池同批等价类
        cross_limit=1, forbidden=set(),
        lsl=-1.0, usl=1.0, center=0.3, k_out=2.0,
        guard={"mode": "multiple", "multiple": 0.0})
    for c in combos:
        batches = {pools[p]["instances"][c["indices"][p]]["batch_id"]
                   for p in range(P)}
        assert len(batches) == 1
    assert prune["cross_batch_limit"] + prune["same_batch_group"] > 0
