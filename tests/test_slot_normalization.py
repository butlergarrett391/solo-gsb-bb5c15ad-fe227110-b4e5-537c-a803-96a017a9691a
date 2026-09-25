"""自定义槽位几何规范化回归测试: 乱序提交按编号求解, 重复/缺失/非有限角度拒绝。

仅依赖标准库: python -m unittest discover -s tests
"""
import math
import unittest

from app.balancer import normalize_slots, run_plan


def shuffled_payload(order, **over):
    angle_of = {0: 0.0, 1: 95.0, 2: 180.0, 3: 272.0}
    p = {
        "slot_count": 4,
        "slots": [
            {"index": i, "angle": angle_of[i], "enabled": True}
            for i in order
        ],
        "radius_mm": 100.0,
        "rpm": 3000,
        "restarts": 4,
        "chain_steps": 600,
    }
    p.update(over)
    return p


class TestNormalizeSlots(unittest.TestCase):
    def test_default_generation_when_omitted(self):
        out = normalize_slots(None, 12)
        self.assertEqual([s["index"] for s in out], list(range(12)))
        self.assertEqual([s["angle"] for s in out], [30.0 * i for i in range(12)])
        self.assertTrue(all(s["enabled"] for s in out))

    def test_empty_list_uses_slot_count(self):
        out = normalize_slots([], 6)
        self.assertEqual(len(out), 6)
        self.assertAlmostEqual(out[5]["angle"], 300.0)

    def test_shuffled_input_sorted_by_index(self):
        raw = [
            {"index": 3, "angle": 272}, {"index": 0, "angle": 0},
            {"index": 2, "angle": 180}, {"index": 1, "angle": 95},
        ]
        out = normalize_slots(raw, 4)
        self.assertEqual([s["index"] for s in out], [0, 1, 2, 3])
        self.assertEqual([s["angle"] for s in out], [0.0, 95.0, 180.0, 272.0])

    def test_default_angle_by_slot_number_when_missing_or_null(self):
        # angle 省略或显式置空(null)均按槽号 360·index/n 生成, 与提交顺序无关
        raw = [
            {"index": 2}, {"index": 0, "angle": None},
            {"index": 3}, {"index": 1, "angle": 90},
        ]
        out = normalize_slots(raw, 4)
        self.assertEqual([s["angle"] for s in out], [0.0, 90.0, 180.0, 270.0])

    def test_duplicate_index_rejected(self):
        raw = [{"index": i, "angle": 90.0 * i} for i in range(3)]
        raw.append({"index": 0, "angle": 330.0})
        with self.assertRaises(ValueError):
            normalize_slots(raw, 4)

    def test_missing_index_rejected(self):
        raw = [{"index": i, "angle": 90.0 * i} for i in (0, 1, 3)]
        with self.assertRaises(ValueError):
            normalize_slots(raw, 4)

    def test_out_of_range_index_rejected(self):
        for bad_idx in (-1, 4):
            raw = [{"index": bad_idx, "angle": 0.0}]
            raw += [{"index": i, "angle": 90.0 * i} for i in range(1, 4)]
            with self.assertRaises(ValueError):
                normalize_slots(raw, 4)

    def test_too_many_slots_rejected(self):
        raw = [{"index": i, "angle": 360.0 * i / 129} for i in range(129)]
        with self.assertRaises(ValueError):
            normalize_slots(raw, 129)

    def test_non_finite_angle_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            raw = [{"index": 0, "angle": bad}]
            raw += [{"index": i, "angle": 90.0 * i} for i in range(1, 4)]
            with self.assertRaises(ValueError):
                normalize_slots(raw, 4)


class TestShuffledGeometrySolving(unittest.TestCase):
    def test_locked_tube_uses_named_slot_angle(self):
        # 旧缺陷: 列表下标被当成槽号, 锁定到槽 2 的管按提交位置 2 的角度(95°)计算
        payload = shuffled_payload([3, 0, 2, 1])
        payload["tubes"] = [
            {"id": 1, "mass": 100, "locked": True, "slot": 2},
        ]
        r = run_plan(payload)
        placement = r["placements"][0]
        self.assertEqual(placement["slot"], 2)
        self.assertEqual(placement["angle"], 180.0)
        # U₁ 方向沿槽 2 的实际角度 180°, 模长 100 g × 100 mm
        self.assertAlmostEqual(r["metrics"]["u1_mag_g_mm"], 10000.0, places=6)
        self.assertAlmostEqual(r["metrics"]["u1_angle_deg"], 180.0, places=6)
        by_slot = {s["slot"]: s["angle"] for s in r["slot_state"]}
        self.assertEqual(by_slot, {0: 0.0, 1: 95.0, 2: 180.0, 3: 272.0})

    def test_shuffled_matches_ordered_submission(self):
        def make(order, tubes):
            return shuffled_payload(
                order, tubes=tubes, seed=20260923,
            )

        tubes = [
            {"id": 1, "mass": 50}, {"id": 2, "mass": 50},
            {"id": 3, "mass": 51}, {"id": 4, "mass": 49},
        ]
        orders = ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1])
        results = []
        for order in orders:
            r = run_plan(make(order, tubes))
            results.append(sorted(
                (p["tube_id"], p["slot"], round(p["angle"], 6))
                for p in r["placements"]
            ))
        for res in results[1:]:
            self.assertEqual(res, results[0])

    def test_irregular_angles_shuffled_picks_opposite_pair(self):
        # 0°/180° 是最接近对径的一对, 与提交顺序无关
        for order in ((0, 1, 2, 3), (3, 1, 0, 2)):
            payload = shuffled_payload(order)
            payload["tubes"] = [
                {"id": 1, "mass": 100}, {"id": 2, "mass": 100},
            ]
            r = run_plan(payload)
            slots = sorted(p["slot"] for p in r["placements"])
            self.assertEqual(slots, [0, 2])

    def test_shuffled_request_saved_geometry_is_canonical(self):
        # 求解结果内部按编号取角; 规范化后的槽位列表按下标即槽号
        payload = shuffled_payload([2, 3, 0, 1])
        payload["tubes"] = [{"id": 1, "mass": 100, "locked": True, "slot": 3}]
        r = run_plan(payload)
        self.assertAlmostEqual(r["metrics"]["u1_angle_deg"], 272.0, places=6)
        slot_angles = {s["slot"]: s["angle"] for s in r["slot_state"]}
        self.assertEqual(slot_angles[3], 272.0)

    def test_duplicate_index_raises(self):
        payload = shuffled_payload([0, 1, 2, 3])
        payload["slots"].append({"index": 0, "angle": 10.0})
        payload["tubes"] = [{"id": 1, "mass": 10}]
        with self.assertRaises(ValueError):
            run_plan(payload)

    def test_missing_index_raises(self):
        payload = shuffled_payload([0, 1, 3])
        payload["slot_count"] = 3
        payload["tubes"] = [{"id": 1, "mass": 10}]
        with self.assertRaises(ValueError):
            run_plan(payload)

    def test_non_finite_angle_raises(self):
        payload = shuffled_payload([0, 1, 2, 3])
        payload["slots"][0]["angle"] = float("nan")
        payload["tubes"] = [{"id": 1, "mass": 10}]
        with self.assertRaises(ValueError):
            run_plan(payload)

    def test_mutex_adjacency_follows_actual_angles(self):
        # 非均匀角度乱序提交: 互斥对不得落在按角度环形排序后的物理邻槽
        defs = [(0, 0.0), (1, 95.0), (2, 180.0), (3, 200.0),
                (4, 272.0), (5, 300.0)]
        slots = [{"index": i, "angle": a} for i, a in
                 sorted(defs, key=lambda kv: (kv[1] * 7 + kv[0]) % 13)]
        payload = {
            "slot_count": 6,
            "slots": slots,
            "tubes": [{"id": i, "mass": m}
                      for i, m in zip(range(1, 6), (50, 50, 51, 49, 30))],
            "radius_mm": 100.0, "rpm": 4000,
            "restarts": 4, "chain_steps": 200,
            "mutex_pairs": [{"tube_a": 1, "tube_b": 2}],
        }
        r = run_plan(payload)
        loc = {p["tube_id"]: p["slot"] for p in r["placements"]}
        order = sorted(defs, key=lambda kv: (kv[1] % 360.0, kv[0]))
        adj = {i: set() for i, _ in defs}
        m = len(order)
        for k, (idx, _) in enumerate(order):
            nxt = order[(k + 1) % m][0]
            adj[idx].add(nxt)
            adj[nxt].add(idx)
        self.assertNotIn(loc[2], adj[loc[1]])
        # 返回的相邻关系与角度环一致
        edges = {tuple(sorted((a, b)))
                 for a, nbs in adj.items() for b in nbs}
        self.assertEqual(set(map(tuple, r["exact_search"]["adjacent_slot_pairs"])),
                         edges)
        # 摆放槽号对应的角度与几何一致, 且指标由实际角度复算
        for p in r["placements"]:
            self.assertEqual(p["angle"], dict(defs)[p["slot"]])
        ux = uy = 0.0
        for p in r["placements"]:
            a = math.radians(p["angle"])
            ux += p["mass"] * math.cos(a)
            uy += p["mass"] * math.sin(a)
        self.assertAlmostEqual(
            r["metrics"]["u1_mag_g_mm"], 100.0 * math.hypot(ux, uy), places=6
        )


if __name__ == "__main__":
    unittest.main()
