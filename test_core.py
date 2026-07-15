"""Tests for milkyEdgeFlowTools pure-logic core (core.py).

core.py has no bpy dependency; run with any Python:
  python test_core.py

Canon TDD test list
-------------------
Chain decomposition:
  [x] open chain is ordered end-to-end
  [x] closed loop is detected
  [x] branched component is skipped and counted
  [x] multiple components are returned separately
Curve (centripetal Catmull-Rom, arc-length parameterized):
  [x] passes through knots (open)
  [x] point_at(0) / point_at(total_length) are the endpoints
  [x] closed curve is periodic
  [x] knot params are strictly increasing
  [x] closest param to a ray hitting a straight curve
  [x] ray pointing away falls back to closest approach at ray origin
Flow extrapolation:
  [x] direction from a straight incoming ring
  [x] insufficient ring points -> None
  [x] target delta on a straight curve
  [x] target delta is clamped to clamp_span
  [x] side blend mixes major (more rings) and minor side
Smoothing solver:
  [x] all pinned -> zero displacement
  [x] uniform targets, no pins -> targets reached
  [x] pin influence decays with distance
  [x] missing targets are interpolated smoothly
  [x] closed chain: pin influence wraps symmetrically
Ordering:
  [x] min spacing enforcement keeps sequence monotone
Chain application order (order_chains):
  [x] chains seen on the dominant side are applied first
  [x] cyclic dependencies fall back to stable input order
  [x] independent chains keep their input order
Integration (relax_chain):
  [x] straight grid with shifted crossing flows slides vertices onto the flows
  [x] pinned vertex stays, factor blends result
  [x] relax_chain_step on a fixed curve matches relax_chain
  [x] relax_chain_step is idempotent once converged
  [x] extra iterations do not change an already-converged result
Constraint propagation (v0.5.0):
  [x] no pins -> zero deltas; influence 0 -> pinned rows only
  [x] a pinned displacement decays monotonically into neighbors
  [x] higher influence spreads further
  [x] propagate_flow_constraints displaces neighboring flows per rail
Regeneration core (M1):
  [x] order_rails orders a path from unordered adjacency; rejects branches
  [x] common_sample_ratios: straight rails -> uniform; bias 0 -> uniform
  [x] common_sample_ratios: bias 1 packs samples around a corner
  [x] smooth_flow_on_rails straightens a kinked flow; pins hold
  [x] generate_flows builds a uniform grid; locked ratios and vertex
      constraints are honored
"""

import math
import unittest

import core
from core import (
    CatmullRomCurve,
    common_sample_ratios,
    compute_vertex_target,
    decompose_chains,
    enforce_min_spacing,
    flow_direction,
    generate_flows,
    order_chains,
    order_rails,
    propagate_deltas,
    propagate_flow_constraints,
    relax_chain,
    relax_chain_step,
    smooth_flow_on_rails,
    solve_relaxed_params,
)


def vlen(a, b):
    return math.dist(a, b)


class TestDecomposeChains(unittest.TestCase):
    def test_open_chain_ordered(self):
        chains, skipped = decompose_chains([(0, 1), (1, 2), (2, 3)])
        self.assertEqual(skipped, 0)
        self.assertEqual(len(chains), 1)
        verts, closed = chains[0]
        self.assertFalse(closed)
        self.assertIn(verts, ([0, 1, 2, 3], [3, 2, 1, 0]))

    def test_closed_loop(self):
        chains, skipped = decompose_chains([(0, 1), (1, 2), (2, 0)])
        self.assertEqual(skipped, 0)
        self.assertEqual(len(chains), 1)
        verts, closed = chains[0]
        self.assertTrue(closed)
        self.assertEqual(set(verts), {0, 1, 2})
        self.assertEqual(len(verts), 3)

    def test_branch_skipped(self):
        chains, skipped = decompose_chains([(0, 1), (0, 2), (0, 3)])
        self.assertEqual(chains, [])
        self.assertEqual(skipped, 1)

    def test_multiple_components(self):
        chains, skipped = decompose_chains(
            [(0, 1), (1, 2), (10, 11), (11, 12), (12, 10)])
        self.assertEqual(skipped, 0)
        self.assertEqual(len(chains), 2)
        kinds = sorted(closed for _, closed in chains)
        self.assertEqual(kinds, [False, True])


def straight_curve():
    pts = [(0, 0, 0), (2.5, 0, 0), (5, 0, 0), (7.5, 0, 0), (10, 0, 0)]
    return CatmullRomCurve(pts, closed=False), pts


class TestCatmullRomCurve(unittest.TestCase):
    def test_passes_through_knots_open(self):
        pts = [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, -1, 0)]
        curve = CatmullRomCurve(pts, closed=False)
        for i, p in enumerate(pts):
            got = curve.point_at(curve.knot_params[i])
            self.assertLess(vlen(got, p), 1e-6)

    def test_endpoints(self):
        curve, pts = straight_curve()
        self.assertLess(vlen(curve.point_at(0.0), pts[0]), 1e-6)
        self.assertLess(vlen(curve.point_at(curve.total_length), pts[-1]),
                        1e-6)

    def test_closed_periodic(self):
        pts = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
        curve = CatmullRomCurve(pts, closed=True)
        a = curve.point_at(0.0)
        b = curve.point_at(curve.total_length)
        self.assertLess(vlen(a, b), 1e-6)

    def test_knot_params_increasing(self):
        pts = [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, -1, 0)]
        curve = CatmullRomCurve(pts, closed=False)
        for a, b in zip(curve.knot_params, curve.knot_params[1:]):
            self.assertLess(a, b)

    def test_closest_param_to_point_extension(self):
        curve, _ = straight_curve()  # along X, length 10
        s, _d = curve.closest_param_to_point((12.0, 1.0, 0.0))
        self.assertAlmostEqual(s, 10.0, delta=1e-6)
        s, d = curve.closest_param_to_point((12.0, 1.0, 0.0), extend=5.0)
        self.assertAlmostEqual(s, 12.0, delta=0.05)
        self.assertAlmostEqual(d, 1.0, delta=0.01)
        s, _d = curve.closest_param_to_point((-3.0, 0.5, 0.0), extend=5.0)
        self.assertAlmostEqual(s, -3.0, delta=0.05)

    def test_closest_param_to_ray_hit(self):
        curve, _ = straight_curve()
        s, dist = curve.closest_param_to_ray((3, 5, 0), (0, -1, 0))
        self.assertAlmostEqual(s, 3.0, delta=0.05)
        self.assertLess(dist, 0.05)

    def test_closest_param_to_ray_pointing_away(self):
        curve, _ = straight_curve()
        s, dist = curve.closest_param_to_ray((3, 5, 0), (0, 1, 0))
        self.assertAlmostEqual(s, 3.0, delta=0.05)
        self.assertAlmostEqual(dist, 5.0, delta=0.05)


class TestFlowExtrapolation(unittest.TestCase):
    def test_direction_straight_ring(self):
        d = flow_direction([(1, 0, 0), (2, 0, 0), (3, 0, 0)])
        self.assertLess(vlen(d, (-1, 0, 0)), 1e-6)

    def test_direction_insufficient(self):
        self.assertIsNone(flow_direction([(1, 0, 0)]))
        self.assertIsNone(flow_direction([]))

    def test_target_straight(self):
        curve, _ = straight_curve()
        sides = [[(3, 1, 0), (3, 2, 0)]]
        delta = compute_vertex_target(curve, sides, s_orig=5.0,
                                      side_blend=0.0, clamp_span=100.0)
        self.assertAlmostEqual(delta, -2.0, delta=0.05)

    def test_target_clamped(self):
        curve, _ = straight_curve()
        sides = [[(3, 1, 0), (3, 2, 0)]]
        delta = compute_vertex_target(curve, sides, s_orig=5.0,
                                      side_blend=0.0, clamp_span=1.0)
        self.assertAlmostEqual(delta, -1.0, delta=1e-6)

    def test_side_blend(self):
        curve, _ = straight_curve()
        major = [(3, 1, 0), (3, 2, 0), (3, 3, 0)]   # 3 rings -> delta -2
        minor = [(6, -1, 0), (6, -2, 0)]            # 2 rings -> delta +1
        for blend, expect in ((0.0, -2.0), (1.0, 1.0), (0.5, -0.5)):
            delta = compute_vertex_target(curve, [minor, major], s_orig=5.0,
                                          side_blend=blend, clamp_span=100.0)
            self.assertAlmostEqual(delta, expect, delta=0.1)

    def test_no_sides(self):
        curve, _ = straight_curve()
        self.assertIsNone(compute_vertex_target(curve, [], 5.0, 0.0, 100.0))


class TestSolver(unittest.TestCase):
    def test_all_pinned(self):
        out = solve_relaxed_params([2.0] * 5, [True] * 5,
                                   stiffness=1.0, closed=False)
        for d in out:
            self.assertAlmostEqual(d, 0.0, delta=1e-4)

    def test_uniform_targets_no_pins(self):
        out = solve_relaxed_params([2.0] * 5, [False] * 5,
                                   stiffness=1.0, closed=False)
        for d in out:
            self.assertAlmostEqual(d, 2.0, delta=1e-4)

    def test_pin_influence_decays(self):
        n = 11
        out = solve_relaxed_params([1.0] * n, [True] + [False] * (n - 1),
                                   stiffness=4.0, closed=False)
        self.assertAlmostEqual(out[0], 0.0, delta=1e-3)
        self.assertLess(out[1], out[5])
        self.assertLess(out[5], out[10])
        self.assertLess(out[10], 1.0 + 1e-6)

    def test_interpolates_missing(self):
        targets = [0.0] + [None] * 9 + [10.0]
        out = solve_relaxed_params(targets, [False] * 11,
                                   stiffness=1.0, closed=False)
        self.assertAlmostEqual(out[5], 5.0, delta=1.0)
        for a, b in zip(out, out[1:]):
            self.assertLessEqual(a, b + 1e-6)

    def test_closed_wraps_symmetrically(self):
        n = 8
        out = solve_relaxed_params([1.0] * n, [True] + [False] * (n - 1),
                                   stiffness=2.0, closed=True)
        self.assertAlmostEqual(out[0], 0.0, delta=1e-3)
        self.assertAlmostEqual(out[1], out[7], delta=1e-6)
        self.assertAlmostEqual(out[2], out[6], delta=1e-6)


class TestOrdering(unittest.TestCase):
    def test_min_spacing_open(self):
        s = enforce_min_spacing([0.0, 2.0, 1.9, 5.0], closed=False,
                                total_length=10.0, min_gap=0.05)
        for a, b in zip(s, s[1:]):
            self.assertGreaterEqual(b - a, 0.05 - 1e-9)
        self.assertAlmostEqual(s[0], 0.0)
        self.assertAlmostEqual(s[3], 5.0)


class TestOrderChains(unittest.TestCase):
    def test_dominant_side_chains_first(self):
        # Chain 1 sees chain 0 on its dominant side: 0 must be applied first.
        self.assertEqual(order_chains([set(), {0}]), [0, 1])
        # Chain 0 sees chain 1: 1 first.
        self.assertEqual(order_chains([{1}, set()]), [1, 0])

    def test_cycle_falls_back_to_stable_order(self):
        self.assertEqual(order_chains([{1}, {0}]), [0, 1])

    def test_independent_chains_keep_input_order(self):
        self.assertEqual(order_chains([set(), set(), {1}]), [0, 1, 2])
        self.assertEqual(order_chains([]), [])


class TestRelaxChain(unittest.TestCase):
    def make_inputs(self):
        n = 7
        points = [(float(x), 0.0, 0.0) for x in range(n)]
        # Crossing flows come straight down at x = i + 0.5 for vertices 0..5;
        # the last vertex has no crossing flow.
        sides = []
        for i in range(n - 1):
            x = i + 0.5
            sides.append([[(x, 1.0, 0.0), (x, 2.0, 0.0)]])
        sides.append([])
        return points, sides

    def test_slides_onto_flows(self):
        points, sides = self.make_inputs()
        out = relax_chain(points, closed=False, sides=sides,
                          pinned=[False] * 7, side_blend=0.0,
                          stiffness=1.0, factor=1.0)
        for i in range(1, 5):
            self.assertAlmostEqual(out[i][0], i + 0.5, delta=0.15)
            self.assertAlmostEqual(out[i][1], 0.0, delta=1e-6)

    def test_pin_and_factor(self):
        points, sides = self.make_inputs()
        pinned = [False] * 7
        pinned[3] = True
        out = relax_chain(points, closed=False, sides=sides, pinned=pinned,
                          side_blend=0.0, stiffness=1.0, factor=1.0)
        self.assertAlmostEqual(out[3][0], 3.0, delta=1e-3)

        half = relax_chain(points, closed=False, sides=sides,
                           pinned=[False] * 7, side_blend=0.0,
                           stiffness=1.0, factor=0.5)
        full = relax_chain(points, closed=False, sides=sides,
                           pinned=[False] * 7, side_blend=0.0,
                           stiffness=1.0, factor=1.0)
        for h, f, p in zip(half, full, points):
            self.assertAlmostEqual(h[0], (f[0] + p[0]) / 2.0, delta=1e-6)

    def test_step_matches_relax_chain(self):
        points, sides = self.make_inputs()
        curve = CatmullRomCurve(points, closed=False)
        params = relax_chain_step(curve, list(curve.knot_params), sides,
                                  [False] * 7, side_blend=0.0, stiffness=1.0)
        full = relax_chain(points, closed=False, sides=sides,
                           pinned=[False] * 7, side_blend=0.0,
                           stiffness=1.0, factor=1.0)
        for s, expected in zip(params, full):
            self.assertLess(vlen(curve.point_at(s), expected), 1e-9)

    def test_step_idempotent_once_converged(self):
        points, sides = self.make_inputs()
        curve = CatmullRomCurve(points, closed=False)
        params = relax_chain_step(curve, list(curve.knot_params), sides,
                                  [False] * 7, side_blend=0.0, stiffness=1.0)
        again = relax_chain_step(curve, params, sides,
                                 [False] * 7, side_blend=0.0, stiffness=1.0)
        for a, b in zip(params, again):
            self.assertAlmostEqual(a, b, delta=1e-3)

    def test_iterations_stable_when_converged(self):
        points, sides = self.make_inputs()
        once = relax_chain(points, closed=False, sides=sides,
                           pinned=[False] * 7, side_blend=0.0,
                           stiffness=1.0, factor=1.0)
        many = relax_chain(points, closed=False, sides=sides,
                           pinned=[False] * 7, side_blend=0.0,
                           stiffness=1.0, factor=1.0, iterations=10)
        for a, b in zip(once, many):
            self.assertLess(vlen(a, b), 1e-3)


def straight_rails(count=3, spacing=1.0, length=4):
    """Vertical straight rails in the XZ..: x = i*spacing, y in [0, length]."""
    rails = []
    for i in range(count):
        pts = [(i * spacing, float(y), 0.0) for y in range(length + 1)]
        rails.append(CatmullRomCurve(pts, closed=False))
    return rails


class TestOrderRails(unittest.TestCase):
    def test_orders_a_path(self):
        order = order_rails(3, [(2, 1), (0, 2)])
        self.assertIn(order, ([0, 2, 1], [1, 2, 0]))

    def test_rejects_branch(self):
        self.assertIsNone(order_rails(4, [(0, 1), (0, 2), (0, 3)]))

    def test_rejects_disconnected(self):
        self.assertIsNone(order_rails(4, [(0, 1), (2, 3)]))

    def test_trivial(self):
        self.assertEqual(order_rails(1, []), [0])
        self.assertIn(order_rails(2, [(0, 1)]), ([0, 1], [1, 0]))


class TestSampleRatios(unittest.TestCase):
    def test_straight_rails_uniform(self):
        rails = straight_rails(2)
        ratios = common_sample_ratios([rails[0], rails[1]], 5, bias=0.5)
        self.assertEqual(len(ratios), 5)
        for got, want in zip(ratios, [0.0, 0.25, 0.5, 0.75, 1.0]):
            self.assertAlmostEqual(got, want, delta=0.02)

    def test_bias_zero_uniform_even_when_curved(self):
        bent = CatmullRomCurve([(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                (2, 1, 0), (2, 2, 0)], closed=False)
        ratios = common_sample_ratios([bent, bent], 5, bias=0.0)
        for got, want in zip(ratios, [0.0, 0.25, 0.5, 0.75, 1.0]):
            self.assertAlmostEqual(got, want, delta=0.02)

    def test_full_bias_packs_around_corner(self):
        # L-shaped rail: the corner sits near the middle of the arc length,
        # so with bias 1 the gaps around the middle must shrink well below
        # the uniform gap.
        bent = CatmullRomCurve([(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                (2, 1, 0), (2, 2, 0)], closed=False)
        ratios = common_sample_ratios([bent, bent], 7, bias=1.0)
        self.assertAlmostEqual(ratios[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(ratios[-1], 1.0, delta=1e-9)
        for a, b in zip(ratios, ratios[1:]):
            self.assertGreater(b, a)
        uniform_gap = 1.0 / 6.0
        mid_gaps = [b - a for a, b in zip(ratios, ratios[1:])
                    if a > 0.25 and b < 0.75]
        self.assertTrue(mid_gaps)
        self.assertLess(min(mid_gaps), uniform_gap * 0.8)


class TestFlowSmoothing(unittest.TestCase):
    def test_straightens_kinked_flow(self):
        rails = straight_rails(3)
        # Endpoints at y=1 (left) and y=3 (right); middle starts at y=0.
        params = smooth_flow_on_rails(rails, [1.0, 0.0, 3.0],
                                      [True, False, True])
        self.assertAlmostEqual(params[1], 2.0, delta=0.05)

    def test_pinned_middle_stays(self):
        rails = straight_rails(3)
        params = smooth_flow_on_rails(rails, [1.0, 0.0, 3.0],
                                      [True, True, True])
        self.assertAlmostEqual(params[1], 0.0, delta=1e-9)


class TestPropagation(unittest.TestCase):
    def test_no_pins_and_zero_influence(self):
        self.assertEqual(propagate_deltas(5, {}, 2.0), [0.0] * 5)
        out = propagate_deltas(5, {2: 1.0}, 0.0)
        self.assertEqual(out[2], 1.0)
        for i in (0, 1, 3, 4):
            self.assertAlmostEqual(out[i], 0.0, delta=1e-9)

    def test_decays_monotonically(self):
        out = propagate_deltas(7, {3: 1.0}, 2.0)
        self.assertAlmostEqual(out[3], 1.0, delta=1e-9)
        self.assertGreater(out[2], out[1])
        self.assertGreater(out[1], out[0])
        self.assertGreater(out[0], -1e-9)
        self.assertAlmostEqual(out[2], out[4], delta=1e-9)  # symmetry

    def test_higher_influence_spreads_further(self):
        near = propagate_deltas(9, {4: 1.0}, 1.0)
        far = propagate_deltas(9, {4: 1.0}, 4.0)
        self.assertGreater(far[1], near[1])
        self.assertGreater(far[6], near[6])

    def test_flow_constraints_displace_neighbors(self):
        base = [[float(i), float(i)] for i in range(6)]  # 6 flows, 2 rails
        moved = {1: [2.0, 2.0]}  # flow 1 dragged from 1.0 to 2.0
        out = propagate_flow_constraints(base, moved, influence=2.0)
        self.assertAlmostEqual(out[1][0], 2.0, delta=1e-9)
        self.assertGreater(out[2][0], base[2][0])   # neighbor pulled along
        self.assertGreater(out[2][0] - base[2][0],
                           out[4][0] - base[4][0])  # decaying
        self.assertAlmostEqual(out[5][0], 5.0, delta=0.35)

    def test_zero_influence_keeps_others(self):
        base = [[float(i)] for i in range(4)]
        out = propagate_flow_constraints(base, {1: [2.5]}, influence=0.0)
        self.assertAlmostEqual(out[1][0], 2.5, delta=1e-9)
        for i in (0, 2, 3):
            self.assertAlmostEqual(out[i][0], base[i][0], delta=1e-9)


class TestGenerateFlows(unittest.TestCase):
    def test_uniform_grid(self):
        rails = straight_rails(3)
        flows = generate_flows(rails, count=5, bias=0.5)
        self.assertEqual(len(flows), 5)
        for i, flow in enumerate(flows):
            self.assertEqual(len(flow), 3)
            for j, s in enumerate(flow):
                self.assertAlmostEqual(s, float(i), delta=0.05)

    def test_locked_ratios_override_count(self):
        rails = straight_rails(3)
        flows = generate_flows(rails, count=99, bias=0.5,
                               locked_ratios=[0.0, 0.1, 1.0])
        self.assertEqual(len(flows), 3)
        self.assertAlmostEqual(flows[1][0], 0.4, delta=0.05)  # 0.1 * len 4
        self.assertAlmostEqual(flows[1][2], 0.4, delta=0.05)

    def test_intermediate_rails_follow_smoothing_not_ratios(self):
        # Ratios bind only the outer rails; a free intermediate rail must
        # settle where the smoothing puts it, not at ratio * its length.
        rail0 = CatmullRomCurve([(0.0, float(y), 0.0) for y in range(5)],
                                closed=False)   # length 4
        rail1 = CatmullRomCurve([(1.0, float(y), 0.0) for y in range(9)],
                                closed=False)   # length 8
        rail2 = CatmullRomCurve([(2.0, float(y), 0.0) for y in range(9)],
                                closed=False)   # length 8
        flows = generate_flows([rail0, rail1, rail2], count=3,
                               locked_ratios=[0.0, 0.25, 1.0])
        # Endpoints at ratio 0.25: y=1 on rail0, y=2 on rail2. The smooth
        # middle sits near their midpoint (y=1.5), not at 0.25 * 8 = 2.
        self.assertAlmostEqual(flows[1][0], 1.0, delta=1e-6)
        self.assertAlmostEqual(flows[1][2], 2.0, delta=1e-6)
        self.assertAlmostEqual(flows[1][1], 1.5, delta=0.05)

    def test_vertex_constraint_is_honored(self):
        rails = straight_rails(3)
        flows = generate_flows(rails, count=5, bias=0.5,
                               constraints={(1, 1): 3.5})
        self.assertAlmostEqual(flows[1][1], 3.5, delta=1e-9)
        # Other flows unaffected.
        self.assertAlmostEqual(flows[2][1], 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
