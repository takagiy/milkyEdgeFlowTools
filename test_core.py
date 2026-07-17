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
  [x] bisect_flows resolves anchored segments; locked ratios and vertex
      constraints are honored
"""

import math
import unittest

import core
from core import (
    CatmullRomCurve,
    blend_flow_curve,
    copy_flow_curve,
    common_sample_ratios,
    compute_vertex_target,
    copy_flows,
    decompose_chains,
    enforce_min_spacing,
    flow_direction,
    bisect_flows,
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

    def test_closest_param_to_polyline(self):
        curve, _ = straight_curve()
        s, dist = curve.closest_param_to_polyline([(3.0, 2.0, 0.0),
                                                   (3.0, -2.0, 0.0)])
        self.assertAlmostEqual(s, 3.0, delta=0.05)
        self.assertLess(dist, 0.05)

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


def s_rail_points(x_base=0.0, amp=0.4, length=5.0, twist=0.25, n=9):
    """porta-like chain: S-curved, non-planar, unevenly spaced verts."""
    pts = []
    for k in range(n):
        t = (k / (n - 1)) ** 1.2
        pts.append((x_base + amp * math.sin(2.0 * math.pi * t),
                    length * t,
                    twist * math.cos(math.pi * t)))
    return pts


def brute_closest_on_curve(curve, target_fn, steps=2000):
    """Reference argmin over a dense param scan; target_fn(point) -> dist."""
    best_s, best_d = 0.0, math.inf
    for i in range(steps + 1):
        s = curve.total_length * i / steps
        d = target_fn(curve.point_at(s))
        if d < best_d:
            best_s, best_d = s, d
    return best_s, best_d


def brute_seg_seg(p1, q1, p2, q2, steps=200):
    """Reference closest distance / first-segment param by grid scan."""
    best_d, best_s = math.inf, 0.0
    for i in range(steps + 1):
        s = i / steps
        a = core._lerp(p1, q1, s)
        for j in range(steps + 1):
            d = math.dist(a, core._lerp(p2, q2, j / steps))
            if d < best_d:
                best_d, best_s = d, s
    return best_d, best_s


def dist_point_to_segment(point, a, b):
    ab = core._sub(b, a)
    denom = core._dot(ab, ab)
    if denom < 1e-18:
        return math.dist(point, a)
    u = min(max(core._dot(ab, core._sub(point, a)) / denom, 0.0), 1.0)
    return math.dist(point, core._add(a, core._mul(ab, u)))


def dist_point_to_polyline(point, points):
    return min(dist_point_to_segment(point, points[k], points[k + 1])
               for k in range(len(points) - 1))


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


# ---------------------------------------------------------------------------
# Branch-coverage suites for the numpy-migration surface
# ---------------------------------------------------------------------------

class TestRotateToward(unittest.TestCase):
    V = (0.12, 0.55, -0.2)

    def test_general_rotation_preserves_geometry(self):
        f, t = (1.0, 0.25, 0.1), (0.3, 1.0, -0.4)
        out = core._rotate_toward(self.V, f, t)
        self.assertAlmostEqual(core._length(out), core._length(self.V),
                               places=9)
        fn, tn = core._normalize(f), core._normalize(t)
        # angle to the frame direction is preserved...
        self.assertAlmostEqual(core._dot(out, tn), core._dot(self.V, fn),
                               places=9)
        # ...and the component along the rotation axis is untouched.
        axis = core._normalize(core._cross(fn, tn))
        self.assertAlmostEqual(core._dot(out, axis),
                               core._dot(self.V, axis), places=9)

    def test_parallel_directions_return_vector(self):
        out = core._rotate_toward(self.V, (2.0, 1.0, 0.4), (5.0, 2.5, 1.0))
        self.assertEqual(out, self.V)

    def test_antiparallel_mirror_near_x_axis(self):
        f = (1.0, 0.2, 0.05)
        out = core._rotate_toward(self.V, f, core._mul(f, -1.0))
        fn = core._normalize(f)
        self.assertAlmostEqual(core._length(out), core._length(self.V),
                               places=9)
        self.assertAlmostEqual(core._dot(out, fn),
                               -core._dot(self.V, fn), places=9)

    def test_antiparallel_mirror_off_axis(self):
        f = (0.1, 0.95, 0.35)
        out = core._rotate_toward(self.V, f, core._mul(f, -1.0))
        fn = core._normalize(f)
        self.assertAlmostEqual(core._length(out), core._length(self.V),
                               places=9)
        self.assertAlmostEqual(core._dot(out, fn),
                               -core._dot(self.V, fn), places=9)

    def test_degenerate_directions_return_vector(self):
        self.assertEqual(
            core._rotate_toward(self.V, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            self.V)
        self.assertEqual(
            core._rotate_toward(self.V, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            self.V)


class TestClosestSegmentSegment(unittest.TestCase):
    def check(self, p1, q1, p2, q2, d_delta=5e-3, s_delta=0.02):
        dist, s = core._closest_segment_segment(p1, q1, p2, q2)
        b_dist, b_s = brute_seg_seg(p1, q1, p2, q2)
        self.assertAlmostEqual(dist, b_dist, delta=d_delta)
        self.assertAlmostEqual(s, b_s, delta=s_delta)
        return dist, s

    def test_skew_interior(self):
        self.check((0.0, 0.0, 0.0), (2.0, 1.0, 0.6),
                   (1.2, -0.4, 1.0), (0.8, 1.4, -0.6))

    def test_closest_before_second_segment(self):
        # Second segment leads away; its closest point is its start.
        self.check((0.0, 0.0, 0.0), (3.0, 0.5, 0.0),
                   (1.5, 2.0, 0.3), (1.8, 4.0, 0.6))

    def test_closest_past_second_segment(self):
        # Second segment approaches from afar; its closest point is its end.
        self.check((0.0, 0.0, 0.0), (3.0, 0.5, 0.0),
                   (1.8, 4.0, 0.6), (1.5, 2.0, 0.3))

    def test_parallel_segments(self):
        # The minimum is attained over a whole overlap interval, so only
        # the distance is well defined; the returned param just needs to
        # be a valid minimizer.
        off = (0.1, 0.9, 0.05)
        p1, q1 = (0.0, 0.0, 0.0), (2.0, 0.4, 0.2)
        p2, q2 = core._add(p1, off), core._add(q1, off)
        dist, s = core._closest_segment_segment(p1, q1, p2, q2)
        b_dist, _b_s = brute_seg_seg(p1, q1, p2, q2)
        self.assertAlmostEqual(dist, b_dist, delta=5e-3)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_first_segment_degenerate(self):
        self.check((0.7, 0.3, 0.2), (0.7, 0.3, 0.2),
                   (0.0, 1.0, 0.0), (2.0, 1.0, 0.5))

    def test_second_segment_degenerate(self):
        self.check((0.0, 1.0, 0.0), (2.0, 1.0, 0.5),
                   (0.7, 0.3, 0.2), (0.7, 0.3, 0.2))

    def test_both_degenerate(self):
        dist, s = core._closest_segment_segment(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
            (1.5, 0.5, 0.5), (1.5, 0.5, 0.5))
        self.assertAlmostEqual(dist, 1.0, places=12)
        self.assertEqual(s, 0.0)


class TestCurveGeometry(unittest.TestCase):
    def test_open_s_rail_interpolates_knots(self):
        pts = s_rail_points()
        curve = CatmullRomCurve(pts, closed=False)
        self.assertEqual(len(curve.knot_params), len(pts))
        self.assertAlmostEqual(curve.knot_params[0], 0.0)
        self.assertAlmostEqual(curve.knot_params[-1], curve.total_length)
        self.assertTrue(all(b > a for a, b in
                            zip(curve.knot_params, curve.knot_params[1:])))
        for s, p in zip(curve.knot_params, pts):
            self.assertLess(vlen(curve.point_at(s), p), 1e-6)

    def test_closed_loop_wraps_and_interpolates(self):
        pts = [(1.3 * math.cos(k * math.pi / 4),
                0.9 * math.sin(k * math.pi / 4),
                0.15 * (-1.0) ** k) for k in range(8)]
        curve = CatmullRomCurve(pts, closed=True)
        self.assertEqual(len(curve.knot_params), 8)
        length = curve.total_length
        p = curve.point_at(0.3 * length)
        self.assertLess(vlen(curve.point_at(0.3 * length + length), p), 1e-9)
        self.assertLess(vlen(curve.point_at(0.3 * length - length), p), 1e-9)
        for s, pt in zip(curve.knot_params, pts):
            self.assertLess(vlen(curve.point_at(s), pt), 1e-6)

    def test_duplicate_vertex_chain_is_finite(self):
        pts = s_rail_points()
        pts.insert(4, pts[4])  # merged verts happen on real meshes
        curve = CatmullRomCurve(pts, closed=False)
        self.assertTrue(math.isfinite(curve.total_length))
        self.assertTrue(all(b >= a for a, b in
                            zip(curve.knot_params, curve.knot_params[1:])))
        self.assertLess(vlen(curve.point_at(curve.total_length), pts[-1]),
                        1e-6)

    def test_point_at_degenerate_curve(self):
        curve = CatmullRomCurve([(1.0, 2.0, 3.0)] * 3, closed=False)
        self.assertEqual(curve.total_length, 0.0)
        self.assertEqual(curve.point_at(0.7), (1.0, 2.0, 3.0))

    def test_point_at_clamps_open_ends(self):
        curve = CatmullRomCurve(s_rail_points(), closed=False)
        self.assertLess(vlen(curve.point_at(-3.0), curve.point_at(0.0)),
                        1e-9)
        self.assertLess(vlen(curve.point_at(curve.total_length + 3.0),
                             curve.point_at(curve.total_length)), 1e-9)


class TestClosestParamToPoint(unittest.TestCase):
    def test_matches_brute_force_on_s_rail(self):
        curve = CatmullRomCurve(s_rail_points(), closed=False)
        probe = (0.55, 2.3, 0.4)
        s, dist = curve.closest_param_to_point(probe)
        b_s, b_dist = brute_closest_on_curve(
            curve, lambda p: math.dist(p, probe))
        self.assertAlmostEqual(dist, b_dist, delta=1e-3)
        self.assertAlmostEqual(s, b_s, delta=0.02)

    def test_probe_beyond_end_clamps(self):
        curve = CatmullRomCurve(s_rail_points(), closed=False)
        s, _dist = curve.closest_param_to_point((0.0, -2.0, 0.3))
        self.assertLess(s, 0.05)

    def test_degenerate_curve(self):
        curve = CatmullRomCurve([(1.0, 1.0, 0.0)] * 2, closed=False)
        s, dist = curve.closest_param_to_point((3.0, 1.0, 0.0))
        self.assertEqual(s, 0.0)
        self.assertAlmostEqual(dist, 2.0, places=12)


class TestClosestParamToPolyline(unittest.TestCase):
    CROSSING = [(-1.0 + 0.5 * k, 2.0 + 0.15 * math.sin(1.7 * k),
                 0.2 - 0.05 * k) for k in range(7)]

    def test_matches_brute_force_on_s_rail(self):
        curve = CatmullRomCurve(s_rail_points(), closed=False)
        s, dist = curve.closest_param_to_polyline(self.CROSSING)
        b_s, b_dist = brute_closest_on_curve(
            curve, lambda p: dist_point_to_polyline(p, self.CROSSING))
        self.assertAlmostEqual(dist, b_dist, delta=2e-3)
        self.assertAlmostEqual(s, b_s, delta=0.02)

    def test_far_polyline_hits_curve_end(self):
        curve = CatmullRomCurve(s_rail_points(), closed=False)
        far = [core._add(p, (0.0, 10.0, 0.0)) for p in self.CROSSING]
        s, _dist = curve.closest_param_to_polyline(far)
        self.assertGreater(s, 0.9 * curve.total_length)

    def test_degenerate_curve(self):
        curve = CatmullRomCurve([(1.0, 1.0, 0.0)] * 2, closed=False)
        s, dist = curve.closest_param_to_polyline([(3.0, 1.0, 0.0),
                                                   (3.0, 2.0, 0.0)])
        self.assertEqual(s, 0.0)
        self.assertAlmostEqual(dist, 2.0, places=12)


class TestClosestParamToRay(unittest.TestCase):
    def test_aiming_matches_brute_force(self):
        far = CatmullRomCurve(s_rail_points(x_base=3.0, amp=0.3),
                              closed=False)
        origin, direction = (0.0, 1.2, 0.1), (1.0, 0.15, -0.05)
        dn = core._normalize(direction)

        def ray_dist(p):
            t = max(0.0, core._dot(dn, core._sub(p, origin)))
            return math.dist(p, core._add(origin, core._mul(dn, t)))

        s, dist = far.closest_param_to_ray(origin, direction)
        b_s, b_dist = brute_closest_on_curve(far, ray_dist)
        self.assertAlmostEqual(dist, b_dist, delta=5e-3)
        self.assertAlmostEqual(s, b_s, delta=0.05)

    def test_ray_parallel_to_straight_rail(self):
        rail = CatmullRomCurve([(2.0, 0.0, 0.0), (2.0, 6.0, 0.0)],
                               closed=False)
        s, dist = rail.closest_param_to_ray((0.0, 1.0, 0.0),
                                            (0.0, 1.0, 0.0))
        self.assertAlmostEqual(dist, 2.0, places=9)
        self.assertGreaterEqual(s, 0.9)
        self.assertLessEqual(s, 1.6)

    def test_backward_ray_clamps_to_origin(self):
        rail = CatmullRomCurve([(2.0, 0.0, 0.0), (2.0, 6.0, 0.0)],
                               closed=False)
        s, dist = rail.closest_param_to_ray((0.0, 1.0, 0.0),
                                            (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(dist, 2.0, places=6)
        self.assertAlmostEqual(s, 1.0, delta=0.2)

    def test_zero_direction_uses_nearest_sample(self):
        rail = CatmullRomCurve([(2.0, 0.0, 0.0), (2.0, 6.0, 0.0)],
                               closed=False)
        s, dist = rail.closest_param_to_ray((2.05, 2.7, 0.0),
                                            (0.0, 0.0, 0.0))
        self.assertAlmostEqual(s, 2.7, delta=0.4)
        self.assertLess(dist, 0.5)

    def test_degenerate_curve_has_no_candidates(self):
        curve = CatmullRomCurve([(1.0, 1.0, 0.0)] * 2, closed=False)
        s, dist = curve.closest_param_to_ray((0.0, 0.0, 0.0),
                                             (1.0, 0.0, 0.0))
        self.assertEqual(s, 0.0)
        self.assertTrue(math.isinf(dist))


class TestBlendFlowCurve(unittest.TestCase):
    ROW_A = [(0.0, 0.0, 0.0), (1.0, 0.45, 0.2), (2.1, 0.6, 0.1),
             (3.0, 0.2, -0.1), (4.0, 0.0, 0.0)]
    ROW_B = [(0.2, 3.0, 0.4), (1.2, 3.6, 0.5), (2.2, 3.9, 0.3),
             (3.2, 3.5, 0.2), (4.1, 3.1, 0.4)]
    START, END = (0.5, 1.5, 0.1), (4.6, 1.9, 0.3)

    def test_endpoints_and_linear_weight(self):
        out = blend_flow_curve(self.ROW_A, self.ROW_B, 0.35,
                               self.START, self.END)
        self.assertEqual(len(out), 33)
        self.assertLess(vlen(out[0], self.START), 1e-9)
        self.assertLess(vlen(out[-1], self.END), 1e-9)
        only_a = blend_flow_curve(self.ROW_A, self.ROW_B, 0.0,
                                  self.START, self.END)
        only_b = blend_flow_curve(self.ROW_A, self.ROW_B, 1.0,
                                  self.START, self.END)
        for pa, pb, pm in zip(only_a, only_b, out):
            expected = core._add(core._mul(pa, 0.65), core._mul(pb, 0.35))
            self.assertLess(vlen(pm, expected), 1e-9)

    def test_degenerate_source_contributes_nothing(self):
        degenerate = [(1.0, 1.0, 1.0)] * 5
        out = blend_flow_curve(degenerate, self.ROW_B, 0.5,
                               self.START, self.END)
        only_b = blend_flow_curve(self.ROW_A, self.ROW_B, 1.0,
                                  self.START, self.END)
        for k, p in enumerate(out):
            chord_pt = core._lerp(self.START, self.END, k / 32)
            expected = core._lerp(chord_pt, only_b[k], 0.5)
            self.assertLess(vlen(p, expected), 1e-9)

    def test_degenerate_chord_collapses_to_start(self):
        out = blend_flow_curve(self.ROW_A, self.ROW_B, 0.5,
                               self.START, self.START)
        for p in out:
            self.assertLess(vlen(p, self.START), 1e-9)


class TestCopyFlowCurve(unittest.TestCase):
    REF = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.25), (2.0, 0.8, 0.3),
           (3.1, 0.4, 0.0), (4.0, 0.1, -0.2)]

    def test_deviation_scaled_never_rotated(self):
        start = (1.0, 5.0, 0.3)
        src_chord = core._sub(self.REF[-1], self.REF[0])
        end = core._add(start, core._mul(src_chord, 1.6))
        out = copy_flow_curve(self.REF, start, end)
        curve = CatmullRomCurve(self.REF, closed=False)
        for k, p in enumerate(out):
            t = k / 32
            src_pt = curve.point_at(curve.total_length * t)
            src_dev = core._sub(
                src_pt, core._add(self.REF[0], core._mul(src_chord, t)))
            dev = core._sub(p, core._lerp(start, end, t))
            self.assertLess(vlen(dev, core._mul(src_dev, 1.6)), 1e-9)

    def test_rotated_chord_keeps_deviation_orientation(self):
        start = (0.0, 0.0, 0.0)
        end = (1.5, 3.5, 0.8)   # chord direction differs from the ref's
        src_chord = core._sub(self.REF[-1], self.REF[0])
        scale = core._length(core._sub(end, start)) / \
            core._length(src_chord)
        out = copy_flow_curve(self.REF, start, end)
        curve = CatmullRomCurve(self.REF, closed=False)
        for k, p in enumerate(out):
            t = k / 32
            src_pt = curve.point_at(curve.total_length * t)
            src_dev = core._sub(
                src_pt, core._add(self.REF[0], core._mul(src_chord, t)))
            dev = core._sub(p, core._lerp(start, end, t))
            self.assertLess(vlen(dev, core._mul(src_dev, scale)), 1e-9)

    def test_degenerate_reference_gives_chord(self):
        out = copy_flow_curve([(2.0, 2.0, 2.0)] * 4,
                              (0.0, 0.0, 0.0), (3.0, 1.0, 0.0))
        for k, p in enumerate(out):
            self.assertLess(
                vlen(p, core._lerp((0.0, 0.0, 0.0), (3.0, 1.0, 0.0),
                                   k / 32)), 1e-9)


class TestCopyFlows(unittest.TestCase):
    def test_parallel_rails_translate_profile(self):
        rails = straight_rails(4, spacing=1.0, length=10)
        # Reference row: a straight diagonal, +0.5 in y per rail.
        ref = [(float(j), 2.0 + 0.5 * j, 0.0) for j in range(4)]
        flows = copy_flows(rails, 0, [0.0, 4.0, 8.0], ref)
        for d, row in zip((0.0, 4.0, 8.0), flows):
            for rj in range(4):
                self.assertAlmostEqual(row[rj], d + 0.5 * rj, delta=0.05)

    def test_deviation_copied_without_rotation(self):
        rails = straight_rails(3, spacing=1.0, length=10)
        # Reference row bulges +1 in y at the middle rail.
        ref = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)]
        flows = copy_flows(rails, 0, [5.0], ref)
        row = flows[0]
        self.assertAlmostEqual(row[0], 5.0, delta=1e-9)
        self.assertAlmostEqual(row[2], 5.0, delta=0.05)
        self.assertAlmostEqual(row[1], 6.0, delta=0.1)

    def test_chord_direction_fixed_on_fan(self):
        near = CatmullRomCurve([(0.0, float(y), 0.0) for y in range(11)],
                               closed=False)
        far = CatmullRomCurve([(4.0 + 0.3 * y, float(y), 0.0)
                               for y in range(11)], closed=False)
        ref = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]  # horizontal chord
        flows = copy_flows([near, far], 0, [2.0, 6.0], ref)
        # The fixed horizontal ray keeps every row horizontal even though
        # the far rail leans away (the hit stays at the anchor's height).
        for s_anchor, row in zip((2.0, 6.0), flows):
            hit = far.point_at(row[1])
            self.assertAlmostEqual(hit[1], s_anchor, delta=0.05)

    def test_intermediate_anchor_reaches_both_outers(self):
        rails = straight_rails(3, spacing=1.0, length=10)
        ref = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, 1.0, 0.0)]
        flows = copy_flows(rails, 1, [3.0], ref)
        row = flows[0]
        self.assertAlmostEqual(row[1], 3.0, delta=1e-9)
        self.assertAlmostEqual(row[0], 2.5, delta=0.05)
        self.assertAlmostEqual(row[2], 3.5, delta=0.05)

    def test_anchor_on_last_rail(self):
        rails = straight_rails(3, spacing=1.0, length=10)
        ref = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, 1.0, 0.0)]
        flows = copy_flows(rails, 2, [4.0], ref)
        row = flows[0]
        self.assertAlmostEqual(row[2], 4.0, delta=1e-9)
        self.assertAlmostEqual(row[1], 3.5, delta=0.05)
        self.assertAlmostEqual(row[0], 3.0, delta=0.05)

    def test_degenerate_reference_direction(self):
        rails = straight_rails(3, spacing=1.0, length=10)
        ref = [(1.0, 4.0, 0.0)] * 3    # zero chord: no ray direction
        flows = copy_flows(rails, 1, [5.0], ref)
        row = flows[0]
        self.assertAlmostEqual(row[1], 5.0, delta=1e-9)
        self.assertEqual(row[0], 0.0)
        self.assertEqual(row[2], 0.0)


class TestBisectFlows(unittest.TestCase):
    def test_uniform_parallel_rails_match_knots(self):
        rails = straight_rails(3)  # x = 0,1,2; y in [0,4]; uniform knots
        flows = bisect_flows(rails, {0: [0.0, 1.0, 2.0, 3.0, 4.0]})
        self.assertEqual(len(flows), 5)
        for i, flow in enumerate(flows):
            for rj, s in enumerate(flow):
                self.assertAlmostEqual(s, float(i), delta=0.05)

    def test_fan_aims_blended_chord_direction(self):
        locked = CatmullRomCurve([(0.0, float(y), 0.0) for y in range(5)],
                                 closed=False)          # length 4
        far = CatmullRomCurve([(2.0, float(y), 0.0) for y in range(9)],
                              closed=False)             # length 8
        flows = bisect_flows([locked, far],
                             {0: [0.0, 1.0, 2.0, 3.0, 4.0]})
        # Middle row: blended direction of the two end chords, aimed from
        # (0, 2), lands at y ~ 3.24 on the far rail (ratio copy would say
        # 4.0).
        self.assertAlmostEqual(flows[2][1], 3.24, delta=0.15)
        self.assertAlmostEqual(flows[1][1], 1.57, delta=0.15)
        self.assertAlmostEqual(flows[3][1], 5.19, delta=0.20)
        for a, b in zip(flows, flows[1:]):
            self.assertGreater(b[1], a[1])

    def test_shape_blend_reaches_middle_rail(self):
        locked = CatmullRomCurve([(0.0, float(y), 0.0) for y in range(5)],
                                 closed=False)
        middle = CatmullRomCurve([(1.0, -0.5, 0.0), (1.0, 0.5, 0.0),
                                  (1.0, 1.5, 0.0), (1.0, 2.5, 0.0),
                                  (1.0, 4.0, 0.0)], closed=False)
        far = CatmullRomCurve([(2.0, float(y), 0.0) for y in range(5)],
                              closed=False)
        flows = bisect_flows([locked, middle, far],
                             {0: [0.0, 1.0, 2.0, 3.0, 4.0]})
        # Row 0 bows down to (1, -0.5); row 4 is straight. The middle row
        # blends half the bow: fitted point (1, 1.75) -> middle-rail arc
        # 1.75 - (-0.5) = 2.25.
        self.assertAlmostEqual(flows[2][1], 2.25, delta=0.12)

    def test_both_end_anchors_skip_the_ray(self):
        # Both outer rails anchored (the no-lock case): the fan gets the
        # anchored quantiles on each rail, not ray landings.
        locked = CatmullRomCurve([(0.0, float(y), 0.0) for y in range(5)],
                                 closed=False)          # length 4
        far = CatmullRomCurve([(2.0, float(y), 0.0) for y in range(9)],
                              closed=False)             # length 8
        flows = bisect_flows(
            [locked, far],
            {0: [0.0, 1.0, 2.0, 3.0, 4.0],
             1: [0.0, 2.0, 4.0, 6.0, 8.0]})
        for i, flow in enumerate(flows):
            self.assertAlmostEqual(flow[0], float(i), delta=1e-9)
            self.assertAlmostEqual(flow[1], 2.0 * i, delta=1e-9)

    def test_both_end_anchors_blend_middle_rail_shape(self):
        # Same bowed geometry as the shape test, but anchored on both
        # outer rails: the fit runs between known endpoints and the
        # middle rail still receives the blended bow.
        outer_a = CatmullRomCurve([(0.0, float(y), 0.0) for y in range(5)],
                                  closed=False)
        middle = CatmullRomCurve([(1.0, -0.5, 0.0), (1.0, 0.5, 0.0),
                                  (1.0, 1.5, 0.0), (1.0, 2.5, 0.0),
                                  (1.0, 4.0, 0.0)], closed=False)
        outer_b = CatmullRomCurve([(2.0, float(y), 0.0) for y in range(5)],
                                  closed=False)
        anchor = [0.0, 1.0, 2.0, 3.0, 4.0]
        flows = bisect_flows([outer_a, middle, outer_b],
                             {0: anchor, 2: list(anchor)})
        self.assertAlmostEqual(flows[2][1], 2.25, delta=0.12)

    def test_intermediate_anchor_splits_segments(self):
        # Anchoring the middle rail resolves both sides outward by rays.
        rails = straight_rails(3)
        flows = bisect_flows(rails, {1: [0.0, 1.0, 2.0, 3.0, 4.0]})
        for i, flow in enumerate(flows):
            for rj, s in enumerate(flow):
                self.assertAlmostEqual(s, float(i), delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
