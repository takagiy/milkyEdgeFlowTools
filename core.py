"""Pure-logic core for milkyEdgeFlowTools.

No bpy dependency; everything operates on plain tuples and lists so it can
be tested with any Python interpreter (see test_core.py).

Pipeline (see requirements.md):
  decompose_chains -> CatmullRomCurve -> compute_vertex_target (per vertex)
  -> solve_relaxed_params -> enforce_min_spacing -> positions on the curve.
Each stage is a standalone function/class so alternative relax definitions
or curve types can be swapped in later.
"""

import math
from bisect import bisect_right

# Weight used to emulate a hard constraint for pinned vertices in the
# quadratic solver.
_PIN_WEIGHT = 1.0e8
# Tiny diagonal regularization so vertices with no data term and zero
# stiffness still yield a (zero) solution.
_EPSILON = 1.0e-9


# ---------------------------------------------------------------------------
# Small vector helpers (3-tuples)
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _length(a)
    if n < 1.0e-12:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


# ---------------------------------------------------------------------------
# Chain decomposition
# ---------------------------------------------------------------------------

def decompose_chains(edges):
    """Split edges (pairs of hashable vertex ids) into connected chains.

    Returns (chains, skipped) where each chain is (ordered_verts, closed).
    Components containing a vertex of degree > 2 (branches) are skipped and
    counted instead of processed.
    """
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited = set()
    chains = []
    skipped = 0
    for start in adj:
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            stack.extend(adj[v])
        visited |= comp

        if any(len(adj[v]) > 2 for v in comp):
            skipped += 1
            continue

        ends = [v for v in comp if len(adj[v]) == 1]
        closed = not ends
        first = next(iter(comp)) if closed else ends[0]

        verts = [first]
        prev, cur = None, first
        while True:
            nxts = [w for w in adj[cur] if w != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == first:
                break
            verts.append(nxt)
            prev, cur = cur, nxt
        chains.append((verts, closed))
    return chains, skipped


# ---------------------------------------------------------------------------
# Centripetal Catmull-Rom curve, arc-length parameterized
# ---------------------------------------------------------------------------

def _cr_point(p0, p1, p2, p3, u):
    """Barry-Goldman centripetal Catmull-Rom; u in [0, 1] maps to [p1, p2]."""
    alpha = 0.5
    t0 = 0.0
    t1 = t0 + max(math.dist(p0, p1), 1.0e-9) ** alpha
    t2 = t1 + max(math.dist(p1, p2), 1.0e-9) ** alpha
    t3 = t2 + max(math.dist(p2, p3), 1.0e-9) ** alpha
    t = t1 + (t2 - t1) * u

    def lp(pa, pb, ta, tb):
        if tb - ta < 1.0e-12:
            return pa
        w = (t - ta) / (tb - ta)
        return _lerp(pa, pb, w)

    a1 = lp(p0, p1, t0, t1)
    a2 = lp(p1, p2, t1, t2)
    a3 = lp(p2, p3, t2, t3)
    b1 = lp(a1, a2, t0, t2)
    b2 = lp(a2, a3, t1, t3)
    return lp(b1, b2, t1, t2)


class CatmullRomCurve:
    """Interpolating spline through the given points.

    The curve is flattened into a dense polyline (samples_per_segment per
    knot span) and parameterized by arc length along that polyline.
    """

    def __init__(self, points, closed, samples_per_segment=16):
        pts = [tuple(map(float, p)) for p in points]
        n = len(pts)
        self.closed = closed

        def ctrl(i):
            if closed:
                return pts[i % n]
            if i < 0:
                return _sub(_mul(pts[0], 2.0), pts[1])
            if i >= n:
                return _sub(_mul(pts[-1], 2.0), pts[-2])
            return pts[i]

        segments = n if closed else n - 1
        samples = []
        knot_sample_idx = []
        for i in range(segments):
            p0, p1, p2, p3 = ctrl(i - 1), ctrl(i), ctrl(i + 1), ctrl(i + 2)
            knot_sample_idx.append(len(samples))
            for k in range(samples_per_segment):
                samples.append(_cr_point(p0, p1, p2, p3,
                                         k / samples_per_segment))
        # Terminal sample: wrap point for closed, last input point for open.
        end_idx = len(samples)
        samples.append(pts[0] if closed else pts[-1])
        if not closed:
            knot_sample_idx.append(end_idx)

        cum = [0.0]
        for a, b in zip(samples, samples[1:]):
            cum.append(cum[-1] + math.dist(a, b))

        self._samples = samples
        self._cum = cum
        self.total_length = cum[-1]
        self.knot_params = [cum[i] for i in knot_sample_idx]

    def point_at(self, s):
        length = self.total_length
        if length <= 0.0:
            return self._samples[0]
        if self.closed:
            s = s % length
        else:
            s = min(max(s, 0.0), length)
        cum = self._cum
        j = min(bisect_right(cum, s), len(cum) - 1) - 1
        j = max(j, 0)
        span = cum[j + 1] - cum[j]
        u = 0.0 if span < 1.0e-12 else (s - cum[j]) / span
        return _lerp(self._samples[j], self._samples[j + 1], u)

    def closest_param_to_point(self, point):
        """Arc-length param of the curve point closest to `point`.

        Returns (s, distance).
        """
        samples, cum = self._samples, self._cum
        best_s, best_dist = 0.0, math.inf
        for j in range(len(samples) - 1):
            a = samples[j]
            ab = _sub(samples[j + 1], a)
            denom = _dot(ab, ab)
            if denom < 1.0e-18:
                u = 0.0
            else:
                u = min(max(_dot(ab, _sub(point, a)) / denom, 0.0), 1.0)
            candidate = _add(a, _mul(ab, u))
            dist = math.dist(candidate, point)
            if dist < best_dist:
                best_dist = dist
                best_s = cum[j] + u * (cum[j + 1] - cum[j])
        return best_s, best_dist

    def closest_param_to_ray(self, origin, direction):
        """Arc-length param of the curve point closest to the ray.

        In 3D a ray and the curve generally do not intersect, so the
        closest-point pair between the ray (t >= 0) and the sample polyline
        is used. Returns (s, distance).
        """
        d = _normalize(direction)
        samples, cum = self._samples, self._cum
        if d is None:
            # Degenerate direction: closest sample to the origin.
            best_j = min(range(len(samples)),
                         key=lambda j: math.dist(samples[j], origin))
            return cum[best_j], math.dist(samples[best_j], origin)

        best_s, best_dist = 0.0, math.inf
        for j in range(len(samples) - 1):
            a = samples[j]
            ab = _sub(samples[j + 1], a)
            seg_len2 = _dot(ab, ab)
            if seg_len2 < 1.0e-18:
                continue
            b_ = _dot(ab, d)
            r = _sub(a, origin)
            denom = seg_len2 - b_ * b_
            if denom > 1.0e-12:
                u = (b_ * _dot(d, r) - _dot(ab, r)) / denom
                u = min(max(u, 0.0), 1.0)
            else:
                u = 0.0  # segment parallel to ray
            t = max(0.0, _dot(d, _sub(_add(a, _mul(ab, u)), origin)))
            u = min(max(_dot(ab, _sub(_add(origin, _mul(d, t)), a))
                        / seg_len2, 0.0), 1.0)
            p_seg = _add(a, _mul(ab, u))
            p_ray = _add(origin, _mul(d, t))
            dist = math.dist(p_seg, p_ray)
            if dist < best_dist:
                best_dist = dist
                best_s = cum[j] + u * (cum[j + 1] - cum[j])
        return best_s, best_dist


# ---------------------------------------------------------------------------
# Flow extrapolation (data term)
# ---------------------------------------------------------------------------

def flow_direction(ring_points):
    """Approach direction of a crossing flow.

    ring_points are ordered nearest-first: [w1, w2, w3, ...] where w1 is the
    far vertex of the crossing edge. The direction points from the ring
    toward the chain, from the weighted average of the last two segments
    before the crossing edge (0.7 nearest / 0.3 next).
    """
    if len(ring_points) < 2:
        return None
    d1 = _normalize(_sub(ring_points[0], ring_points[1]))
    if d1 is None:
        return None
    if len(ring_points) >= 3:
        d2 = _normalize(_sub(ring_points[1], ring_points[2]))
        if d2 is not None:
            mixed = _normalize(_add(_mul(d1, 0.7), _mul(d2, 0.3)))
            if mixed is not None:
                return mixed
    return d1


def compute_vertex_target(curve, sides, s_orig, side_blend, clamp_span):
    """Target arc-length displacement for one chain vertex, or None.

    sides: ring polylines (nearest-first) for each side of the vertex.
    The side with more rings is the major side (side_blend 0); the other is
    minor (side_blend 1). The displacement is clamped to +-clamp_span.
    """
    candidates = []
    for ring in sides:
        direction = flow_direction(ring)
        if direction is None:
            continue
        s, _dist = curve.closest_param_to_ray(ring[0], direction)
        delta = s - s_orig
        if curve.closed:
            length = curve.total_length
            delta = (delta + length / 2.0) % length - length / 2.0
        delta = min(max(delta, -clamp_span), clamp_span)
        candidates.append((len(ring), delta))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]
    candidates.sort(key=lambda c: -c[0])
    major, minor = candidates[0][1], candidates[1][1]
    return major * (1.0 - side_blend) + minor * side_blend


# ---------------------------------------------------------------------------
# 1D smoothing solver
# ---------------------------------------------------------------------------

def _thomas(sub, diag, sup, rhs):
    n = len(diag)
    cp = [0.0] * n
    dp = [0.0] * n
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / m
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m
    x = [0.0] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _cyclic_thomas(sub, diag, sup, rhs, corner):
    """Solve a cyclic tridiagonal system (Sherman-Morrison).

    corner is the value at [0][n-1] and [n-1][0] (symmetric).
    """
    n = len(diag)
    gamma = -diag[0]
    diag2 = list(diag)
    diag2[0] -= gamma
    diag2[n - 1] -= corner * corner / gamma
    y = _thomas(sub, diag2, sup, rhs)
    u = [0.0] * n
    u[0] = gamma
    u[n - 1] = corner
    z = _thomas(sub, diag2, sup, u)
    num = y[0] + corner * y[n - 1] / gamma
    den = 1.0 + z[0] + corner * z[n - 1] / gamma
    factor = num / den
    return [y[i] - factor * z[i] for i in range(n)]


def solve_relaxed_params(targets, pinned, stiffness, closed):
    """Solve for arc-length displacements delta_i.

    Minimizes  sum w_i (d_i - target_i)^2 + stiffness * sum (d_{i+1} - d_i)^2
    where w_i = 1 for vertices with a target, 0 otherwise, and pinned
    vertices are held at 0 via a large penalty weight. Pinned influence
    propagates through the smoothness term and decays with distance.
    """
    n = len(targets)
    if n == 0:
        return []
    if n == 1:
        return [0.0 if pinned[0] or targets[0] is None else targets[0]]

    lam = max(0.0, stiffness)
    weights = [0.0] * n
    rhs = [0.0] * n
    for i in range(n):
        if pinned[i]:
            weights[i] = _PIN_WEIGHT
        elif targets[i] is not None:
            weights[i] = 1.0
            rhs[i] = targets[i]

    use_cyclic = closed and n >= 3 and lam > _EPSILON
    diag = [0.0] * n
    sub = [0.0] * n
    sup = [0.0] * n
    for i in range(n):
        degree = 2 if (use_cyclic or 0 < i < n - 1) else 1
        diag[i] = weights[i] + lam * degree + _EPSILON
        rhs[i] *= weights[i]
        if i > 0:
            sub[i] = -lam
        if i < n - 1:
            sup[i] = -lam

    if use_cyclic:
        return _cyclic_thomas(sub, diag, sup, rhs, -lam)
    return _thomas(sub, diag, sup, rhs)


# ---------------------------------------------------------------------------
# Ordering safety
# ---------------------------------------------------------------------------

def enforce_min_spacing(params, closed, total_length, min_gap):
    """Keep arc-length params monotone so vertices cannot swap order."""
    out = list(params)
    n = len(out)
    for i in range(1, n):
        if out[i] < out[i - 1] + min_gap:
            out[i] = out[i - 1] + min_gap
    if closed and n >= 2:
        limit = out[0] + total_length - min_gap
        if out[n - 1] > limit:
            out[n - 1] = limit
            for i in range(n - 2, 0, -1):
                if out[i] > out[i + 1] - min_gap:
                    out[i] = out[i + 1] - min_gap
    return out


# ---------------------------------------------------------------------------
# Chain application order
# ---------------------------------------------------------------------------

def order_chains(dominant_sets):
    """Application order for multiple chains.

    dominant_sets[i] is the set of chain indices visible from chain i on the
    dominant blend side. Those chains are applied before chain i, so that
    chain i extrapolates its flows from already-relaxed geometry. Stable
    topological order; dependency cycles fall back to input order.
    """
    n = len(dominant_sets)
    deps = [set(d) - {i} for i, d in enumerate(dominant_sets)]
    order = []
    placed = set()
    while len(order) < n:
        ready = [i for i in range(n)
                 if i not in placed and not (deps[i] - placed)]
        if not ready:
            ready = [i for i in range(n) if i not in placed]
        order.append(ready[0])
        placed.add(ready[0])
    return order


# ---------------------------------------------------------------------------
# Regeneration core (M1) — see requirements.md chapter 11
# ---------------------------------------------------------------------------

def order_rails(count, adjacency_pairs):
    """Order rails into a row using crossing adjacency between them.

    adjacency_pairs are unordered (a, b) rail-index pairs that share
    crossing flows. Returns the ordered rail indices, or None when the
    adjacency does not form a single simple path (branching, cycles, or
    disconnected selections).
    """
    if count == 0:
        return []
    if count == 1:
        return [0]
    adjacency = {i: set() for i in range(count)}
    for a, b in adjacency_pairs:
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    ends = [i for i in range(count) if len(adjacency[i]) == 1]
    if len(ends) != 2 or any(len(n) > 2 for n in adjacency.values()):
        return None
    order = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxts = [x for x in adjacency[cur] if x != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order if len(order) == count else None


def _curvature_profile(curve, resolution):
    """Smoothed curvature at uniform normalized-arc positions.

    Returns (profile, total_turning_angle). The total turning angle (in
    radians) tells callers whether the curve is effectively straight.
    """
    samples = curve._samples
    cum = curve._cum
    n = len(samples)
    if n < 3 or curve.total_length <= 0.0:
        return [0.0] * resolution, 0.0

    kappas = [0.0] * n
    total_turn = 0.0
    for j in range(1, n - 1):
        v1 = _sub(samples[j], samples[j - 1])
        v2 = _sub(samples[j + 1], samples[j])
        l1, l2 = _length(v1), _length(v2)
        if l1 < 1.0e-12 or l2 < 1.0e-12:
            continue
        cosang = min(max(_dot(v1, v2) / (l1 * l2), -1.0), 1.0)
        angle = math.acos(cosang)
        total_turn += angle
        kappas[j] = angle / ((l1 + l2) / 2.0)
    kappas[0] = kappas[1]
    kappas[-1] = kappas[-2]
    smoothed = [(kappas[max(j - 1, 0)] + kappas[j]
                 + kappas[min(j + 1, n - 1)]) / 3.0 for j in range(n)]

    length = curve.total_length
    profile = []
    for i in range(resolution):
        s = length * i / (resolution - 1)
        j = min(max(bisect_right(cum, s) - 1, 0), n - 2)
        span = cum[j + 1] - cum[j]
        u = 0.0 if span < 1.0e-12 else (s - cum[j]) / span
        profile.append(smoothed[j] + (smoothed[j + 1] - smoothed[j]) * u)
    return profile, total_turn


def common_sample_ratios(curves, count, bias, resolution=128):
    """Arc-length ratios shared by all curves for subdividing rails.

    Density is w(t) = (1 - bias) + bias * normalized curvature, averaged
    over the curves so paired rails are sampled at the same quantiles and
    the flows stay parallel. Effectively straight curves fall back to a
    uniform density. Endpoints 0 and 1 are always included.
    """
    count = max(2, count)
    bias = min(max(bias, 0.0), 1.0)

    profiles = []
    for curve in curves:
        profile, total_turn = _curvature_profile(curve, resolution)
        mean = sum(profile) / len(profile)
        if total_turn < 0.01 or mean < 1.0e-12:
            profiles.append([1.0] * resolution)
        else:
            profiles.append([k / mean for k in profile])
    averaged = [sum(p[i] for p in profiles) / len(profiles)
                for i in range(resolution)]
    weights = [max((1.0 - bias) + bias * a, 1.0e-4) for a in averaged]

    cumulative = [0.0]
    for i in range(1, resolution):
        cumulative.append(cumulative[-1] + (weights[i - 1] + weights[i]) / 2)

    ratios = []
    for i in range(count):
        q = cumulative[-1] * i / (count - 1)
        j = min(max(bisect_right(cumulative, q) - 1, 0), resolution - 2)
        span = cumulative[j + 1] - cumulative[j]
        u = 0.0 if span < 1.0e-12 else (q - cumulative[j]) / span
        ratios.append((j + u) / (resolution - 1))
    ratios[0] = 0.0
    ratios[-1] = 1.0
    return ratios


def opposite_shore_params(points, curve):
    """Arc params on `curve` geometrically opposite the given points.

    Used when a locked chain dictates the flows: normalized arc ratios of
    a strongly curved chain skew away from the visual correspondence (its
    curvature inflates the denominator), so each locked vertex is instead
    projected to its closest point on the rail. The first and last params
    are forced onto the rail endpoints (the end flows connect endpoints by
    construction) and the sequence is kept strictly increasing.
    """
    length = curve.total_length
    params = [curve.closest_param_to_point(tuple(p))[0] for p in points]
    n = len(params)
    if n == 0:
        return []
    params[0] = 0.0
    if n > 1:
        params[-1] = length
    gap = 1.0e-3 * (length / max(1, n - 1))
    for i in range(1, n - 1):
        params[i] = max(params[i], params[i - 1] + gap)
    for i in range(n - 2, 0, -1):
        if params[i] > params[i + 1] - gap:
            params[i] = params[i + 1] - gap
    return params


def smooth_flow_on_rails(rails, params, pinned, iterations=10):
    """Slide flow vertices along their rail curves to minimize bending.

    params[j] is the arc-length position of the flow vertex on rails[j].
    Each free vertex is pulled toward the point on its own rail closest to
    the midpoint of its neighbors (rail-constrained Laplacian smoothing),
    so vertices always stay on their rail curves. Pinned vertices (always
    including both endpoints, per the caller) do not move.
    """
    m = len(rails)
    out = list(params)
    if m < 3:
        return out
    for _ in range(max(1, iterations)):
        for j in range(1, m - 1):
            if pinned[j]:
                continue
            a = rails[j - 1].point_at(out[j - 1])
            b = rails[j + 1].point_at(out[j + 1])
            out[j], _ = rails[j].closest_param_to_point(_lerp(a, b, 0.5))
    return out


def generate_flows(rails, count, bias=0.5, locked_ratios=None,
                   constraints=None, iterations=10):
    """Generate crossing flows across ordered rail curves.

    rails: CatmullRomCurve list in row order (outermost first and last).
    locked_ratios: when given (locked chain), these normalized arc ratios
    replace both `count` and the density sampling.
    constraints: optional {(flow_index, rail_index): arc_param} pass-through
    points (dragged/locked vertices); they are pinned during smoothing.
    Returns one list of per-rail arc-length params for each flow.
    """
    if locked_ratios is not None:
        ratios = list(locked_ratios)
    else:
        ratios = common_sample_ratios([rails[0], rails[-1]], count, bias)
    constraints = constraints or {}

    flows = []
    for i, ratio in enumerate(ratios):
        params = [ratio * rail.total_length for rail in rails]
        pinned = [False] * len(rails)
        pinned[0] = pinned[-1] = True
        for (flow_i, rail_j), s in constraints.items():
            if flow_i == i:
                params[rail_j] = s
                pinned[rail_j] = True
        flows.append(smooth_flow_on_rails(rails, params, pinned, iterations))
    return flows


# ---------------------------------------------------------------------------
# Chain relaxation (orchestration)
# ---------------------------------------------------------------------------

def relax_chain_step(curve, params, sides, pinned, side_blend, stiffness):
    """One relax pass over a chain on a fixed, prefitted curve.

    params are the current arc-length positions of the chain vertices on
    the curve; the returned list is the relaxed positions. Keeping the
    curve fixed across steps means iterating can never drift the shape of
    the loop.
    """
    knots = curve.knot_params
    n = len(knots)
    length = curve.total_length

    # Original knot gaps in arc length, used to clamp runaway targets.
    gaps = []
    for i in range(n - 1):
        gaps.append(knots[i + 1] - knots[i])
    if curve.closed:
        gaps.append(length - knots[-1])

    targets = []
    for i in range(n):
        prev_gap = gaps[i - 1] if (i > 0 or curve.closed) else gaps[0]
        next_gap = gaps[i] if i < len(gaps) else gaps[-1]
        span = 2.0 * max(prev_gap, next_gap)
        targets.append(compute_vertex_target(curve, sides[i], params[i],
                                             side_blend, span))

    deltas = solve_relaxed_params(targets, pinned, stiffness, curve.closed)
    new_params = [params[i] + deltas[i] for i in range(n)]

    seg_count = n if curve.closed else n - 1
    min_gap = 0.01 * (length / max(seg_count, 1))
    return enforce_min_spacing(new_params, curve.closed, length, min_gap)


def relax_chain(points, closed, sides, pinned, side_blend=0.0,
                stiffness=1.0, factor=1.0, iterations=1):
    """Compute relaxed positions for one chain.

    points: ordered chain vertex positions.
    sides: per vertex, a list of crossing-flow ring polylines (see
           compute_vertex_target); empty list -> no data term.
    pinned: per vertex, True to hold the vertex in place.
    The relax pass runs `iterations` times on the same fitted curve; factor
    blends the final result once at the end.
    Returns the new positions (same length/order as points).
    """
    n = len(points)
    if n < 2:
        return [tuple(map(float, p)) for p in points]

    curve = CatmullRomCurve(points, closed)
    params = list(curve.knot_params)
    for _ in range(max(1, iterations)):
        params = relax_chain_step(curve, params, sides, pinned,
                                  side_blend, stiffness)

    result = []
    for i in range(n):
        if pinned[i]:
            result.append(tuple(map(float, points[i])))
            continue
        p = curve.point_at(params[i])
        if factor != 1.0:
            p = _lerp(tuple(map(float, points[i])), p, factor)
        result.append(p)
    return result
