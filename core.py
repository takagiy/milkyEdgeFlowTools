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
# Weight of the one-sided anti-crowding springs added by the IRLS passes.
_CROWD_WEIGHT = 10.0
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


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _closest_segment_segment(p1, q1, p2, q2):
    """Closest points between segments [p1,q1] and [p2,q2].

    Returns (distance, param on the first segment in [0, 1]).
    """
    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    r = _sub(p1, p2)
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)

    def clamp01(x):
        return min(max(x, 0.0), 1.0)

    if a < 1.0e-18 and e < 1.0e-18:
        return math.dist(p1, p2), 0.0
    if a < 1.0e-18:
        s, t = 0.0, clamp01(f / e)
    else:
        c = _dot(d1, r)
        if e < 1.0e-18:
            s, t = clamp01(-c / a), 0.0
        else:
            b = _dot(d1, d2)
            denom = a * e - b * b
            s = clamp01((b * f - c * e) / denom) if denom > 1.0e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = clamp01(-c / a), 0.0
            elif t > 1.0:
                s, t = clamp01((b - c) / a), 1.0
    pt1 = _add(p1, _mul(d1, s))
    pt2 = _add(p2, _mul(d2, t))
    return math.dist(pt1, pt2), s


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

    def _segment_ranges(self, s_center, s_window):
        """Sample-segment index ranges covering s_center +- s_window."""
        seg_count = len(self._samples) - 1
        length = self.total_length
        if (s_center is None or s_window is None
                or 2.0 * s_window >= length):
            return [(0, seg_count)]

        def index_range(lo, hi):
            j0 = max(0, bisect_right(self._cum, lo) - 1)
            j1 = min(seg_count, bisect_right(self._cum, hi))
            return (j0, j1)

        if not self.closed:
            return [index_range(max(0.0, s_center - s_window),
                                min(length, s_center + s_window))]
        lo = (s_center - s_window) % length
        hi = (s_center + s_window) % length
        if lo <= hi:
            return [index_range(lo, hi)]
        return [index_range(lo, length), index_range(0.0, hi)]

    def closest_param_to_path(self, path, s_center=None, s_window=None):
        """Arc-length param of the curve point closest to a polyline.

        Optionally restricts the search to params within s_window of
        s_center (targets outside that range would be clamped anyway).
        Returns (s, distance).
        """
        samples, cum = self._samples, self._cum
        best_s, best_dist = 0.0, math.inf
        for j0, j1 in self._segment_ranges(s_center, s_window):
            for j in range(j0, j1):
                a, b = samples[j], samples[j + 1]
                for k in range(len(path) - 1):
                    dist, u = _closest_segment_segment(a, b, path[k],
                                                       path[k + 1])
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


def extrapolation_path(ring_points, extent):
    """Continuation of a crossing flow beyond its nearest point, or None.

    Extrapolates as a curvature-decaying spiral: the discrete curvature of
    the last three ring points is kept initially and decays exponentially
    (kappa(s) = kappa0 * exp(-s / lam)), so curved flows keep their bend
    near the loop but straighten out with distance. The total turning is
    bounded by kappa0 * lam <= 90 degrees, so the path can never loop
    back. With fewer than three points, or negligible curvature, this
    degenerates to a straight ray of length `extent`.
    """
    direction = flow_direction(ring_points)
    if direction is None:
        return None
    origin = tuple(map(float, ring_points[0]))

    def straight():
        return [origin, _add(origin, _mul(direction, extent))]

    if len(ring_points) < 3:
        return straight()

    a, b, c = ring_points[2], ring_points[1], ring_points[0]
    u = _sub(b, a)
    v = _sub(c, b)
    lu, lv, lw = _length(u), _length(v), math.dist(a, c)
    binormal = _cross(u, v)
    area2 = _length(binormal)
    if min(lu, lv, lw) < 1.0e-12 or area2 < 1.0e-12:
        return straight()

    seg_mean = (lu + lv) / 2.0
    kappa = min(2.0 * area2 / (lu * lv * lw), 2.0 / seg_mean)
    if kappa * seg_mean < 1.0e-4:
        return straight()
    lam = min(3.0 * seg_mean, (math.pi / 2.0) / kappa)

    # Initial heading: the last chord rotated by half its subtended angle
    # approximates the tangent at the ring's nearest point.
    bin_n = _normalize(binormal)
    v_hat = _normalize(v)
    side = _normalize(_cross(bin_n, v_hat))
    phi = kappa * lv / 2.0
    d0 = _normalize(_add(_mul(v_hat, math.cos(phi)),
                         _mul(side, math.sin(phi))))
    n0 = _normalize(_cross(bin_n, d0))
    if d0 is None or n0 is None:
        return straight()

    points = [origin]
    pos = origin
    steps = 12
    step = 4.0 * lam / steps
    for k in range(steps):
        s_mid = (k + 0.5) * step
        theta = kappa * lam * (1.0 - math.exp(-s_mid / lam))
        heading = _add(_mul(d0, math.cos(theta)), _mul(n0, math.sin(theta)))
        pos = _add(pos, _mul(heading, step))
        points.append(pos)
    theta_end = kappa * lam * (1.0 - math.exp(-4.0))
    heading = _add(_mul(d0, math.cos(theta_end)),
                   _mul(n0, math.sin(theta_end)))
    points.append(_add(pos, _mul(heading, extent)))
    return points


def compute_vertex_target(curve, sides, s_orig, side_blend, clamp_span):
    """Target arc-length displacement for one chain vertex, or None.

    sides: ring polylines (nearest-first) for each side of the vertex.
    The side with more rings is the major side (side_blend 0); the other is
    minor (side_blend 1). The displacement is clamped to +-clamp_span.
    """
    candidates = []
    for ring in sides:
        anchor_dist = math.dist(tuple(map(float, ring[0])),
                                curve.point_at(s_orig)) if ring else 0.0
        path = extrapolation_path(ring, 2.0 * (anchor_dist + clamp_span))
        if path is None:
            continue
        s, _dist = curve.closest_param_to_path(path, s_center=s_orig,
                                               s_window=1.5 * clamp_span)
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


def solve_relaxed_params(targets, pinned, stiffness, closed, spacing=None):
    """Solve for arc-length displacements delta_i.

    Minimizes  sum w_i (d_i - target_i)^2 + stiffness * sum (d_{i+1} - d_i)^2
    where w_i = 1 for vertices with a target, 0 otherwise, and pinned
    vertices are held at 0 via a large penalty weight. Pinned influence
    propagates through the smoothness term and decays with distance.

    spacing, if given, maps a pair index i (the pair (i, i+1); i = n-1 is
    the wrap pair of a closed chain) to (weight, rest) and adds the term
    weight * (d_{i+1} - d_i - rest)^2 — the anti-crowding springs.
    """
    n = len(targets)
    if n == 0:
        return []
    if n == 1:
        return [0.0 if pinned[0] or targets[0] is None else targets[0]]

    lam = max(0.0, stiffness)
    spacing = spacing or {}
    weights = [0.0] * n
    rhs = [0.0] * n
    for i in range(n):
        if pinned[i]:
            weights[i] = _PIN_WEIGHT
        elif targets[i] is not None:
            weights[i] = 1.0
            rhs[i] = targets[i]

    has_wrap_spring = closed and (n - 1) in spacing
    use_cyclic = closed and n >= 3 and (lam > _EPSILON or has_wrap_spring)
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

    corner = -lam
    for i, (w, rest) in spacing.items():
        if i < n - 1:
            diag[i] += w
            diag[i + 1] += w
            sup[i] -= w
            sub[i + 1] -= w
            rhs[i] -= w * rest
            rhs[i + 1] += w * rest
        elif use_cyclic and i == n - 1:
            diag[n - 1] += w
            diag[0] += w
            corner -= w
            rhs[n - 1] -= w * rest
            rhs[0] += w * rest

    if use_cyclic:
        return _cyclic_thomas(sub, diag, sup, rhs, corner)
    return _thomas(sub, diag, sup, rhs)


# ---------------------------------------------------------------------------
# Minimum-spacing projection
# ---------------------------------------------------------------------------

def _pava(values):
    """Isotonic (non-decreasing) L2 regression, pool-adjacent-violators."""
    pooled = []
    counts = []
    for x in values:
        pooled.append(x)
        counts.append(1)
        while len(pooled) > 1 and pooled[-2] > pooled[-1]:
            v = pooled.pop()
            c = counts.pop()
            pooled[-1] = (pooled[-1] * counts[-1] + v * c) / (counts[-1] + c)
            counts[-1] += c
    out = []
    for v, c in zip(pooled, counts):
        out.extend([v] * c)
    return out


def _project_open(params, mins, pinned):
    """Project params onto { t[i+1] - t[i] >= mins[i] }, keeping pins.

    Substituting u_i = t_i - cumsum(mins) turns the constraints into plain
    monotonicity, so the L2-closest feasible point is isotonic regression
    (PAVA), solved independently on each run between pinned vertices and
    clipped to the pinned boundary values.
    """
    n = len(params)
    offsets = [0.0] * n
    for i in range(1, n):
        offsets[i] = offsets[i - 1] + mins[i - 1]
    u = [params[i] - offsets[i] for i in range(n)]

    i = 0
    while i < n:
        if pinned[i]:
            i += 1
            continue
        j = i
        while j < n and not pinned[j]:
            j += 1
        lo = u[i - 1] if i > 0 else None
        hi = u[j] if j < n else None
        for k, value in enumerate(_pava(u[i:j])):
            if lo is not None and value < lo:
                value = lo
            if hi is not None and value > hi:
                value = hi
            u[i + k] = value
        i = j
    return [u[i] + offsets[i] for i in range(n)]


def project_min_spacing(params, gaps, pinned, beta, closed, total_length):
    """Closest params where no gap shrinks below beta * its original size.

    gaps are the original arc-length gaps (n-1 entries for open chains, n
    including the wrap gap for closed ones). A tiny absolute floor is kept
    even at beta = 0 so vertices can never swap order. Closed chains are
    cut at the first pinned vertex (or anchored at vertex 0) and projected
    as an open run whose both ends are fixed.
    """
    n = len(params)
    if n < 2:
        return list(params)
    mean_gap = sum(gaps) / len(gaps)
    floor = 1.0e-3 * mean_gap if mean_gap > 0 else 1.0e-9
    mins = [max(beta * g, floor) for g in gaps]

    if not closed:
        return _project_open(list(params), mins, list(pinned))

    anchor = next((i for i, p in enumerate(pinned) if p), 0)
    rotated = [params[(anchor + k) % n]
               + (total_length if anchor + k >= n else 0.0)
               for k in range(n)]
    rotated.append(params[anchor] + total_length)
    pins = [pinned[(anchor + k) % n] for k in range(n)] + [True]
    pins[0] = True
    mins_rot = [mins[(anchor + k) % n] for k in range(n)]
    projected = _project_open(rotated, mins_rot, pins)

    result = list(params)
    for k in range(n):
        result[(anchor + k) % n] = (projected[k]
                                    - (total_length
                                       if anchor + k >= n else 0.0))
    return result


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
# Chain relaxation (orchestration)
# ---------------------------------------------------------------------------

def relax_chain_step(curve, params, sides, pinned, side_blend, stiffness,
                     min_spacing=0.3):
    """One relax pass over a chain on a fixed, prefitted curve.

    params are the current arc-length positions of the chain vertices on
    the curve; the returned list is the relaxed positions. Keeping the
    curve fixed across steps means iterating can never drift the shape of
    the loop.

    Crowding protection (converging crossing flows aiming neighboring
    vertices at nearly the same spot) is two-layered: IRLS passes add
    one-sided springs to pairs compressed below min_spacing * their
    original gap, then a PAVA projection enforces the floor exactly.
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

    springs = {}
    new_params = params
    for _ in range(3):
        deltas = solve_relaxed_params(targets, pinned, stiffness,
                                      curve.closed, spacing=springs or None)
        new_params = [params[i] + deltas[i] for i in range(n)]
        added = False
        for i in range(len(gaps)):
            nxt = (i + 1) % n
            wrap = length if nxt == 0 else 0.0
            gap_now = new_params[nxt] + wrap - new_params[i]
            if (gap_now < min_spacing * gaps[i] - 1.0e-9
                    and i not in springs):
                base = params[nxt] + wrap - params[i]
                springs[i] = (_CROWD_WEIGHT, min_spacing * gaps[i] - base)
                added = True
        if not added:
            break

    return project_min_spacing(new_params, gaps, pinned, min_spacing,
                               curve.closed, length)


def relax_chain(points, closed, sides, pinned, side_blend=0.0,
                stiffness=1.0, factor=1.0, iterations=1, min_spacing=0.3):
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
                                  side_blend, stiffness, min_spacing)

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
