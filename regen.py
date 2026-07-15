# milkyEdgeFlowTools -- Regenerate Crossing Flows (requirements.md ch. 11)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Strip analysis and mesh rebuild live in this module and are callable
# without any UI (the operator's execute() path uses them directly, which
# is what the headless integration tests exercise). The modal adjustment
# mode is layered on top in this module as well.

from collections import deque

import bmesh
import bpy

from . import core

END_WALK_MAX_STEPS = 256
ADJACENCY_WALK_MAX_STEPS = 64


class StripError(Exception):
    """Analysis failure with a translatable message."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _edge_key(edge):
    return frozenset((edge.verts[0].index, edge.verts[1].index))


def _next_path_edge(vert, incoming):
    """Continue a crossing path through a vertex.

    Unlike the relax tool's ring walk this also works at 3-valence
    boundary vertices: the continuation is the unique edge sharing no
    face with the incoming edge. Poles (several candidates) return None.
    """
    incoming_faces = set(incoming.link_faces)
    candidates = [e for e in vert.link_edges
                  if e is not incoming
                  and not (set(e.link_faces) & incoming_faces)]
    return candidates[0] if len(candidates) == 1 else None


def _walk_crossing(vert, edge, stop_pred, max_steps):
    """Walk a crossing path from `vert` through `edge`.

    Returns the vertices visited after `vert` (nearest first), ending with
    the first vertex satisfying stop_pred, or None when the walk dies or
    exceeds max_steps without satisfying it.
    """
    path = []
    cur_vert = edge.other_vert(vert)
    cur_edge = edge
    for _ in range(max_steps):
        path.append(cur_vert)
        if stop_pred(cur_vert):
            return path
        nxt = _next_path_edge(cur_vert, cur_edge)
        if nxt is None:
            return None
        cur_vert = nxt.other_vert(cur_vert)
        cur_edge = nxt
    return None


def _edge_between(vert_a, vert_b):
    for edge in vert_a.link_edges:
        if edge.other_vert(vert_a) is vert_b:
            return edge
    return None


# ---------------------------------------------------------------------------
# Strip analysis
# ---------------------------------------------------------------------------

class StripData:
    """Everything the generator and the rebuild need."""

    def __init__(self):
        self.rails = []          # ordered+oriented vert index lists
        self.curves = []         # CatmullRomCurve per rail (object space)
        self.strip_faces = set()  # face indices to delete
        self.end_paths = []      # two vert-index paths across each end
        self.interior_verts = set()
        self.material_index = 0
        self.use_smooth = True


def analyze_strip(bm):
    """Validate the selection and describe the strip. Raises StripError."""
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    selected = [e for e in bm.edges if e.select]
    if not selected:
        raise StripError("Select two or more parallel edge chains")
    chains, skipped = core.decompose_chains(
        [(e.verts[0].index, e.verts[1].index) for e in selected])
    if skipped or len(chains) < 2:
        raise StripError("Selected chains must form a row of parallel "
                         "open loops")
    if any(closed for _, closed in chains):
        raise StripError("Closed loops are not supported yet")

    vert_chain = {}
    for ci, (vert_indices, _closed) in enumerate(chains):
        for vi in vert_indices:
            vert_chain[vi] = ci
    chain_keys = {_edge_key(e) for e in selected}

    # Rail adjacency via crossing walks from mid-chain vertices.
    pairs = set()
    for ci, (vert_indices, _closed) in enumerate(chains):
        vert = bm.verts[vert_indices[len(vert_indices) // 2]]
        chain_faces = set()
        for e in vert.link_edges:
            if _edge_key(e) in chain_keys:
                chain_faces.update(e.link_faces)
        for edge in vert.link_edges:
            if _edge_key(edge) in chain_keys:
                continue
            if not (set(edge.link_faces) & chain_faces):
                continue
            hit = _walk_crossing(
                vert, edge,
                lambda v: vert_chain.get(v.index) not in (None, ci),
                ADJACENCY_WALK_MAX_STEPS)
            if hit:
                pairs.add(frozenset((ci, vert_chain[hit[-1].index])))
    order = core.order_rails(len(chains), [tuple(p) for p in pairs])
    if order is None:
        raise StripError("Selected chains must form a row of parallel "
                         "open loops")

    # Orient rails consistently (minimize summed endpoint distances).
    rails = [list(chains[ci][0]) for ci in order]
    for k in range(1, len(rails)):
        prev, cur = rails[k - 1], rails[k]
        a0 = bm.verts[prev[0]].co
        a1 = bm.verts[prev[-1]].co
        b0 = bm.verts[cur[0]].co
        b1 = bm.verts[cur[-1]].co
        if ((a0 - b0).length + (a1 - b1).length
                > (a0 - b1).length + (a1 - b0).length):
            cur.reverse()

    data = StripData()
    data.rails = rails

    # End paths: crossing walks between the outermost rails' endpoints.
    first, last = rails[0], rails[-1]
    for end in (0, -1):
        start_vert = bm.verts[first[end]]
        target = last[end]
        path = None
        for edge in start_vert.link_edges:
            if _edge_key(edge) in chain_keys:
                continue
            walked = _walk_crossing(
                start_vert, edge,
                lambda v: v.index in (last[0], last[-1]),
                END_WALK_MAX_STEPS)
            if walked and walked[-1].index == target:
                path = [first[end]] + [v.index for v in walked]
                break
        if path is None:
            raise StripError("Could not trace the strip ends; the end "
                             "rows must be walkable crossing paths")
        on_path = set(path)
        for rail in rails[1:-1]:
            if rail[end] not in on_path:
                raise StripError("Could not trace the strip ends; the end "
                                 "rows must be walkable crossing paths")
        data.end_paths.append(path)

    # Flood fill the strip faces from the corner face (the unique face
    # sharing both the first rail edge and the first end-path edge), with
    # rail and end-path edges as barriers.
    barrier = set(chain_keys)
    for path in data.end_paths:
        for a, b in zip(path, path[1:]):
            barrier.add(frozenset((a, b)))

    corner = bm.verts[first[0]]
    rail_edge = _edge_between(corner, bm.verts[first[1]])
    end_edge = _edge_between(corner, bm.verts[data.end_paths[0][1]])
    if rail_edge is None or end_edge is None:
        raise StripError("Could not trace the strip ends; the end "
                         "rows must be walkable crossing paths")
    seeds = set(rail_edge.link_faces) & set(end_edge.link_faces)
    if len(seeds) != 1:
        raise StripError("Could not trace the strip ends; the end "
                         "rows must be walkable crossing paths")

    stack = list(seeds)
    visited = set()
    while stack:
        face = stack.pop()
        if face.index in visited:
            continue
        visited.add(face.index)
        for edge in face.edges:
            if _edge_key(edge) in barrier:
                continue
            for other in edge.link_faces:
                if other.index not in visited:
                    stack.append(other)
    data.strip_faces = visited

    rail_vert_all = {vi for rail in rails for vi in rail}
    end_vert_all = {vi for path in data.end_paths for vi in path}
    for fi in data.strip_faces:
        for vert in bm.faces[fi].verts:
            if (vert.index not in rail_vert_all
                    and vert.index not in end_vert_all):
                data.interior_verts.add(vert.index)

    # Hole check: any strip-region boundary edge that is neither a rail
    # edge nor an end-path edge means the region has extra boundaries
    # (interior holes, or a fill that escaped a malformed strip).
    for fi in data.strip_faces:
        for edge in bm.faces[fi].edges:
            inside = sum(1 for f in edge.link_faces
                         if f.index in data.strip_faces)
            if inside == 1 and _edge_key(edge) not in barrier:
                raise StripError("The strip contains a hole; fill it or "
                                 "exclude it first")

    materials = {}
    smooth_votes = 0
    for fi in data.strip_faces:
        face = bm.faces[fi]
        materials[face.material_index] = materials.get(
            face.material_index, 0) + 1
        smooth_votes += 1 if face.smooth else -1
    if materials:
        data.material_index = max(materials, key=materials.get)
    data.use_smooth = smooth_votes >= 0

    data.curves = [core.CatmullRomCurve(
        [tuple(bm.verts[vi].co) for vi in rail], closed=False)
        for rail in rails]
    return data


def default_flow_count(data):
    outer = (len(data.rails[0]) + len(data.rails[-1])) / 2.0
    return max(2, round(outer))


def generate(data, count, bias, locked_rails=(), constraints=None):
    """Generate flows: per-flow lists of arc params, one per rail."""
    locked_ratios = None
    merged = dict(constraints or {})
    if locked_rails:
        ratio_lists = []
        for rj in locked_rails:
            curve = data.curves[rj]
            ratio_lists.append([s / curve.total_length
                                for s in curve.knot_params])
        locked_ratios = [sum(rs[i] for rs in ratio_lists) / len(ratio_lists)
                         for i in range(len(ratio_lists[0]))]
        for rj in locked_rails:
            for i, s in enumerate(data.curves[rj].knot_params):
                merged.setdefault((i, rj), s)
        # Flow endpoints on the outer rails copy the locked vertices'
        # normalized arc ratios (averaged over the locked chains).
        # Closest-point projection was tried here and retired: combined
        # with the default end locks and the constraint propagation, the
        # plain ratios produced the better strips in practice. Explicit
        # user constraints (drags) take precedence.
        for side in (0, len(data.curves) - 1):
            if side in locked_rails:
                continue
            side_length = data.curves[side].total_length
            for i, ratio in enumerate(locked_ratios):
                merged.setdefault((i, side), ratio * side_length)
    return core.generate_flows(data.curves, count, bias=bias,
                               locked_ratios=locked_ratios,
                               constraints=merged)


def default_end_constraints(data, flow_count):
    """Per-vertex locks holding the end rows on the rail endpoints.

    Applied when entering the adjustment mode (and re-applied on count
    changes); the user can unlock them per vertex to let the strip ends
    regenerate too.
    """
    constraints = {}
    for rj, curve in enumerate(data.curves):
        constraints[(0, rj)] = 0.0
        constraints[(flow_count - 1, rj)] = curve.total_length
    return constraints


def compose_flows(data, count, bias, locked_rails=(), constraints=None,
                  influence=2.0):
    """Base generation plus decaying propagation of vertex constraints.

    The base is generated without vertex constraints; each constrained
    flow is then re-smoothed through its constraints, and per rail the
    constrained rows' displacements are propagated to the free rows with
    a falloff of roughly `influence` flows.
    """
    base = generate(data, count, bias, locked_rails)
    constraints = constraints or {}
    rail_count = len(data.curves)

    rows = {}
    for (i, rj), s in constraints.items():
        if 0 <= i < len(base) and 0 <= rj < rail_count:
            rows.setdefault(i, {})[rj] = s
    constrained_rows = {}
    for i, row_constraints in rows.items():
        params = list(base[i])
        pinned = [False] * rail_count
        pinned[0] = pinned[-1] = True
        for rj in locked_rails:
            pinned[rj] = True
        for rj, s in row_constraints.items():
            params[rj] = s
            pinned[rj] = True
        constrained_rows[i] = core.smooth_flow_on_rails(
            data.curves, params, pinned)

    flows = core.propagate_flow_constraints(base, constrained_rows,
                                            influence)
    # Keep each rail's rows ordered and on the curve.
    for rj, curve in enumerate(data.curves):
        length = curve.total_length
        column = [flows[i][rj] for i in range(len(flows))]
        gap = 1.0e-3 * (length / max(1, len(flows) - 1))
        column = core.enforce_min_spacing(column, False, length, gap)
        for i in range(len(flows)):
            flows[i][rj] = min(max(column[i], 0.0), length)
    return flows


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def _monotone_nearest(old_params, new_params):
    """Map old params to new-param indices, order preserving."""
    mapping = []
    floor = 0
    for p in old_params:
        best = floor
        best_dist = abs(new_params[floor] - p)
        for j in range(floor, len(new_params)):
            dist = abs(new_params[j] - p)
            if dist < best_dist:
                best, best_dist = j, dist
        mapping.append(best)
        floor = best
    return mapping


def _split_end_path(path, endpoints):
    """Interior vert runs of an end path between consecutive rails."""
    position = {vi: k for k, vi in enumerate(path)}
    segments = []
    for rj in range(len(endpoints) - 1):
        a = position[endpoints[rj]]
        b = position[endpoints[rj + 1]]
        if a <= b:
            segments.append(path[a + 1:b])
        else:
            segment = path[b + 1:a]
            segment.reverse()
            segments.append(segment)
    return segments


def _match_orientation(bm, new_faces):
    """Flip new faces so their winding matches the surviving neighbors."""
    pending = set(new_faces)
    fixed = set()

    def loop_start(face, edge):
        for loop in face.loops:
            if loop.edge is edge:
                return loop.vert
        return None

    queue = deque()
    for face in new_faces:
        for edge in face.edges:
            for ref in edge.link_faces:
                if ref is not face and ref not in pending:
                    queue.append((face, ref, edge))
    while queue:
        face, ref, edge = queue.popleft()
        if face in fixed:
            continue
        a = loop_start(face, edge)
        b = loop_start(ref, edge)
        if a is not None and b is not None and a is b:
            face.normal_flip()
        fixed.add(face)
        for edge2 in face.edges:
            for nxt in edge2.link_faces:
                if nxt in pending and nxt not in fixed:
                    queue.append((nxt, face, edge2))
    leftovers = [f for f in new_faces if f not in fixed]
    if leftovers:
        bmesh.ops.recalc_face_normals(bm, faces=leftovers)


def apply_regeneration(bm, data, flows, locked_rails=()):
    """Delete the strip and rebuild it from the generated flows."""
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    rail_count = len(data.rails)
    flow_count = len(flows)
    end_vert_all = {vi for path in data.end_paths for vi in path}

    doomed = set(data.interior_verts)
    for rj, rail in enumerate(data.rails):
        if rj not in locked_rails:
            doomed.update(rail[1:-1])
    doomed -= end_vert_all

    # Tokenize outside faces that touch doomed rail verts, before deleting.
    rail_param = {}
    for rj, rail in enumerate(data.rails):
        for i, vi in enumerate(rail):
            rail_param[vi] = (rj, data.curves[rj].knot_params[i])
    outside_tokens = []
    for face in bm.faces:
        if face.index in data.strip_faces:
            continue
        if not any(v.index in doomed for v in face.verts):
            continue
        tokens = []
        for vert in face.verts:
            if vert.index in rail_param:
                rj, param = rail_param[vert.index]
                tokens.append(('r', rj, param))
            elif vert.index in doomed:
                raise StripError("The strip contains a hole; fill it or "
                                 "exclude it first")
            else:
                tokens.append(('v', vert))
        # Rotate so a run of rail tokens never wraps around the list end.
        pivot = next((k for k, t in enumerate(tokens) if t[0] == 'v'), 0)
        tokens = tokens[pivot:] + tokens[:pivot]
        outside_tokens.append((tokens, face.material_index, face.smooth))

    segment_verts = []
    for end, path in enumerate(data.end_paths):
        endpoints = [rail[0 if end == 0 else -1] for rail in data.rails]
        segment_verts.append(
            [[bm.verts[vi] for vi in seg]
             for seg in _split_end_path(path, endpoints)])
    rail_end_verts = [(bm.verts[rail[0]], bm.verts[rail[-1]])
                      for rail in data.rails]
    locked_rail_verts = {rj: [bm.verts[vi] for vi in data.rails[rj]]
                         for rj in locked_rails}

    bmesh.ops.delete(bm, geom=[bm.verts[vi] for vi in doomed],
                     context='VERTS')

    # New grid vertices. End rows sitting on the rail endpoints reuse the
    # preserved endpoint verts; locked rails reuse all their original
    # verts; everything else is created on the rail curve.
    grid = [[None] * rail_count for _ in range(flow_count)]
    new_rail_params = [[] for _ in range(rail_count)]
    for rj in range(rail_count):
        curve = data.curves[rj]
        eps = 1.0e-6 * max(curve.total_length, 1.0e-9)
        for i in range(flow_count):
            if rj in locked_rails:
                grid[i][rj] = locked_rail_verts[rj][i]
                new_rail_params[rj].append(curve.knot_params[i])
            elif i == 0 and flows[i][rj] <= eps:
                grid[i][rj] = rail_end_verts[rj][0]
                new_rail_params[rj].append(0.0)
            elif (i == flow_count - 1
                    and flows[i][rj] >= curve.total_length - eps):
                grid[i][rj] = rail_end_verts[rj][1]
                new_rail_params[rj].append(curve.total_length)
            else:
                grid[i][rj] = bm.verts.new(curve.point_at(flows[i][rj]))
                new_rail_params[rj].append(flows[i][rj])

    # Rail vertex sequences including the preserved endpoint verts when an
    # end row moved inward; used for outside-face mapping and selection.
    rail_seq_verts = []
    rail_seq_params = []
    for rj in range(rail_count):
        verts_seq = []
        params_seq = []
        if grid[0][rj] is not rail_end_verts[rj][0]:
            verts_seq.append(rail_end_verts[rj][0])
            params_seq.append(0.0)
        for i in range(flow_count):
            verts_seq.append(grid[i][rj])
            params_seq.append(new_rail_params[rj][i])
        if grid[flow_count - 1][rj] is not rail_end_verts[rj][1]:
            verts_seq.append(rail_end_verts[rj][1])
            params_seq.append(data.curves[rj].total_length)
        rail_seq_verts.append(verts_seq)
        rail_seq_params.append(params_seq)

    new_faces = []

    def make_face(verts, material, smooth):
        unique = []
        for v in verts:
            if not unique or unique[-1] is not v:
                unique.append(v)
        if len(unique) > 1 and unique[0] is unique[-1]:
            unique.pop()
        if len(unique) < 3:
            return
        try:
            face = bm.faces.new(unique)
        except ValueError:
            return  # face already exists
        face.material_index = material
        face.smooth = smooth
        new_faces.append(face)

    # Strip faces. Extreme rows still sitting on the endpoints absorb the
    # kept end-path verts as n-gons; end rows that moved inward instead
    # get a separate band n-gon connecting them back to the preserved end
    # boundary (endpoint verts + end path).
    def row_on_endpoints(row, end, rj):
        return (grid[row][rj] is rail_end_verts[rj][end]
                and grid[row][rj + 1] is rail_end_verts[rj + 1][end])

    for i in range(flow_count - 1):
        for rj in range(rail_count - 1):
            verts = [grid[i][rj]]
            if i == 0 and row_on_endpoints(0, 0, rj):
                verts.extend(segment_verts[0][rj])
            verts.append(grid[i][rj + 1])
            verts.append(grid[i + 1][rj + 1])
            if (i + 1 == flow_count - 1
                    and row_on_endpoints(flow_count - 1, 1, rj)):
                verts.extend(reversed(segment_verts[1][rj]))
            verts.append(grid[i + 1][rj])
            make_face(verts, data.material_index, data.use_smooth)

    for end, row in ((0, 0), (1, flow_count - 1)):
        for rj in range(rail_count - 1):
            if row_on_endpoints(row, end, rj):
                continue
            verts = ([rail_end_verts[rj][end]] + segment_verts[end][rj]
                     + [rail_end_verts[rj + 1][end],
                        grid[row][rj + 1], grid[row][rj]])
            make_face(verts, data.material_index, data.use_smooth)

    # Rebuild the tokenized outside faces: rail runs are replaced by the
    # new rail verts via a monotone-nearest param mapping.
    for tokens, material, smooth in outside_tokens:
        verts = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token[0] == 'v':
                verts.append(token[1])
                idx += 1
                continue
            rj = token[1]
            run = []
            while (idx < len(tokens) and tokens[idx][0] == 'r'
                   and tokens[idx][1] == rj):
                run.append(tokens[idx][2])
                idx += 1
            reverse = len(run) > 1 and run[0] > run[-1]
            params = list(reversed(run)) if reverse else run
            mapped = [rail_seq_verts[rj][i] for i in
                      _monotone_nearest(params, rail_seq_params[rj])]
            if reverse:
                mapped.reverse()
            verts.extend(mapped)
        make_face(verts, material, smooth)

    _match_orientation(bm, new_faces)

    # Select the regenerated rails (and nothing else).
    for vert in bm.verts:
        vert.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    for rj in range(rail_count):
        seq = rail_seq_verts[rj]
        for a, b in zip(seq, seq[1:]):
            edge = _edge_between(a, b)
            if edge:
                edge.select = True
                a.select = True
                b.select = True


def run_regeneration(obj, count=None, bias=0.5, locked_rails=(),
                     constraints=None, influence=2.0):
    """Full UI-independent pipeline on an edit-mesh object.

    The end rows get their default endpoint locks; explicit constraints
    override them.
    """
    bm = bmesh.from_edit_mesh(obj.data)
    data = analyze_strip(bm)
    if locked_rails:
        count = len(data.rails[locked_rails[0]])
    if count is None:
        count = default_flow_count(data)
    merged = default_end_constraints(data, count)
    merged.update(constraints or {})
    flows = compose_flows(data, count, bias, locked_rails, merged,
                          influence)
    apply_regeneration(bm, data, flows, locked_rails)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    return len(flows)


# ---------------------------------------------------------------------------
# Modal adjustment mode
# ---------------------------------------------------------------------------

COLOR_RAIL = (0.55, 0.55, 0.55, 1.0)
COLOR_RAIL_LOCKED = (1.0, 0.85, 0.2, 1.0)
COLOR_FLOW = (0.3, 0.85, 1.0, 1.0)
COLOR_VERT_FILL = (1.0, 1.0, 1.0, 1.0)
COLOR_VERT_BORDER = (0.3, 0.85, 1.0, 1.0)
COLOR_VERT_LOCKED = (1.0, 0.85, 0.2, 1.0)
CURVE_DRAW_SAMPLES = 64
PICK_VERT_PX = 12.0
PICK_CURVE_PX = 9.0

_session = None
_suppress_updates = False


class _Session:
    def __init__(self, obj, data):
        self.object_name = obj.name
        self.matrix = obj.matrix_world.copy()
        self.data = data
        self.count = default_flow_count(data)
        self.bias = 0.5
        self.influence = 2.0
        self.locked = set()
        self.constraints = {}
        self.flows = []
        self.message = ""
        self.rail_lines = []   # world-space polylines per rail
        self.flow_lines = []   # world-space polylines per flow
        self.handles = {}      # (flow_i, rail_j) -> world position
        self.drag = None       # (flow_i, rail_j) while dragging
        self.request = None    # 'APPLY' / 'CANCEL' from the panel
        self.draw_3d = None
        self.draw_2d = None
        self.timer = None


def _world(session, point):
    from mathutils import Vector
    return session.matrix @ Vector(point)


def _refresh_caches(session):
    session.rail_lines = []
    for curve in session.data.curves:
        length = curve.total_length
        line = [_world(session, curve.point_at(
            length * k / CURVE_DRAW_SAMPLES))
            for k in range(CURVE_DRAW_SAMPLES + 1)]
        session.rail_lines.append(line)
    session.flow_lines = []
    session.handles = {}
    for i, flow in enumerate(session.flows):
        line = []
        for rj, s in enumerate(flow):
            pos = _world(session, session.data.curves[rj].point_at(s))
            line.append(pos)
            session.handles[(i, rj)] = pos
        session.flow_lines.append(line)


def _reset_constraints(session):
    session.constraints = default_end_constraints(session.data,
                                                  session.count)


def _regenerate(session):
    session.flows = compose_flows(session.data, session.count, session.bias,
                                  tuple(session.locked), session.constraints,
                                  session.influence)
    session.count = len(session.flows)
    _refresh_caches(session)


def _set_count(session, count):
    if session.locked:
        session.message = "Flow count is fixed by a locked chain"
        return
    count = max(2, count)
    if count == session.count:
        return
    session.count = count
    _reset_constraints(session)
    session.message = ""
    _regenerate(session)
    _sync_settings(session)


def _toggle_lock(session, rail_j):
    if rail_j in session.locked:
        session.locked.discard(rail_j)
        session.message = ""
        _reset_constraints(session)
        _regenerate(session)
        _sync_settings(session)
        return
    rail_len = len(session.data.rails[rail_j])
    for other in session.locked:
        if len(session.data.rails[other]) != rail_len:
            session.message = (
                "Locked chains must have the same vertex count "
                "(%d vs %d)" % (len(session.data.rails[other]), rail_len))
            return
    session.locked.add(rail_j)
    session.count = rail_len
    _reset_constraints(session)
    session.message = ""
    _regenerate(session)
    _sync_settings(session)


def _sync_settings(session):
    global _suppress_updates
    _suppress_updates = True
    try:
        settings = bpy.context.window_manager.milky_regen
        settings.flow_count = session.count
        settings.curvature_bias = session.bias
        settings.influence = session.influence
    finally:
        _suppress_updates = False


def _settings_changed(_self, _context):
    if _suppress_updates or _session is None:
        return
    settings = bpy.context.window_manager.milky_regen
    if settings.flow_count != _session.count:
        _set_count(_session, settings.flow_count)
    if abs(settings.curvature_bias - _session.bias) > 1.0e-6:
        _session.bias = settings.curvature_bias
        _regenerate(_session)
    if abs(settings.influence - _session.influence) > 1.0e-6:
        _session.influence = settings.influence
        _regenerate(_session)


# --- drawing ---------------------------------------------------------------

def _draw_view3d():
    session = _session
    if session is None:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')

    def draw_lines(line, color, width):
        if len(line) < 2:
            return
        gpu.state.line_width_set(width)
        coords = []
        for a, b in zip(line, line[1:]):
            coords.extend((a, b))
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        shader.uniform_float("color", color)
        batch.draw(shader)

    def draw_points(coords, color, size):
        if not coords:
            return
        gpu.state.point_size_set(size)
        batch = batch_for_shader(shader, 'POINTS', {"pos": coords})
        shader.uniform_float("color", color)
        batch.draw(shader)

    for rj, line in enumerate(session.rail_lines):
        locked = rj in session.locked
        draw_lines(line, COLOR_RAIL_LOCKED if locked else COLOR_RAIL,
                   3.0 if locked else 2.0)
    for line in session.flow_lines:
        draw_lines(line, COLOR_FLOW, 2.0)

    plain = []
    locked_pts = []
    for (i, rj), pos in session.handles.items():
        if (i, rj) in session.constraints or rj in session.locked:
            locked_pts.append(pos)
        else:
            plain.append(pos)
    draw_points(plain, COLOR_VERT_BORDER, 10.0)
    draw_points(locked_pts, COLOR_VERT_LOCKED, 10.0)
    draw_points(plain + locked_pts, COLOR_VERT_FILL, 6.0)

    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.blend_set('NONE')


def _draw_hud():
    session = _session
    if session is None:
        return
    import blf
    iface = bpy.app.translations.pgettext_iface
    font = 0
    lines = [
        iface("Regenerate Crossing Flows"),
        iface("Flows: %d") % session.count
        + (iface(" (locked)") if session.locked else ""),
        iface("Drag: move vertex   Shift+Click: lock/unlock   "
              "+/-: flow count   Enter: apply   Esc: cancel"),
    ]
    if session.message:
        lines.insert(2, iface(session.message))
    blf.size(font, 13)
    y = 24
    for line in reversed(lines):
        blf.position(font, 20, y, 0)
        blf.color(font, 1.0, 1.0, 1.0, 0.9)
        blf.draw(font, line)
        y += 20


# --- picking ---------------------------------------------------------------

def _region_data(context, event):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    return region, rv3d, (event.mouse_region_x, event.mouse_region_y)


def _pick_handle(context, event):
    from bpy_extras import view3d_utils
    hit = _region_data(context, event)
    if hit is None or _session is None:
        return None
    region, rv3d, coord = hit
    best = None
    best_dist = PICK_VERT_PX
    for key, pos in _session.handles.items():
        screen = view3d_utils.location_3d_to_region_2d(region, rv3d, pos)
        if screen is None:
            continue
        dist = ((screen[0] - coord[0]) ** 2
                + (screen[1] - coord[1]) ** 2) ** 0.5
        if dist < best_dist:
            best, best_dist = key, dist
    return best


def _pick_rail(context, event):
    from bpy_extras import view3d_utils
    hit = _region_data(context, event)
    if hit is None or _session is None:
        return None
    region, rv3d, coord = hit
    best = None
    best_dist = PICK_CURVE_PX
    for rj, line in enumerate(_session.rail_lines):
        for pos in line:
            screen = view3d_utils.location_3d_to_region_2d(region, rv3d,
                                                           pos)
            if screen is None:
                continue
            dist = ((screen[0] - coord[0]) ** 2
                    + (screen[1] - coord[1]) ** 2) ** 0.5
            if dist < best_dist:
                best, best_dist = rj, dist
    return best


def _drag_param(context, event, rail_j):
    from bpy_extras import view3d_utils
    hit = _region_data(context, event)
    if hit is None or _session is None:
        return None
    region, rv3d, coord = hit
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    inverse = _session.matrix.inverted()
    obj_origin = inverse @ origin
    obj_dir = (inverse.to_3x3() @ direction).normalized()
    curve = _session.data.curves[rail_j]
    s, _dist = curve.closest_param_to_ray(tuple(obj_origin), tuple(obj_dir))
    return s


def _mouse_in_sidebar(context, event):
    for region in context.area.regions:
        if region.type in {'UI', 'HEADER', 'TOOL_HEADER'}:
            if (region.x <= event.mouse_x < region.x + region.width
                    and region.y <= event.mouse_y
                    < region.y + region.height):
                return True
    return False


# --- operator / panel ------------------------------------------------------

class MilkyRegenSettings(bpy.types.PropertyGroup):
    flow_count: bpy.props.IntProperty(
        name="Flow Count",
        description="Number of crossing flows to generate",
        min=2, default=8, update=_settings_changed,
    )
    curvature_bias: bpy.props.FloatProperty(
        name="Curvature Bias",
        description=("Bias of the subdivision density toward curved "
                     "regions (0 = uniform)"),
        min=0.0, max=1.0, default=0.5, subtype='FACTOR',
        update=_settings_changed,
    )
    influence: bpy.props.FloatProperty(
        name="Influence",
        description=("How many neighboring flows a locked or dragged "
                     "vertex influences (0 = constrained flows only)"),
        min=0.0, max=10.0, default=2.0,
        update=_settings_changed,
    )


class MESH_OT_milky_regen_request(bpy.types.Operator):
    """Apply or cancel the running adjustment mode from the panel."""
    bl_idname = "mesh.milky_regen_request"
    bl_label = "Regenerate Crossing Flows Request"
    bl_options = {'INTERNAL'}

    action: bpy.props.EnumProperty(items=[('APPLY', "Apply", ""),
                                          ('CANCEL', "Cancel", "")])

    def execute(self, context):
        if _session is not None:
            _session.request = self.action
        return {'FINISHED'}


class VIEW3D_PT_milky_regen(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "milkyEdgeFlow"
    bl_label = "Regenerate Crossing Flows"

    @classmethod
    def poll(cls, context):
        return _session is not None

    def draw(self, context):
        layout = self.layout
        settings = context.window_manager.milky_regen
        col = layout.column()
        col.enabled = not _session.locked
        col.prop(settings, "flow_count")
        col.prop(settings, "curvature_bias")
        layout.prop(settings, "influence")
        if _session.message:
            layout.label(text=_session.message, icon='ERROR')
        row = layout.row()
        row.operator("mesh.milky_regen_request",
                     text="Apply").action = 'APPLY'
        row.operator("mesh.milky_regen_request",
                     text="Cancel").action = 'CANCEL'


class MESH_OT_milky_regenerate_crossing_flows(bpy.types.Operator):
    bl_idname = "mesh.milky_regenerate_crossing_flows"
    bl_label = "Regenerate Crossing Flows"
    bl_description = ("Delete the strip between the outermost selected "
                      "chains and regenerate crossing flows on fitted "
                      "curves")
    bl_options = {'REGISTER', 'UNDO'}

    flow_count: bpy.props.IntProperty(
        name="Flow Count",
        description=("Number of crossing flows to generate "
                     "(0 = keep a similar density)"),
        min=0, default=0,
    )
    curvature_bias: bpy.props.FloatProperty(
        name="Curvature Bias",
        description=("Bias of the subdivision density toward curved "
                     "regions (0 = uniform)"),
        min=0.0, max=1.0, default=0.5, subtype='FACTOR',
    )
    influence: bpy.props.FloatProperty(
        name="Influence",
        description=("How many neighboring flows a locked or dragged "
                     "vertex influences (0 = constrained flows only)"),
        min=0.0, max=10.0, default=2.0,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and context.mode == 'EDIT_MESH' and _session is None)

    # Headless / redo path: apply immediately with the given parameters.
    def execute(self, context):
        obj = context.active_object
        try:
            count = self.flow_count if self.flow_count >= 2 else None
            run_regeneration(obj, count, self.curvature_bias,
                             influence=self.influence)
        except StripError as exc:
            self.report({'ERROR'},
                        bpy.app.translations.pgettext_rpt(exc.message))
            return {'CANCELLED'}
        return {'FINISHED'}

    def invoke(self, context, event):
        global _session
        if bpy.app.background:
            return self.execute(context)
        obj = context.active_object
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            data = analyze_strip(bm)
        except StripError as exc:
            self.report({'ERROR'},
                        bpy.app.translations.pgettext_rpt(exc.message))
            return {'CANCELLED'}

        session = _Session(obj, data)
        session.bias = self.curvature_bias
        session.influence = self.influence
        if self.flow_count >= 2:
            session.count = self.flow_count
        _session = session
        _reset_constraints(session)
        _regenerate(session)
        _sync_settings(session)

        session.draw_3d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_view3d, (), 'WINDOW', 'POST_VIEW')
        session.draw_2d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (), 'WINDOW', 'POST_PIXEL')
        session.timer = context.window_manager.event_timer_add(
            0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        session = _session
        if session is None:
            return {'CANCELLED'}
        context.area.tag_redraw()

        if event.type == 'TIMER':
            if session.request == 'APPLY':
                return self._finish(context, apply_result=True)
            if session.request == 'CANCEL':
                return self._finish(context, apply_result=False)
            return {'RUNNING_MODAL'}

        if _mouse_in_sidebar(context, event):
            return {'PASS_THROUGH'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}

        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            return self._finish(context, apply_result=True)
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._finish(context, apply_result=False)

        if (event.type in {'EQUAL', 'NUMPAD_PLUS'}
                and event.value == 'PRESS'):
            _set_count(session, session.count + 1)
            return {'RUNNING_MODAL'}
        if (event.type in {'MINUS', 'NUMPAD_MINUS'}
                and event.value == 'PRESS'):
            _set_count(session, session.count - 1)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if event.shift:
                handle = _pick_handle(context, event)
                if handle is not None:
                    flow_i, rail_j = handle
                    if rail_j in session.locked:
                        return {'RUNNING_MODAL'}
                    if handle in session.constraints:
                        del session.constraints[handle]
                    else:
                        session.constraints[handle] = \
                            session.flows[flow_i][rail_j]
                    session.message = ""
                    _regenerate(session)
                    return {'RUNNING_MODAL'}
                rail = _pick_rail(context, event)
                if rail is not None:
                    _toggle_lock(session, rail)
                return {'RUNNING_MODAL'}
            handle = _pick_handle(context, event)
            if handle is not None:
                flow_i, rail_j = handle
                if rail_j not in session.locked:
                    session.drag = handle
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and session.drag is not None:
            flow_i, rail_j = session.drag
            param = _drag_param(context, event, rail_j)
            if param is not None:
                session.constraints[(flow_i, rail_j)] = param
                _regenerate(session)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            session.drag = None
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _finish(self, context, apply_result):
        global _session
        session = _session
        result = {'CANCELLED'}
        if apply_result and session is not None:
            obj = bpy.data.objects.get(session.object_name)
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                apply_regeneration(bm, session.data, session.flows,
                                   tuple(session.locked))
                bmesh.update_edit_mesh(obj.data, loop_triangles=True,
                                       destructive=True)
                result = {'FINISHED'}
            except StripError as exc:
                self.report({'ERROR'},
                            bpy.app.translations.pgettext_rpt(exc.message))
        if session is not None:
            if session.draw_3d is not None:
                bpy.types.SpaceView3D.draw_handler_remove(
                    session.draw_3d, 'WINDOW')
            if session.draw_2d is not None:
                bpy.types.SpaceView3D.draw_handler_remove(
                    session.draw_2d, 'WINDOW')
            if session.timer is not None:
                context.window_manager.event_timer_remove(session.timer)
        _session = None
        context.area.tag_redraw()
        return result


classes = (
    MilkyRegenSettings,
    MESH_OT_milky_regen_request,
    MESH_OT_milky_regenerate_crossing_flows,
    VIEW3D_PT_milky_regen,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.milky_regen = bpy.props.PointerProperty(
        type=MilkyRegenSettings)


def unregister():
    del bpy.types.WindowManager.milky_regen
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
