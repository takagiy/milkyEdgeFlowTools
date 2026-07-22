"""Integration test for the operator.

Runs against the official bpy wheel (uv run poe test-blender) or inside a
real Blender (uv run poe test-blender-app).

Builds a 7x7 grid whose middle column is shifted along the loop direction,
selects that column as an edge loop, runs Relax Crossing Flows, and checks
that the crossing rows pulled the vertices back while boundary vertices
stayed pinned.
"""

import json
import math
import os
import sys

# bpy must be imported before bmesh: the standalone bpy wheel sets up the
# search path for its bundled modules on first import.
import bpy
import bmesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import milkyEdgeFlowTools  # noqa: E402
from milkyEdgeFlowTools import regen as milky_regen  # noqa: E402

N = 7            # verts per side
COL = 3          # selected column (edge loop along +Y)
SHIFT = 0.35     # displacement of the interior column verts along the loop


def vid(col, row):
    return row * N + col


def build_grid_object():
    verts = []
    for row in range(N):
        for col in range(N):
            y = float(row)
            if col == COL and 0 < row < N - 1:
                y += SHIFT
            verts.append((float(col), y, 0.0))
    faces = []
    for row in range(N - 1):
        for col in range(N - 1):
            faces.append((vid(col, row), vid(col + 1, row),
                          vid(col + 1, row + 1), vid(col, row + 1)))
    mesh = bpy.data.meshes.new("grid")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("grid", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def select_column_loop(obj, row_from=0, row_to=N - 1):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    loop_keys = {frozenset((vid(COL, r), vid(COL, r + 1)))
                 for r in range(row_from, row_to)}
    for e in bm.edges:
        key = frozenset((e.verts[0].index, e.verts[1].index))
        e.select = key in loop_keys
    for v in bm.verts:
        v.select = any(e.select for e in v.link_edges)
    bm.select_flush(True)
    bmesh.update_edit_mesh(obj.data)


def fresh_grid():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing)
    return build_grid_object()


def test_lock_ends_and_iterations():
    """Partial selection (rows 1..5): normally both chain ends could move,
    with Lock Ends they must stay, and iterations must keep converging."""
    obj = fresh_grid()
    bpy.ops.object.mode_set(mode='EDIT')
    select_column_loop(obj, row_from=1, row_to=N - 2)

    result = bpy.ops.mesh.milky_relax_crossing_flows(lock_ends=True,
                                                     iterations='5')
    assert result == {'FINISHED'}, f"operator returned {result}"

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for row in (1, N - 2):
        co = bm.verts[vid(COL, row)].co
        assert abs(co.y - (row + SHIFT)) < 1e-6, \
            f"locked end at row {row} moved, y={co.y}"
    mid = bm.verts[vid(COL, N // 2)].co
    assert mid.y < N // 2 + SHIFT - 0.05, \
        f"middle vertex did not relax with locked ends, y={mid.y}"
    bpy.ops.object.mode_set(mode='OBJECT')


REG_COLS = 6
REG_ROWS = 5


def build_plain_grid(cols, rows):
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing)
    verts = [(float(c), float(r), 0.0)
             for r in range(rows) for c in range(cols)]
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            faces.append((r * cols + c, r * cols + c + 1,
                          (r + 1) * cols + c + 1, (r + 1) * cols + c))
    mesh = bpy.data.meshes.new("regen_grid")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("regen_grid", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def select_grid_columns(obj, cols, grid_cols, grid_rows):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    keys = {frozenset((r * grid_cols + c, (r + 1) * grid_cols + c))
            for c in cols for r in range(grid_rows - 1)}
    for e in bm.edges:
        e.select = frozenset(
            (e.verts[0].index, e.verts[1].index)) in keys
    for v in bm.verts:
        v.select = any(e.select for e in v.link_edges)
    # No select_flush: it would also select the rungs between adjacent
    # selected columns (all their verts are selected), turning the rails
    # into a branched ladder.
    bmesh.update_edit_mesh(obj.data)


def test_regenerate_basic():
    """6x5 grid, rails at x=1 and x=4, regenerate with 3 flows."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

    result = bpy.ops.mesh.milky_regenerate_crossing_flows(
        flow_count=3, curvature_bias=0.0)
    assert result == {'FINISHED'}, f"operator returned {result}"

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    assert len(bm.verts) == 20, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 10, f"face count {len(bm.faces)}"

    def verts_at_x(x):
        return sorted(round(v.co.y, 3) for v in bm.verts
                      if abs(v.co.x - x) < 1e-4)

    assert verts_at_x(1.0) == [0.0, 2.0, 4.0], verts_at_x(1.0)
    assert verts_at_x(4.0) == [0.0, 2.0, 4.0], verts_at_x(4.0)
    # Kept end-path verts survive; strip interior is gone.
    assert verts_at_x(2.0) == [0.0, 4.0], verts_at_x(2.0)
    assert verts_at_x(3.0) == [0.0, 4.0], verts_at_x(3.0)
    # Outside columns untouched.
    assert verts_at_x(0.0) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert verts_at_x(5.0) == [0.0, 1.0, 2.0, 3.0, 4.0]

    # Only rail edges are selected afterwards.
    for e in bm.edges:
        on_rail = all(abs(v.co.x - 1.0) < 1e-4 for v in e.verts) \
            or all(abs(v.co.x - 4.0) < 1e-4 for v in e.verts)
        assert e.select == on_rail, \
            f"selection mismatch at {[tuple(v.co) for v in e.verts]}"

    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_same_density():
    """flow_count matching the original density keeps positions."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

    result = bpy.ops.mesh.milky_regenerate_crossing_flows(
        flow_count=5, curvature_bias=0.0)
    assert result == {'FINISHED'}, f"operator returned {result}"

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == 24, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 12, f"face count {len(bm.faces)}"
    for x in (1.0, 4.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [0.0, 1.0, 2.0, 3.0, 4.0], ys
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_locked_rail():
    """A locked rail keeps its original vertices and dictates the count;
    the opposite rail is resampled at the geometric opposites."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

    count = milky_regen.run_regeneration(obj, locked_rails=(0,))
    assert count == REG_ROWS, f"flow count {count}"

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == 24, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 12, f"face count {len(bm.faces)}"
    for x in (1.0, 4.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [0.0, 1.0, 2.0, 3.0, 4.0], (x, ys)
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_trims_overshoot():
    """A chain running past the shared end row is trimmed back to the
    row instead of rejecting the strip (loop-select overshoot)."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    # Column 1 spans rows 0..3 only; column 4 runs on to row 4, one edge
    # past the shared end row at y=3.
    keys = {frozenset((r * REG_COLS + 1, (r + 1) * REG_COLS + 1))
            for r in range(3)}
    keys |= {frozenset((r * REG_COLS + 4, (r + 1) * REG_COLS + 4))
             for r in range(REG_ROWS - 1)}
    for e in bm.edges:
        e.select = frozenset((e.verts[0].index, e.verts[1].index)) in keys
    for v in bm.verts:
        v.select = any(e.select for e in v.link_edges)
    bmesh.update_edit_mesh(obj.data)

    count = milky_regen.run_regeneration(obj, bias=0.0)
    assert count == 4, f"flow count {count}"

    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == 26, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 14, f"face count {len(bm.faces)}"
    # Rows 0..3 are the regenerated strip; the y=4 verts sit outside it
    # (column 1's was never selected, column 4's was trimmed off).
    for x in (1.0, 4.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [0.0, 1.0, 2.0, 3.0, 4.0], (x, ys)
    assert not any(len(e.link_faces) > 2 for e in bm.edges), "non-manifold"
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_moved_end_row():
    """Unlocked (constrained inward) end rows move the shared boundary
    verts — rail endpoints and end-path interiors — onto the new row
    instead of bridging with a band n-gon; propagation off for
    exactness."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

    count = milky_regen.run_regeneration(
        obj, count=3, bias=0.0,
        constraints={(0, 0): 1.0, (0, 1): 1.0}, influence=0.0)
    assert count == 3, f"flow count {count}"

    bm = bmesh.from_edit_mesh(obj.data)
    # Rails hold exactly the moved row + middle + far endpoint; the old
    # endpoint verts themselves moved to y=1.
    for x in (1.0, 4.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [1.0, 2.0, 4.0], (x, ys)
    # End-path interior verts followed the row.
    for x in (2.0, 3.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [1.0, 4.0], (x, ys)
    assert len(bm.verts) == 20, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 10, f"face count {len(bm.faces)}"
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_three_rails():
    """Multi-bay strips: the fill must cross intermediate rails, and the
    unselected interior column between rails must be deleted."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 2, 4), REG_COLS, REG_ROWS)

    result = bpy.ops.mesh.milky_regenerate_crossing_flows(
        flow_count=3, curvature_bias=0.0)
    assert result == {'FINISHED'}, f"operator returned {result}"

    bm = bmesh.from_edit_mesh(obj.data)

    def verts_at_x(x):
        return sorted(round(v.co.y, 3) for v in bm.verts
                      if abs(v.co.x - x) < 1e-4)

    for x in (1.0, 2.0, 4.0):
        assert verts_at_x(x) == [0.0, 2.0, 4.0], (x, verts_at_x(x))
    # The unselected interior column survives only at the kept end verts.
    assert verts_at_x(3.0) == [0.0, 4.0], verts_at_x(3.0)
    assert len(bm.verts) == 21, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 12, f"face count {len(bm.faces)}"
    assert not any(len(e.link_faces) > 2 for e in bm.edges), "non-manifold"
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_denser_count():
    """Increasing the flow count must not create chord (diagonal) edges
    that skip new rail vertices in the rebuilt outside faces."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

    result = bpy.ops.mesh.milky_regenerate_crossing_flows(
        flow_count=7, curvature_bias=0.0)
    assert result == {'FINISHED'}, f"operator returned {result}"

    bm = bmesh.from_edit_mesh(obj.data)
    for x in (1.0, 4.0):
        rail = sorted((round(v.co.y, 5), v.index) for v in bm.verts
                      if abs(v.co.x - x) < 1e-4)
        order = {idx: k for k, (_y, idx) in enumerate(rail)}
        for e in bm.edges:
            if all(abs(v.co.x - x) < 1e-4 for v in e.verts):
                a, b = (order[v.index] for v in e.verts)
                assert abs(a - b) == 1, \
                    f"diagonal rail edge at x={x}: positions {a}-{b}"
    assert not any(len(e.link_faces) > 2 for e in bm.edges), "non-manifold"
    boundary = sum(1 for e in bm.edges if len(e.link_faces) < 2)
    assert boundary == 2 * (REG_COLS - 1) + 2 * (REG_ROWS - 1), boundary
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_locked_intermediate_rail():
    """Locking the middle rail splits the strip into two aimed segments;
    the locked rail keeps its original vertices."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 2, 4), REG_COLS, REG_ROWS)

    bm = bmesh.from_edit_mesh(obj.data)
    data = milky_regen.analyze_strip(bm)
    x_mid = [round(c.point_at(0.0)[0], 3) for c in data.curves].index(2.0)
    count = milky_regen.run_regeneration(obj, locked_rails=(x_mid,))
    assert count == REG_ROWS, f"flow count {count}"

    bm = bmesh.from_edit_mesh(obj.data)
    for x in (1.0, 2.0, 4.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [0.0, 1.0, 2.0, 3.0, 4.0], (x, ys)
    assert len(bm.verts) == 27, f"vert count {len(bm.verts)}"
    assert len(bm.faces) == 16, f"face count {len(bm.faces)}"
    assert not any(len(e.link_faces) > 2 for e in bm.edges), "non-manifold"
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_copy_flow_shape():
    """COPY mode: free rows are parallel copies of the reference row
    (fixed chord direction, scale only); end rows stay pinned by the
    default end locks with influence 0."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

    # Sculpt row 2 into a mild diagonal: y=2 on the left rail, y=2.5 on
    # the right one; every free row must repeat that +0.5 offset.
    count = milky_regen.run_regeneration(
        obj, count=5, bias=0.0,
        constraints={(2, 0): 2.0, (2, 1): 2.5},
        influence=0.0, mode='COPY', copy_row=2)
    assert count == 5, f"flow count {count}"

    bm = bmesh.from_edit_mesh(obj.data)
    ys1 = sorted(round(v.co.y, 2) for v in bm.verts
                 if abs(v.co.x - 1.0) < 1e-4)
    ys4 = sorted(round(v.co.y, 2) for v in bm.verts
                 if abs(v.co.x - 4.0) < 1e-4)
    assert ys1 == [0.0, 1.0, 2.0, 3.0, 4.0], ys1
    assert ys4 == [0.0, 1.5, 2.5, 3.5, 4.0], ys4
    assert not any(len(e.link_faces) > 2 for e in bm.edges), "non-manifold"
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), "mesh needed corrections"


def test_regenerate_copy_flow_shape_defaults_and_errors():
    """Without copy_row COPY falls back to the lowest constrained row
    (the default-locked end row); >1 locked chains are rejected."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 2, 4), REG_COLS, REG_ROWS)

    count = milky_regen.run_regeneration(obj, count=5, bias=0.0,
                                         mode='COPY')
    assert count == 5, f"flow count {count}"
    bm = bmesh.from_edit_mesh(obj.data)
    for x in (1.0, 2.0, 4.0):
        ys = sorted(round(v.co.y, 3) for v in bm.verts
                    if abs(v.co.x - x) < 1e-4)
        assert ys == [0.0, 1.0, 2.0, 3.0, 4.0], (x, ys)

    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 2, 4), REG_COLS, REG_ROWS)
    try:
        milky_regen.run_regeneration(obj, locked_rails=(0, 2),
                                     mode='COPY', copy_row=1)
        raise AssertionError("two locked chains were accepted")
    except milky_regen.StripError as exc:
        assert "at most one locked chain" in exc.message, exc.message
    bpy.ops.object.mode_set(mode='OBJECT')


def test_regenerate_rejects_single_chain():
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1,), REG_COLS, REG_ROWS)
    try:
        result = bpy.ops.mesh.milky_regenerate_crossing_flows(flow_count=3)
        assert result == {'CANCELLED'}, f"operator returned {result}"
    except RuntimeError:
        pass  # error report raises in background mode
    bpy.ops.object.mode_set(mode='OBJECT')


# ---------------------------------------------------------------------------
# Equalize Loop Spacing
# ---------------------------------------------------------------------------

def build_loop_band(gaps, slant=0.3, closed=False):
    """Quad band between two loops; loop A is straight (open) or an
    octagon (closed), loop B sits gap[i] away with slanted rungs."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing)
    n = len(gaps)
    verts = []
    if closed:
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            verts.append((math.cos(ang), math.sin(ang), 0.05 * (k % 2)))
        for k, g in enumerate(gaps):
            ang = 2.0 * math.pi * k / n
            r = 1.0 + g
            verts.append((r * math.cos(ang), r * math.sin(ang),
                          0.05 * (k % 2)))
    else:
        for k in range(n):
            verts.append((float(k), 0.0, 0.0))
        for k, g in enumerate(gaps):
            dx = slant if k < n - 1 else -slant
            verts.append((k + dx, g, 0.0))
    faces = []
    pairs = n if closed else n - 1
    for k in range(pairs):
        nxt = (k + 1) % n
        faces.append((k, nxt, n + nxt, n + k))
    mesh = bpy.data.meshes.new("loop_band")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("loop_band", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    keys = set()
    for k in range(pairs):
        nxt = (k + 1) % n
        keys.add(frozenset((k, nxt)))
        keys.add(frozenset((n + k, n + nxt)))
    for e in bm.edges:
        e.select = frozenset((e.verts[0].index, e.verts[1].index)) in keys
    for v in bm.verts:
        v.select = any(e.select for e in v.link_edges)
    bmesh.update_edit_mesh(obj.data)
    return obj


def test_equalize_open_fixed_side():
    from milkyEdgeFlowTools import spacing as milky_spacing
    gaps = [0.5, 0.9, 0.7, 1.2, 0.6]
    obj = build_loop_band(gaps)
    n = len(gaps)
    bm = bmesh.from_edit_mesh(obj.data)
    orig = {v.index: v.co.copy() for v in bm.verts}

    used = milky_spacing.run_equalize(obj, distance=1.0, fixed_vert=0)
    assert abs(used - 1.0) < 1e-9, used

    bm = bmesh.from_edit_mesh(obj.data)
    for vi in range(n):   # fixed loop untouched
        assert (bm.verts[vi].co - orig[vi]).length < 1e-9, vi
    for k in range(n):    # moving loop: perpendicular distance == 1
        co = bm.verts[n + k].co
        assert abs(math.hypot(co.y, co.z) - 1.0) < 2e-3, (k, co)
        d_old = (orig[n + k] - orig[k]).normalized()
        d_new = (co - orig[k]).normalized()
        assert (d_old - d_new).length < 1e-6, (k, "rung direction")
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True)


def test_equalize_symmetric_median():
    from milkyEdgeFlowTools import spacing as milky_spacing
    gaps = [0.8, 0.95, 0.85, 1.05, 0.9]
    obj = build_loop_band(gaps)
    n = len(gaps)
    bm = bmesh.from_edit_mesh(obj.data)
    orig = {v.index: v.co.copy() for v in bm.verts}

    milky_spacing.run_equalize(obj, distance=1.0, fixed_vert=None)

    bm = bmesh.from_edit_mesh(obj.data)
    for k in range(n):    # rung midpoints preserved
        mid_old = (orig[k] + orig[n + k]) / 2.0
        mid_new = (bm.verts[k].co + bm.verts[n + k].co) / 2.0
        assert (mid_old - mid_new).length < 1e-9, k
        assert (bm.verts[k].co - orig[k]).length > 1e-6, "fixed side moved"
    bpy.ops.object.mode_set(mode='OBJECT')


def test_equalize_closed_rings():
    from milkyEdgeFlowTools import core as milky_core
    from milkyEdgeFlowTools import spacing as milky_spacing
    gaps = [0.4, 0.9, 0.6, 1.1, 0.5, 0.8, 1.0, 0.7]
    obj = build_loop_band(gaps, closed=True)
    n = len(gaps)
    bm = bmesh.from_edit_mesh(obj.data)
    inner = [tuple(bm.verts[k].co) for k in range(n)]

    milky_spacing.run_equalize(obj, distance=0.8, fixed_vert=0)

    bm = bmesh.from_edit_mesh(obj.data)
    curve = milky_core.CatmullRomCurve(inner, closed=True)
    for k in range(n):
        _s, dist = curve.closest_param_to_point(tuple(bm.verts[n + k].co))
        assert abs(dist - 0.8) < 3e-3, (k, dist)
    bpy.ops.object.mode_set(mode='OBJECT')


def test_equalize_auto_distance_and_operator():
    from milkyEdgeFlowTools import spacing as milky_spacing
    gaps = [0.5, 0.9, 0.7, 1.2, 0.6]
    obj = build_loop_band(gaps)
    n = len(gaps)
    bm = bmesh.from_edit_mesh(obj.data)
    orig = {v.index: v.co.copy() for v in bm.verts}

    # Auto distance = current mean perpendicular gap.
    used = milky_spacing.run_equalize(obj, distance=None, fixed_vert=0)
    assert 0.4 < used < 1.3, used

    # Operator path with the Active Element pivot: loop B stays fixed.
    obj = build_loop_band(gaps)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bpy.context.scene.tool_settings.transform_pivot_point = \
        'ACTIVE_ELEMENT'
    bm.select_history.add(bm.verts[n + 2])
    bmesh.update_edit_mesh(obj.data)
    orig_b = [bm.verts[n + k].co.copy() for k in range(n)]
    result = bpy.ops.mesh.milky_equalize_loop_spacing(distance=0.9)
    assert result == {'FINISHED'}, result
    bm = bmesh.from_edit_mesh(obj.data)
    for k in range(n):
        assert (bm.verts[n + k].co - orig_b[k]).length < 1e-9, k
    bpy.context.scene.tool_settings.transform_pivot_point = 'MEDIAN_POINT'
    bpy.ops.object.mode_set(mode='OBJECT')


def test_equalize_rejects_bad_selection():
    from milkyEdgeFlowTools import spacing as milky_spacing

    # Three loops selected (grid columns).
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 2, 4), REG_COLS, REG_ROWS)
    try:
        milky_spacing.run_equalize(obj)
        raise AssertionError("three loops were accepted")
    except milky_spacing.StripError as exc:
        assert "exactly two" in exc.message, exc.message

    # Two loops that are not bridged (columns 1 and 4 of the grid).
    select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)
    try:
        milky_spacing.run_equalize(obj)
        raise AssertionError("unbridged loops were accepted")
    except milky_spacing.StripError as exc:
        assert "bridged" in exc.message, exc.message
    bpy.ops.object.mode_set(mode='OBJECT')


# ---------------------------------------------------------------------------
# End-to-end regression cases on curved "real" meshes
# ---------------------------------------------------------------------------

def build_coons_strip(cols, rows, variant=0):
    """Curved test patch whose four boundary edges are four distinct
    non-straight 3D curves; the interior is their Coons interpolation."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing)
    amp = 1.0 + 0.25 * variant
    width = 4.2 + 0.4 * variant
    height = 4.6 + 0.3 * variant

    def bottom(u):
        return (width * u,
                0.55 * amp * math.sin(math.pi * u + 0.4),
                0.30 * math.sin(2.0 * math.pi * u))

    def top(u):
        return (width * u + 0.35 * math.sin(math.pi * u),
                height + 0.40 * amp * math.sin(2.0 * math.pi * u + 1.1),
                0.10 - 0.25 * math.cos(math.pi * u))

    def side(v, corner_b, corner_t, sx, sy, sz):
        base = [corner_b[i] + (corner_t[i] - corner_b[i]) * v
                for i in range(3)]
        return (base[0] + sx * math.sin(math.pi * v)
                + 0.15 * math.sin(3.0 * math.pi * v),
                base[1] + sy * math.sin(2.0 * math.pi * v),
                base[2] + sz * math.sin(2.0 * math.pi * v + 0.6))

    def left(v):
        return side(v, bottom(0.0), top(0.0), 0.45 * amp, 0.20, 0.30)

    def right(v):
        return side(v, bottom(1.0), top(1.0), -0.50 * amp, 0.25, -0.20)

    c00, c10, c01, c11 = bottom(0.0), bottom(1.0), top(0.0), top(1.0)

    def coons(u, v):
        b, t, le, r = bottom(u), top(u), left(v), right(v)
        return tuple(
            (1 - v) * b[i] + v * t[i] + (1 - u) * le[i] + u * r[i]
            - ((1 - u) * (1 - v) * c00[i] + u * (1 - v) * c10[i]
               + (1 - u) * v * c01[i] + u * v * c11[i])
            for i in range(3))

    verts = [coons(c / (cols - 1), r / (rows - 1))
             for r in range(rows) for c in range(cols)]
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            faces.append((r * cols + c, r * cols + c + 1,
                          (r + 1) * cols + c + 1, (r + 1) * cols + c))
    mesh = bpy.data.meshes.new("e2e_strip")
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new("e2e_strip", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _select_column_ranges(obj, specs, grid_cols):
    """specs: (col, first_row, last_row) vert ranges, edges in between."""
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    keys = set()
    for col, r0, r1 in specs:
        for r in range(r0, r1):
            keys.add(frozenset((r * grid_cols + col,
                                (r + 1) * grid_cols + col)))
    for e in bm.edges:
        e.select = frozenset((e.verts[0].index, e.verts[1].index)) in keys
    for v in bm.verts:
        v.select = any(e.select for e in v.link_edges)
    bmesh.update_edit_mesh(obj.data)


def _boundary_loop_count(bm):
    bm.edges.ensure_lookup_table()
    remaining = {e for e in bm.edges if len(e.link_faces) == 1}
    vert_edges = {}
    for e in remaining:
        for v in e.verts:
            vert_edges.setdefault(v.index, []).append(e)
    loops = 0
    while remaining:
        loops += 1
        seed = next(iter(remaining))
        remaining.discard(seed)
        stack = [seed]
        while stack:
            e = stack.pop()
            for v in e.verts:
                for e2 in vert_edges[v.index]:
                    if e2 in remaining:
                        remaining.discard(e2)
                        stack.append(e2)
    return loops


def _rail_pos_index(data, pos):
    m = len(data.rails)
    if pos == 'mid':
        return m // 2
    x_first = data.curves[0].point_at(
        0.5 * data.curves[0].total_length)[0]
    x_last = data.curves[m - 1].point_at(
        0.5 * data.curves[m - 1].total_length)[0]
    left_idx = 0 if x_first <= x_last else m - 1
    return left_idx if pos == 'left' else (m - 1) - left_idx


E2E_GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "test_data", "e2e_golden.json")
# Nearest-neighbor tolerance for golden vertex positions. Loose enough
# to survive float reordering (e.g. a numpy migration), tight enough
# that any real behavior change trips it.
E2E_GOLDEN_TOL = 1e-4


def _mesh_snapshot(obj):
    mesh = obj.data
    return {
        "verts": sorted([round(v.co.x, 6), round(v.co.y, 6),
                         round(v.co.z, 6)] for v in mesh.vertices),
        "face_sizes": sorted(len(f.vertices) for f in mesh.polygons),
    }


def _compare_snapshot(name, snap, golden):
    assert golden["face_sizes"] == snap["face_sizes"], \
        (name, "face structure changed")
    gv, sv = golden["verts"], snap["verts"]
    assert len(gv) == len(sv), (name, "vert count", len(sv), len(gv))
    for side_a, side_b in ((gv, sv), (sv, gv)):
        worst = max(min(math.dist(a, b) for b in side_b) for a in side_a)
        assert worst < E2E_GOLDEN_TOL, (name, "vert drift", worst)


M7, M8, M6, M10 = (7, 6, 0), (8, 7, 1), (6, 5, 2), (10, 8, 3)

E2E_CASES = [
    dict(name="b2-default", mesh=M7, sel=[1, 5]),
    dict(name="b2-count4-bias0", mesh=M7, sel=[1, 5],
         kwargs=dict(count=4, bias=0.0)),
    dict(name="b2-count9-bias1", mesh=M7, sel=[1, 5],
         kwargs=dict(count=9, bias=1.0)),
    dict(name="b2-count3", mesh=M7, sel=[1, 5], kwargs=dict(count=3)),
    dict(name="b3-default", mesh=M7, sel=[1, 3, 5]),
    dict(name="b4-default", mesh=M8, sel=[1, 3, 5, 6]),
    dict(name="b2-boundary-rails", mesh=M7, sel=[0, 6]),
    dict(name="b2-lock-left", mesh=M7, sel=[1, 5], lock=['left']),
    dict(name="b2-lock-right", mesh=M7, sel=[1, 5], lock=['right']),
    dict(name="b3-lock-left", mesh=M7, sel=[1, 3, 5], lock=['left']),
    dict(name="b4-lock-mid", mesh=M8, sel=[1, 3, 5, 6], lock=['mid']),
    dict(name="b3-lock-mid", mesh=M7, sel=[1, 3, 5], lock=['mid']),
    dict(name="b3-lock-both-outer", mesh=M7, sel=[1, 3, 5],
         lock=['left', 'right']),
    dict(name="b4-lock-left-mid", mesh=M8, sel=[1, 3, 5, 6],
         lock=['left', 'mid']),
    dict(name="b3-lock-all", mesh=M7, sel=[1, 3, 5],
         lock=['left', 'mid', 'right']),
    dict(name="b2-endrow-moved-inf0", mesh=M7, sel=[1, 5],
         cons=[(0, 'left', 0.15), (0, 'right', 0.20)],
         kwargs=dict(influence=0.0), check_cons=True),
    dict(name="b2-endrow-moved-inf2", mesh=M7, sel=[1, 5],
         cons=[(0, 'left', 0.15), (0, 'right', 0.20)],
         kwargs=dict(influence=2.0)),
    dict(name="b2-interior-row-inf0", mesh=M7, sel=[1, 5],
         cons=[(2, 'left', 0.50)], kwargs=dict(influence=0.0),
         check_cons=True),
    dict(name="b2-multi-rows-inf1", mesh=M7, sel=[1, 5],
         cons=[(1, 'left', 0.30), (3, 'right', 0.70)],
         kwargs=dict(influence=1.0)),
    dict(name="b2-lock-right-plus-vertex", mesh=M7, sel=[1, 5],
         lock=['right'], cons=[(2, 'left', 0.45)],
         kwargs=dict(influence=0.0), check_cons=True),
    dict(name="b2-top-endrow-moved", mesh=M7, sel=[1, 5],
         cons=[(5, 'left', 0.85), (5, 'right', 0.80)],
         kwargs=dict(influence=0.0), check_cons=True),
    dict(name="b2-both-endrows-moved", mesh=M7, sel=[1, 5],
         cons=[(0, 'left', 0.12), (0, 'right', 0.15),
               (5, 'left', 0.88), (5, 'right', 0.85)],
         kwargs=dict(influence=0.0), check_cons=True),
    dict(name="overshoot-top", mesh=M7, sel=[(1, 0, 5), (5, 0, 4)]),
    dict(name="overshoot-both-ends", mesh=M8, sel=[(1, 0, 6), (5, 1, 5)]),
    dict(name="copy-default", mesh=M7, sel=[1, 5],
         kwargs=dict(mode='COPY')),
    dict(name="copy-ref-row2", mesh=M7, sel=[1, 5],
         cons=[(2, 'left', 0.40), (2, 'right', 0.50)],
         kwargs=dict(mode='COPY', copy_row=2, influence=0.0),
         check_cons=True),
    dict(name="copy-lock-left", mesh=M7, sel=[1, 5], lock=['left'],
         kwargs=dict(mode='COPY')),
    dict(name="copy-lock-mid", mesh=M7, sel=[1, 3, 5], lock=['mid'],
         kwargs=dict(mode='COPY')),
    dict(name="copy-count8", mesh=M7, sel=[1, 5],
         kwargs=dict(mode='COPY', count=8)),
    dict(name="copy-bias1", mesh=M6, sel=[1, 4],
         kwargs=dict(mode='COPY', bias=1.0)),
    dict(name="b3-M6-lock-right", mesh=M6, sel=[1, 3, 4], lock=['right']),
    dict(name="b3-M10-default", mesh=M10, sel=[2, 5, 8]),
]


def _run_e2e_case(case):
    from mathutils import Vector
    name = case["name"]
    cols, rows, variant = case["mesh"]
    obj = build_coons_strip(cols, rows, variant)
    bpy.ops.object.mode_set(mode='EDIT')
    specs = [(c, 0, rows - 1) if isinstance(c, int) else c
             for c in case["sel"]]
    _select_column_ranges(obj, specs, cols)

    bm = bmesh.from_edit_mesh(obj.data)
    assert _boundary_loop_count(bm) == 1, (name, "builder boundary")
    data = milky_regen.analyze_strip(bm)
    m = len(data.rails)

    locked = tuple(sorted(_rail_pos_index(data, p)
                          for p in case.get("lock", ())))
    constraints = {}
    expected_points = []
    for row, pos, ratio in case.get("cons", ()):
        rj = _rail_pos_index(data, pos)
        curve = data.curves[rj]
        s = ratio * curve.total_length
        constraints[(row, rj)] = s
        expected_points.append(Vector(curve.point_at(s)))
    locked_coords = [[bm.verts[vi].co.copy() for vi in data.rails[rj]]
                     for rj in locked]

    kwargs = dict(case.get("kwargs", {}))
    count = milky_regen.run_regeneration(
        obj, locked_rails=locked, constraints=constraints or None,
        **kwargs)

    if locked:
        expected_count = len(data.rails[locked[0]])
    elif kwargs.get("count"):
        expected_count = kwargs["count"]
    else:
        expected_count = milky_regen.default_flow_count(data)
    assert count == expected_count, (name, count, expected_count)

    bm = bmesh.from_edit_mesh(obj.data)
    assert not any(len(e.link_faces) > 2 for e in bm.edges), \
        (name, "non-manifold")
    assert _boundary_loop_count(bm) == 1, (name, "boundary loops")
    sel_verts = sum(1 for v in bm.verts if v.select)
    sel_edges = sum(1 for e in bm.edges if e.select)
    assert sel_verts == m * count, (name, sel_verts, m * count)
    assert sel_edges == m * (count - 1), (name, sel_edges)
    for coords in locked_coords:
        for co in coords:
            nearest = min((v.co - co).length for v in bm.verts)
            assert nearest < 1e-6, (name, "locked vert moved", nearest)
    if case.get("check_cons"):
        for point in expected_points:
            nearest = min((v.co - point).length for v in bm.verts)
            assert nearest < 1e-4, (name, "constraint missed", nearest)
    bpy.ops.object.mode_set(mode='OBJECT')
    assert not obj.data.validate(verbose=True), (name, "validate")
    return _mesh_snapshot(obj)


def test_e2e_curved_mesh_cases():
    write = bool(os.environ.get("E2E_WRITE_GOLDEN"))
    golden = {}
    if not write:
        with open(E2E_GOLDEN_PATH, encoding="utf-8") as handle:
            golden = json.load(handle)
    written = {}
    for case in E2E_CASES:
        snap = _run_e2e_case(case)
        if write:
            written[case["name"]] = snap
        else:
            assert case["name"] in golden, (case["name"], "no golden entry")
            _compare_snapshot(case["name"], snap, golden[case["name"]])
        print(f"  e2e ok: {case['name']}")
    if write:
        os.makedirs(os.path.dirname(E2E_GOLDEN_PATH), exist_ok=True)
        with open(E2E_GOLDEN_PATH, "w", encoding="utf-8") as handle:
            json.dump(written, handle, separators=(",", ":"))
        print(f"  golden snapshots written: {E2E_GOLDEN_PATH}")


def main():
    milkyEdgeFlowTools.register()
    obj = fresh_grid()
    bpy.ops.object.mode_set(mode='EDIT')
    select_column_loop(obj)

    result = bpy.ops.mesh.milky_relax_crossing_flows()
    assert result == {'FINISHED'}, f"operator returned {result}"

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Boundary vertices of the loop are pinned by their boundary crossing
    # edges and must not move.
    for row in (0, N - 1):
        co = bm.verts[vid(COL, row)].co
        assert abs(co.x - COL) < 1e-6 and abs(co.y - row) < 1e-6, \
            f"pinned vertex at row {row} moved to {tuple(co)}"

    # Interior vertices must slide back toward the crossing rows (y = row)
    # while staying on the (straight, x = COL) fitted curve.
    for row in range(1, N - 1):
        co = bm.verts[vid(COL, row)].co
        assert abs(co.x - COL) < 1e-5, f"row {row}: left the curve, x={co.x}"
        assert co.y < row + SHIFT - 0.05, \
            f"row {row}: did not move toward the flow, y={co.y}"
    mid = bm.verts[vid(COL, N // 2)].co
    assert abs(mid.y - N // 2) < 0.15, \
        f"middle vertex not relaxed onto the flow, y={mid.y}"

    # Vertices off the selected loop must be untouched.
    for row in range(N):
        for col in (COL - 1, COL + 1):
            co = bm.verts[vid(col, row)].co
            assert abs(co.x - col) < 1e-9 and abs(co.y - row) < 1e-9, \
                f"unselected vertex ({col},{row}) moved to {tuple(co)}"

    bpy.ops.object.mode_set(mode='OBJECT')

    test_lock_ends_and_iterations()

    test_regenerate_basic()
    test_regenerate_same_density()
    test_regenerate_locked_rail()
    test_regenerate_trims_overshoot()
    test_regenerate_moved_end_row()
    test_regenerate_three_rails()
    test_regenerate_denser_count()
    test_regenerate_locked_intermediate_rail()
    test_regenerate_copy_flow_shape()
    test_regenerate_copy_flow_shape_defaults_and_errors()
    test_regenerate_rejects_single_chain()
    test_equalize_open_fixed_side()
    test_equalize_symmetric_median()
    test_equalize_closed_rings()
    test_equalize_auto_distance_and_operator()
    test_equalize_rejects_bad_selection()
    test_e2e_curved_mesh_cases()

    milkyEdgeFlowTools.unregister()
    print("test_blender: ALL ASSERTIONS PASSED")


main()
