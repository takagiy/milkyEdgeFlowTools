"""Integration test for the operator.

Runs against the official bpy wheel (uv run poe test-blender) or inside a
real Blender (uv run poe test-blender-app).

Builds a 7x7 grid whose middle column is shifted along the loop direction,
selects that column as an edge loop, runs Relax Crossing Flows, and checks
that the crossing rows pulled the vertices back while boundary vertices
stayed pinned.
"""

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


def test_regenerate_locked_free_fit_modes():
    """RATIO/DENSITY anchor the free outer chain instead of ray aiming;
    on the uniform grid every mode reproduces the straight rows."""
    for mode in ('RATIO', 'DENSITY'):
        obj = build_plain_grid(REG_COLS, REG_ROWS)
        bpy.ops.object.mode_set(mode='EDIT')
        select_grid_columns(obj, (1, 4), REG_COLS, REG_ROWS)

        count = milky_regen.run_regeneration(obj, locked_rails=(0,),
                                             free_fit=mode)
        assert count == REG_ROWS, f"[{mode}] flow count {count}"

        bm = bmesh.from_edit_mesh(obj.data)
        assert len(bm.verts) == 24, f"[{mode}] vert count {len(bm.verts)}"
        assert len(bm.faces) == 12, f"[{mode}] face count {len(bm.faces)}"
        for x in (1.0, 4.0):
            ys = sorted(round(v.co.y, 3) for v in bm.verts
                        if abs(v.co.x - x) < 1e-4)
            assert ys == [0.0, 1.0, 2.0, 3.0, 4.0], (mode, x, ys)
        bpy.ops.object.mode_set(mode='OBJECT')
        assert not obj.data.validate(verbose=True), \
            f"[{mode}] mesh needed corrections"


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


def test_regenerate_copy_flow_shape_errors():
    """COPY mode rejects a missing reference row and >1 locked chains."""
    obj = build_plain_grid(REG_COLS, REG_ROWS)
    bpy.ops.object.mode_set(mode='EDIT')
    select_grid_columns(obj, (1, 2, 4), REG_COLS, REG_ROWS)

    try:
        milky_regen.run_regeneration(obj, mode='COPY')
        raise AssertionError("missing copy_row was accepted")
    except milky_regen.StripError as exc:
        assert "locked or dragged flow row" in exc.message, exc.message

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
    test_regenerate_locked_free_fit_modes()
    test_regenerate_trims_overshoot()
    test_regenerate_moved_end_row()
    test_regenerate_three_rails()
    test_regenerate_denser_count()
    test_regenerate_locked_intermediate_rail()
    test_regenerate_copy_flow_shape()
    test_regenerate_copy_flow_shape_errors()
    test_regenerate_rejects_single_chain()

    milkyEdgeFlowTools.unregister()
    print("test_blender: ALL ASSERTIONS PASSED")


main()
