"""Integration test for the operator, run inside Blender:

  blender --background --factory-startup --python-exit-code 1 \
          --python test_blender.py

Builds a 7x7 grid whose middle column is shifted along the loop direction,
selects that column as an edge loop, runs Relax Crossing Flows, and checks
that the crossing rows pulled the vertices back while boundary vertices
stayed pinned.
"""

import os
import sys

import bmesh
import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import milkyEdgeFlowTools  # noqa: E402

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


def select_column_loop(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    loop_keys = {frozenset((vid(COL, r), vid(COL, r + 1)))
                 for r in range(N - 1)}
    for e in bm.edges:
        key = frozenset((e.verts[0].index, e.verts[1].index))
        e.select = key in loop_keys
    for v in bm.verts:
        v.select = any(e.select for e in v.link_edges)
    bm.select_flush(True)
    bmesh.update_edit_mesh(obj.data)


def main():
    milkyEdgeFlowTools.register()
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing)

    obj = build_grid_object()
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
    milkyEdgeFlowTools.unregister()
    print("test_blender: ALL ASSERTIONS PASSED")


main()
