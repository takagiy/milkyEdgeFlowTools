# milkyEdgeFlowTools -- edge flow tools for Blender 5.0+
# SPDX-License-Identifier: GPL-3.0-or-later
#
# See requirements.md for the full specification. Pure geometry lives in
# core.py (no bpy); this module extracts topology from the edit mesh, runs
# the core pipeline, and writes the result back.

import math

import bmesh
import bpy
from bpy.props import FloatProperty

from . import core

MAX_RING_STEPS = 16


# ---------------------------------------------------------------------------
# Mesh topology extraction
# ---------------------------------------------------------------------------

def _edge_key(edge):
    return frozenset((edge.verts[0].index, edge.verts[1].index))


def _next_ring_edge(vert, incoming):
    """Continue a crossing loop through a 4-valence manifold vertex."""
    if len(vert.link_edges) != 4:
        return None
    incoming_faces = set(incoming.link_faces)
    candidates = [e for e in vert.link_edges
                  if e is not incoming and not (set(e.link_faces)
                                                & incoming_faces)]
    return candidates[0] if len(candidates) == 1 else None


def _walk_ring(vert, edge, max_steps=MAX_RING_STEPS):
    """Points of the crossing loop beyond `edge`, nearest-first."""
    points = []
    cur_vert = edge.other_vert(vert)
    cur_edge = edge
    points.append(tuple(cur_vert.co))
    for _ in range(max_steps - 1):
        nxt = _next_ring_edge(cur_vert, cur_edge)
        if nxt is None:
            break
        cur_vert = nxt.other_vert(cur_vert)
        cur_edge = nxt
        points.append(tuple(cur_vert.co))
    return points


def _is_shape_edge(edge, angle_limit):
    """True when a crossing edge defines the object's shape (pin trigger)."""
    if len(edge.link_faces) < 2:
        return True  # open boundary or wire edge
    if not edge.smooth:
        return True  # marked sharp
    face_angle = edge.calc_face_angle(None)
    if face_angle is None:
        return True  # non-manifold; be conservative
    interior = math.pi - face_angle
    return interior <= angle_limit


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class MESH_OT_milky_relax_crossing_flows(bpy.types.Operator):
    bl_idname = "mesh.milky_relax_crossing_flows"
    bl_label = "Relax Crossing Flows"
    bl_description = ("Resample selected edge loops on a fitted curve "
                      "so that crossing edge flows relax")
    bl_options = {'REGISTER', 'UNDO'}

    factor: FloatProperty(
        name="Factor",
        description="Blend between original and relaxed positions",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    side_blend: FloatProperty(
        name="Side Blend",
        description=("Blend between the flow of the side with more rings "
                     "(0) and the side with fewer rings (1)"),
        default=0.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    angle_limit: FloatProperty(
        name="Face Angle Limit",
        description=("Crossing edges whose faces meet at this interior "
                     "angle or less are treated as shape-defining and "
                     "pinned"),
        default=math.radians(90.0), min=0.0, max=math.pi, subtype='ANGLE',
    )
    stiffness: FloatProperty(
        name="Stiffness",
        description=("Smoothness of the redistribution; higher values "
                     "spread the influence of pinned vertices further"),
        default=1.0, min=0.0, max=100.0,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def execute(self, context):
        moved = 0
        skipped = 0
        for obj in context.objects_in_mode_unique_data:
            if obj.type != 'MESH':
                continue
            obj_moved, obj_skipped = self._process_object(obj)
            moved += obj_moved
            skipped += obj_skipped

        pgettext = bpy.app.translations.pgettext_rpt
        if skipped:
            self.report({'WARNING'},
                        pgettext("Skipped %d branched selection(s)")
                        % skipped)
        if not moved:
            self.report({'INFO'},
                        pgettext("No movable edge loops in selection"))
            return {'CANCELLED'}
        return {'FINISHED'}

    def _process_object(self, obj):
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        bm.verts.ensure_lookup_table()

        selected = [e for e in bm.edges if e.select]
        if not selected:
            return 0, 0
        chain_keys = {_edge_key(e) for e in selected}
        chains, skipped = core.decompose_chains(
            [(e.verts[0].index, e.verts[1].index) for e in selected])

        moved = 0
        for vert_indices, closed in chains:
            if self._relax_chain(bm, vert_indices, closed, chain_keys):
                moved += 1
        if moved:
            bmesh.update_edit_mesh(mesh, loop_triangles=True,
                                   destructive=False)
        return moved, skipped

    def _relax_chain(self, bm, vert_indices, closed, chain_keys):
        verts = [bm.verts[i] for i in vert_indices]
        points = [tuple(v.co) for v in verts]

        sides = []
        pinned = []
        for vert in verts:
            chain_edges = [e for e in vert.link_edges
                           if _edge_key(e) in chain_keys]
            chain_faces = set()
            for e in chain_edges:
                chain_faces.update(e.link_faces)
            crossing = [e for e in vert.link_edges
                        if _edge_key(e) not in chain_keys
                        and set(e.link_faces) & chain_faces]

            pinned.append(any(_is_shape_edge(e, self.angle_limit)
                              for e in crossing))
            if len(crossing) <= 2:
                sides.append([_walk_ring(vert, e) for e in crossing])
            else:
                sides.append([])  # pole on the chain: no reliable flow

        new_points = core.relax_chain(
            points, closed, sides, pinned,
            side_blend=self.side_blend,
            stiffness=self.stiffness,
            factor=self.factor,
        )

        changed = False
        for vert, old, new in zip(verts, points, new_points):
            if math.dist(old, new) > 1.0e-9:
                vert.co = new
                changed = True
        return changed


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

class VIEW3D_MT_milky_edge_flow_tools(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_milky_edge_flow_tools"
    bl_label = "milkyEdgeFlowTools"

    def draw(self, context):
        self.layout.operator(MESH_OT_milky_relax_crossing_flows.bl_idname)


def _draw_context_menu(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_milky_edge_flow_tools.bl_idname)


# ---------------------------------------------------------------------------
# Translations (English base, Japanese)
# ---------------------------------------------------------------------------

_translations = {
    "ja_JP": {
        ("Operator", "Relax Crossing Flows"):
            "交差するフローをリラックス",
        ("*", "Relax Crossing Flows"):
            "交差するフローをリラックス",
        ("*", "Resample selected edge loops on a fitted curve so that "
              "crossing edge flows relax"):
            "選択エッジループを近似曲線上でリサンプルし、"
            "交差するエッジフローをリラックスさせる",
        ("*", "Factor"): "強度",
        ("*", "Blend between original and relaxed positions"):
            "元の位置とリラックス後の位置のブレンド",
        ("*", "Side Blend"): "サイドブレンド",
        ("*", "Blend between the flow of the side with more rings (0) and "
              "the side with fewer rings (1)"):
            "リング数が多い側のフロー (0) と少ない側のフロー (1) のブレンド",
        ("*", "Face Angle Limit"): "面角度のしきい値",
        ("*", "Crossing edges whose faces meet at this interior angle or "
              "less are treated as shape-defining and pinned"):
            "交差エッジの 2 面の内角がこの値以下の場合、"
            "形状を定義するエッジとみなして固定する",
        ("*", "Stiffness"): "剛性",
        ("*", "Smoothness of the redistribution; higher values spread the "
              "influence of pinned vertices further"):
            "再配置の滑らかさ。値が大きいほど固定頂点の影響が遠くまで及ぶ",
        ("*", "Skipped %d branched selection(s)"):
            "分岐のある選択を %d 個スキップしました",
        ("*", "No movable edge loops in selection"):
            "選択内に移動可能なエッジループがありません",
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    MESH_OT_milky_relax_crossing_flows,
    VIEW3D_MT_milky_edge_flow_tools,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(_draw_context_menu)
    bpy.app.translations.register(__name__, _translations)


def unregister():
    bpy.app.translations.unregister(__name__)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(_draw_context_menu)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
