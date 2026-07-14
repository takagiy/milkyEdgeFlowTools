# milkyEdgeFlowTools -- edge flow tools for Blender 5.0+
# SPDX-License-Identifier: GPL-3.0-or-later
#
# See requirements.md for the full specification. Pure geometry lives in
# core.py (no bpy); this module extracts topology from the edit mesh, runs
# the core pipeline, and writes the result back.

import math

import bmesh
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty

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
    """Vertices of the crossing loop beyond `edge`, nearest-first."""
    ring = []
    cur_vert = edge.other_vert(vert)
    cur_edge = edge
    ring.append(cur_vert)
    for _ in range(max_steps - 1):
        nxt = _next_ring_edge(cur_vert, cur_edge)
        if nxt is None:
            break
        cur_vert = nxt.other_vert(cur_vert)
        cur_edge = nxt
        ring.append(cur_vert)
    return ring


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
    min_spacing: FloatProperty(
        name="Min Spacing",
        description=("Minimum spacing between neighboring vertices, as a "
                     "fraction of their original spacing"),
        default=0.3, min=0.0, max=1.0, subtype='FACTOR',
    )
    iterations: EnumProperty(
        name="Iterations",
        description="Number of times the relax pass is applied",
        items=[(v, v, "") for v in ("1", "5", "10", "15", "20", "25", "30")],
        default='1',
    )
    lock_ends: BoolProperty(
        name="Lock Ends",
        description=("Keep both end vertices of each open edge loop "
                     "in place"),
        default=False,
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
        if not chains:
            return 0, skipped

        vert_chain = {}
        for ci, (vert_indices, _closed) in enumerate(chains):
            for vi in vert_indices:
                vert_chain[vi] = ci

        data = [self._extract_chain(bm, ci, chains[ci], chain_keys,
                                    vert_chain)
                for ci in range(len(chains))]
        order = core.order_chains([d["dominant"] for d in data])

        for _ in range(int(self.iterations)):
            for ci in order:
                self._relax_step(data[ci])

        moved = 0
        for d in data:
            changed = False
            for vert, orig, pin in zip(d["verts"], d["points"], d["pinned"]):
                new = orig
                if not pin:
                    new = tuple(orig[k] + (vert.co[k] - orig[k]) * self.factor
                                for k in range(3))
                if math.dist(orig, new) > 1.0e-9:
                    changed = True
                vert.co = new
            if changed:
                moved += 1
        if moved:
            bmesh.update_edit_mesh(mesh, loop_triangles=True,
                                   destructive=False)
        return moved, skipped

    def _extract_chain(self, bm, ci, chain, chain_keys, vert_chain):
        vert_indices, closed = chain
        verts = [bm.verts[i] for i in vert_indices]

        sides = []
        pinned = []
        dominant = set()
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
                rings = [_walk_ring(vert, e) for e in crossing]
            else:
                rings = []  # pole on the chain: no reliable flow
            sides.append(rings)

            # Other chains visible on the dominant blend side must be
            # relaxed before this one.
            if len(rings) == 2:
                major = 0 if len(rings[0]) >= len(rings[1]) else 1
                dom_ring = rings[major if self.side_blend <= 0.5
                                 else 1 - major]
            else:
                dom_ring = rings[0] if rings else []
            for ring_vert in dom_ring:
                cj = vert_chain.get(ring_vert.index)
                if cj is not None and cj != ci:
                    dominant.add(cj)

        if self.lock_ends and not closed and len(verts) >= 2:
            pinned[0] = True
            pinned[-1] = True

        points = [tuple(v.co) for v in verts]
        curve = core.CatmullRomCurve(points, closed)
        return {
            "verts": verts,
            "points": points,
            "pinned": pinned,
            "sides": sides,
            "curve": curve,
            "params": list(curve.knot_params),
            "dominant": dominant,
        }

    def _relax_step(self, d):
        side_coords = [[[tuple(v.co) for v in ring] for ring in rings]
                       for rings in d["sides"]]
        d["params"] = core.relax_chain_step(
            d["curve"], d["params"], side_coords, d["pinned"],
            self.side_blend, self.stiffness, self.min_spacing)
        for vert, s, pin, orig in zip(d["verts"], d["params"], d["pinned"],
                                      d["points"]):
            vert.co = orig if pin else d["curve"].point_at(s)


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
        ("*", "Min Spacing"): "最小間隔",
        ("*", "Minimum spacing between neighboring vertices, as a fraction "
              "of their original spacing"):
            "隣接頂点間の間隔の下限（元の間隔に対する割合）",
        ("*", "Iterations"): "イテレーション回数",
        ("*", "Number of times the relax pass is applied"):
            "リラックス処理を適用する回数",
        ("*", "Lock Ends"): "両端をロック",
        ("*", "Keep both end vertices of each open edge loop in place"):
            "開いたエッジループの両端の頂点を固定する",
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
