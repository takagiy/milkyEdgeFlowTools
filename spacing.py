# milkyEdgeFlowTools -- Equalize Loop Spacing (requirements.md ch. 12)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Analysis and application are UI-independent (run_equalize) so the
# headless integration tests exercise the same code path as the modal
# operator, which only adds the drag interaction on top.

import bmesh
import bpy

from . import core
from .regen import StripError, _edge_key


class LoopPairData:
    """Two bridged loops: loop_b is aligned pairwise to loop_a."""

    def __init__(self):
        self.loop_a = []
        self.loop_b = []
        self.closed = False


def analyze_loop_pair(bm):
    """Validate the selection and pair the loops. Raises StripError."""
    bm.verts.ensure_lookup_table()
    selected = [e for e in bm.edges if e.select]
    if not selected:
        raise StripError("Select exactly two parallel edge loops")
    chains, skipped = core.decompose_chains(
        [(e.verts[0].index, e.verts[1].index) for e in selected])
    if skipped or len(chains) != 2:
        raise StripError("Select exactly two parallel edge loops")
    (verts_a, closed_a), (verts_b, closed_b) = chains
    if closed_a != closed_b:
        raise StripError("Loops must both be open or both closed")
    if len(verts_a) != len(verts_b):
        raise StripError("Loops must be bridged by a single quad strip")

    chain_keys = {_edge_key(e) for e in selected}
    in_b = set(verts_b)
    paired = []
    for vi in verts_a:
        vert = bm.verts[vi]
        rungs = [e for e in vert.link_edges
                 if _edge_key(e) not in chain_keys
                 and e.other_vert(vert).index in in_b]
        if len(rungs) != 1:
            raise StripError("Loops must be bridged by a single quad "
                             "strip")
        paired.append(rungs[0].other_vert(vert).index)
    if len(set(paired)) != len(verts_b):
        raise StripError("Loops must be bridged by a single quad strip")

    data = LoopPairData()
    data.loop_a = verts_a
    data.loop_b = paired
    data.closed = closed_a
    return data


def _sides(bm, data, fixed_vert):
    """(fixed_ids, moving_ids, symmetric) for the requested mode."""
    if fixed_vert is None:
        return data.loop_a, data.loop_b, True
    if fixed_vert in set(data.loop_a):
        return data.loop_a, data.loop_b, False
    if fixed_vert in set(data.loop_b):
        return data.loop_b, data.loop_a, False
    return data.loop_a, data.loop_b, True


def _coords(bm, ids):
    return [tuple(bm.verts[vi].co) for vi in ids]


def mean_gap(bm, data, fixed_vert=None):
    """Current mean perpendicular gap for the given mode."""
    fixed_ids, moving_ids, symmetric = _sides(bm, data, fixed_vert)
    gaps = core.perpendicular_gaps(
        _coords(bm, fixed_ids), _coords(bm, moving_ids), data.closed,
        symmetric)
    return sum(gaps) / len(gaps)


def _positions(fixed_ids, fixed_pts, moving_ids, moving_pts, closed,
               distance, symmetric):
    f_out, m_out = core.equalize_loop_spacing(
        fixed_pts, moving_pts, closed, distance, symmetric)
    out = dict(zip(moving_ids, m_out))
    if symmetric:
        out.update(zip(fixed_ids, f_out))
    return out


def run_equalize(obj, distance=None, fixed_vert=None):
    """Full UI-independent pipeline on an edit-mesh object.

    fixed_vert: a vertex index on the loop to keep fixed; None keeps
    the rung midpoints fixed and moves both loops symmetrically.
    Returns the distance used.
    """
    bm = bmesh.from_edit_mesh(obj.data)
    data = analyze_loop_pair(bm)
    if distance is None or distance <= 0.0:
        distance = mean_gap(bm, data, fixed_vert)
    fixed_ids, moving_ids, symmetric = _sides(bm, data, fixed_vert)
    positions = _positions(fixed_ids, _coords(bm, fixed_ids),
                           moving_ids, _coords(bm, moving_ids),
                           data.closed, distance, symmetric)
    for vi, co in positions.items():
        bm.verts[vi].co = co
    bmesh.update_edit_mesh(obj.data)
    return distance


# ---------------------------------------------------------------------------
# Modal operator (Shrink/Fatten-style)
# ---------------------------------------------------------------------------

COLOR_FIXED = (1.0, 0.85, 0.2, 1.0)


def _draw_fixed_loop(op):
    if not op._fixed_lines:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.line_width_set(2.5)
    batch = batch_for_shader(shader, 'LINE_STRIP',
                             {"pos": op._fixed_lines})
    shader.uniform_float("color", COLOR_FIXED)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_hud(op, context):
    import blf
    rpt = bpy.app.translations.pgettext_iface
    mode = rpt("Active locked") if not op._symmetric else rpt("Median")
    text = (f"{rpt('Distance')}: {op._w:.4f}   {mode}   "
            f"{rpt('Shift: precise   LMB/Enter: apply   RMB/Esc: cancel')}")
    font = 0
    blf.size(font, 14.0)
    blf.position(font, 60.0, 30.0, 0.0)
    blf.color(font, 1.0, 1.0, 1.0, 0.9)
    blf.draw(font, text)


class MESH_OT_milky_equalize_loop_spacing(bpy.types.Operator):
    bl_idname = "mesh.milky_equalize_loop_spacing"
    bl_label = "Equalize Loop Spacing"
    bl_description = ("Slide two bridged edge loops so their "
                      "perpendicular gap is uniform")
    bl_options = {'REGISTER', 'UNDO'}

    distance: bpy.props.FloatProperty(
        name="Distance",
        description=("Target perpendicular distance between the loops "
                     "(0 = keep the current average)"),
        min=0.0, default=0.0, subtype='DISTANCE',
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and context.mode == 'EDIT_MESH')

    def _active_fixed_vert(self, context, bm, data):
        """Vertex id pinning a loop when the pivot is Active Element."""
        pivot = context.scene.tool_settings.transform_pivot_point
        if pivot != 'ACTIVE_ELEMENT':
            return None
        active = bm.select_history.active
        candidates = []
        if isinstance(active, bmesh.types.BMVert):
            candidates = [active.index]
        elif isinstance(active, bmesh.types.BMEdge):
            candidates = [v.index for v in active.verts]
        loop_verts = set(data.loop_a) | set(data.loop_b)
        for vi in candidates:
            if vi in loop_verts:
                return vi
        return None

    # Headless / redo path.
    def execute(self, context):
        obj = context.active_object
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            data = analyze_loop_pair(bm)
            fixed_vert = self._active_fixed_vert(context, bm, data)
            used = run_equalize(obj, self.distance or None, fixed_vert)
        except StripError as exc:
            self.report({'ERROR'},
                        bpy.app.translations.pgettext_rpt(exc.message))
            return {'CANCELLED'}
        self.distance = used
        return {'FINISHED'}

    def invoke(self, context, event):
        if bpy.app.background:
            return self.execute(context)
        obj = context.active_object
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            data = analyze_loop_pair(bm)
        except StripError as exc:
            self.report({'ERROR'},
                        bpy.app.translations.pgettext_rpt(exc.message))
            return {'CANCELLED'}

        fixed_vert = self._active_fixed_vert(context, bm, data)
        self._obj = obj
        self._data = data
        fixed_ids, moving_ids, symmetric = _sides(bm, data, fixed_vert)
        self._fixed_ids = fixed_ids
        self._moving_ids = moving_ids
        self._symmetric = symmetric
        self._fixed_pts = _coords(bm, fixed_ids)
        self._moving_pts = _coords(bm, moving_ids)
        self._orig = {vi: bm.verts[vi].co.copy()
                      for vi in fixed_ids + moving_ids}
        self._w = mean_gap(bm, data, fixed_vert)
        self._last_x = event.mouse_x
        self._per_px = self._pixel_scale(context)

        if symmetric:
            self._fixed_lines = []
        else:
            matrix = obj.matrix_world
            ids = fixed_ids + fixed_ids[:1] if data.closed else fixed_ids
            self._fixed_lines = [matrix @ bm.verts[vi].co for vi in ids]

        self._apply(context)
        self._draw_3d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_fixed_loop, (self,), 'WINDOW', 'POST_VIEW')
        self._draw_2d = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (self, context), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def _pixel_scale(self, context):
        from bpy_extras import view3d_utils
        from mathutils import Vector
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return 0.01
        center = Vector((0.0, 0.0, 0.0))
        for co in self._orig.values():
            center += co
        center = self._obj.matrix_world @ (center / max(len(self._orig), 1))
        mid = (region.width / 2.0, region.height / 2.0)
        a = view3d_utils.region_2d_to_location_3d(region, rv3d, mid, center)
        b = view3d_utils.region_2d_to_location_3d(
            region, rv3d, (mid[0] + 1.0, mid[1]), center)
        scale = (a - b).length
        return scale if scale > 1.0e-9 else 0.01

    def _apply(self, context):
        bm = bmesh.from_edit_mesh(self._obj.data)
        positions = _positions(self._fixed_ids, self._fixed_pts,
                               self._moving_ids, self._moving_pts,
                               self._data.closed, self._w,
                               self._symmetric)
        for vi, co in positions.items():
            bm.verts[vi].co = co
        bmesh.update_edit_mesh(self._obj.data)

    def _cleanup(self, context):
        if self._draw_3d is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_3d,
                                                      'WINDOW')
        if self._draw_2d is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_2d,
                                                      'WINDOW')
        self._draw_3d = self._draw_2d = None
        context.area.tag_redraw()

    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type == 'MOUSEMOVE':
            step = self._per_px * (0.1 if event.shift else 1.0)
            self._w = max(0.0, self._w
                          + (event.mouse_x - self._last_x) * step)
            self._last_x = event.mouse_x
            self._apply(context)
            return {'RUNNING_MODAL'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        if (event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}
                and event.value == 'PRESS'):
            self.distance = self._w
            self._cleanup(context)
            return {'FINISHED'}
        if (event.type in {'RIGHTMOUSE', 'ESC'}
                and event.value == 'PRESS'):
            bm = bmesh.from_edit_mesh(self._obj.data)
            for vi, co in self._orig.items():
                bm.verts[vi].co = co
            bmesh.update_edit_mesh(self._obj.data)
            self._cleanup(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}


_classes = (MESH_OT_milky_equalize_loop_spacing,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
