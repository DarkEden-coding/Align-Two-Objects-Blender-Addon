bl_info = {
    "name": "Align Object To Point Pairs",
    "author": "ScytheEden",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Align",
    "description": "Create vertex pairs between two mesh objects and align the source to the target via best-fit similarity transform.",
    "tracker_url": "https://github.com/DarkEden-coding/Align-3d-Scan-Blender/issues",
    "doc_url": "https://github.com/DarkEden-coding/Align-3d-Scan-Blender/blob/main/ReadME.md",
    "category": "Object",
}

import typing as _t

import bpy
import bmesh
from bpy.types import Context, Operator, Panel, PropertyGroup, UIList
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree
import time

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]


class AlignPairItem(PropertyGroup):
    """Pair of vertex indices linking a source object vertex to a target object vertex."""

    source_object: StringProperty(name="Source Object")
    target_object: StringProperty(name="Target Object")
    source_index: IntProperty(name="Source Vertex Index", min=0)
    target_index: IntProperty(name="Target Vertex Index", min=0)


class ALIGN_UL_pairs(UIList):
    """UI list displaying stored vertex index pairs."""

    def draw_item(  # type: ignore[override]
        self,
        context: Context,
        layout: bpy.types.UILayout,
        data: _t.Any,
        item: AlignPairItem,
        icon: int,
        active_data: _t.Any,
        active_propname: str,
        index: int = 0,
    ) -> None:
        row = layout.row(align=True)
        if item is None:
            row.label(text="<invalid>")
            return
        text = f"{item.source_object}[{item.source_index}] → {item.target_object}[{item.target_index}]"
        row.label(text=text)


class ALIGN_OT_pick_active_vertex(Operator):
    """Capture the active or selected vertex index from the active mesh in Edit Mode."""

    bl_idname = "align_pairs.pick_active_vertex"
    bl_label = "Pick Active Vertex"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(
        name="Role",
        items=(
            ("SOURCE", "Source", "Pick from source object"),
            ("TARGET", "Target", "Pick from target object"),
        ),
        default="SOURCE",
    )

    def execute(self, context: Context) -> set[str]:
        scene = context.scene
        active_obj = context.view_layer.objects.active
        if active_obj is None or active_obj.type != "MESH":
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        if active_obj.mode != "EDIT":
            self.report({"ERROR"}, "Enter Edit Mode and select a vertex")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(active_obj.data)
        active_element = bm.select_history.active
        vertex_index = -1
        if isinstance(active_element, bmesh.types.BMVert):
            vertex_index = active_element.index
        else:
            selected_indices = [v.index for v in bm.verts if v.select]
            vertex_index = selected_indices[0] if selected_indices else -1

        if vertex_index < 0:
            self.report({"ERROR"}, "Select a vertex in Edit Mode")
            return {"CANCELLED"}

        if self.role == "SOURCE":
            if scene.align_pointcloud_source_object is None:
                scene.align_pointcloud_source_object = active_obj
            scene.align_pointcloud_current_source_index = int(vertex_index)
            self.report({"INFO"}, f"Source vertex set to {vertex_index}")
        else:
            if scene.align_pointcloud_target_object is None:
                scene.align_pointcloud_target_object = active_obj
            scene.align_pointcloud_current_target_index = int(vertex_index)
            self.report({"INFO"}, f"Target vertex set to {vertex_index}")

        return {"FINISHED"}


class ALIGN_OT_add_pair(Operator):
    """Add a vertex index pair for the current source and target objects."""

    bl_idname = "align_pairs.add_pair"
    bl_label = "Add Pair"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        scene = context.scene
        src = scene.align_pointcloud_source_object
        tgt = scene.align_pointcloud_target_object
        src_idx = scene.align_pointcloud_current_source_index
        tgt_idx = scene.align_pointcloud_current_target_index

        if src is None or tgt is None:
            self.report({"ERROR"}, "Select both source and target objects")
            return {"CANCELLED"}
        if src.type != "MESH" or tgt.type != "MESH":
            self.report({"ERROR"}, "Both objects must be meshes")
            return {"CANCELLED"}
        if src_idx < 0 or tgt_idx < 0:
            self.report({"ERROR"}, "Pick both source and target vertex indices")
            return {"CANCELLED"}

        pair = scene.align_pointcloud_pairs.add()
        pair.source_object = src.name
        pair.target_object = tgt.name
        pair.source_index = int(src_idx)
        pair.target_index = int(tgt_idx)
        scene.align_pointcloud_pairs_index = len(scene.align_pointcloud_pairs) - 1

        scene.align_pointcloud_current_source_index = -1
        scene.align_pointcloud_current_target_index = -1
        self.report({"INFO"}, "Pair added")
        return {"FINISHED"}


class ALIGN_OT_remove_pair(Operator):
    """Remove the selected vertex index pair from the list."""

    bl_idname = "align_pairs.remove_pair"
    bl_label = "Remove Pair"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        scene = context.scene
        idx = scene.align_pointcloud_pairs_index
        if 0 <= idx < len(scene.align_pointcloud_pairs):
            scene.align_pointcloud_pairs.remove(idx)
            scene.align_pointcloud_pairs_index = max(0, idx - 1)
            return {"FINISHED"}
        self.report({"ERROR"}, "No pair selected")
        return {"CANCELLED"}


class ALIGN_OT_clear_pairs_for_objects(Operator):
    """Clear all pairs that belong to the current source and target objects."""

    bl_idname = "align_pairs.clear_pairs_for_objects"
    bl_label = "Clear Pairs For Objects"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        scene = context.scene
        src = scene.align_pointcloud_source_object
        tgt = scene.align_pointcloud_target_object
        if src is None or tgt is None:
            self.report({"ERROR"}, "Select both source and target objects")
            return {"CANCELLED"}
        to_remove = [
            i
            for i, p in enumerate(scene.align_pointcloud_pairs)
            if p.source_object == src.name and p.target_object == tgt.name
        ]
        for i in reversed(to_remove):
            scene.align_pointcloud_pairs.remove(i)
        scene.align_pointcloud_pairs_index = min(scene.align_pointcloud_pairs_index, len(scene.align_pointcloud_pairs) - 1)
        self.report({"INFO"}, "Pairs cleared for current objects")
        return {"FINISHED"}


def _collect_world_points(
    obj: bpy.types.Object,
    indices: list[int],
) -> list[Vector]:
    """Return world-space coordinates for the given vertex indices.

    Args:
        obj: Mesh object whose vertices are accessed.
        indices: List of vertex indices to sample.

    Returns:
        List of world-space vertex coordinates.
    """
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    world_mat = obj.matrix_world.copy()
    verts = obj.data.vertices  # type: ignore[assignment]
    result: list[Vector] = []
    for i in indices:
        if i < 0 or i >= len(verts):
            continue
        result.append(world_mat @ verts[i].co)
    return result


def _compute_similarity_transform(
    source_points: "np.ndarray",
    target_points: "np.ndarray",
    allow_scaling: bool,
) -> "np.ndarray":
    """Compute best-fit similarity transform mapping source to target.

    Args:
        source_points: Nx3 array of source points in world space.
        target_points: Nx3 array of target points in world space.
        allow_scaling: Whether to estimate uniform scale.

    Returns:
        4x4 homogeneous transform matrix as a NumPy array that maps source to target.
    """
    n = source_points.shape[0]
    mu_src = source_points.mean(axis=0)
    mu_tgt = target_points.mean(axis=0)

    x = source_points - mu_src
    y = target_points - mu_tgt

    cov = (x.T @ y) / float(n)
    u, s_vals, vt = np.linalg.svd(cov)
    v = vt.T
    r = v @ u.T
    if np.linalg.det(r) < 0.0:
        v[:, -1] *= -1.0
        r = v @ u.T

    if allow_scaling:
        var_src = np.sum(x * x) / float(n)
        scale = 0.0 if var_src == 0.0 else float(np.sum(s_vals) / var_src)
    else:
        scale = 1.0

    t = mu_tgt - scale * (r @ mu_src)

    a = np.eye(4, dtype=float)
    a[:3, :3] = scale * r
    a[:3, 3] = t
    return a


class ALIGN_OT_align_now(Operator):
    """Align the source object to the target using stored pairs via least-squares similarity transform."""

    bl_idname = "align_pairs.align_now"
    bl_label = "Align Source To Target"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        scene = context.scene
        src = scene.align_pointcloud_source_object
        tgt = scene.align_pointcloud_target_object
        if src is None or tgt is None:
            self.report({"ERROR"}, "Select both source and target objects")
            return {"CANCELLED"}
        if src.type != "MESH" or tgt.type != "MESH":
            self.report({"ERROR"}, "Both objects must be meshes")
            return {"CANCELLED"}
        if np is None:
            self.report({"ERROR"}, "NumPy is required for alignment")
            return {"CANCELLED"}
        start_time = time.monotonic()
        timeout = float(scene.align_pointcloud_timeout)

        pairs = [
            p
            for p in scene.align_pointcloud_pairs
            if p.source_object == src.name and p.target_object == tgt.name
        ]
        if len(pairs) < 3:
            self.report({"ERROR"}, "At least 3 pairs are required for alignment")
            return {"CANCELLED"}

        src_indices = [p.source_index for p in pairs]
        tgt_indices = [p.target_index for p in pairs]

        src_points = _collect_world_points(src, src_indices)
        tgt_points = _collect_world_points(tgt, tgt_indices)
        if len(src_points) != len(tgt_points) or len(src_points) < 3:
            self.report({"ERROR"}, "Invalid pairs or indices out of range")
            return {"CANCELLED"}

        src_arr = np.array([[v.x, v.y, v.z] for v in src_points], dtype=float)
        tgt_arr = np.array([[v.x, v.y, v.z] for v in tgt_points], dtype=float)

        if time.monotonic() - start_time > timeout:
            self.report({"ERROR"}, "Alignment timed out")
            return {"CANCELLED"}

        a_np = _compute_similarity_transform(
            source_points=src_arr,
            target_points=tgt_arr,
            allow_scaling=bool(scene.align_pointcloud_use_scale),
        )

        a = Matrix(
            (
                (float(a_np[0, 0]), float(a_np[0, 1]), float(a_np[0, 2]), float(a_np[0, 3])),
                (float(a_np[1, 0]), float(a_np[1, 1]), float(a_np[1, 2]), float(a_np[1, 3])),
                (float(a_np[2, 0]), float(a_np[2, 1]), float(a_np[2, 2]), float(a_np[2, 3])),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        src.matrix_world = a @ src.matrix_world
        self.report({"INFO"}, "Alignment applied")
        return {"FINISHED"}


class ALIGN_OT_icp_refine(Operator):
    """Refine alignment by iteratively matching nearest vertices and optimizing transform."""

    bl_idname = "align_pairs.icp_refine"
    bl_label = "Refine (ICP)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        scene = context.scene
        source_object = scene.align_pointcloud_source_object
        target_object = scene.align_pointcloud_target_object
        if source_object is None or target_object is None:
            self.report({"ERROR"}, "Select both source and target objects")
            return {"CANCELLED"}
        if source_object.type != "MESH" or target_object.type != "MESH":
            self.report({"ERROR"}, "Both objects must be meshes")
            return {"CANCELLED"}
        if np is None:
            self.report({"ERROR"}, "NumPy is required for ICP refine")
            return {"CANCELLED"}

        max_iterations = max(1, int(scene.align_pointcloud_icp_max_iterations))
        tolerance = float(scene.align_pointcloud_icp_tolerance)
        distance_threshold = float(scene.align_pointcloud_icp_distance_threshold)
        trim_fraction = float(scene.align_pointcloud_icp_trim_fraction)
        trim_fraction = min(max(0.0, trim_fraction), 0.49)
        sample_count = max(0, int(scene.align_pointcloud_icp_sample_count))
        allow_scaling = bool(scene.align_pointcloud_use_scale)
        use_modifiers = bool(scene.align_pointcloud_use_modifiers)
        start_time = time.monotonic()
        timeout = float(scene.align_pointcloud_timeout)

        depsgraph = context.evaluated_depsgraph_get()

        def get_world_vertices(obj: bpy.types.Object) -> list[Vector]:
            if use_modifiers:
                obj_eval = obj.evaluated_get(depsgraph)
                mesh = obj_eval.to_mesh()
                try:
                    world = obj_eval.matrix_world.copy()
                    return [world @ v.co for v in mesh.vertices]
                finally:
                    obj_eval.to_mesh_clear()
            if obj.mode == "EDIT":
                obj.update_from_editmode()
            world = obj.matrix_world.copy()
            return [world @ v.co for v in obj.data.vertices]  # type: ignore[attr-defined]

        target_world_points = get_world_vertices(target_object)
        if len(target_world_points) < 3:
            self.report({"ERROR"}, "Target must have at least 3 vertices")
            return {"CANCELLED"}

        target_kdtree = KDTree(len(target_world_points))
        for idx, vec in enumerate(target_world_points):
            target_kdtree.insert(vec, idx)
        target_kdtree.balance()

        previous_error = float("inf")
        iterations_performed = 0

        for _ in range(max_iterations):
            iterations_performed += 1
            source_world_points = get_world_vertices(source_object)
            if len(source_world_points) < 3:
                self.report({"ERROR"}, "Source must have at least 3 vertices")
                return {"CANCELLED"}

            if sample_count > 0 and sample_count < len(source_world_points):
                step = max(1, len(source_world_points) // sample_count)
                sampled_source_indices = list(range(0, len(source_world_points), step))[:sample_count]
            else:
                sampled_source_indices = list(range(len(source_world_points)))

            matched_source: list[Vector] = []
            matched_target: list[Vector] = []
            distances: list[float] = []
            for si in sampled_source_indices:
                sv = source_world_points[si]
                loc, index, dist = target_kdtree.find(sv)
                if distance_threshold > 0.0 and dist > distance_threshold:
                    continue
                matched_source.append(sv)
                matched_target.append(loc)
                distances.append(dist)

            if len(matched_source) < 3:
                break

            if trim_fraction > 0.0 and len(distances) >= 4:
                order = np.argsort(np.asarray(distances))
                keep = int((1.0 - trim_fraction) * len(order))
                keep = max(3, keep)
                order = order[:keep]
                matched_source = [matched_source[i] for i in order]
                matched_target = [matched_target[i] for i in order]
                distances = [distances[i] for i in order]

            source_arr = np.array([[v.x, v.y, v.z] for v in matched_source], dtype=float)
            target_arr = np.array([[v.x, v.y, v.z] for v in matched_target], dtype=float)
            mean_error = float(np.mean(np.asarray(distances))) if distances else float("inf")
            if time.monotonic() - start_time > timeout:
                self.report({"ERROR"}, "ICP refine timed out, maximum distance may be too low")
                return {"CANCELLED"}

            a_np = _compute_similarity_transform(
                source_points=source_arr,
                target_points=target_arr,
                allow_scaling=allow_scaling,
            )

            a = Matrix(
                (
                    (float(a_np[0, 0]), float(a_np[0, 1]), float(a_np[0, 2]), float(a_np[0, 3])),
                    (float(a_np[1, 0]), float(a_np[1, 1]), float(a_np[1, 2]), float(a_np[1, 3])),
                    (float(a_np[2, 0]), float(a_np[2, 1]), float(a_np[2, 2]), float(a_np[2, 3])),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
            source_object.matrix_world = a @ source_object.matrix_world

            if previous_error - mean_error < tolerance:
                previous_error = mean_error
                break
            previous_error = mean_error

        self.report({"INFO"}, f"ICP refine done in {iterations_performed} iterations, mean error {previous_error:.6f}")
        return {"FINISHED"}


class ALIGN_PT_panel(Panel):
    """Sidebar panel for creating pairs and aligning objects."""

    bl_label = "Align To Point Pairs"
    bl_idname = "ALIGN_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Align"

    def draw(self, context: Context) -> None:  # type: ignore[override]
        scene = context.scene
        layout = self.layout

        col = layout.column(align=True)
        col.prop(scene, "align_pointcloud_source_object", text="Source")
        col.prop(scene, "align_pointcloud_target_object", text="Target")

        box = layout.box()
        row = box.row(align=True)
        row.prop(scene, "align_pointcloud_current_source_index", text="Src Index")
        op = row.operator(ALIGN_OT_pick_active_vertex.bl_idname, text="Pick Src")
        op.role = "SOURCE"

        row = box.row(align=True)
        row.prop(scene, "align_pointcloud_current_target_index", text="Tgt Index")
        op = row.operator(ALIGN_OT_pick_active_vertex.bl_idname, text="Pick Tgt")
        op.role = "TARGET"

        box.operator(ALIGN_OT_add_pair.bl_idname, text="Add Pair", icon="ADD")

        row = layout.row()
        row.template_list(
            ALIGN_UL_pairs.__name__,
            "ALIGN_UL_pairs",
            scene,
            "align_pointcloud_pairs",
            scene,
            "align_pointcloud_pairs_index",
            rows=4,
        )
        col_buttons = row.column(align=True)
        col_buttons.operator(ALIGN_OT_remove_pair.bl_idname, text="", icon="REMOVE")
        col_buttons.operator(ALIGN_OT_clear_pairs_for_objects.bl_idname, text="", icon="TRASH")

        col = layout.column(align=True)
        col.prop(scene, "align_pointcloud_use_scale", text="Use Uniform Scale")
        col.prop(scene, "align_pointcloud_use_modifiers", text="Use Modifiers")
        col.prop(scene, "align_pointcloud_timeout", text="Timeout (s)")
        col.operator(ALIGN_OT_align_now.bl_idname, text="Align Source To Target", icon="ORIENTATION_GIMBAL")

        ref_box = layout.box()
        ref_box.label(text="Refine (ICP)")
        ref_box.prop(scene, "align_pointcloud_icp_max_iterations", text="Max Iterations")
        ref_box.prop(scene, "align_pointcloud_icp_tolerance", text="Tolerance")
        ref_box.prop(scene, "align_pointcloud_icp_distance_threshold", text="Max Distance")
        ref_box.prop(scene, "align_pointcloud_icp_trim_fraction", text="Trim Fraction")
        ref_box.prop(scene, "align_pointcloud_icp_sample_count", text="Sample Count")
        ref_box.operator(ALIGN_OT_icp_refine.bl_idname, text="Refine", icon="MOD_SMOOTH")


_classes: tuple[type, ...] = (
    AlignPairItem,
    ALIGN_UL_pairs,
    ALIGN_OT_pick_active_vertex,
    ALIGN_OT_add_pair,
    ALIGN_OT_remove_pair,
    ALIGN_OT_clear_pairs_for_objects,
    ALIGN_OT_align_now,
    ALIGN_OT_icp_refine,
    ALIGN_PT_panel,
)


def register() -> None:
    """Register add-on classes and scene properties."""
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.align_pointcloud_pairs = CollectionProperty(type=AlignPairItem)
    bpy.types.Scene.align_pointcloud_pairs_index = IntProperty(default=0)
    bpy.types.Scene.align_pointcloud_source_object = PointerProperty(type=bpy.types.Object)
    bpy.types.Scene.align_pointcloud_target_object = PointerProperty(type=bpy.types.Object)
    bpy.types.Scene.align_pointcloud_current_source_index = IntProperty(default=-1, min=-1)
    bpy.types.Scene.align_pointcloud_current_target_index = IntProperty(default=-1, min=-1)
    bpy.types.Scene.align_pointcloud_use_scale = BoolProperty(default=True)
    bpy.types.Scene.align_pointcloud_use_modifiers = BoolProperty(default=False)
    bpy.types.Scene.align_pointcloud_timeout = bpy.props.FloatProperty(default=45.0, min=0.0)
    bpy.types.Scene.align_pointcloud_icp_max_iterations = IntProperty(default=20, min=1)
    bpy.types.Scene.align_pointcloud_icp_tolerance = bpy.props.FloatProperty(default=1e-6, min=0.0)
    bpy.types.Scene.align_pointcloud_icp_distance_threshold = bpy.props.FloatProperty(default=0.0, min=0.0)
    bpy.types.Scene.align_pointcloud_icp_trim_fraction = bpy.props.FloatProperty(default=0.2, min=0.0, max=0.49)
    bpy.types.Scene.align_pointcloud_icp_sample_count = IntProperty(default=0, min=0)


def unregister() -> None:
    """Unregister add-on classes and scene properties."""
    del bpy.types.Scene.align_pointcloud_icp_sample_count
    del bpy.types.Scene.align_pointcloud_icp_trim_fraction
    del bpy.types.Scene.align_pointcloud_icp_distance_threshold
    del bpy.types.Scene.align_pointcloud_icp_tolerance
    del bpy.types.Scene.align_pointcloud_icp_max_iterations
    del bpy.types.Scene.align_pointcloud_use_modifiers
    del bpy.types.Scene.align_pointcloud_timeout
    del bpy.types.Scene.align_pointcloud_use_scale
    del bpy.types.Scene.align_pointcloud_current_target_index
    del bpy.types.Scene.align_pointcloud_current_source_index
    del bpy.types.Scene.align_pointcloud_target_object
    del bpy.types.Scene.align_pointcloud_source_object
    del bpy.types.Scene.align_pointcloud_pairs_index
    del bpy.types.Scene.align_pointcloud_pairs

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
