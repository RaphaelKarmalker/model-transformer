from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import omni.ext
import omni.ui as ui
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


# --------------------------
# Configuration
# --------------------------
# Support both "compound" and the user's typo "compond"
KEYWORD_ALIASES = {
    "compond": "compound",
}

# keyword -> (group token in prim path, RGBA)
KEYWORD_TO_GROUP_AND_COLOR: Dict[str, Tuple[str, Gf.Vec4f]] = {
    "compound": ("compound_gear", Gf.Vec4f(0.10, 0.80, 0.20, 1.0)),  # green
    "small": ("group_small",      Gf.Vec4f(0.20, 0.50, 1.00, 1.0)),  # blue
    "big": ("group_big",          Gf.Vec4f(1.00, 0.60, 0.10, 1.0)),  # orange
}

LOOKS_ROOT_PATH = Sdf.Path("/World/Looks")


class ColorSwitcherExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._window: Optional[ui.Window] = None
        self._status_label: Optional[ui.Label] = None

        self._keyword_model: Optional[ui.SimpleStringModel] = None
        self._dx_model: Optional[ui.SimpleFloatModel] = None
        self._dy_model: Optional[ui.SimpleFloatModel] = None
        self._dz_model: Optional[ui.SimpleFloatModel] = None

        self._build_ui()

    def on_shutdown(self) -> None:
        if self._window:
            self._window.visible = False
            self._window = None

    # --------------------------
    # UI
    # --------------------------
    def _build_ui(self) -> None:
        self._window = ui.Window("Color Switcher", width=520, height=240, visible=True)

        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Keyword:", height=18)
                ui.Label("Supported: compound (or compond) / small / big", height=18)

                self._keyword_model = ui.SimpleStringModel("compound")
                ui.StringField(self._keyword_model, height=28)

                with ui.HStack(height=30, spacing=8):
                    ui.Button("Apply Color", clicked_fn=self._on_apply_color_clicked)
                    ui.Button("Reset (clear bindings)", clicked_fn=self._on_reset_clicked)

                ui.Separator(height=10)

                ui.Label("Translate target group (units follow your USD stage units):", height=18)
                with ui.HStack(height=28, spacing=8):
                    ui.Label("dx", width=20)
                    self._dx_model = ui.SimpleFloatModel(0.0)
                    ui.FloatField(self._dx_model, width=120)

                    ui.Label("dy", width=20)
                    self._dy_model = ui.SimpleFloatModel(0.0)
                    ui.FloatField(self._dy_model, width=120)

                    ui.Label("dz", width=20)
                    self._dz_model = ui.SimpleFloatModel(0.0)
                    ui.FloatField(self._dz_model, width=120)

                    ui.Button("Translate", clicked_fn=self._on_translate_clicked, width=110)

                self._status_label = ui.Label("", height=42, word_wrap=True)

        # Best-effort: Enter to apply color (if available in this Kit build)
        try:
            self._keyword_model.add_end_edit_fn(lambda *_: self._on_apply_color_clicked())
        except Exception:
            pass

    # --------------------------
    # Actions
    # --------------------------
    def _normalize_keyword(self, raw: str) -> str:
        k = raw.strip().lower()
        return KEYWORD_ALIASES.get(k, k)

    def _get_stage(self) -> Optional[Usd.Stage]:
        return omni.usd.get_context().get_stage()

    def _on_apply_color_clicked(self) -> None:
        keyword = self._normalize_keyword(self._keyword_model.as_string if self._keyword_model else "")
        if keyword not in KEYWORD_TO_GROUP_AND_COLOR:
            self._set_status(f"Unsupported keyword '{keyword}'. Use: compound/compond/small/big.")
            return

        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return

        group_token, rgba = KEYWORD_TO_GROUP_AND_COLOR[keyword]
        mesh_prims = self._find_meshes_under_groups(stage, group_token)
        if not mesh_prims:
            self._set_status(f"No Mesh prims found under groups matching '{group_token}'.")
            return

        self._ensure_xform(stage, LOOKS_ROOT_PATH)
        material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{keyword}")
        material = self._get_or_create_preview_material(stage, material_path, rgba)

        bound_count = 0
        for prim in mesh_prims:
            if self._bind_material_to_prim(prim, material):
                bound_count += 1

        self._set_status(f"Applied '{keyword}' color to {bound_count} mesh prim(s).")

    def _on_translate_clicked(self) -> None:
        keyword = self._normalize_keyword(self._keyword_model.as_string if self._keyword_model else "")
        if keyword not in KEYWORD_TO_GROUP_AND_COLOR:
            self._set_status(f"Unsupported keyword '{keyword}'. Use: compound/compond/small/big.")
            return

        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return

        dx = float(self._dx_model.as_float if self._dx_model else 0.0)
        dy = float(self._dy_model.as_float if self._dy_model else 0.0)
        dz = float(self._dz_model.as_float if self._dz_model else 0.0)

        group_token, _ = KEYWORD_TO_GROUP_AND_COLOR[keyword]

        # We translate the "group root(s)" (Xform prims that match token),
        # not every mesh, to keep the assembly coherent.
        roots = self._find_group_roots(stage, group_token)
        if not roots:
            self._set_status(f"No group root prim found matching '{group_token}'.")
            return

        moved = 0
        for root_prim in roots:
            if self._apply_translate_delta(root_prim, Gf.Vec3d(dx, dy, dz)):
                moved += 1

        self._set_status(f"Translated '{keyword}' on {moved} group root prim(s) by ({dx}, {dy}, {dz}).")

    def _on_reset_clicked(self) -> None:
        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return

        total_cleared = 0
        for _, (group_token, _) in KEYWORD_TO_GROUP_AND_COLOR.items():
            for prim in self._find_meshes_under_groups(stage, group_token):
                if self._clear_material_binding(prim):
                    total_cleared += 1

        self._set_status(f"Cleared bindings on {total_cleared} mesh prim(s).")

    # --------------------------
    # USD helpers (finding)
    # --------------------------
    def _find_group_roots(self, stage: Usd.Stage, group_token: str) -> List[Usd.Prim]:
        """
        Return prims that look like the "root" of a group.
        Strategy:
        - prim path contains group_token
        - prim is Xformable (Xform / Mesh / etc.)
        - prefer Xform prims (UsdGeom.Xform) when possible
        Deduplicate results.
        """
        candidates: List[Usd.Prim] = []
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if group_token not in prim.GetPath().pathString:
                continue
            # Prefer Xform prim types as "group root"
            if prim.IsA(UsdGeom.Xform):
                candidates.append(prim)

        # Fallback: if no Xform found, allow any Xformable prim
        if not candidates:
            for prim in stage.Traverse():
                if not prim.IsValid():
                    continue
                if group_token not in prim.GetPath().pathString:
                    continue
                if UsdGeom.Xformable(prim):
                    candidates.append(prim)

        # Deduplicate by path
        seen = set()
        roots: List[Usd.Prim] = []
        for p in candidates:
            ps = p.GetPath().pathString
            if ps not in seen:
                seen.add(ps)
                roots.append(p)
        return roots

    def _find_meshes_under_groups(self, stage: Usd.Stage, group_token: str) -> List[Usd.Prim]:
        targets: List[Usd.Prim] = []
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if group_token not in prim.GetPath().pathString:
                continue
            targets.extend(self._collect_mesh_descendants(prim))

        # Deduplicate by path
        seen = set()
        unique: List[Usd.Prim] = []
        for p in targets:
            ps = p.GetPath().pathString
            if ps not in seen:
                seen.add(ps)
                unique.append(p)
        return unique

    def _collect_mesh_descendants(self, root: Usd.Prim) -> List[Usd.Prim]:
        meshes: List[Usd.Prim] = []
        if root.IsA(UsdGeom.Mesh):
            meshes.append(root)
        for child in Usd.PrimRange(root):
            if child == root:
                continue
            if child.IsA(UsdGeom.Mesh):
                meshes.append(child)
        return meshes

    # --------------------------
    # USD helpers (authoring)
    # --------------------------
    def _apply_translate_delta(self, prim: Usd.Prim, delta: Gf.Vec3d) -> bool:
        """
        Adds delta to a translate op on the given prim.
        If no translate op exists, create one.
        """
        if not prim or not prim.IsValid():
            return False

        try:
            xf = UsdGeom.Xformable(prim)

            # Try to find an existing translate op
            ops = xf.GetOrderedXformOps()
            translate_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op is None:
                translate_op = xf.AddTranslateOp()

            current = translate_op.Get()
            if current is None:
                current = Gf.Vec3d(0.0, 0.0, 0.0)

            translate_op.Set(Gf.Vec3d(current[0] + delta[0], current[1] + delta[1], current[2] + delta[2]))
            return True
        except Exception:
            return False

    def _ensure_xform(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
        return UsdGeom.Xform.Define(stage, path).GetPrim()

    def _get_or_create_preview_material(
        self, stage: Usd.Stage, material_path: Sdf.Path, rgba: Gf.Vec4f
    ) -> UsdShade.Material:
        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            material = UsdShade.Material.Define(stage, material_path)

        shader_path = material_path.AppendChild("PreviewShader")
        shader = UsdShade.Shader.Get(stage, shader_path)
        if not shader:
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(rgba[0], rgba[1], rgba[2])
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _bind_material_to_prim(self, prim: Usd.Prim, material: UsdShade.Material) -> bool:
        if not prim or not prim.IsValid():
            return False
        try:
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            return True
        except Exception:
            return False

    def _clear_material_binding(self, prim: Usd.Prim) -> bool:
        if not prim or not prim.IsValid():
            return False
        try:
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
            return True
        except Exception:
            return False

    def _set_status(self, msg: str) -> None:
        if self._status_label:
            self._status_label.text = msg

