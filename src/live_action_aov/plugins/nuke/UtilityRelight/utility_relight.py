"""
UtilityRelight v1.11
Part of the LiveActionAOV toolkit - https://github.com/lettidude/LiveActionAOV

Copyright (c) 2026 Leonardo Paolini
Developed with Claude (Anthropic)
License: MIT

==================================================================

A standalone-render-layer relight node for Nuke, with always-on 3D
light placement via Nuke's 3D viewer.

v1.11 changes:
  - Removed the Switch and View Mode dropdown. 3D scene preview is
    now always visible when the viewer is pointed at the node: press
    Tab to enter 3D mode any time. Implementation: ScanlineRender
    branch runs alongside the kernel; Merge2 (copy=B) passes only the
    kernel through as the 2D output but keeps the 3D branch attached
    to the output graph for the 3D viewer to see.

v1.10.1: 2D picker now applies preview X/Y flip before writing axis.
v1.10: Switch + Key Color bugfix + preview flip compensation.
v1.9:  ScanlineRender preview (first version).
v1.8:  3D preview with PositionToPoints.
v1.7:  Normal convention selector.
v1.6:  Key Plate Mix + Fog layer.

Six independent light layers: Key, Spec, Rim, Bounce, Glow, Fog.

Install:
    1. Copy BOTH files into ~/.nuke/:
         utility_relight.py
         UtilityRelightKernel.blink
    2. Add to ~/.nuke/menu.py:
         import utility_relight
         utility_relight.register()
    3. Restart Nuke. Menu -> UtilityPasses -> UtilityRelight
"""

import os
import nuke


# ---------------------------------------------------------------------
# Locate kernel file next to this .py
# ---------------------------------------------------------------------
def _kernel_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "UtilityRelightKernel.blink").replace("\\", "/")


# ---------------------------------------------------------------------
# Build internal DAG inside a Group
# ---------------------------------------------------------------------
def _build_internal_dag(group):
    with group:
        src_in  = nuke.nodes.Input(name="Input_src",  label="src",  number=0)
        aov_in  = nuke.nodes.Input(name="Input_aov",  label="aov",  number=1)

        sh_n   = nuke.nodes.Shuffle(name="Shuffle_N",   label="N")
        sh_n.setInput(0, aov_in)
        sh_p   = nuke.nodes.Shuffle(name="Shuffle_P",   label="P")
        sh_p.setInput(0, aov_in)
        sh_z   = nuke.nodes.Shuffle(name="Shuffle_Z",   label="Z")
        sh_z.setInput(0, aov_in)
        sh_ao  = nuke.nodes.Shuffle(name="Shuffle_AO",  label="AO")
        sh_ao.setInput(0, aov_in)
        sh_alb = nuke.nodes.Shuffle(name="Shuffle_Alb", label="albedo")
        sh_alb.setInput(0, aov_in)

        # LightAxis: sensible default translate; ZERO default rotation
        # (per user request - arbitrary rotation confuses 3D manipulation).
        # Auto-scale overrides translate on AOV connect.
        light_axis = nuke.nodes.Axis2(name="LightAxis")
        light_axis["translate"].setValue([0.0, 0.0, 0.3])
        light_axis["rotate"].setValue([0.0, 0.0, 0.0])

        bs = nuke.nodes.BlinkScript(name="RelightKernel")
        bs["kernelSourceFile"].setValue(_kernel_path())
        bs["reloadKernelSourceFile"].execute()

        bs.setInput(0, src_in)
        bs.setInput(1, sh_n)
        bs.setInput(2, sh_p)
        bs.setInput(3, sh_z)
        bs.setInput(4, sh_ao)
        bs.setInput(5, sh_alb)

        # ---- 3D Preview branch (v1.10) ----
        # v1.11 DAG: always-on 3D preview, no Switch, no View Mode dropdown.
        #
        # The ScanlineRender cheat: we build the 3D preview branch end-to-end
        # (PositionToPoints2 -> TransformGeo -> ScanlineRender) and Merge its
        # output INVISIBLY into the main output path via Merge2 (operation=B).
        # The 2D output is always the kernel's 2D relight (B side wins).
        # But because the ScanlineRender is upstream of Output, Nuke's 3D
        # viewer sees the preview geometry + LightAxis continuously: press
        # Tab in the viewer to drop into 3D mode any time.
        #
        # Final DAG:
        #   RelightKernel (bs) ------+--> Merge2 in 1 (B, PASSED THROUGH)
        #                            |
        #                            +--> PositionToPoints2 in 0 (COLOR, relit)
        #   Shuffle_P  ------------------> PositionToPoints2 in 1 (pos)
        #   Shuffle_N  ------------------> PositionToPoints2 in 2 (norm)
        #                                      |
        #                                 TransformGeo (X/Y flip)
        #                                      |
        #                                 ScanlineRender (bg=none)
        #                                      |
        #                                 Merge2 in 0 (A, IGNORED)
        #                                      |
        #                                   Output1
        #
        # LightAxis stays DAG-orphaned; kernel reads it via world_matrix
        # expressions, and Nuke's 3D viewer picks it up automatically.

        p2p = nuke.nodes.PositionToPoints2(name="PreviewCloud")
        p2p.setInput(0, bs)      # color = relit pixels (LIVE)
        p2p.setInput(1, sh_p)    # pos
        p2p.setInput(2, sh_n)    # norm
        try: p2p["detail"].setValue(0.1)
        except Exception: pass
        try: p2p["pointSize"].setValue(5)
        except Exception: pass

        # TransformGeo for preview X/Y flip. Scale driven by expressions
        # linked to preview_flip_x / preview_flip_y toggles.
        tflip = nuke.nodes.TransformGeo(name="PreviewYFlip")
        tflip.setInput(0, p2p)

        # ScanlineRender: renders the 3D branch to a 2D image. We don't
        # actually USE its output, but having it in the graph is what keeps
        # Nuke's 3D viewer engaged with the preview geometry whenever the
        # output is being viewed.
        sr = nuke.nodes.ScanlineRender(name="PreviewScanline")
        sr.setInput(0, None)    # bg (black is fine; result is discarded)
        sr.setInput(1, tflip)   # obj/scene

        # Merge2 operation=B: B (kernel) wins, A (ScanlineRender) ignored.
        # The ScanlineRender is only here to keep the 3D branch attached
        # to the output graph. Readable intent for future maintainers.
        mrg = nuke.nodes.Merge2(name="PreviewMerge")
        mrg.setInput(0, sr)     # A - ignored
        mrg.setInput(1, bs)     # B - kernel (this is what reaches output)
        try: mrg["operation"].setValue("copy")  # copy = B straight through
        except Exception: pass

        out = nuke.nodes.Output(name="Output1")
        out.setInput(0, mrg)

    return {
        "src": src_in, "aov": aov_in,
        "shuffles": {"N": sh_n, "P": sh_p, "Z": sh_z, "AO": sh_ao, "Alb": sh_alb},
        "light": light_axis,
        "kernel": bs,
        "output": out,
        "p2p": p2p,
        "preview_flip": tflip,
        "scanline": sr,
        "merge": mrg,
    }


# ---------------------------------------------------------------------
# User knobs on the outer Group
# ---------------------------------------------------------------------
def _add_user_knobs(group):
    def tab(name, label=None):
        group.addKnob(nuke.Tab_Knob(name, label or name))

    def text(txt):
        group.addKnob(nuke.Text_Knob("", "", txt))

    def enum(name, label, default, opts, tip=""):
        k = nuke.Enumeration_Knob(name, label, opts)
        k.setValue(default if default in opts else opts[0])
        if tip: k.setTooltip(tip)
        group.addKnob(k)

    def dbl(name, label, default, lo=0.0, hi=1.0, tip=""):
        k = nuke.Double_Knob(name, label)
        k.setRange(lo, hi); k.setValue(default)
        if tip: k.setTooltip(tip)
        group.addKnob(k)

    def color(name, label, default=(1.0, 1.0, 1.0), tip=""):
        k = nuke.Color_Knob(name, label)
        k.setValue(list(default))
        if tip: k.setTooltip(tip)
        group.addKnob(k)

    def integer(name, label, default):
        k = nuke.Int_Knob(name, label)
        k.setValue(default)
        group.addKnob(k)

    def pybtn(name, label, code, tip=""):
        k = nuke.PyScript_Knob(name, label, code)
        if tip: k.setTooltip(tip)
        group.addKnob(k)

    # Counter for unique picker-copy knob names across tabs.
    picker_counter = [0]

    def picker_copy():
        """Render-layer panels need the 2D picker visible too. We can't put
        the same XY_Knob in two places; instead expose Link_Knobs that
        mirror the real ones on the Lighting tab. Each call produces a
        pair of uniquely-named Link_Knobs (Nuke requires unique names)
        pointing at the same underlying light_2d_pos / light_depth_offset."""
        picker_counter[0] += 1
        i = picker_counter[0]
        text("<b>Light placement</b> (link to Lighting tab).")
        lk = nuke.Link_Knob("pick_link_" + str(i), "Light 2D Position")
        lk.setLink("light_2d_pos")
        group.addKnob(lk)
        lkd = nuke.Link_Knob("depth_link_" + str(i), "Depth Offset")
        lkd.setLink("light_depth_offset")
        group.addKnob(lkd)

    # ---------- Channels (first; needed before anything lights) ----------
    tab("tab_channels", "Channels")
    text("AOV layer selection. 'none' disables that input.")

    enum("layer_N",   "Normals",           "N",    ["N", "none"],
         "Normals layer. Required for all shading.")
    enum("layout_N",  "Normals layout",    "xyz",  ["xyz", "rgb"],
         "Which channels of the Normals layer carry X/Y/Z.")

    text("<b>Normals convention</b> - align the estimator's output with the "
         "Position pass frame. Defaults match NormalCrafter (OpenGL view-space "
         "flipped into image-space). If shading looks inverted on some surfaces, "
         "toggle these.")
    k = nuke.Boolean_Knob("n_flip_x", "Flip N.x")
    k.setValue(False); k.setTooltip("Negate N.x. Enable if left/right lighting looks swapped.")
    group.addKnob(k)
    k = nuke.Boolean_Knob("n_flip_y", "Flip N.y")
    k.setValue(True);  k.setTooltip("Negate N.y. Default ON for NormalCrafter (Y-up -> Y-down).")
    group.addKnob(k)
    k = nuke.Boolean_Knob("n_flip_z", "Flip N.z")
    k.setValue(True);  k.setTooltip("Negate N.z. Default ON for NormalCrafter (+Z toward cam -> +Z away).")
    group.addKnob(k)
    k = nuke.Boolean_Knob("n_swap_yz", "Swap N.y <-> N.z")
    k.setValue(False); k.setTooltip("Swap Y and Z axes. Enable if your estimator uses Z-up world frame.")
    group.addKnob(k)
    enum("layer_P",   "Position",          "P",    ["P", "none"],
         "World-space position layer. Required for positional falloff.")
    enum("layout_P",  "Position layout",   "xyz",  ["xyz", "rgb"],
         "Which channels of the Position layer carry X/Y/Z.")
    enum("layer_Z",   "Depth",             "Z",    ["Z", "none"],
         "Scalar depth layer. Used as fallback when Position is unavailable.")
    enum("layout_Z",  "Depth component",   "r",    ["r", "z", "g", "b", "a"],
         "Which channel of the Depth layer carries the scalar depth.")
    enum("layer_AO",  "Ambient Occlusion", "none", ["ao", "none"],
         "AO layer (0=exposed, 1=occluded). Optional.")
    enum("layout_AO", "AO component",      "r",    ["r", "g", "b", "a"])
    enum("layer_alb", "Ext. Albedo",       "none", ["albedo", "none"],
         "External albedo layer. Reserved for future use.")
    enum("layout_alb","Ext. Albedo layout","rgb",  ["xyz", "rgb"])

    pybtn("refresh_layers", "Refresh Layer Lists",
          "utility_relight.sync(nuke.thisNode())",
          "Re-scan the connected AOV input for available layers.")

    text("<b>Camera intrinsics</b> (used when Position = 'none').")
    dbl("fx", "fx (px)", 1000.0, 100, 5000,
        "Focal length in pixels, x axis.")
    dbl("fy", "fy (px)", 1000.0, 100, 5000,
        "Focal length in pixels, y axis.")
    dbl("cx", "cx (px)",  960.0,   0, 4096,
        "Principal point x coordinate in pixels.")
    dbl("cy", "cy (px)",  540.0,   0, 2160,
        "Principal point y coordinate in pixels.")

    # ---------- Lighting ----------
    tab("tab_lighting", "Lighting")
    enum("light_type", "Light Type", "Point", ["Point", "Directional", "Spot"],
         "Point: omnidirectional with gaussian falloff. "
         "Directional: constant direction from LightAxis rotation, no falloff. "
         "Spot: point + cone gated by Spot parameters below.")

    text("<b>Placement.</b> Drag the <i>Light 2D Position</i> point on the viewer "
         "to sample the plate and drop the light onto the surface you clicked. "
         "Push it forward/back with <i>Light Depth Offset</i>.")
    k = nuke.XY_Knob("light_2d_pos", "Light 2D Position")
    k.setValue([960.0, 540.0])
    k.setTooltip("Drag this point on the viewer to place the light. "
                 "The AOV's P is sampled at the clicked pixel, then pushed "
                 "toward the camera by Depth Offset.")
    group.addKnob(k)
    dbl("light_depth_offset", "Light Depth Offset", 0.01, -1.0, 1.0,
        "Offset along view ray toward camera from sampled surface P. "
        "Default 0.01 keeps the light just above the clicked surface; "
        "for 3D placement, drag the LightAxis directly in 3D Scene mode. "
        "Auto-scaled on first AOV connection.")

    text("<b>Manual transform</b> (driven by picker above; also editable).")
    lk_t = nuke.Link_Knob("light_translate", "Translate"); lk_t.setLink("LightAxis.translate")
    lk_t.setTooltip("Direct world-space position of the light.")
    group.addKnob(lk_t)
    lk_r = nuke.Link_Knob("light_rotate",    "Rotate");    lk_r.setLink("LightAxis.rotate")
    lk_r.setTooltip("Rotation of the LightAxis. Drives Directional light "
                    "direction and Spot cone axis.")
    group.addKnob(lk_r)

    text("<b>Falloff</b> (3D gaussian, no hard cutoff).")
    dbl("light_radius", "Light Radius", 0.3, 0.001, 2.0,
        "Gaussian sigma in P-space units. Typically 0.1-0.5 for normalized "
        "depth sidecars; auto-scaled on AOV connect.")
    dbl("light_softness", "Softness",   0.3, 0.0, 1.0,
        "Area-light approximation via 7-sample disk facing the camera. "
        "0 = single hard point. 1 = samples spread by radius*0.5 for a "
        "large, soft light.")

    text("<b>Spot parameters</b> (only used when Light Type = Spot).")
    dbl("spot_angle",    "Cone Half-Angle (deg)", 30.0, 1.0, 89.0,
        "Outer half-angle of the spotlight cone, in degrees.")
    dbl("spot_softness", "Cone Softness",          0.2, 0.0, 1.0,
        "Feathering at the cone edge. 0 = hard cutoff, 1 = very soft.")

    text("<b>3D positioning</b> - the proper way to place this light.")
    text(
        "Connect the node's output to a Viewer, then press <b>Tab</b> in "
        "the viewer to enter 3D mode. You'll see a point cloud of the "
        "subject (from the P pass) with the <i>LightAxis</i> gizmo among "
        "them. <b>Drag the axis handles</b> to position the light in real "
        "3D space - the cloud re-lights in realtime as you drag. Press "
        "<b>Tab</b> again to return to 2D."
    )
    dbl("preview_detail", "Point Detail", 0.1, 0.01, 1.0,
        "Fraction of source pixels turned into points. 0.1 = 10% (fast, "
        "default). 1.0 = every pixel (dense, slower). Live-linked to "
        "PositionToPoints inside the gizmo.")
    dbl("preview_point_size", "Point Size", 5.0, 1.0, 20.0,
        "Pixel size of each point in the 3D viewer.")
    k = nuke.Boolean_Knob("preview_flip_x", "Flip Preview X")
    k.setValue(True)
    k.setTooltip(
        "Mirror the preview cloud horizontally. Matches NormalCrafter-style "
        "image-space X convention so the cloud reads right-way-round.")
    group.addKnob(k)
    k = nuke.Boolean_Knob("preview_flip_y", "Flip Preview Y")
    k.setValue(True)
    k.setTooltip(
        "Mirror the preview cloud vertically. P-space sidecars use image-"
        "space Y-down, which appears upside-down in Nuke's Y-up 3D viewer. "
        "Both flips are compensated inside the kernel so LightAxis drag "
        "lights what it visually points at.")
    group.addKnob(k)

    # ---------- LAYER 1: Key ----------
    tab("tab_key", "Key")
    text(
        "<b>Key layer.</b> The main light contribution: plate colors lit by "
        "NdotL * falloff. At lit pixels, output = surface * intensity * color. "
        "Plate Mix controls the surface: 0 = use the plate (dark plate stays "
        "dark), 1 = use pure keyColor (light punches through shadows)."
    )
    picker_copy()
    text("<b>Key settings</b>")
    dbl("key_intensity", "Intensity", 2.0, 0.0, 4.0,
        "Brightness multiplier on the lit contribution.")
    color("key_color",   "Color",     (1.0, 1.0, 1.0),
          "Color of the light. Plate Mix controls how much it overrides "
          "plate colors.")
    dbl("key_plate_mix", "Plate Mix", 0.0, 0.0, 1.0,
        "0 = plate-colored light (honors plate luminance, shadows stay dark). "
        "1 = pure key-colored light (ignores plate, punches through shadows). "
        "Start at 0.3-0.5 for night scenes where the plate is nearly black.")
    dbl("key_amount",    "Mix",       1.0, 0.0, 2.0,
        "Master multiplier on top of Intensity. Handy for quick A/B.")

    # ---------- LAYER 2: Spec ----------
    tab("tab_spec", "Spec")
    text(
        "<b>Specular layer.</b> Blinn-Phong highlight that tracks the light "
        "direction. Adds a proper sharp reflection on top of the diffuse key. "
        "Roughness controls how tight the highlight is."
    )
    picker_copy()
    text("<b>Specular settings</b>")
    color("spec_color",    "Color",     (1.0, 1.0, 1.0),
          "Color of the specular highlight.")
    dbl("spec_amount",     "Amount",    0.03, 0.0, 4.0,
        "Specular intensity. 0 = no highlight. Subtle default (0.03); push "
        "to 0.3-1.0 for stronger highlights.")
    dbl("spec_roughness",  "Roughness", 0.4, 0.02, 1.0,
        "Microsurface roughness. 0.02 = mirror-sharp, 1.0 = very broad lobe.")

    # ---------- LAYER 3: Rim ----------
    tab("tab_rim", "Rim")
    text(
        "<b>Rim layer.</b> Additive direction-aware Fresnel. The bright edge "
        "appears on the side of the silhouette FACING AWAY from the light - "
        "classic back-rim. Move the light and the rim rotates with it."
    )
    picker_copy()
    text("<b>Rim settings</b>")
    color("rim_color",   "Color",    (1.0, 0.9, 0.7),
          "Rim color. Default warm; try cool (0.7, 0.9, 1.0) for moonlit rim.")
    dbl("rim_amount",    "Amount",   0.0, 0.0, 4.0,
        "Rim intensity. 0 = disabled. Start around 0.5-1.0.")
    dbl("rim_exp",       "Falloff",  0.5, 0.5, 10.0,
        "Fresnel exponent. Higher = tighter / sharper rim edge.")
    dbl("rim_wrap",      "Wrap",     0.0, 0.0, 1.0,
        "How much the rim wraps toward the lit side. 0 = rim only on the "
        "darkest edge (classic back-rim). 1 = rim wraps around most of "
        "the silhouette.")

    # ---------- LAYER 4: Bounce ----------
    tab("tab_bounce", "Bounce")
    text(
        "<b>Bounce layer.</b> Additive complementary fill on the shadow side. "
        "Simulates bounced environment light. Usually a small, cool complement "
        "to a warm key (or vice versa)."
    )
    picker_copy()
    text("<b>Bounce settings</b>")
    color("bounce_color", "Color",   (0.4, 0.5, 0.7),
          "Bounce color. Pick a hue complementary to the Key color for a "
          "cinematic split-tone feel.")
    dbl("bounce_amount",  "Amount",  0.0, 0.0, 2.0,
        "Bounce fill intensity. 0 = disabled. Start around 0.1-0.3.")
    dbl("bounce_ao_damp", "AO Damp", 0.6, 0.0, 1.0,
        "How much AO darkens the bounce. 0 = AO ignored. 1 = cavities go fully dark.")

    # ---------- LAYER 5: Glow ----------
    tab("tab_glow", "Glow")
    text(
        "<b>Atmospheric glow.</b> Additive gaussian halo around the light in 3D "
        "space. Gives the 'light has volume in the air' feel - visible even on "
        "dark surfaces. Independent of surface orientation."
    )
    picker_copy()
    text("<b>Glow settings</b>")
    color("glow_color",  "Color",  (1.0, 0.9, 0.7),
          "Glow color. Usually matches the Key color.")
    dbl("glow_amount",   "Amount", 0.0, 0.0, 4.0,
        "Glow intensity. 0 = disabled.")
    dbl("glow_radius",   "Radius", 0.8, 0.001, 4.0,
        "Gaussian sigma for the glow halo. Typically 2-4x the Light Radius.")

    # ---------- LAYER 6: Fog / Volume ----------
    tab("tab_fog", "Fog")
    text(
        "<b>Volume / atmospheric haze.</b> Depth-modulated fog lit by the "
        "active light. Creates the cinematic 'beam of light in the air' feel: "
        "brighter near the light, fades into the scene's dark corners. "
        "Combines well with Glow for fog-filled interior shots."
    )
    picker_copy()
    text("<b>Fog settings</b>")
    color("fog_color",   "Color",   (1.0, 1.0, 1.0),
          "Fog tint. Usually white or slightly warm/cool to match the scene. "
          "Modulated by the Key color internally.")
    dbl("fog_amount",    "Amount",  0.0, 0.0, 4.0,
        "Fog intensity. 0 = disabled. Start around 0.3-1.0 for subtle haze.")
    dbl("fog_start",     "Start (Z)", 0.0,  0.0, 2.0,
        "Depth at which fog starts to appear (in P.z units). "
        "Auto-scaled on AOV connect.")
    dbl("fog_end",       "End (Z)",   1.0,  0.0, 2.0,
        "Depth at which fog is fully dense. Auto-scaled on AOV connect.")

    # ---------- Occlusion ----------
    tab("tab_ao", "Occlusion")
    text("<b>AO on key layer.</b> Darkens the key in crevices. Requires AO layer connected.")
    dbl("ao_int", "AO Intensity", 0.0, 0, 2,
        "Default 0 (AO off) - AI-generated AO can produce weird results. "
        "Turn up if AO is clean.")

    # ---------- Output ----------
    tab("tab_output", "Output")
    enum("output_mode", "Output Mode",
         "Combined (all layers)",
         ["Combined (all layers)",
          "Key only",
          "Spec only",
          "Rim only",
          "Bounce only",
          "Glow only",
          "Fog only",
          "Key mask (diagnostic)",
          "Normals (diagnostic)",
          "Depth (diagnostic)",
          "AO (diagnostic)"],
         "Preview a single layer in isolation, or diagnostic views for "
         "checking AOV integrity.")
    dbl("mix_amt", "Mix", 1.0, 0, 1,
        "1 = pure light render layer (for export/downstream merge). "
        "<1 blends the plate back in for in-node A/B preview.")
    text("<b>Depth diagnostic range</b>.")
    dbl("z_near", "Z Near", 0.0,  0, 2.0,
        "Near clip for the Depth diagnostic view.")
    dbl("z_far",  "Z Far",  1.0,  0, 2.0,
        "Far clip for the Depth diagnostic view.")

    # ---------- About (now last) ----------
    tab("tab_about", "About")
    text(
        "<b>UtilityRelight v1.11</b><br/>"
        "Part of the <b>LiveActionAOV</b> toolkit.<br/>"
        "(c) 2026 <i>Leonardo Paolini</i> &middot; "
        "Developed with Claude (Anthropic) &middot; License: MIT<br/><br/>"
        "<b>Standalone render layer.</b> Output is the light contribution ONLY "
        "(black where the light doesn't reach) - merge it onto the plate "
        "downstream with Plus / Screen / ColorDodge. Layer multiple instances "
        "for multi-light setups.<br/><br/>"
        "<b>Inputs:</b> src (plate), aov (LiveActionAOV sidecar: N, P, Z, ao)."
    )

    # ---------- Hidden state ----------
    tab("tab_internal", "Internal")
    integer("has_p",       "has_p",       1)
    integer("has_z",       "has_z",       1)
    integer("has_ao",      "has_ao",      0)
    integer("has_alb",     "has_alb",     0)
    integer("auto_scaled", "auto_scaled", 0)
    group.knob("tab_internal").setFlag(nuke.INVISIBLE)


# ---------------------------------------------------------------------
# Param linking. Kernel knob names are prefixed with kernelName_.
# ---------------------------------------------------------------------
_SCALAR_LINKS = [
    ("lightType",     "parent.light_type"),
    ("lightRadius",   "parent.light_radius"),
    ("softness",      "parent.light_softness"),
    ("spotAngle",     "parent.spot_angle"),
    ("spotSoftness",  "parent.spot_softness"),
    ("keyIntensity",  "parent.key_intensity"),
    ("keyAmount",     "parent.key_amount"),
    ("keyPlateMix",   "parent.key_plate_mix"),
    ("specAmount",    "parent.spec_amount"),
    ("specRoughness", "parent.spec_roughness"),
    ("rimAmount",     "parent.rim_amount"),
    ("rimExp",        "parent.rim_exp"),
    ("rimWrap",       "parent.rim_wrap"),
    ("bounceAmount",  "parent.bounce_amount"),
    ("bounceAODamp",  "parent.bounce_ao_damp"),
    ("glowAmount",    "parent.glow_amount"),
    ("glowRadius",    "parent.glow_radius"),
    ("fogAmount",     "parent.fog_amount"),
    ("fogStart",      "parent.fog_start"),
    ("fogEnd",        "parent.fog_end"),
    ("hasAO",         "parent.has_ao"),
    ("aoInt",         "parent.ao_int"),
    ("hasP",          "parent.has_p"),
    ("hasZ",          "parent.has_z"),
    ("nFlipX",        "parent.n_flip_x"),
    ("nFlipY",        "parent.n_flip_y"),
    ("nFlipZ",        "parent.n_flip_z"),
    ("nSwapYZ",       "parent.n_swap_yz"),
    ("previewFlipX",  "parent.preview_flip_x"),
    ("previewFlipY",  "parent.preview_flip_y"),
    ("fx",            "parent.fx"),
    ("fy",            "parent.fy"),
    ("cx",            "parent.cx"),
    ("cy",            "parent.cy"),
    ("outputMode",    "parent.output_mode"),
    ("mixAmt",        "parent.mix_amt"),
    ("zNear",         "parent.z_near"),
    ("zFar",          "parent.z_far"),
]

_VECTOR_LINKS = [
    ("lightPos",    "parent.LightAxis.translate"),
    ("keyColor",    "parent.key_color"),
    ("specColor",   "parent.spec_color"),
    ("rimColor",    "parent.rim_color"),
    ("bounceColor", "parent.bounce_color"),
    ("glowColor",   "parent.glow_color"),
    ("fogColor",    "parent.fog_color"),
]

# Per-component: lightDir comes from LightAxis rotation (forward axis).
# world_matrix columns: indices 2, 6, 10 = local -Z in world, negated.
_VECTOR_LINKS_PER_COMPONENT = [
    ("lightDir", [
        "-parent.LightAxis.world_matrix.2",
        "-parent.LightAxis.world_matrix.6",
        "-parent.LightAxis.world_matrix.10",
    ]),
]


def _link_kernel_params(group, kernel_node):
    """Wire kernel params to group-level knobs. Idempotent."""
    prefix = kernel_node["kernelName"].value() + "_"

    for param, expr in _SCALAR_LINKS:
        k = kernel_node.knob(prefix + param)
        if k is None: continue
        try: k.setExpression(expr)
        except Exception as e: nuke.tprint("link fail " + param + ": " + str(e))

    for param, expr in _VECTOR_LINKS:
        k = kernel_node.knob(prefix + param)
        if k is None: continue
        try:
            for i in range(3):
                k.setExpression(expr, i)
        except Exception as e:
            nuke.tprint("vec link fail " + param + ": " + str(e))

    for param, exprs in _VECTOR_LINKS_PER_COMPONENT:
        k = kernel_node.knob(prefix + param)
        if k is None: continue
        try:
            for i, ex in enumerate(exprs):
                k.setExpression(ex, i)
        except Exception as e:
            nuke.tprint("per-comp link fail " + param + ": " + str(e))


def _link_preview_nodes(group):
    """Wire the 3D-preview-branch knobs (PositionToPoints detail/size,
    TransformGeo flip-X/Y scale) to outer user-facing knobs. Idempotent:
    uses setExpression, which overwrites."""
    try:
        p2p = group.node("PreviewCloud")
        if p2p is not None:
            # PositionToPoints2 knob names:
            for knob_name, expr in (("detail",     "parent.preview_detail"),
                                    ("pointSize",  "parent.preview_point_size")):
                try:
                    p2p[knob_name].setExpression(expr)
                except Exception as e:
                    nuke.tprint("p2p {} link fail: {}".format(knob_name, e))

        # Flip X/Y on the TransformGeo scale.
        tflip = group.node("PreviewYFlip")
        if tflip is not None:
            expr_x = "parent.preview_flip_x ? -1 : 1"
            expr_y = "parent.preview_flip_y ? -1 : 1"
            # Nuke versions: try "scaling" (newer) then "scale" (older).
            for knob_name in ("scaling", "scale"):
                try:
                    tflip[knob_name].setExpression(expr_x, 0)
                    tflip[knob_name].setExpression(expr_y, 1)
                    tflip[knob_name].setExpression("1", 2)
                    break
                except Exception:
                    continue

    except Exception as e:
        import traceback
        nuke.tprint("_link_preview_nodes error: " + str(e))
        nuke.tprint(traceback.format_exc())


# ---------------------------------------------------------------------
# Shuffle config
# ---------------------------------------------------------------------
def _configure_shuffle(sh, layer, layout=None):
    try:
        sh["in"].setValue(layer if layer != "none" else "none")
    except Exception:
        pass


# ---------------------------------------------------------------------
# Channel sampling helper (tries multiple suffix conventions)
# ---------------------------------------------------------------------
def _sample_any(node, layer, channel_candidates, px, py):
    for suffix in channel_candidates:
        try:
            return node.sample(layer + "." + suffix, px, py)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------
# Auto-scale on aov connect
# ---------------------------------------------------------------------
def _auto_scale(group):
    """Sample the AOV depth range and adapt radii/offsets/glow to match."""
    try:
        flag = group.knob("auto_scaled")
        if flag is not None and flag.value() > 0.5:
            return

        aov = group.input(1)
        if aov is None:
            return

        has_p = group["has_p"].value() > 0
        has_z = group["has_z"].value() > 0
        layer_P = group["layer_P"].value() if has_p else "none"
        layer_Z = group["layer_Z"].value() if has_z else "none"

        try:
            fmt = aov.format()
            W = fmt.width(); H = fmt.height()
        except Exception:
            W, H = 1920, 1080

        samples_z = []
        for gy in (1, 2, 3):
            for gx in (1, 2, 3):
                x = int(W * gx / 4); y = int(H * gy / 4)
                z = None
                if has_p and layer_P != "none":
                    z = _sample_any(aov, layer_P, ("z", "blue"), x, y)
                elif has_z and layer_Z != "none":
                    z = _sample_any(aov, layer_Z, ("red", "z", "x"), x, y)
                if z is not None and z > 1e-4 and z < 1e5:
                    samples_z.append(z)

        if len(samples_z) < 3:
            return

        z_min = min(samples_z); z_max = max(samples_z); z_mid = (z_min + z_max) * 0.5
        scene_scale = max(z_max - z_min, z_mid * 0.5)

        r = scene_scale * 0.3
        try: group["light_radius"].setValue(max(r, 1e-3))
        except Exception: pass
        try: group["glow_radius"].setValue(max(r * 2.5, 1e-3))
        except Exception: pass
        # Push the light well off the surface (30% of scene depth toward camera)
        # so NdotL discriminates properly between top and bottom of the subject.
        # Previously this was 5%, which put the light nearly on the surface
        # and gave almost flat lighting.
        try: group["light_depth_offset"].setValue(max(scene_scale * 0.3, 1e-4))
        except Exception: pass

        axis = group.node("LightAxis")
        if axis is not None:
            axis["translate"].setValue([0.0, z_mid * 0.3, z_mid * 0.7])

        try:
            group["z_near"].setValue(max(z_min - scene_scale * 0.05, 0.0))
            group["z_far"].setValue(z_max + scene_scale * 0.05)
        except Exception:
            pass
        # Fog depth range: bracket the observed scene, slightly beyond near/far.
        try:
            group["fog_start"].setValue(max(z_min, 0.0))
            group["fog_end"].setValue(z_max)
        except Exception:
            pass

        if flag is not None:
            flag.setValue(1)

        nuke.tprint("UtilityRelight: auto-scale z=[{:.3f}, {:.3f}], "
                    "lightRadius={:.3f}, glowRadius={:.3f}, offset={:.4f}".format(
                    z_min, z_max, r, r * 2.5, scene_scale * 0.3))

    except Exception as e:
        import traceback
        nuke.tprint("UtilityRelight._auto_scale error: " + str(e))
        nuke.tprint(traceback.format_exc())


# ---------------------------------------------------------------------
# Live sampling: XY knob drag -> sample P -> reposition LightAxis
# ---------------------------------------------------------------------
def sample_and_place_light(group):
    try:
        aov = group.input(1)
        if aov is None:
            return
        xy_knob = group.knob("light_2d_pos")
        if xy_knob is None:
            return
        px = int(round(xy_knob.value(0)))
        py = int(round(xy_knob.value(1)))

        has_p = group["has_p"].value() > 0
        has_z = group["has_z"].value() > 0
        layer_P = group["layer_P"].value() if has_p else "none"
        layer_Z = group["layer_Z"].value() if has_z else "none"

        Px = Py = Pz = None
        if has_p and layer_P != "none":
            Px = _sample_any(aov, layer_P, ("x", "red"),   px, py)
            Py = _sample_any(aov, layer_P, ("y", "green"), px, py)
            Pz = _sample_any(aov, layer_P, ("z", "blue"),  px, py)
        elif has_z and layer_Z != "none":
            Zv = _sample_any(aov, layer_Z, ("red", "z", "x"), px, py)
            if Zv is None:
                return
            fx = group["fx"].value()
            fy = group["fy"].value()
            cx = group["cx"].value()
            cy = group["cy"].value()
            Px = (px - cx) / max(fx, 1e-4) * Zv
            Py = (py - cy) / max(fy, 1e-4) * Zv
            Pz = Zv
        else:
            return

        if Px is None or Py is None or Pz is None:
            return

        import math
        Plen = math.sqrt(Px*Px + Py*Py + Pz*Pz)
        if Plen < 1e-3:
            return

        off = group["light_depth_offset"].value()
        ux = -Px / Plen; uy = -Py / Plen; uz = -Pz / Plen
        Px += ux * off; Py += uy * off; Pz += uz * off

        # Preview-flip compensation (v1.10.1). The 3D cloud is displayed
        # X/Y-flipped via TransformGeo, and the kernel flips lightPos.x/.y
        # back into the raw P frame for correct lighting. If we wrote the
        # raw sampled P directly to LightAxis.translate, the kernel would
        # flip it away from the surface we sampled. So: pre-flip here to
        # match the display frame. The axis then visually sits on the
        # (flipped) surface the user clicked, and the kernel un-flips it
        # back to P-frame for the math. Net result: light lands on the
        # clicked surface regardless of flip flags.
        try:
            if group["preview_flip_x"].value():
                Px = -Px
            if group["preview_flip_y"].value():
                Py = -Py
        except Exception:
            pass

        axis = group.node("LightAxis")
        if axis is not None:
            axis["translate"].setValue([float(Px), float(Py), float(Pz)])
    except Exception as e:
        import traceback
        nuke.tprint("sample_and_place_light error: " + str(e))
        nuke.tprint(traceback.format_exc())


# ---------------------------------------------------------------------
# Sync callback
# ---------------------------------------------------------------------
def sync(group):
    try:
        kernel = group.node("RelightKernel")
        if kernel is not None:
            prefix = kernel["kernelName"].value() + "_"
            probe = kernel.knob(prefix + "lightType")
            if probe is not None and not probe.hasExpression():
                _link_kernel_params(group, kernel)

        aov_in = group.input(1)
        if aov_in is not None:
            chans = aov_in.channels()
            layers = sorted(set(c.split(".")[0] if "." in c else c for c in chans))
            opts = ["none"] + layers
            defaults = {"layer_N":"N","layer_P":"P","layer_Z":"Z",
                        "layer_AO":"ao","layer_alb":"albedo"}
            for kn in ("layer_N","layer_P","layer_Z","layer_AO","layer_alb"):
                k = group.knob(kn)
                if k is None: continue
                current = k.value()
                k.setValues(opts)
                if current in opts:
                    k.setValue(current)
                else:
                    want = defaults.get(kn, "none")
                    k.setValue(want if want in opts else "none")

        def sel(name):
            k = group.knob(name)
            return k.value() if k is not None else "none"

        group["has_p"].setValue(  0 if sel("layer_P")   == "none" else 1)
        group["has_z"].setValue(  0 if sel("layer_Z")   == "none" else 1)
        group["has_ao"].setValue( 0 if sel("layer_AO")  == "none" else 1)
        group["has_alb"].setValue(0 if sel("layer_alb") == "none" else 1)

        _auto_scale(group)

        pairs = [
            ("Shuffle_N",   sel("layer_N"),   sel("layout_N")),
            ("Shuffle_P",   sel("layer_P"),   sel("layout_P")),
            ("Shuffle_Z",   sel("layer_Z"),   sel("layout_Z")),
            ("Shuffle_AO",  sel("layer_AO"),  sel("layout_AO")),
            ("Shuffle_Alb", sel("layer_alb"), sel("layout_alb")),
        ]
        for sh_name, layer, layout in pairs:
            sh = group.node(sh_name)
            if sh is None: continue
            _configure_shuffle(sh, layer, layout)

    except Exception as e:
        import traceback
        nuke.tprint("sync error: " + str(e))
        nuke.tprint(traceback.format_exc())


_KNOB_CHANGED_SCRIPT = """
import nuke, utility_relight
k = nuke.thisKnob()
if k is not None:
    name = k.name()
    if name in (
        'inputChange','showPanel',
        'layer_N','layer_P','layer_Z','layer_AO','layer_alb',
        'layout_N','layout_P','layout_Z','layout_AO','layout_alb',
        'refresh_layers'):
        utility_relight.sync(nuke.thisNode())
    elif name in ('light_2d_pos', 'light_depth_offset'):
        utility_relight.sample_and_place_light(nuke.thisNode())
"""


def create():
    group = nuke.createNode("Group", inpanel=False)
    group.setName("UtilityRelight1", uncollide=True)
    group["help"].setValue(
        "UtilityRelight v1.11 - part of the LiveActionAOV toolkit. "
        "Standalone light render layer. Connect: 0=plate, 1=AOV sidecar. "
        "Merge (plus) onto plate downstream. Press Tab in viewer for "
        "3D mode - drag the LightAxis to position the light in real 3D."
    )
    nodes = _build_internal_dag(group)
    _add_user_knobs(group)
    _link_kernel_params(group, nodes["kernel"])
    _link_preview_nodes(group)
    group["knobChanged"].setValue(_KNOB_CHANGED_SCRIPT)
    sync(group)
    return group


def register():
    toolbar = nuke.menu("Nodes")
    m = toolbar.addMenu("UtilityPasses")
    m.addCommand("UtilityRelight", "utility_relight.create()")
