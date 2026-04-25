# CLAUDE.md

Context file for AI assistants modifying UtilityRelight. Read this before making changes.

---

## What this tool is

A Nuke node that relights real footage using AI-estimated utility passes (N, P, Z, ao from NormalCrafter / LiveActionAOV). Outputs a standalone render layer the comp artist merges onto the plate downstream.

- **Two files, both ship to `~/.nuke/`:**
  - `utility_relight.py` — Python module, builds the Group node programmatically
  - `UtilityRelightKernel.blink` — BlinkScript GPU kernel doing the lighting math
- **No `.gizmo` file.** Nuke's `.gizmo` TCL parser is fragile; we build the node via `nuke.nodes.*` calls inside a Python `with group:` block instead.
- Target Nuke version: **16.0**. Probably works on 14–15. Uncertain on 13 — see Known Issues below.

---

## Architecture

```
User-facing Group node
├── Tabs: Channels, Lighting, Key, Spec, Rim, Bounce, Glow, Fog,
│         Occlusion, Output, About
├── Hidden Internal tab: has_p, has_z, has_ao, auto_scaled flags
│
└── Internal DAG (v1.11, always-on 3D preview):
    Input_src ─────────────────────────┐
    Input_aov ─────┬──── Shuffle_N ────┤
                   ├──── Shuffle_P ────┤
                   ├──── Shuffle_Z ────┤
                   ├──── Shuffle_AO ───┤
                   └──── Shuffle_Alb ──┘
                                       └── BlinkScript (RelightKernel) ──┬──► Merge2 in 1 (B, PASSED)
                                                                         │
                                                                         └──► PositionToPoints2[in 0=color]
                                                                             Shuffle_P ──► [in 1=pos]
                                                                             Shuffle_N ──► [in 2=norm]
                                                                             │
                                                                         TransformGeo (X/Y flip)
                                                                             │
                                                                         ScanlineRender (bg=none, result discarded)
                                                                             │
                                                                         Merge2 in 0 (A, IGNORED)
                                                                             │
                                                                         Output1

    LightAxis (Axis2, DAG-orphaned; read by kernel via world_matrix expressions)
```

**Why the DAG looks that way:**
- PositionToPoints2 input 0 is fed from the **kernel output** so the 3D preview shows the RELIT points. Drag the LightAxis, cloud re-lights in realtime. This is the killer feature — don't break it.
- PositionToPoints2 has 3 dedicated inputs (color/pos/norm). Wire each explicitly to the right source. DO NOT try to be clever with shuffled channels on a single input. (I tried. It breaks.)
- LightAxis is not scene-connected. Nuke's 3D viewer picks it up automatically from the group namespace, and the kernel reads it through `world_matrix` expressions. No Scene node needed.
- Single `Output1`. **Nuke groups can only have ONE Output node.** If you see `Output2` getting auto-created, it means two `nuke.nodes.Output(...)` calls are happening — find and kill the duplicate.
- **The Merge2 + ScanlineRender cheat (v1.11):** the shipped 2D output is the kernel's relight. The ScanlineRender branch isn't used for pixels — but its presence upstream of Output keeps Nuke's 3D viewer engaged with the PositionToPoints2 + LightAxis geometry continuously. Press Tab in the viewer any time to drop into 3D mode. No dropdown toggle needed.
- **Merge2 operation=copy** means "B straight through" — B is the kernel, A (ScanlineRender) is ignored at pixel level but attached for the DAG-presence effect.

---

## Kernel design

**Output = sum of 6 independent layers** (key + spec + rim + bounce + glow + fog).

All lit where the light reaches, black everywhere else. It's a render layer, not a final comp.

**Critical math invariants:**
1. **N and P must be in the same Y/Z frame** before dot products. The `nFlipX/Y/Z` + `nSwapYZ` params realign the estimator's N to match P. Default `flipY=1, flipZ=1` is for NormalCrafter-style OpenGL view-space normals paired with image-space Y-down positions.
2. **Softness uses avgNdotLAtten directly**, not divided by avgAtten. The divide flattens the NdotL gradient across the subject. If you "optimize" this away, lighting goes flat.
3. **Rim is direction-aware**, gated by `(1 - NdotL)^wrap`. It rotates with the light. A pure Fresnel(view) rim (which I tried first) is always at the camera silhouette regardless of light position — wrong for a CG-style back-rim.
4. **keyColor always tints the light**, independent of `keyPlateMix`. The plateMix only controls whether the surface color is the plate or neutral white. This is the fix for an earlier bug where the Color slider did nothing at default settings.
5. **Preview flip compensation** (`previewFlipX/Y`) mirrors `lightPos` and `lightDir` inside the kernel when the 3D preview is flipped for display. Without this, the artist drags the axis in the visible frame but the lighting computes in the unflipped frame — the axis "lights" something other than what it visually points at.

---

## Nuke-specific gotchas to watch for

### 1. BlinkScript kernel param naming
Kernel params appear as knobs named `{kernelName}_{paramName}`, e.g. `UtilityRelightKernel_lightType`. The UI strips the prefix but the Python API needs the full name. See `_SCALAR_LINKS` etc. in `utility_relight.py`.

### 2. Kernel loading
Setting `kernelSource` directly on a BlinkScript node doesn't work (it's a display proxy). You must write the `.blink` to disk, set `kernelSourceFile`, then call `reloadKernelSourceFile.execute()`. Done automatically in `_build_internal_dag`.

### 3. `Image<>` reads return float3, not float4
In this Nuke version the `src()` call returns `float3`. Don't write `float4 s = src();` — it'll fail to compile. Build alpha separately in `dst() = float4(..., 1.0f)`.

### 4. No `local:` block with inline `float PI`
BlinkScript compiler rejects `local:` blocks containing `float` constants. Inline numeric constants (`3.14159265f`) directly.

### 5. Pure ASCII in the `.blink` file
Em-dashes, curly quotes, non-breaking spaces — all break the BlinkScript compiler. Run a non-ASCII byte check on the file before shipping (the build validation in this repo already does this).

### 6. Vector expression linking
For vec3 knobs, set the same base expression on each of the three components (`i=0,1,2`). Nuke maps component-to-component automatically. DO NOT append `.x/.y/.z` to the expression — that breaks the mapping.

### 7. lightDir comes from `world_matrix`, not `rotate`
For Directional/Spot lights, the kernel reads `lightDir` as `-parent.LightAxis.world_matrix.{2,6,10}` (the 3rd column, the forward axis, negated). This is per-component, not a simple vector link.

### 8. PositionToPoints2 input semantics
Three inputs:
- 0 = image source (color carrier — feed it the kernel output for a relit cloud)
- 1 = pos (position data from Shuffle_P)
- 2 = norm (normals from Shuffle_N)

Don't assume you can feed the same image to multiple inputs and let the node sort it out via channel pickers. Be explicit.

### 9. Switch node `which` knob
Integer. Expression-linked to the `view_mode` enum dropdown. Nuke coerces enum → int (0, 1, 2...) in expressions, so `sw["which"].setExpression("parent.view_mode")` works. If you change the enum, check this still makes sense.

### 10. `TransformGeo` scale knob name
Modern Nuke uses `scaling`. Older versions use `scale`. The code tries both via exception fallback.

### 11. Link_Knob unique names
Picker knobs mirrored across Key/Spec/Rim/Bounce/Glow tabs need unique names per instance. A counter variable (`picker_counter`) generates `pick_link_1`, `pick_link_2`, etc.

### 12. XY_Knob viewer overlay only shows on its own tab
The `light_2d_pos` picker overlay only appears when you're looking at the Lighting tab. Link_Knob copies on other tabs give access to the numeric values but not the overlay. Live with it.

### 13. knobChanged callback string
Set on the Group, runs on every knob change inside it. Filter by `k.name()`. We use this to trigger `sync()` on layer-picker changes and `sample_and_place_light()` on picker drag. See `_KNOB_CHANGED_SCRIPT` in `utility_relight.py`.

### 14. Display frame vs data frame — keep them in sync everywhere
There are TWO frames in this node:
- **Data frame** — the raw P from the sidecar (image-space, Y-down).
- **Display frame** — what the user sees in the 3D viewer (flipped via TransformGeo if `preview_flip_x/y` are on, which they are by default).

**The LightAxis lives in the display frame.** The kernel flips `lightPos.x/y` back into the data frame before computing lighting (`previewFlipX/Y` params). Any code that *writes* to LightAxis from raw data — like `sample_and_place_light` reading a P sample — MUST also apply the display-frame flip before writing, or the light will land mirrored.

Bug fixed in v1.10.1: the 2D picker was writing raw P to LightAxis, skipping the flip. Fix at the end of `sample_and_place_light`:
```python
if group["preview_flip_x"].value(): Px = -Px
if group["preview_flip_y"].value(): Py = -Py
```

Any future feature that sets LightAxis from data (e.g. "snap light to surface", "move light along normal") needs the same treatment. Rule of thumb: **LightAxis.translate is always in display frame; kernel handles the conversion.**

---

## Why no Switch / View Mode dropdown

Earlier versions (v1.8–v1.10.1) used a Switch node with a "View Mode" dropdown on the Output tab to toggle between 2D relight and 3D scene preview. That was removed in v1.11: users found toggling back and forth tedious, and Nuke's 3D viewer already has Tab as the 2D/3D toggle.

The current design keeps both the 2D output and the 3D scene graph continuously present in the DAG via the Merge2 + ScanlineRender cheat above. Viewer shows 2D by default; press Tab to see 3D and drag the axis; press Tab again to return. No knob toggle, no round-trip through Python.

If you ever want to bring the Switch back: keep the Merge2 branch as-is and add the Switch in parallel — do NOT use both; Switch fighting against Merge2 for the output path will confuse Nuke's DAG evaluation.

---

## Things to NOT do

- **Don't add a Scene node back.** It was there in v1.8. It's not needed — TransformGeo feeds the output path directly via ScanlineRender, and LightAxis shows up in the 3D viewer from the namespace alone.
- **Don't put keyColor inside the plateMix lerp.** That's the bug from v1.9 where the Color slider did nothing at default settings. Keep `keyColor` as a multiplier on the final keyLit, separate from the plate/white blend.
- **Don't remove the avgNdotLAtten-direct usage in softness.** The old divide-by-atten version flattens lighting and users will complain about "light not discriminating top/bottom."
- **Don't hardcode light_radius / glow_radius / fog_start / fog_end defaults.** Those are auto-scaled from the sidecar's P.z range in `_auto_scale()`. Leaving them at hardcoded values would be wrong for sidecars with different depth scales.
- **Don't create two Output nodes** inside the group — Nuke will rename one to `Output2` and you'll get two output pipes. Nuke groups support ONE output.
- **Don't re-introduce the Switch/View Mode dropdown.** The always-on 3D preview (ScanlineRender + Merge2) was a deliberate v1.11 simplification after user feedback. Tab in the viewer is the native Nuke 2D/3D toggle — don't duplicate it with a knob.
- **Don't try to use `.gizmo` TCL serialization instead of the Python build-from-scratch approach.** Nuke's TCL parser is fragile and the gizmo workflow lost us a lot of time in early iterations.

---

## Nuke version compatibility

- **Nuke 16.0:** Primary target, fully tested.
- **Nuke 15.x:** Likely works. Untested. The main risks are `Axis2`/`Camera2` naming and `PositionToPoints2` availability.
- **Nuke 14.x:** Uncertain. `PositionToPoints2` was introduced fairly recently; Nuke 14 may only have the older `PositionToPoints` which has different inputs and knob names.
- **Nuke 13.x:** Almost certainly broken. `Axis2` might be just `Axis` there. `PositionToPoints2` likely doesn't exist. BlinkScript syntax may differ. **Don't promise users this works on 13 without testing.**

If you need Nuke 13 support, the likely required changes are:
- Replace `Axis2` with `Axis`
- Replace `PositionToPoints2` with `PositionToPoints` (different knob names for position/color — check Nuke 13 docs)
- Possibly update some BlinkScript syntax for the older compiler
- Add fallbacks for the `detail` / `pointSize` / `scaling` knob names

---

## Validation before shipping

Always run these checks before handing off a new version:

```bash
# Python syntax
python3 -c "import ast; ast.parse(open('utility_relight.py').read())"

# Brace balance in kernel
python3 -c "k=open('UtilityRelightKernel.blink').read(); assert k.count('{')==k.count('}')"

# Non-ASCII in both files (must be zero)
python3 -c "
for f in ['utility_relight.py','UtilityRelightKernel.blink']:
    raw=open(f,'rb').read()
    n=sum(1 for b in raw if b>127)
    assert n==0, f'{f} has {n} non-ASCII bytes'
print('OK')
"

# Node creation calls should match expectations
grep -c "nuke.nodes.Output(" utility_relight.py  # must be 1
grep -c "nuke.nodes.Scene(" utility_relight.py   # must be 0
grep -c "nuke.nodes.Copy(" utility_relight.py    # must be 0 (was used before, now removed)
```

If those pass, it will at least LOAD in Nuke. Actual behavior testing requires the live Nuke + real sidecar EXRs.

---

## Version history (high-level)

- **v1.0** — initial `.gizmo` attempt. Abandoned (TCL parser fragility).
- **v1.1** — Python-built group; 25 params linked.
- **v1.2** — Layered approach (Key, Rim, Bounce, Glow). Dropped multiply-albedo.
- **v1.3** — Softness via 7-sample disk, proper key-intensity vs. color split.
- **v1.4** — Standalone render-layer output (no plate passthrough).
- **v1.5** — Specular layer restored, rim made direction-aware, picker mirrored on all tabs.
- **v1.6** — Fog layer restored, Key Plate Mix slider.
- **v1.7** — Normal convention selector (flipX/Y/Z + swapYZ).
- **v1.8** — 3D scene preview via PositionToPoints (original version, ScanlineRender-based; superseded).
- **v1.9** — ScanlineRender + Camera preview branch (superseded by Switch approach in v1.10).
- **v1.10** — Switch-based single-Output design, PositionToPoints2 proper 3-input wiring, kernel-lit preview cloud, Key Color always tints, preview X/Y flip with kernel compensation.
- **v1.10.1** — Bugfix: 2D picker now applies preview X/Y flip to sampled P before writing LightAxis. Previously the light landed mirrored from the clicked surface when flips were on (the default).
- **v1.11** — Removed the Switch and View Mode dropdown. Always-on 3D preview via ScanlineRender + Merge2(copy) so the 3D geometry stays attached to the output graph continuously. Press Tab in viewer to toggle 2D/3D the Nuke-native way.

---

## If the user reports a bug

Before changing code:

1. **Ask which Nuke version.** 16.0 is supported; anything else is uncertain.
2. **Ask for the `.nk` copy-paste** of their UtilityRelight node so you can read the actual wired state.
3. **Ask for a sidecar EXR** if the bug involves shading/positioning — you need to verify the P/N conventions in their data match what the defaults assume.
4. **Look at the `.blink` file kernelSource inside the .nk**, not the kernel file on disk. If the user edited it in the BlinkScript UI, it diverges from the file on disk.
5. **Validate assumptions empirically.** The user's sidecar may have different conventions than ours. Don't assume.

Good luck.
