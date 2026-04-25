# UtilityRelight

**Nuke node for physically-motivated relighting from AOV passes.**
Six layered light contributions, live 3D axis placement, single BlinkScript kernel on the GPU.

Part of the [LiveActionAOV](https://github.com/lettidude/LiveActionAOV) toolkit — an open-source pipeline that produces AI-estimated utility passes (depth, normals, position, AO, mattes) from real footage as sidecar EXRs. UtilityRelight consumes those sidecars to let comp artists relight footage in Nuke.

---

- **Version:** 1.11
- **Copyright** (c) 2026 Leonardo Paolini
- **Developed with** Claude (Anthropic)
- **License:** MIT (see `LICENSE`)
- **Tested on:** Nuke 16.0

---

## Install

1. Copy these files into your `~/.nuke/` folder (directly, NOT in subfolders):
   - `utility_relight.py`
   - `UtilityRelightKernel.blink`

2. Add to `~/.nuke/menu.py`:
   ```python
   import utility_relight
   utility_relight.register()
   ```

3. Restart Nuke. The node appears at `Nodes → UtilityPasses → UtilityRelight`.

---

## Quick start

1. Create the node. Connect plate to input `src`, sidecar EXR to input `aov`.
2. Connect the output to a Viewer.
3. Press **Tab** in the viewer to enter 3D mode.
4. You see a point cloud of the subject with the **LightAxis** gizmo among them.
5. **Drag the axis arrows** to position the light — the cloud re-lights in realtime.
6. Press **Tab** again to return to 2D and see the final render-layer output.
7. Downstream: merge (plus) the output onto your plate.

---

## Inputs

| # | Name | Content |
|---|------|---------|
| 0 | `src` | beauty plate (RGB) |
| 1 | `aov` | LiveActionAOV sidecar EXR (`N`, `P`, `Z`, optional `ao`) |

The node is layer-agnostic — pick which layer is which from the **Channels** tab. Defaults assume LiveActionAOV naming.

---

## Six light layers

| Layer | Type | Purpose |
|-------|------|---------|
| **Key** | Diffuse | Plate-colored NdotL × falloff lift, with optional Plate Mix to blend toward pure light color |
| **Spec** | Blinn-Phong | Sharp highlight tracking the light direction, Roughness-controlled |
| **Rim** | Fresnel | Direction-aware back-rim that rotates with the light |
| **Bounce** | Complementary | Shadow-side fill in a complementary color, AO-damped |
| **Glow** | 3D Gaussian | Atmospheric halo around the light in space |
| **Fog** | Volume | Depth-modulated haze lit by the active light |

Output is the **sum of all layers** (black where nothing reaches) — a standalone render layer to merge over the plate downstream. This is the right mental model: these are lights you're adding, not a finished comp.

---

## Light types

- **Point** — omnidirectional with 3D gaussian falloff (`exp(-(d/r)²)`)
- **Directional** — constant direction from LightAxis rotation, no falloff
- **Spot** — point falloff + cone test using LightAxis rotation as forward axis

All types support **Softness** (0..1) — area-light approximation via 7-sample disk facing the camera. No accidental back-lighting.

---

## Output modes (diagnostic views)

The **Output Mode** dropdown on the Output tab lets you preview any single layer in isolation plus diagnostics:

- Combined (all layers)
- Key only / Spec only / Rim only / Bounce only / Glow only / Fog only
- Key mask (grayscale NdotL × atten)
- Normals (after convention flags applied)
- Depth (bracket via Z Near / Z Far)
- AO

---

## Conventions and adaptation

**Sidecar data conventions** (typical for LiveActionAOV / NormalCrafter):
- Normals: camera-space OpenGL (+Y up, +Z toward camera), unit length
- Position: image-space (+Y down, +Z away from camera), normalized or metric
- AO: 0 = exposed, 1 = occluded
- Depth: scalar on `.r` or `.x`, same units as P.z

**Normal convention selector** (Channels tab):
- **Flip N.x** (default OFF) — enable if left/right lighting reads swapped
- **Flip N.y** (default ON) — default for NormalCrafter (Y-up → Y-down)
- **Flip N.z** (default ON) — default for NormalCrafter (+Z toward cam → +Z away)
- **Swap N.y ↔ N.z** (default OFF) — for Z-up world-frame estimators

If you swap to a different normal estimator (Marigold, DSINE, GeoWizard, etc.) and shading looks inverted on some surfaces, toggle these.

---

## Files

| File | Purpose |
|------|---------|
| `utility_relight.py` | Python module that builds the node |
| `UtilityRelightKernel.blink` | Single BlinkScript kernel (GPU shader) |
| `CLAUDE.md` | Context file for AI assistants making changes |
| `LICENSE` | MIT license text |
| `README.md` | This file |

---

## Typical downstream comp

```
[plate] ──────────────── Merge (plus) ──── Viewer
                               ↑
         [UtilityRelight] ─────┘
             ↑    ↑
         plate  aov
```

Layer multiple UtilityRelight nodes for multi-light setups (key + fill + rim):

```
[plate] ──── Merge (plus) ──── Merge (plus) ──── Merge (plus) ──── Viewer
                  ↑                  ↑                  ↑
         [Key UtilityRelight]   [Fill]             [Rim]
```
