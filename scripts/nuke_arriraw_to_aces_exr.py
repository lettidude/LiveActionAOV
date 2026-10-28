"""Turn ARRIRAW camera files into ACEScg EXR test plates, inside Nuke.

    "C:\\Program Files\\Nuke16.0v8\\Nuke16.0.exe" -t scripts\\nuke_arriraw_to_aces_exr.py <src> <outdir> <name> off=0 n=100 [width=4096] [squeeze=1.00]

`src` is an ARRIRAW .mxf or any one .ari frame of a sequence. The ARRI SDK
debayers straight to ACES AP0 linear, OCIO (ACES 1.3 cg config) writes
ACEScg, half float, Zip1. `off`/`n` pick the frames, `width` rescales
(ALEXA 65 open gate is 6.5K). Arguments are key=value because Nuke -t
treats bare numbers as frame ranges.

Note: Nuke writes no chromaticities and carries the camera's
`ColorGammaSxS = LOGC_WIDEGAMUT` tag into the EXR, which is now false - a
useful worst case for colourspace detection.
"""

import os
import sys

import nuke

src, outdir, name = sys.argv[1], sys.argv[2], sys.argv[3]
kv = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
off, n = int(kv.get("off", 0)), int(kv.get("n", 100))
width = int(kv.get("width", 0))
squeeze = kv.get("squeeze", "1.00")
rt = nuke.root()
rt["colorManagement"].setValue("OCIO")
rt["OCIO_config"].setValue("fn-nuke_cg-config-v1.0.0_aces-v1.3_ocio-v2.1")
if src.lower().endswith(".ari"):
    import glob
    import re

    fs = sorted(glob.glob(os.path.join(os.path.dirname(src), "*.ari")))
    m = re.match(r"(.*\.)(\d+)\.ari$", fs[0].replace("\\", "/"))
    pad = len(m.group(2))
    nums = [int(re.search(r"\.(\d+)\.ari$", f).group(1)) for f in fs]
    src = f"{m.group(1)}{'#' * pad}.ari {min(nums)}-{max(nums)}"
r = nuke.createNode("Read")
r["file"].fromUserText(src)
r["arri_colorspace"].setValue("ACES AP0 / ~D60 / linear")
if squeeze != "1.00":
    r["arri_lens_squeeze"].setValue(squeeze)
r["colorspace"].setValue("ACES2065-1")
first = int(r["first"].value()) + off
last = min(first + n - 1, int(r["last"].value()))
node = r
if width:
    rf = nuke.createNode("Reformat")
    rf["type"].setValue("scale")
    rf["scale"].setValue(width / float(r.width()))
    rf["filter"].setValue("Lanczos4")
    rf.setInput(0, r)
    node = rf
os.makedirs(outdir, exist_ok=True)
w = nuke.createNode("Write")
w.setInput(0, node)
w["file"].setValue(f"{outdir}/{name}.####.exr".replace("\\", "/"))
w["file_type"].setValue("exr")
w["datatype"].setValue("16 bit half")
w["compression"].setValue("Zip (1 scanline)")
w["colorspace"].setValue("ACEScg")
w["metadata"].setValue("all metadata")
print("RANGE", first, last, r.width(), r.height())
nuke.execute(w, first, last)
