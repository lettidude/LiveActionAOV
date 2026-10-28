"""Contact sheet of ACES2065-1 EXR frames, to scout shots in a long sequence.

    .venv\\Scripts\\python.exe scripts\\aces_contact_sheet.py <dir> <out.jpg> [cols=6] [w=420]

AP0 -> Rec.709, a Reinhard curve and sRGB: just to see the frames.
"""

import glob
import os
import sys

import cv2
import numpy as np
import OpenImageIO as oiio

d, out = sys.argv[1], sys.argv[2]
kv = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
cols, W = int(kv.get("cols", 6)), int(kv.get("w", 420))
# AP0 -> linear Rec.709 (Bradford D60->D65)
M = np.array(
    [[2.52169, -1.13413, -0.38756], [-0.27648, 1.37272, -0.09624], [-0.01538, -0.15298, 1.16835]],
    np.float32,
)
tiles = []
for p in sorted(glob.glob(d + "/*.exr")):
    i = oiio.ImageInput.open(p)
    px = i.read_image(0, 0, 0, 3, "float")
    i.close()
    s = 4
    rgb = np.maximum(px[::s, ::s] @ M.T, 0)
    rgb = rgb / (1 + rgb)  # Reinhard: just to see the frame
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    h = int(W * rgb.shape[0] / rgb.shape[1])
    t = cv2.resize(
        (np.clip(rgb, 0, 1) * 255).astype(np.uint8), (W, h), interpolation=cv2.INTER_AREA
    )
    cv2.putText(t, os.path.basename(p)[12:-4], (6, 22), 0, 0.6, (255, 230, 60), 2)
    tiles.append(t)
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k : k + cols]) for k in range(0, len(tiles), cols)]
cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out)
