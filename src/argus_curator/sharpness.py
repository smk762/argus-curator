"""How sharp an image is, for whole images (``scanner``) and face boxes (``faces``)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Sharpness is measured with the long side at this many pixels. The 3x3 edge
# kernel reads per-pixel contrast, which falls as resolution rises (the same
# edge spans more pixels), so measured at native size a sharp 4800px studio
# frame scores about a quarter of its 2048px copy and well under the default
# blur_threshold of 100. Fixing the size gives blur_threshold and
# sharpness_ref one meaning for every file size. Smaller images are measured
# as they are.
SHARPNESS_SIDE = 1024


def sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian edge response — high = sharp, low = blurry.

    Measured with the long side scaled to :data:`SHARPNESS_SIDE` so the value
    does not depend on the file's resolution, over the interior only (the 1px
    frame's edge response is an artefact of the filter), so a flat image of
    any brightness scores 0.
    """
    gray = img.convert("L")
    w, h = gray.size
    if max(w, h) > SHARPNESS_SIDE:
        scale = SHARPNESS_SIDE / max(w, h)
        gray = gray.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    # PIL's 3x3 filter answers the 1px frame with a response that grows with
    # the image's brightness, so on a low-detail picture the border dominated
    # the variance and a bright blurred (even flat) frame passed as sharp
    # (#147). Only the interior is measured; an image under 3px has none, so
    # it has no measurable edges either.
    if edges.shape[0] < 3 or edges.shape[1] < 3:
        return 0.0
    return float(edges[1:-1, 1:-1].var())
