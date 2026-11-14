# Crops: watermarks and borders, without touching the source

A watermark or a border band is something every check reads and a trainer
learns. With crops on, a boxed image is cropped once, in memory, right after
it is read, so scoring, faces, NudeNet, EraX, scenes, garments and SingGuard
all judge the cropped frame, and export writes it cropped. The source file is
only ever read.

## Boxes

A box is `{"left", "top", "right", "bottom"}` in pixels of the image **as
displayed** (after its EXIF orientation; `right`/`bottom` exclusive), which is
what an editor or a browser shows when the crop is drawn. It comes from a
sidecar beside the source, `photo.jpg.crop.json`, or inline on the request,
keyed by `rel_path`; an inline box wins:

```jsonc
"crops": { "enabled": true, "sidecars": true,
           "boxes": { "set1/IMG_0001.jpg": { "left": 0, "top": 0, "right": 3000, "bottom": 3800 } } }
```

An image with no box is scanned as it is. The cropped image is re-encoded in
its own format (JPEG and lossy WebP at quality 95, keeping the colour profile
and the EXIF less its orientation), upright. `ImageResult.crop` records the
box; `width`, `height` and every face and person box are of the cropped frame.
A box that leaves the frame, is empty, or a sidecar that cannot be read makes
the image a reject (`crop:invalid`, `crop:unreadable`, `crop:failed`) that is
never kept or exported, as with the prefilter: it would otherwise ship with
the watermark. Like any reject, it skips NudeNet, EraX and SingGuard.

## Export

Export writes each cropped image cropped, so a scan that cropped anything
exports only by `mode: "copy"` (HTTP 400 otherwise; a link or a move would
hand on the source). `ExportResult.cropped` counts them and each manifest row
carries its `crop` ([manifest](manifest.md) 2.2). The `.crop.json` sidecar is not carried.
With `carry_sidecars` and without `write_xmp`, a carried `.xmp` keeps face
regions drawn on the uncropped frame; with `write_xmp` they are replaced by
the cropped frame's. `--xmp-sources` writes no regions for a cropped image, as
its sidecar describes the uncropped original.

`/thumb?scan_id=` shows the cropped frame, which the row's boxes fit. `/image`
stays the original file: offset boxes by `crop.left` and `crop.top` to draw on it.

## Finding watermarks

With `"detect": true` the boxes are found rather than given. Every image
without a box by hand goes to a watermark detector, `ARGUS_CURATOR_WATERMARK_URL`:
a server speaking NudeNet's `POST /infer` shape, as the EraX sidecar does
(argus-cull's `watermark` service runs fancyfeast's YOLO11x, the detector
behind JoyCaption's watermark tags). Curator sends each frame upright, so the
boxes come back in the pixels it crops by, and never imports Ultralytics.
Configure it in the environment, like the NudeNet server (`<LAN_HOST>` is a
placeholder):

```bash
ARGUS_CURATOR_WATERMARK_URL=http://<LAN_HOST>:8080   # its origin; /infer is appended
ARGUS_CURATOR_WATERMARK_TIMEOUT=60                     # seconds per image (default 60)
ARGUS_CURATOR_WATERMARK_WORKERS=2                      # concurrent requests (default 2)
```

Each hit of at least `min_conf` (0.4) is cut away along one of the four
edges, plus `margin` pixels (4). Every way of assigning the hits to edges is
tried, and the one that keeps the most of the frame wins, provided no edge
loses more than `max_cut` of its side (0.15). A hit that no such cut removes,
or more than six hits (a tiled mark), is an overlay: with `reject_overlay`
(the default) the image is a `crop:overlay` reject, else it is kept uncropped.
An image the detector could not check is a `crop:unchecked` reject, so a
detector that is down never lets a watermark through. `ImageResult.watermarks`
lists the hits on the uncropped frame (empty: none found), and `crop_source`
says whether a crop was `manual` or `detected`. A streaming scan reports the
pass as phase `watermarks`, before scoring. Detection runs only when a scan
asks for it; asking without a detector is HTTP 400, and asking while it is not
answering (`GET /health` fails or returns 5xx, as while it loads) is HTTP 503,
so the scan never starts. A server with no `/health` route still counts as up.
`/detectors` reports `watermark`, `/health` reports `watermark_detect`.

```jsonc
"crops": { "enabled": true, "detect": true, "max_cut": 0.15, "reject_overlay": true }
```

The summary records the setting as `crop_config`, and `/health` reports
`crops`. On the CLI, `--crops` reads the sidecars, and `--detect-watermarks`
(with `--watermark-url`, `--max-cut`, `--keep-overlays`) finds them:

```bash
# Same folder with and without crops: two scans to compare verdicts
argus-curator scan /data/inbox --detect-watermarks --watermark-url http://watermark:8080 --json cropped.json
argus-curator scan /data/inbox --json uncropped.json
```
