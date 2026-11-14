# XMP sidecars

`write_xmp: true` (`--xmp`) writes an XMP sidecar beside each exported file,
`photo.jpg.xmp`, through argus-cortex's shared writer, the one argus-lens uses:

- `mwg-rs:Regions`: every detected face, with the name given for its cluster.
- `Iptc4xmpExt:PersonInImage`: the names, from `face_names`
  (`--face-names face_1=Ada,face_3=Ada`; two clusters can be one person). An
  unnamed cluster gets an unnamed region and no person entry.
- `argus:ContentRating`: the NudeNet [content rating](nudity.md#content-rating).

The source's own sidecar (lens's captions, a photo app's stars and edits) is
the starting point and everything curator does not set is kept; the source's
sidecar itself is never changed. Age estimates are never written: a MiVOLO
guess in a photo library reads as a fact. A file curator knows nothing about
gets no sidecar; `xmp_written` counts the ones written. A transfer can succeed
while a sidecar fails (`copied` greater than `xmp_written`); the image still
landed, and the failure is logged rather than undoing it.

## Writing beside the originals

`--xmp-sources` (CLI only) merges the same fields into the sidecars beside
**every scanned original** curator has something for — including rejected
images — independent of `--xmp` / `--copy-to`. That writes into the scanned
folder; the server's write paths stay under the export root. Regions are
replaced as a whole, so regions drawn by hand on an image with detected faces
are lost.
