# Sidecars and run stats

`carry_sidecars: true` (`--carry-sidecars`) transfers the sidecars beside each
exported source image with it, in the same mode (copy or move), so a moved
keep set does not leave its captions orphaned in the source folder. Under
symlink the sidecars are *copied*, not linked: argus-lens and other tools write
captions beside the exported image, and through a link that write would land in
your source library. Each of these that exists is carried, named after the *destination* file (so a
flattened, de-collided `img-1a2b3c4d.png` gets `img-1a2b3c4d.txt`):

| Beside the source | Written by | Beside the destination |
|---|---|---|
| `photo.txt` | argus-lens captions | `<dest stem>.txt` |
| `photo.jpg.xmp` | argus-lens, digiKam | `<dest name>.xmp` |
| `photo.xmp` | Lightroom | `<dest stem>.xmp` |
| `photo.json` | | `<dest stem>.json` |

Sidecars are carried only for images whose transfer succeeded, and an existing
destination sidecar is replaced, as an existing destination image is. A
sidecar that fails to transfer does not undo its image: it is logged and
counted in `ExportResult.sidecars_failed`; `sidecars_carried` counts the ones
that landed. Two images that share a stem in one folder (`a.jpg`, `a.png`)
share `a.txt`, and both get it in every mode. Under move it is copied to each
image but the last, which takes the original; if any of them fails to transfer,
the original stays in the source folder. The export's own files keep their
names: with sidecars carried, an image at the destination root with the stem
`run_stats`, `manifest` or `curation_report` gets a hash suffix, even with
`preserve_structure`, so its `.json` cannot be overwritten by `run_stats.json`.

## With XMP

With `write_xmp` as well, sidecars are carried first and curator's data is then
merged ([XMP sidecars](xmp.md)) onto the carried `photo.jpg.xmp` (or, when only a Lightroom `photo.xmp`
was carried, onto a `photo.jpg.xmp` seeded from it). Under move that leaves one
merged sidecar at the destination and nothing behind in the source folder.
Under symlink the merge lands on the carried copy, so the source's sidecar is
never changed. argus-forge reads captions from beside the
manifest's `abs_path`, which under move is the destination, so a moved export
with `carry_sidecars` keeps its captions in reach.

## Run stats

`run_stats` (`--run-stats FILE` on the CLI) is opaque JSON from the client, for
example the cull session that produced the keep set. It is written
pretty-printed to `<dest>/run_stats.json`, beside the manifest, whenever the
export produced a destination folder: at least one file was transferred, or
`write_manifest` is on. `ExportResult.run_stats_path` names the file, or is
null when nothing was written. The serialised JSON is capped at 5 MB; a larger
body is refused with 422.

## Move exports and the scan cache

A server `mode: "move"` export also drops the moved images from the cached
scan, as `POST /scan/{id}/delete` does: their sources are gone, so the rows
would only point thumbnails at missing files. Only rows whose transfer
succeeded are dropped, and near-duplicate groups and identities are
recounted. An export that fails after files moved still drops their rows, and
its error says how many left the source. A move export of the scan that is
still running is refused with 409, since that scan's next save would put the rows back. `GET /health` says
`export_sidecars: true` on a server with all of this.
