# Standalone moderation and dataset auditing

Everything curator records about an image is usable without the rest of the
suite. A scan never modifies the source folder; it produces a per-image record
(`--json`, `--csv`, or `GET /scan/{scan_id}`), and an export applies filters
to it, giving every excluded image a `keep_reason`. That record doubles as an
audit trail: you can show which images were removed and on what grounds.

| Check | What is recorded | Filter on it |
|---|---|---|
| Technical quality | sharpness, resolution, aspect ratio, compression artifacts | `--min-score`, hard-filter `reject_reason` |
| Near-duplicates | pHash groups, best representative kept | `--keep-similar` to keep the rest |
| People | face count, identity clusters, head pose | `--face-clusters`, `--pose` |
| Apparent age | MiVOLO age in years and its bracket, per face (needs `mivolo-convert`) | `--min-age` / `--max-age`, `--age` |
| Apparent gender | MiVOLO gender with its probability, pooled per identity | `--gender`, `--min-attr-conf` |
| Setting | Places365 scene top-k, indoor/outdoor | `--scene`, `--environment` |
| Clothing | Fashionpedia garments: type, colour, cut off at the frame edge | `--garment`, `--exclude-garment`, `--uncropped` |
| Nudity | NudeNet detections per body-part class, exposed score (needs a [NudeNet server](nudity.md)) | `--nsfw-threshold`, `--nsfw-label` |
| Policy | SingGuard verdict and the rules that fired: sexual content, crime, gore and self-harm, … (needs a [SingGuard server](policy-check.md)) | `--guard-unsafe`, `--guard-rules` |

A typical moderation pass is scan, review, act:

```bash
# 1. Scan and report only: nothing is copied, moved or deleted.
argus-curator scan /data/inbox --mivolo --scenes --garments --csv report.csv --json scan.json

# 2. Review report.csv: one row per image with every tag and its confidence.

# 3. Keep only images whose primary face reads as clearly adult. --min-score 0
#    and --keep-similar stop score and duplicates from removing anything, so
#    among the images that passed the hard filters age is the only criterion.
#    An image that failed them is never kept, and shows its own reject reason.
#    Images where no age could be estimated (no face found) are excluded as
#    `face-attributes-missing`: the filter fails closed rather than letting
#    unknowns through.
argus-curator scan /data/inbox --mivolo --min-age 25 \
    --min-score 0 --keep-similar \
    --copy-to /data/approved --csv decisions.csv
```

`decisions.csv` then lists every image with `keep` and its `keep_reason`
(`face-age-out-of-range`, `face-attributes-missing`, `garment-excluded`, …),
which is the list to send for human review or to quarantine. The same scans and
filters are available over HTTP (`POST /scan/folder`, `GET /scan/{scan_id}`,
`POST /export`; see [HTTP API](http-api.md)), so curator can sit in front of an
upload or moderation queue.

## Limits: read before relying on it for moderation

- **Content checks need their servers, and both miss things.** NudeNet
  detects exposed body parts only, and misses stylised, partial or unusual
  poses; the optional EraX sidecar is a second pass on those misses.
  SingGuard judges a whole policy (violence, gore, self-harm, …) but
  gives a verdict without a confidence, so nothing is "near the threshold" to
  review. Set NSFW thresholds conservatively and have a person review flagged
  and borderline images.
- **Age and gender are model estimates from appearance**, not facts
  about anyone. Age estimates are typically off by a few years, more so from
  the face alone (the default). `--mivolo-body` adds each face's body from
  InsightFace's person detector; a person with no face still gets a body-only
  estimate. Filter with a margin above any legal threshold (`--min-age 25`
  rather than 18) and send borderline images to a person. See
  [Face attributes](face-attributes.md).
- Only the **primary** face of each image is filtered on; other faces are
  recorded per face (`faces[].attributes`) but do not exclude the image.
- Scene and garment models make mistakes too; see the measured accuracy in
  [Scene tags](scene-tags.md) and [Garments](garments.md).

Curator is a filter and an audit trail to put in front of human review, not a
replacement for it.

## Related pages

- [Nudity detection (NudeNet)](nudity.md)
- [Policy check (SingGuard)](policy-check.md)
- [Face attributes (MiVOLO, FairFace)](face-attributes.md)
- [Prefilter](prefilter.md): drop images before the slow checks
