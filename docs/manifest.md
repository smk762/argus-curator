# Manifest reference

An export writes `manifest.jsonl` into its destination folder: one JSON object
per line, one line per **exported** image. It is the handoff contract the rest
of the suite reads: argus-lens batch-captions it (an upload to its
`/caption/manifest`, or a path under its `export_root`), and argus-forge
builds training sets from it. Curator and lens share one `TargetProfile`, so a
manifest written here is captioned downstream with no remapping.

This page describes manifest **2.8**, the version current releases write, and
what changed in each earlier version. A speech export writes no `manifest.jsonl`:
its clips go in `speech_manifest.jsonl`, a row shape of its own
(`SpeechManifestRow`) with its own version (`SPEECH_MANIFEST_VERSION`, 1.1),
described in [Speech scans](speech.md#the-speech-manifest).

## Where it is written

- `POST /export` and `POST /export/stream` write it when `write_manifest` is
  true (the default), as `<dest>/manifest.jsonl`. `ExportResult.manifest_path`
  names the file, or is null when none was written.
- `argus-curator scan ... --copy-to DIR` (or `--move-to`, `--symlink-to`)
  writes it as `DIR/manifest.jsonl`. A scan with no destination writes none.
- With `caption_url` set on the export request, curator also POSTs the file to
  that URL as a multipart upload (field `manifest`, type
  `application/x-ndjson`), for a one-click curate-then-caption run. The
  response's `captioned` says whether that POST succeeded. An internal host
  must be listed in `CURATOR_CAPTION_ALLOWED_HOSTS` (see the
  [HTTP API](http-api.md)).

The file is UTF-8, one row per line, each row a complete JSON object. Rows
exist only for files whose transfer succeeded: the manifest is a work list,
and never claims a file that is not on disk. Re-exporting into the same
destination overwrites the files and rewrites the manifest; each export plans
in isolation.

## Example row

Pretty-printed here; in the file each row is one line.

```json
{
  "manifest_version": "2.8",
  "rel_path": "shoots/2026-07/IMG_0412.jpg",
  "abs_path": "/data/images/shoots/2026-07/IMG_0412.jpg",
  "exported_path": "shoots/2026-07/IMG_0412.jpg",
  "exported_abs_path": "/data/out/training-set-v1/shoots/2026-07/IMG_0412.jpg",
  "target_profile": {
    "modality": "image",
    "target_style": "photo",
    "target_backend": "sdxl",
    "checkpoint": null,
    "target_category": "identity"
  },
  "primary_face_cluster": "face_2",
  "primary_face_pose": "three_quarter",
  "score": 0.8712,
  "similar_group": 3,
  "crop": { "left": 0, "top": 0, "right": 3000, "bottom": 3800 },
  "primary_face": {
    "bbox": [412.0, 188.0, 301.5, 380.0],
    "yaw": -24.1,
    "pitch": 3.2,
    "roll": -1.8,
    "sharpness": 812.4,
    "eye_open": 0.29,
    "mouth_open": 0.02,
    "cluster_distance": 0.12,
    "reference_distance": null,
    "age_years": 31.4,
    "landmarks": null
  },
  "set_age": { "years": 31.0, "cluster_id": "face_2", "faces": 42, "share": 0.95 },
  "split": null,
  "consent_release_sha256": null,
  "consent_record_sha256": null,
  "provenance": null,
  "pose_labels": { "stance": "standing", "arms": "down", "facing": "front" },
  "pose_cluster": 0
}
```

## Row fields (2.8)

Every key is always present in a 2.8 row; a field with no value is written as
`null`, not left out.

| Field | Type | Since | Meaning |
|---|---|---|---|
| `manifest_version` | string | 1.0 | The manifest version this row was written as, `"<major>.<minor>"`. See [Compatibility](#compatibility). |
| `rel_path` | string | 1.0 | The image's path relative to the scanned folder (posix). Its original location within the scan. |
| `abs_path` | string | 1.0 | Where the image can be **read**. Under `copy` and `symlink` that is the source; under `move` the source is gone, so it equals `exported_abs_path` (since 2.1). |
| `exported_path` | string | 2.0 | The path actually written under the export root (posix, relative to it). Use it rather than deriving a location from `rel_path`; see [Paths](#paths). |
| `exported_abs_path` | string | 2.1 | The exported file's absolute location. |
| `target_profile` | object | 1.0 | The scan's [`TargetProfile`](#target_profile), the same on every row. |
| `primary_face_cluster` | string or null | 1.0 | The identity cluster of the image's primary face (`face_1`, `face_2`, …), or null without one. |
| `primary_face_pose` | string or null | 1.0 | The primary face's head-pose bucket from its absolute yaw: `frontal` (up to 15°), `three_quarter` (up to 45°) or `profile`. Null without a face. |
| `score` | number | 1.0 | The composite training-suitability score, 0 to 1, rounded to 4 places. See [How scoring works](scoring.md). |
| `similar_group` | integer | 1.0 | The image's near-duplicate group. Near-duplicates share a group id; an image with none has a group of its own. |
| `crop` | object or null | 2.2 | The box baked into the exported file, or null when the file is the source as it was. See [`crop`](#crop). |
| `primary_face` | object or null | 2.3 | The primary face's measurements, or null for an image with no face. See [`primary_face`](#primary_face). |
| `set_age` | object or null | 2.3 | The scan's set age, when the scan read one. See [`set_age`](#set_age). |
| `split` | string or null | 2.4 | `train` or `held_out` when the export asked for a train/held-out split (`split`, exporting `split_part`), else null. See the [HTTP API](http-api.md#ranking-max_keep-and-splits). |
| `consent_release_sha256` | string or null | 2.6 | A consented face set only (`face_source.consent`): the consent record's key, the sha256 of the signed release. Null otherwise. See [Consented face sets](face-sources.md#consented-face-sets-of-a-real-person). |
| `consent_record_sha256` | string or null | 2.6 | A consented face set only: the sha256 of the consent record file as read at export. Null otherwise. |
| `provenance` | object or null | 2.6 | A consented face set only: where the image came from, as argus-cortex's `Provenance` (`kind`, `source`, `license`, `subject`, `release_sha256`, `cleared`…; since 2.8 also the optional `lookalike` and the derived `demo_ready` of argus-cortex 0.16). It is the `provenance` in the image's `<stem>.json` sidecar when there is one, else `kind: "consented"` under the record's release. Either way `subject` is cortex's subject ref `consent:<release_sha256>`, never the person's name (cortex's `consent_matches` accepts it). Null otherwise. |
| `pose_labels` | object or null | 2.7 | The primary person's body-pose labels from argus-puppet: `stance` (`standing`, `jumping`, `crouching`, `seated`, `lying`, `unknown`), `arms` (`up`, `out`, `down`, `mixed`, `unknown`) and `facing` (`front`, `back`, `left`, `right`, `unknown`; the subject's, as the camera sees it). `unknown` is puppet's own reading (the joints its rule needs were not confident). Null when the pose stage did not run, found nobody, or failed. See [Body pose](body-pose.md). |
| `pose_cluster` | integer or null | 2.7 | The image's scan-level body-pose cluster, 0 for the largest. Clusters are formed over the **whole scan**, not the export. Null when the pose stage did not run, found nobody, failed, the pose was unusable, or the scan was not clustered. See [Body pose](body-pose.md). |

### `target_profile`

| Field | Type | Values |
|---|---|---|
| `modality` | string | always `image` (2.5+; absent before, read as `image`) |
| `target_style` | string | `photo` or `anime` |
| `target_backend` | string or null | the diffusion backend family, e.g. `sdxl`, `flux-dev-1` |
| `checkpoint` | string or null | the target checkpoint, when given |
| `target_category` | string | `identity`, `wardrobe`, `pose_composition`, `setting` or `concept` |

The style and category values are argus-cortex's shared taxonomy. Curator
curates images, so its profile is cortex's narrowed to the image modality: a
speech profile (`modality: "speech"`) is refused at the request, and a
manifest never carries a speech category.

### `crop`

`{"left", "top", "right", "bottom"}`, integers, in pixels of the **source** as
displayed (after its EXIF orientation), with `right` and `bottom` exclusive.
The exported file is that region of the source, written upright. A scan that
cropped anything exports only by copy. See [Crops](crops.md).

### `primary_face`

Measured on the exported file's frame: when the image was cropped, boxes and
landmarks are in the cropped image's pixels.

| Field | Type | Meaning |
|---|---|---|
| `bbox` | number[4] | The face box, `[x, y, w, h]` in pixels. |
| `yaw`, `pitch`, `roll` | number or null | Head pose in degrees, from InsightFace. |
| `sharpness` | number or null | Laplacian variance over the face box (the image-level blur check measures the whole frame). |
| `eye_open` | number or null | The eye aspect ratio from the 68 landmarks; the face-source preset counts 0.18 and over as open. |
| `mouth_open` | number or null | The inner-lip gap over the mouth width; the preset counts 0.1 and under as closed. |
| `cluster_distance` | number or null | Cosine distance from the face's embedding to its identity's centroid. |
| `reference_distance` | number or null | Cosine distance to the reference identity, when the scan had one: the centroid of the reference images' primary faces (`--face-reference`, `faces.reference_paths`), or a centroid the caller supplied (`faces.reference_embedding`). |
| `age_years` | number or null | MiVOLO's apparent age in years, when MiVOLO ran. A model's estimate, not a fact. |
| `landmarks` | object or null | Null unless the export asked (`--landmarks`, `"landmarks": true`). Then `kps`, the detector's 5 keypoints (eyes, nose tip, mouth corners) as `[x, y]`, and `landmark_3d_68`, the 68 iBUG points as `[x, y, z]`; either can be null. |

See [Face sources](face-sources.md) for how these are measured and used.

### `set_age`

An age read off the set for frames with no face (`--set-age`). It is the
scan's, so when present it is the same object on every row, faces or not.

| Field | Type | Meaning |
|---|---|---|
| `years` | number | The median MiVOLO age of the scan's dominant identity. |
| `cluster_id` | string | That identity. Read before the prefilter, so it can name an identity no longer in `face_clusters`. |
| `faces` | integer | How many ages the median is read from. |
| `share` | number | The fraction of the scan's faces in that identity. |

It is an inference about the set, not a reading of any one frame. See
[An age for the frames with no face](face-attributes.md#an-age-for-the-frames-with-no-face).

## Paths

`exported_path` is the path actually written under the export root. Consumers
must use it rather than re-deriving a location from `rel_path`, because the
two can differ:

- Flattened exports (`preserve_structure: false`) de-collide duplicate
  basenames with a short hash suffix (`stem-<hash>.ext`). Collisions are
  detected case-insensitively and Unicode-normalised, so the result is safe on
  case-insensitive filesystems, and the export fails rather than overwrite if
  a unique name cannot be generated.
- With `carry_sidecars`, an image at the destination root whose stem is
  `manifest`, `run_stats` or `curation_report` gets the same suffix in any
  export, so its sidecars cannot clash with the export's own files.

Prefer `exported_abs_path`: joining the export root onto `exported_path` means
rebuilding the server's layout rule somewhere that does not own it, and the
de-collision suffix cannot be reproduced by a client. A manifest is also often
read somewhere other than the folder it was written into (argus-lens receives
it as an upload), and no row names that folder.

The export response carries the same mappings as `exported_paths` and
`exported_abs_paths` (keyed by `rel_path`), so they are available even with
`write_manifest: false`.

## Versions

| Version | First release | Change |
|---|---|---|
| 1.0 | 0.1.0 | Initial rows: `manifest_version`, `rel_path`, `abs_path`, `target_profile`, `primary_face_cluster`, `primary_face_pose`, `score`, `similar_group`. A row per **selected** image, including any whose transfer failed. |
| 2.0 | 0.2.0 | **Breaking.** Rows only for files whose transfer succeeded, each with `exported_path`. The row shape is published in the wire schema as `ManifestRow`. |
| 2.1 | 0.2.1 | Adds `exported_abs_path`. Fixes `abs_path` under `move`, which named the deleted source; it is now the destination. |
| 2.2 | 0.14.0 | Adds `crop`. |
| 2.3 | 0.15.0 | Adds `primary_face` and `set_age`. |
| 2.4 | 0.19.0 | Adds `split`. |
| 2.5 | 0.19.0 | Adds `target_profile.modality`, always `image` (argus-cortex 0.12). |
| 2.6 | 0.19.0 | Adds `consent_release_sha256`, `consent_record_sha256` and `provenance` (consented face sets, #133); `provenance.subject` is the subject ref `consent:<release_sha256>`, never a name. |
| 2.7 | 0.20.0 | Adds `pose_labels` and `pose_cluster` (argus-puppet's body pose). |
| 2.8 | 0.24.0 | `provenance` is argus-cortex 0.16's `Provenance`: it gains the optional `lookalike` (`who`, `at`, `how`, `found`) and the derived `demo_ready`. No field of curator's own changed. A reader that validates `provenance` with cortex's model needs argus-cortex 0.16 or later, since earlier releases refuse the unknown `demo_ready`. |

2.0 is a new major because a 1.x consumer that derives destinations from
`rel_path`, or trusts every row to name a file on disk, misreads a 2.x manifest.

## Compatibility

- A **major** bump (1 to 2) breaks the row's shape or meaning. A **minor**
  bump only adds fields or fixes behaviour to match what was documented.
- A consumer reads `manifest_version` and **refuses a major it does not
  know**, rather than misreading the rows. Within a known major it accepts any
  minor, ignores fields it does not know, and treats a field missing from an
  older row as null.
- A consumer that needs a field added in a minor version checks that minor
  too (for example, `primary_face` needs 2.3).
- To check before exporting, read `manifest_version` from `GET /health` (it
  sits beside the package `version`; the two move independently) or from any
  `ExportResult`, which declares it whether or not a manifest was written.

argus-forge, for example, accepts majors in its `SUPPORTED_MANIFEST_MAJORS`
and requires `exported_path` from 2.0 on.

## Schema

The row is published as `ManifestRow` (with `ManifestFace`, `FaceLandmarks`,
`CropBox`, `SetAge`, `TargetProfile` and `Provenance`) in the wire-contract JSON Schema,
[`schema/curator-wire.schema.json`](../schema/curator-wire.schema.json).
`argus-curator schema` regenerates it from the models, and CI fails if the
committed copy is stale. In Python, `argus_curator.models.ManifestRow` reads a
row:

```python
from argus_curator.models import ManifestRow

with open("manifest.jsonl", encoding="utf-8") as f:
    rows = [ManifestRow.model_validate_json(line) for line in f if line.strip()]
```

`MANIFEST_VERSION` in `argus_curator.models` is the version this package
writes.
