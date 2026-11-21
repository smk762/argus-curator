# Face sources (argus-mask)

With faces on, every face also carries its head **roll** (beside yaw and
pitch), its own **sharpness** (Laplacian variance over the face box, so a
sharp face on a plain background isn't judged by the background), **eye_open**
and **mouth_open** (the eye aspect ratio and the inner-lip gap over the mouth
width, from InsightFace's 68 landmarks), and **cluster_distance**, the cosine
distance from its identity's centroid. Name a reference and each face also
gets a **reference_distance** from it:

```bash
argus-curator scan views/ --mivolo --face-reference portrait.png \
    --face-source --blur-threshold 0 --csv views.csv --copy-to source/
```

## The `--face-source` preset

`--face-source` (`face_source: {}` on `POST /export` and `/scan/{id}/select`)
is a preset over those measurements:

- one face, of one identity;
- yaw and pitch within 15°, and roll within 10°;
- a face box at least 256 px on its short side;
- face sharpness at least 100;
- eyes open (`eye_open >= 0.18`) and mouth closed (`mouth_open <= 0.1`);
- an identity distance within 0.4: to the reference when there is one, else
  to the identity's centroid;
- a MiVOLO age of 18 or over (with an adult set age when the scan read one).

Which identity it keeps:

- `face_source.cluster`, when given;
- else, when the scan has a reference (any face has a `reference_distance`),
  no identity is picked: the reference distance decides which faces are the
  reference's identity;
- else the largest identity among the passing images.

Set any check to `null` to turn it off. An image that fails gets
`face-source:<check>` as its keep reason, and the kept images come back ranked
best first (by `rank`: score by default, or `"identity"` for the closest face
first; see [Ranking, max_keep and splits](http-api.md#ranking-max_keep-and-splits)). A scan without face detection, or (for the age check) without
MiVOLO, is refused with a 400 rather than keeping nothing. The measurements are
not an expression classifier: a closed-lip smile passes.

## References

A reference is one of two things (not both):

- **Images in the scan**: `faces.reference_paths` (rel paths), or
  `--face-reference` with a path relative to the scanned folder. Their primary
  faces' embeddings are averaged. A reference the scan's own filters reject (a
  studio portrait on a flat background often fails the blur check) is still
  embedded, from the bytes the scan read, and its row stays rejected.
- **A centroid you supply**: `faces.reference_embedding`, a list of floats in
  the face model's embedding space (512 for `buffalo_l`). Build one in Python
  from images outside the set, such as a consenting person's reference photos:

  ```python
  from argus_curator.faces import reference_distance, reference_embedding
  from argus_curator.models import FaceConfig, ScanRequest

  centroid = reference_embedding([p.read_bytes() for p in consent_photos])
  req = ScanRequest(folder="set", faces=FaceConfig(mivolo=True, reference_embedding=centroid))
  # or compare one embedding yourself: reference_distance(embedding, centroid)
  ```

  `--face-reference` with a file outside the scanned folder does the same from
  the CLI. The centroid is a biometric template: it is used for the scan and
  never stored. The scan summary drops it, so neither the scan cache nor
  `GET /scan/{id}` holds it, and a rescan has to supply it again.

`reference_skipped` on the summary says why a reference gave no distances (no
such image in the scan, none with a face, or an embedding of the wrong length).
The face-source preset refuses such a scan rather than fall back to the
cluster distance.

Embeddings are not stored, only the distances. The 5 keypoints and 68
landmarks are kept in the scan cache (not served by `GET /scan/{id}`) and
written into the manifest when an export asks (`--landmarks`,
`landmarks: true`; see [Manifest](manifest.md)).
[Synthetic source portraits](synthetic-face-sets.md) walks through a
face-source shortlist end to end.
