# Face sources (argus-mask)

With faces on, every face also carries its head **roll** (beside yaw and
pitch), its own **sharpness** (Laplacian variance over the face box, so a
sharp face on a plain background isn't judged by the background), **eye_open**
and **mouth_open** (the eye aspect ratio and the inner-lip gap over the mouth
width, from InsightFace's 68 landmarks), and **cluster_distance**, the cosine
distance from its identity's centroid. Name reference images in the scan and
each face also gets a **reference_distance** from them:

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
best first. A scan without face detection, or (for the age check) without
MiVOLO, is refused with a 400 rather than keeping nothing. The measurements are
not an expression classifier: a closed-lip smile passes.

## References

A reference must pass the scan's filters, as a rejected image is never
face-checked; `reference_skipped` on the summary says why none was used. A
plain studio portrait often fails the default blur check, so pass
`--blur-threshold 0` (as above) when the reference has a flat background.

Embeddings are not stored, only the distances. The 5 keypoints and 68
landmarks are kept in the scan cache (not served by `GET /scan/{id}`) and
written into the manifest when an export asks (`--landmarks`,
`landmarks: true`; see [Manifest](manifest.md)).
[Synthetic source portraits](synthetic-face-sets.md) walks through a
face-source shortlist end to end.
