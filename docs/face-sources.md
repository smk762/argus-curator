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

## Refused sources

Every face-source selection reads each candidate's `<stem>.json` sidecar again
when it runs, and keeps out an image whose sidecar says
`"likeness_consent": false` (`face-source:not-consented`). argus-quarry writes
that mark on every photo of a real person: a licence to a photo is not its
subject's consent to become a face.

## Consented face sets of a real person

The docs lead with synthetic faces, but some routes need a real, consenting
person (the multi-view captures of the Gaussian-splat route, say). Curator
makes sure every image in such a set is of the person who consented, adult,
sourced and current. It does not collect consent and gives no legal advice.

The consent is argus-mask's record, argus-cortex's `ConsentRecord`, in a file
beside the images (`consent.json` at the folder root, say). It binds the
signed release by its sha256 and lists the sha256 of each reference image of
the person; the signed document itself stays outside curator.

```bash
argus-curator scan shoot/ --consent consent.json --face-source-consent \
    --blur-threshold 0 --copy-to faceset/
```

Over HTTP, scan with `"consent": {"record": "consent.json"}` and export (or
select) with `"face_source": {"consent": true}`.

**The scan reads the record** and checks it with cortex's
`consent_valid(record, use="avatar", likeness="face", at=now)`. A missing,
invalid, revoked, not-yet-signed, expired or out-of-scope record makes the set
ineligible, and `consent.reason` and `consent.detail` on the scan say why. A
consent scan turns on face detection, MiVOLO and the set age.

**It binds the set to one identity.** The record's reference images are found
in the scanned folder by their sha256, and the centroid of their primary faces
is the identity reference (`faces.reference_embedding`, held in memory only;
a consent scan refuses a reference of its own). Each image then gets
`consent`: `bound`, the primary face's `distance`, and a `reason` when not
bound:

| reason | meaning |
|---|---|
| `not-consented` | its sidecar marks its subject as not consenting (argus-quarry) |
| `provenance` | its sidecar's `provenance` is unreadable, outside the folder, or names another source than this release |
| `unchecked` | face detection failed on it |
| `no-face` | no face |
| `bystander` | someone else is in the frame: another face, or another person the body detector found |
| `mismatch` | the primary face is further than `consent.max_distance` (0.4; `--consent-max-distance`) from the reference |

The distances stay on the scan, so the borderline ones can be reviewed.

**The export is gated** (`face_source.consent`, `--face-source-consent`). It
runs the face-source preset over the bound images only, and refuses (400; exit
2 on the CLI) when:

- the scan read no record, or the record made the set ineligible;
- the scan read no adult set age, or any face bound to the record reads
  under 18 (one young estimate refuses the whole set);
- the record, read again now, no longer covers the set: revoked or expired
  since the scan, gone, or replaced by another release;
- no image qualifies.

`face_source.consent` cannot be combined with `require_adult: false` or a
named `selection`. Each manifest row then carries `consent_release_sha256`,
`consent_record_sha256` (the record file as read at export) and `provenance`:
the image's `<stem>.json` sidecar `provenance` when it has one (it must be
this release's), else `kind: "consented"` under the record's release. See
[Manifest](manifest.md).

The export never names the person. Every row's `provenance.subject` is
argus-cortex's subject ref, `consent:<release_sha256>` (`subject_ref`,
applied with cortex's `pseudonymise`), not the name in the record or the
sidecar, and the default `source` doesn't name who collected the release.
The ref is the release's own key, already on the row as
`consent_release_sha256`; unlike a hash of the name, guessing names can't
reverse it. cortex's `consent_matches` still matches a row to its record.
With `carry_sidecars`, a carried `<stem>.json` is written as a copy with its
`provenance` pseudonymised the same way, under every mode: the source sidecar
is only read, so under `move` it stays in the source folder. `source`,
`license` and `notes` are free text and are carried as the sidecar's writer
wrote them, so keep names out of them. So are carried captions (`.txt`) and
the source's own `.xmp`, which hold whatever you or another tool put there.

Nothing about the consent outlives the scan: the scan keeps only the verdict,
the record's hashes, which images are the references and each image's
binding, inside its own cached summary, and never the record or the reference
embedding. Deleting or purging the scan removes it all. Over HTTP the record
resolves under the scanned folder, as images do; the CLI takes any path.
