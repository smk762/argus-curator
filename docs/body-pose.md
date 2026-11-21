# Body pose (argus-puppet)

With `ARGUS_CURATOR_PUPPET_URL` set, every scan sends each passing image to an
[argus-puppet](https://github.com/smk762/argus-puppet) server and records, per
image, the primary person's coarse body-pose labels and a **scan-level** pose
cluster. Curator never imports argus-puppet: it speaks puppet's HTTP wire
(`puppet_version` 1.x) and keeps only plain values. **No keypoints are stored**
anywhere in curator: not on a row, in the scan cache, in a response, the CSV
report or the manifest.

## Turning it on

| Variable | Default | Meaning |
|---|---|---|
| `ARGUS_CURATOR_PUPPET_URL` | unset (stage off) | The puppet server's origin, e.g. `http://<PUPPET_HOST>:8220` |
| `ARGUS_CURATOR_PUPPET_TIMEOUT` | 60 | Seconds per `POST /pose`; `POST /stats` gets five times this |
| `ARGUS_CURATOR_PUPPET_WORKERS` | 2 | Images in flight at once (puppet admits `ARGUS_PUPPET_POSE_CONCURRENCY`, default 2) |
| `ARGUS_CURATOR_PUPPET_MAX_SET` | 5000 | Largest scan clustered; match puppet's `ARGUS_PUPPET_MAX_SET_RESULTS` |
| `ARGUS_CURATOR_PUPPET_MAX_JSON_BYTES` | 134217728 | Largest `/stats` body; match puppet's `ARGUS_PUPPET_MAX_JSON_BYTES` |

The rule is the moderation checks' rule: the stage is on whenever the URL is
set, unless a scan opts out with `"pose": {"enabled": false}` (CLI
`--no-pose`). `"enabled": true` (`--pose`) makes it required: a scan is refused
with 400 when the URL is unset. A malformed URL refuses the scan with 400 naming
the variable, never its value (it can hold a credential). The URL is process
configuration only: no request field can point the stage at another host. The
scan records whether the stage ran in `pose_config.enabled`, and
`GET /detectors` reports `puppet: true` when a URL is set (reachability is not
probed).

## What happens in a scan

The stage runs after the faces (and the prefilter), before SingGuard, as
progress phase `pose`. For each passing image curator sends `POST /pose`:

- the bytes every other check saw (cropped, when `crops` cut the frame), raw,
  with `name=` set to the file's base name;
- the primary face as the hint `face=x,y,w,h`, so puppet's primary person is the
  body that face belongs to. Curator finds faces on the stored pixels and puppet
  applies the EXIF orientation first, so the box is turned into the upright
  frame before it is sent. Without a face, puppet's own rule (largest box by
  detector score) picks the primary person, which is curator's face rule too;
- `X-Argus-Priority: batch`. Puppet takes its own argus-iris GPU lease.

A `503` (puppet busy, or its GPU lease refused) is retried after its
`Retry-After` (capped at 10 s), at most 3 times; then the image counts as failed.

When every image is in, curator sends **one** `POST /stats` with the scan's
posed images inline, each named by its `rel_path` and trimmed to its primary
person (the only one the set metrics read). The answer gives each image's
cluster and the scan's numbers. Then the keypoints are dropped.

## Per image

| Field | Meaning |
|---|---|
| `pose_check` | How it went: `ok` (a primary person; labelled, and clustered unless the scan's clustering was skipped), `unusable` (a person, but too few confident body joints to cluster), `no_person`, `failed` (the request failed or the server was unreachable). `null`: not checked (stage off, or the image was rejected before it) |
| `pose_labels` | `{stance, arms, facing}` of the primary person, or `null` (no person, failed, not checked). Labels are puppet's geometry rules ([puppet's metrics.md](https://github.com/smk762/argus-puppet/blob/dev/docs/metrics.md#labels)): `stance` is `standing`, `jumping`, `crouching`, `seated`, `lying` or `unknown`; `arms` `up`, `out`, `down`, `mixed` or `unknown`; `facing` `front`, `back`, `left`, `right` or `unknown`, the subject's facing as the camera sees it. `unknown` is a real reading: the joints a rule needs were not confident. An unusable pose keeps whatever labels the rules could decide |
| `pose_cluster` | The scan-level cluster id, `0` for the largest, or `null` (unusable, no person, failed, not checked, or the scan was not clustered) |

A UI must never render `null` as a label: `pose_check` says why there is none.

The CSV report has `pose_check`, `pose_stance`, `pose_arms`, `pose_facing` and
`pose_cluster` columns; manifest rows (2.7) carry `pose_labels` and
`pose_cluster` ([Manifest](manifest.md)).

## Per scan: `ScanSummary.pose`

`null` when the stage did not run. Otherwise:

| Field | Meaning |
|---|---|
| `puppet_version` | The oldest `puppet_version` among the answers |
| `posed`, `no_person`, `failed` | Images puppet answered (of them, with nobody in frame), and images that failed |
| `usable` | Posed images with a usable primary pose: the clustered ones |
| `diversity` | Puppet's pose diversity, 0 to 1: roughly the chance two poses drawn at random are clearly different |
| `comparable_share` | How many usable pairs puppet could compare (low when crops are mixed) |
| `largest_cluster_share` | The largest cluster's share of `usable` |
| `clusters` | `{id, size, share, medoid_rel_path}`, the medoid being the image whose pose is most central to the cluster |
| `skipped` | Why the scan was not clustered, else `null`: nothing was posed, the scan is over `ARGUS_CURATOR_PUPPET_MAX_SET` or the body over `ARGUS_CURATOR_PUPPET_MAX_JSON_BYTES` (checked before sending, never truncated), puppet refused the set (413), or the call failed. Labels are kept either way |
| `stale` | Rows were dropped (a delete or a move) since: cluster sizes, shares and medoids are recounted from the rows left, `diversity` and `comparable_share` are as measured |

## Clusters span the whole scan

Clusters are formed by puppet's average linkage over **every** posed image in
the scan, at puppet's default threshold (0.25 torso units: jittered copies of
one pose merge, "arms out" and "arms up" stay apart). Curator does not compute
anything per selection. A UI that wants "40% of this selection is the same
pose" counts the selection's rows by `pose_cluster` and takes its most common
cluster's share, client-side. Rows with a `null` cluster are left out of that
count, as puppet leaves unusable poses out of its own.

The vector is in the image plane, so one stance shot frontal and in profile is
two poses to it; see puppet's [metrics.md](https://github.com/smk762/argus-puppet/blob/dev/docs/metrics.md).

## Failure

The stage never fails a scan. A failed image keeps `null` values and
`pose_check: "failed"`; after 8 failures with no success puppet is taken to be
down and the images not yet sent stay failed, as with the moderation checks. A
failed or refused `/stats` keeps every label and sets `skipped`. A server
answering another `puppet_version` major counts each image as failed. A 1.0
server puts no labels on its people: curator then takes them from the `/stats`
items, or leaves them `null` when the scan was not clustered.

## Deployment

Puppet's `POST /pose` never calls curator, so neither waits on the other. (The
only call the other way is puppet's optional library age check, which uses
curator's `POST /faces`; scans do not involve it.) Each service takes its own
argus-iris lease for its own GPU work.
