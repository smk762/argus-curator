# HTTP API (:8101)

The server is a peer to argus-lens (:8100): Studio's `/curate` page drives it, and it exposes the same scans, filters and exports as the CLI.

```bash
pip install "argus-curator[server,faces]"
argus-curator serve --cors --port 8101 --source-root /data/images --export-root /data/out
```

It binds `127.0.0.1` unless `--host` says otherwise. `--host 0.0.0.0` (what the
Docker image runs) puts the scan, export and delete endpoints on the network with
no authentication: a request with no `Origin` (curl) passes the write guard, so
expose it only on a network you trust.

## Roots, moves and deletes

Request-supplied paths are treated as untrusted: scan folders resolve under
`--source-root` (env: `ARGUS_CURATOR_SCAN_ROOT` / `CURATOR_SOURCE_PATH`) and
export destinations under `--export-root` (`ARGUS_CURATOR_EXPORT_ROOT` /
`CURATOR_EXPORT_PATH`). Traversal escapes are refused, and scan/export refuse
outright when their root is not configured. Both roots are needed to export:
the destination resolves under the export root, and the scan being exported
must itself lie under the source root — a `scan_id` is data from a store the
CLI also writes to, so its folder is re-checked rather than trusted (the same
check applies to `/thumb?scan_id=`). The destructive `mode: "move"` is
rejected unless the server is started with `--allow-move`
(`CURATOR_ALLOW_MOVE=1`). Likewise `POST /scan/{id}/delete` removes only
images MiVOLO reads as under 18 with an exposed score (the
[under-18 rule](moderation.md#the-under-18-rule-minor-exposed)) unless the server is
started with `--allow-delete` (`CURATOR_ALLOW_DELETE=1`, or `CURATOR_DEVMODE=1`),
which lets it delete any scanned image. `--cors` allows only the localhost dev
frontend (:3000); allow-list other origins with `--cors-origin <origin>`
(`CURATOR_CORS_ORIGINS`), or use `--cors-any` for a credential-less wildcard
on public demos. A literal `*` in the allow-list is treated as `--cors-any`
(wildcard without credentials), never as origin reflection.

A scan's consent record (`consent.record`) resolves under the scanned folder
the same way, a symlink included, and is re-checked at select and export
since the stored path is data (see
[Consented face sets](face-sources.md#consented-face-sets-of-a-real-person)).

## CORS and cross-site writes

CORS is not a write boundary — `POST /upload` sends `multipart/form-data`, which
browsers deliver with no preflight — so state-changing requests are additionally
gated on `Origin`: it must be absent (curl, the CLI, server-to-server),
same-origin, or on the allow-list, else 403. `--cors-any` therefore permits
anonymous reads from anywhere but no cross-site writes; name an origin with
`--cors-origin` to let it write.

Same-origin passes on its own only when the server is reached by an IP address
or `localhost`. Under DNS rebinding a browser sends the attacker's DNS name as
both `Origin` and `Host`, so a matching name proves nothing. A UI proxied onto
this server under a DNS name (`http://<LAN_HOST>:8101`) must list that name with
`--trusted-host <LAN_HOST>` (`CURATOR_TRUSTED_HOSTS=<LAN_HOST>`, comma-separated,
no port), or its writes get 403.

## Routes

| Route | Description |
|---|---|
| `GET  /health` | Liveness. `faces` is true when `POST /faces` can run (InsightFace and MiVOLO). `minor_exposed` is the under-18 rule in use, `frame` or `overlap` (`ARGUS_CURATOR_MINOR_EXPOSED`). |
| `GET  /detectors` | `{ torch, cuda, clip, insightface, onnxruntime, fairface, mivolo, bodies, places365, garments, nudenet, erax, watermark, singguard, puppet, speech, dnsmos, speaker_model, … }` |
| `POST /faces` | One image in memory (raw bytes or multipart), at most `CURATOR_MAX_UPLOAD_MB` (else 413). `{ source: mivolo, faces: [{ age_estimate_years, bbox }] }`. Writes nothing. |
| `GET  /guard/policies` | Each SingGuard policy a scan can name: `{ singguard, policies: [{ name, recorded, source, text, error }] }` |
| `GET  /folders?path=<rel>` | Browse Docker-mounted folders under the source root (for the UI picker) |
| `POST /scan/folder` | Scan + score + dedup + face-cluster → `ScanSummary`. With `target_profile.modality` `speech`, scores the folder's WAV/FLAC clips instead (rows in `clips`; see [Speech scans](speech.md)) |
| `GET  /scan/{scan_id}` | Cached summary, paginated via `?offset=&limit=` |
| `GET  /thumb?path=<rel>&scan_id=<id>` | `image/webp` thumbnail from the mount, of the cropped frame when that scan cropped it |
| `GET  /image?path=<rel>&scan_id=<id>` | Original image bytes, same containment as `/thumb` |
| `POST /scan/{scan_id}/select` | Dry-run of the export filter, cap, ranking and split → `{total, kept, rank, train, held_out, rows: [{rel_path, abs_path, keep_reason, rank, split}]}`, kept rows first and best first. Writes nothing. See [Ranking, max_keep and splits](#ranking-max_keep-and-splits). A speech scan's rows carry `speaker_id` (and `consent_distance` under `speech.consent_record`, [consented voice binding](speech.md#consented-voice-binding)), and `speech` the hours kept ([Speech scans](speech.md#select-and-export)) |
| `POST /scan/{scan_id}/delete` | Delete source files in the scan (`rel_path` or `rel_paths`, all or nothing: a path not in the scan is a 404, a running scan a 409) and drop their rows → `{rel_paths, total}`. Any image needs `--allow-delete` / `CURATOR_ALLOW_DELETE=1` / `CURATOR_DEVMODE=1` (else 403 unless under-18 exposed); `/health` reports `delete_any` |
| `POST /upload` | Multipart image upload (`files` + `folder`) into a folder under the source root. Each file is capped at `CURATOR_MAX_UPLOAD_MB` (default 50); a larger one is a 413 and nothing is saved |
| `POST /export` | Structure-preserving transfer + manifest → `ExportResult`. A speech scan exports its clips, a `.txt` transcript beside each and `speech_manifest.jsonl` (`modality: speech`; needs `speech.corpus` and `speech.license`, or `speech.consent_record`) |
| `POST /scan/folder/stream` | Same as `/scan/folder`, streaming live progress over SSE |
| `POST /export/stream` | Same as `/export`, streaming per-file transfer progress over SSE |
| `GET  /scans` | Cached scans, newest first, plus `active`: the one running, with its phase, counts, `pending_checks` and `cancelling` |
| `POST /scan/{scan_id}/cancel` | Stop the running scan (404 if that is not the one running). The stream ends with `event: cancelled`; the partial summary is dropped |
| `POST /scans/purge` | Stop the running scan, if any, wait for it to let go, and delete every cached scan → `{cancelled, deleted}`. Source images are not touched |

## Streaming

The `*/stream` variants emit `event: progress` frames (`{phase, done, total}`)
while the work runs, then a single `event: complete` frame carrying the same
payload the non-streaming endpoint returns (or `event: error`). A streamed scan
saves snapshots as it runs, so `GET /scans` lists it while it is `active`; a scan
that fails or is cancelled drops them, so what stays listed is a finished scan.
A scan runs
`scoring` → `faces` and `nsfw` together → `identities` → `prefilter` →
`scenes` → `garments` → `pose` (when argus-puppet is configured) → `guard` → `clustering`: ages and the exposed score come first, so
a viewer can lift a blur as soon as both are known, and their progress events
interleave while both run; identities follow at once, so a set age is known
before the guard pass reads it as context. Every `event: chunk` frame carries
the current `face_clusters`, empty until the identities are read. Scan progress
and `event: chunk` frames carry `pending_checks`, the checks among `faces`, `nsfw` and `guard` whose results
have not all landed yet; a check that could not run is not listed. A row's
`nsfw` is `null` until it is decided: with EraX on, a NudeNet score of 0 is held
back until EraX has answered, so a non-null `nsfw` is a score a viewer can act on.

## Cancelling and deleting scans

A scan stops cooperatively: `POST /scan/{id}/cancel` raises a flag the
scanner reads before each image and between phases, the checks already with
a model or a remote server finish, the ones still queued are dropped, and the
slot frees. Until then `GET /scans` reports the scan as `cancelling`. In the
library, pass `cancelled=` (a callable returning True to stop) to
`scan_folder()`; it raises `ScanCancelled`. `GET /health` says
`scan_control: true` on a server that has these routes.

`DELETE /scan/{id}` removes one cached scan and leaves the source images
alone. It answers 409 for the scan still running (cancel it first) and 404
for an unknown id. `GET /health` says `scan_delete: true` on a server that
has it.

## Request bodies

`POST /scan/folder` body:

```jsonc
{
  "folder": "shoots/2026-07",         // relative to the server's --source-root ("" = the root)
  "target_profile": { "target_style": "photo", "target_category": "identity" },
  "config": { "min_short_side": 512, "max_aspect_ratio": 3.0, "blur_threshold": 100.0,
              "cluster_distance": 10, "max_workers": 4 },
  "faces":  { "enabled": true, "model": "buffalo_l", "min_det_score": 0.5, "cluster_eps": 0.5 },
  "consent": { "record": null, "max_distance": 0.4 },  // optional: a consent record under the folder
  "embeddings": { "enabled": false, "device": "auto",  // CLIP image embeddings (memory only)
                  "reference_paths": [],           // optional likeness reference: images in this scan
                  "reference_embedding": null },   // or a centroid in curator's CLIP space (not both)
  "pose": { "enabled": null }         // body pose on argus-puppet: null = on when ARGUS_CURATOR_PUPPET_URL is set
}
```

`pose` runs the body-pose stage ([Body pose](body-pose.md)): each passing
image's `pose_check`, `pose_labels` and scan-level `pose_cluster`, and the
scan's `pose` summary (clusters, `largest_cluster_share`, `diversity`). Like
NudeNet it is on whenever its URL is set, `false` opts out and `true` refuses
the scan (400) without a URL; a malformed URL is a 400 naming the variable,
never the value. A puppet that is down or failing never fails the scan.

`POST /export` body:

```jsonc
{
  "scan_id": "...",                 // or inline "selection": ["rel_path", ...]
  "dest": "training-set-v1",        // relative to the server's --export-root
  "mode": "copy",                   // "copy" | "symlink" | "move" (move needs --allow-move)
  "preserve_structure": true,
  "min_score": 0.6, "keep_similar": false,
  "face_clusters": ["face_2"],      // optional: export only these identities
  "face_poses": ["frontal", "three_quarter"],  // optional: export only these head poses
  "face_races": null,               // optional: FairFace race filter on the primary face
  "face_genders": null,             // optional: gender filter on the primary face
  "face_ages": null,                // age brackets ("20-29", ...), from MiVOLO's years
  "min_age_years": 18, "max_age_years": 35,     // optional: MiVOLO age range
  "min_attribute_conf": 0.6,        // filtered attributes must be at least this confident
  "scenes": ["beach", "coast"],     // optional: Places365 scenes among the image's top-k
  "scene_environments": ["outdoor"],
  "min_scene_conf": 0.1, "diversify_scenes": false,
  "garments": ["dress"], "exclude_garments": ["bag_wallet"],  // optional: Fashionpedia
  "min_garment_conf": 0.5, "garments_uncropped": false, "diversify_outfits": true,
  "max_keep": null, "diversity_weight": null,  // optional cap; null weight = the scan's
  "rank": "score",                  // "score" | "identity" | "likeness": the order kept images come in
  "max_likeness_distance": null,    // optional: ceiling on each image's likeness_distance
  "exclude_references": false,      // optional: keep the scan's reference images out
  "split": null, "split_part": "train",        // optional train/held-out split; export one part
  "write_manifest": true,
  "caption_url": null,              // optional: POST manifest to argus-lens (internal hosts: CURATOR_CAPTION_ALLOWED_HOSTS)
  "carry_sidecars": false,          // optional: transfer .txt / .xmp / .json sidecars with each image
  "run_stats": null                 // optional: client JSON written to <dest>/run_stats.json
}
```

`write_manifest` writes `manifest.jsonl` into the destination; see
[Manifest](manifest.md) for its rows. `caption_url` then POSTs it to argus-lens
through argus-cortex's `net.fetch`, and only to a host listed in
`CURATOR_CAPTION_ALLOWED_HOSTS`, public or not (comma-separated `host`, any
port, or `host:port`, e.g. `lens:8100`). Any other host, or any `caption_url`
while the variable is empty, is refused with 400 before anything is transferred. A POST is never re-sent to a redirect; a hand-off that
fails after the export leaves `captioned: false` and logs the host only. `carry_sidecars` and `run_stats` are
covered in [Sidecars and run stats](sidecars.md), and every filter field in its
feature's page ([Face attributes](face-attributes.md), [Scene tags](scene-tags.md),
[Garments](garments.md), [Nudity detection](nudity.md),
[Policy check](policy-check.md), [Face sources](face-sources.md)).

## Ranking, max_keep and splits

`POST /scan/{id}/select` and `POST /export` run the same selection
(`selection.plan_selection`): the filters, then the cap, then the ranking,
then the split. A select and an export with the same fields keep the same
images, in the same order, in the same parts.

- **`rank`** orders the kept images, best first. `"score"` (the default): the
  curation score, then the primary face's identity distance. `"identity"`: the
  identity distance (the face's `reference_distance` when the scan had
  a reference, `faces.reference_paths` or `faces.reference_embedding`, else its
  `cluster_distance`; an image with no face or
  no distance comes last), then the score. Ties end on `rel_path`. `"identity"`
  needs a scan that ran face detection (else 400). `"likeness"`: the image's
  `likeness_distance` (see [Likeness](#likeness-subjects-without-a-face);
  none last), then the score. The manifest is written in
  this order, and each select row's `rank` is its 1-based position.
- **`max_keep`** caps the kept images; the rest get `diversity-trimmed`. The
  pick blends merit under `rank` with spread (pHash, plus scenes and outfits
  with `diversify_scenes` / `diversify_outfits`) by `diversity_weight`
  (0 = the best `max_keep` by rank, 1 = the widest spread; null = the scan's
  `config.diversity_weight`).
- **`split`** divides the kept images into `train` and `held_out`:

  ```jsonc
  "split": {
    "held_out": 8,              // or "held_out_fraction": 0.2 (exactly one)
    "group_by": "image",        // "image" | "face_cluster" | "similar_group" | "speaker" (speech): a group never spans both parts
    "method": "spread",         // "spread": evenly through the ranking | "random": shuffled by seed
    "seed": 0
  }
  ```

  Whole groups go to one part. With `group_by: "face_cluster"` one identity
  (the primary face's cluster) is never in both parts; an image with no
  clustered face is a group of its own. With `group_by: "similar_group"` an
  image's near-duplicates (the scan's `similar_group`: pHash, and CLIP with
  `config.semantic_duplicates`) are in its part, so a held-out reference is
  never a near-copy of a training image. A group that would push `held_out`
  past its target is skipped, so it can fall short of the target, never
  exceed it. A split that leaves either part empty (too few images, or one
  group) is refused with 400. The split is deterministic: the same request on
  the same scan always gives the same parts. Select reports both parts (each
  kept row's `split`, and the `train` / `held_out` counts). An export
  transfers only `split_part`, and its manifest rows carry `split`; export
  each part to its own `dest` (argus-forge trains on everything under one).
  A split can't be combined with an inline `selection`. A speech scan's split
  must group by `"speaker"` (and an image scan's can't): it is speaker-disjoint
  and stratified by accent and gender, see
  [Speech scans](speech.md#select-and-export).

argus-mint's identity training, for example, selects with
`{"face_source": {...}, "rank": "identity", "max_keep": train + held_out,
"diversity_weight": 0, "split": {"held_out": held_out}}`, then exports the
`train` part to the training export and the `held_out` part to proof's
references.

## Likeness: subjects without a face

A subject with no usable face (a creature, an object, a masked or stylised
character) has no identity distance, so likeness gives it one from the scan's
CLIP embeddings (#191). A scan with `embeddings` on records each passing
image's `likeness_distance`: the cosine distance (0..2) from its embedding to
the subject. The subject is, in order:

- `embeddings.reference_paths`: images in the scan, their embeddings averaged.
  A reference image the scan rejected is still embedded from the bytes it read
  (its row stays rejected, with no distance);
- `embeddings.reference_embedding`: a centroid the caller supplies, in
  curator's embedding space (`embeddings.EMBEDDING_SPACE`; another CLIP's
  vectors don't compare). Like a face `reference_embedding`, it is
  request-only: the stored and served summary never carries it;
- else the set's own centroid, recentred once on the closest 75% of the
  passing images so a few strays pull it less.

Either reference turns `embeddings` on. `ScanSummary.likeness_source` says
which subject was used (`"reference"` | `"centroid"`), and `likeness_skipped`
why no distance was measured (no reference image in the scan, none could be
embedded, a `reference_embedding` of the wrong length). A reference that gives
nothing is never swapped for the centroid.

Select and export read the recorded distances, never the vectors, so they
still work after a restart:

- `rank: "likeness"` orders kept images closest first (none last), then by score.
- `max_likeness_distance` keeps out an image farther than it, or with no
  distance (keep reason `likeness-distance`).
- `exclude_references` keeps the scan's reference images
  (`embeddings.reference_paths` and `faces.reference_paths`) out, keep reason
  `reference`: they are the yardstick, so a cap or a split never spends a place
  on one. Refused on a scan that named none.

Either one on a scan that didn't compute embeddings, or that measured no
distances, is refused with 400, never ignored. The embedding is of the frame
as the scan saw it: a background that varies more than the subject does
dominates the distance, so crop to the subject first (`crops`, or a
`.crop.json` sidecar) when it is small in the frame.

argus-mint's `likeness` identities, for example, upload the candidates and the
identity's references together, scan with
`{"embeddings": {"reference_paths": anchors}}`, and select with
`{"rank": "likeness", "max_likeness_distance": d, "exclude_references": true,
"max_keep": train + held_out, "diversity_weight": 0,
"split": {"held_out": held_out, "group_by": "similar_group"}}`, and export each
part by the same fields with `split_part`.

## Scan cache

Scans are cached on disk (keyed by `scan_id`, default `~/.cache/argus_curator/scans`,
override with `CURATOR_CACHE_DIR`). This is what makes paginated `GET /scan/{id}`
and export-by-id work without recomputing. `POST /scans/purge` empties it (and
stops a scan that is writing to it first); nothing else on disk is touched.
A streamed scan saves snapshots as it runs, and marks them with a
`<scan_id>.running` file until it finishes. If the server dies mid-scan, the next
start removes that scan, so a partial one is never listed as a result.
