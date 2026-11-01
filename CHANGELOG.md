# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-11-01

### Added

- Optional EraX-NSFW sidecar (`ARGUS_CURATOR_ERAX_URL`). When NudeNet returns
  no exposed score, EraX hits are mapped onto the same labels so filters,
  blur, pairing and cascade still work. `GET /detectors` reports `erax`.

## [0.10.2] - 2026-11-01

### Added

- `scripts/guard_eval.py` scores a `cinematic` expectation (fictional crime
  as a story or still): pass under `adult-cinematic`, flag under the others.
  Text probes cover a heist, pension fraud, dirty cops, war, weapons as a
  prop, and drugs as a plot, plus a how-to robbery that every policy should
  still flag. Commercial adult-entertainment stills stay `adult` (pass under
  every adult policy, including cinematic).

### Changed

- Requires `argus-cortex>=0.4.5` so `adult-cinematic` is in `POLICIES` and the
  wire schema. Policy texts come from cortex; headings dropped Extreme.

## [0.10.1] - 2026-10-30

### Changed

- Under `cascade` thinking, a frame with a MiVOLO age under 18 (a measured
  face, a body-only estimate, or the set's age for a faceless frame) and an
  exposed NudeNet score of at least 0.2 goes straight to SingGuard `slow`
  instead of the `fast` screen. That pairing already holds the frame, so
  reasoning on it costs little.
- `docker-compose.yaml` turns the GPU lease on (`ARGUS_GPU_COORDINATOR=lease`,
  `none` in `.env` to opt out) and mounts its lock directory: host
  `ARGUS_GPU_LOCK_DIR` (`/tmp/argus-gpu`) at `/run/argus-gpu`, lock file
  `/run/argus-gpu/argus-gpu.lock`. argus-lens shares the lock by mounting the
  same host directory and naming the same file.

### Fixed

- `.env.example` listed `ARGUS_LENS_URL`, which nothing reads: an export hands
  its manifest to argus-lens only through the request's `caption_url`. It now
  says so, and lists the GPU lease settings instead.

## [0.10.0] - 2026-10-30

### Added

- Person boxes from `--mivolo-body` are recorded on `results[].people` (bbox,
  matched `face_index`, age). A person with no face still gets a body-only
  MiVOLO age. NudeNet detections are assigned to the overlapping person (else
  face); `minor-exposed` uses that pairing and fails closed on unassigned
  exposure. `POST /faces?body=true` returns `people`. `people_check` (`ok`,
  `detection_failed` or `body_ages_failed`, with `people_error`) says how the
  person check went, on each result and in the `/faces` response, so an empty
  `people` reads as "none found" only when it is `ok`.
- SingGuard context kinds `scene`, `nsfw`, and `garments` send Places365,
  the NudeNet content rating, and Fashionpedia tags with the image. `age`
  includes body-only estimates. Compose defaults
  `ARGUS_CURATOR_SINGGUARD_CONTEXT` to `age,folder,caption,scene`.
  `adult-granular` and `adult-cautious` stay available for comparison.

- MiVOLO can read each face's body as well as its face:
  `FaceConfig.mivolo_body` (`--mivolo-body`, `POST /faces?body=true`). Bodies
  come from InsightFace 2.1's person detector, the one model of the
  `cheetah_l` pack, loaded alone. Faces get bodies by one-to-one assignment
  and body crops are masked as MiVOLO masks them, with MiVOLO's thresholds as
  request fields (`body_min_score`, `body_min_size`, `body_min_visible`,
  `body_face_match`). `FaceAttributes.age_body` records whether a body was
  used. Off by default while it is evaluated: `scripts/age_eval.py` compares
  face-only and face+body ages on labelled images, including how many minors
  each reads as adults.
- On CUDA, curator takes argus-cortex 0.4.3's host-wide GPU lease
  (`argus_cortex.gpu`) around each image's model run (faces with MiVOLO and
  FairFace, Places365 batches, garments) and around each model load. It is off
  unless `ARGUS_GPU_COORDINATOR=lease`, and never taken on CPU. With `[gpu]`
  (now including `argus-cortex[gpu]`, for NVML) it also waits for free VRAM
  across the card. A scan that cannot get the GPU in time fails instead of
  skipping images; the server answers 503 with `Retry-After: 5` (and a
  streaming scan drops any partial cache). `POST /faces` waits at most 30 s,
  then answers the same 503. `GET /health` reports `gpu_coordinator`. The
  lease is not taken when CUDA was requested but onnxruntime has no CUDA
  provider. InsightFace's pack download is off the load lease. The minimum
  argus-cortex is now 0.4.3.
- A CUDA image: `ghcr.io/smk762/argus-curator:<version>-gpu` (and
  `latest-gpu`) beside the CPU image, built with `GPU=1`. It installs `[gpu]`
  plus `onnxruntime-gpu[cuda,cudnn]`, removing the CPU `onnxruntime` that
  insightface pulls in (it shadows the GPU build), and the build fails if the
  CUDA provider is missing. `docker compose` now builds this variant by default
  (`CURATOR_GPU=0` for CPU).

### Fixed

- On a `[gpu]` install whose CUDA libraries came from NVIDIA's pip wheels,
  every model silently ran on CPU: those libraries are off the loader's path.
  `runtime.providers()` now calls onnxruntime's `preload_dlls()` once when it
  picks CUDA.

### Changed

- SingGuard thinking defaults to `cascade` (`fast` on every image, `slow` on
  what `fast` flags), including `docker-compose.yaml`. A server without the
  mode (stock SingGuard answers 400 to `cascade` and `slow`) is retried in
  `fast`; once `fast` works there, the rest of the run is judged in `fast`,
  `singguard_mode_unsupported` is logged once, and each such verdict records
  `thinking: "fast"`. A 400 that `fast` also gets is raised as it came.

- InsightFace is pinned to 2.x (`>=2.1,<3`; it had no upper bound, and a GPU
  image build had picked up 2.1 unasked), with `scikit-image<2.2`, whose 2.2
  removes a call InsightFace's face alignment makes.
- InsightFace loads only detection, recognition and the 3D-68 landmarks. The
  pack's gender/age model (MiVOLO replaces it) and its 2D-106 landmarks were
  loaded and never read: about 28% less time per image on CPU, with identical
  embeddings and pose.

## [0.9.0] - 2026-10-30

### Added

- XMP sidecars on export (#16): `write_xmp` (`--xmp`) writes face regions,
  person names (`face_names` / `--face-names`, by cluster) and the content
  rating beside each exported file, through `argus_cortex.xmp`. The source's
  sidecar is the starting point (`photo.jpg.xmp`, else Lightroom's
  `photo.xmp`), its other properties are kept, and it is never changed itself.
  Age estimates are never written. `ExportResult.xmp_written` counts the
  sidecars; a transfer can succeed while a sidecar fails. `--xmp-sources`
  (CLI only) merges into every scanned original's sidecar, including
  rejected images, independent of `--xmp`.
- A content rating from NudeNet (#18): `ImageResult.content_rating` is
  `safe` / `suggestive` / `explicit` (derived on read, so cached scans have one
  too). It is filtered by `content_ratings` (`--content-ratings`) and appears
  in the CSV report.
- `POST /faces` judges one image from memory (raw bytes or a multipart file)
  and returns MiVOLO `age_estimate_years` plus `bbox` per face. It writes
  nothing, so a caller that must not persist a buffer (argus-darkroom) can
  use it. `GET /health` reports `faces` when InsightFace and MiVOLO can run.
  A missing model, a detector that fails, or a detected face with no estimate
  is HTTP 503. A body over 64 MiB is HTTP 413.
  At most two calls run at once; a third is refused straight away with HTTP
  503 and `Retry-After: 1` rather than queued, because each call holds an
  image in memory. Admission is counted under a lock so a burst that hits a
  free slot cannot queue behind it.

### Fixed

- An unreadable source sidecar no longer leaves a broken copy beside the
  exported file. The seed is removed if the merge fails after copying it.

### Changed

- The SingGuard caption context reads XMP with argus-cortex's parser and
  finds argus-lens's sidecars: it looked only for `photo.xmp`, while lens
  writes `photo.jpg.xmp`, so lens captions never reached it. A sidecar that is
  not valid XMP is now no caption rather than a best-effort regex read.
- The InsightFace app and the onnxruntime sessions (MiVOLO, FairFace,
  Places365, garments) are built under a lock. A scan and a `POST /faces`
  asking for the same model at once used to be able to load it twice.
  InsightFace `app.get` is also serialized: the cached app is shared, and
  concurrent `get()` has mixed bboxes across images.
- The SingGuard policy names and texts (`default`, `adult`, `adult-granular`,
  `adult-cautious`) now come from argus-cortex 0.4.2
  (`argus_cortex.moderation.policies`), which shares them with argus-darkroom.
  The texts are unchanged, so results record the same policy names and
  verdicts are comparable with earlier scans. `guard.POLICIES`, the
  `guard.ADULT_*_POLICY` constants and `GuardPolicy` keep their names. The
  minimum argus-cortex is now 0.4.2.

## [0.8.1] - 2026-10-28

### Changed

- The adult policies name SingGuard's own "Adult Content (Non-criminal)"
  category under Safe. The model files legal adult material under that
  category from its training and cited it as a violation under all three
  adult policies. The "is not this rule" lines under rules A (`adult`,
  `adult-cautious`) and C (all three) are gone: SingGuard reads a bullet under
  a risk heading as a match for that rule, so what passes is said only under
  Safe. Results still record the same policy names; scans made before this
  release were judged under the old wording.

## [0.8.0] - 2026-10-28

### Added

- `GET /guard/policies` serves every SingGuard policy a scan can name with
  its text, its `source` (`builtin`, `file` for the operator's policy file,
  or `singguard` when `default` sends no text and `text` is null) and the
  name results record it under, so a client can show the wording a verdict
  was judged against. An unreadable policy file is listed with an `error`
  naming the variable, not the path. `/health` reports `guard_policy_text`.
  The policy file's text is readable by anyone who can reach the server.

## [0.7.0] - 2026-10-27

### Added

- `FaceConfig.set_age` (`--set-age`): a passing frame with no face takes a
  `set_age` read off the set, when one identity holds at least 90% of the
  scan's faces (`set_age_min_share`) with at least 5 MiVOLO ages
  (`set_age_min_faces`): the identity's median `years`, its `cluster_id`,
  the `faces` counted and its `share`. The summary carries the same
  `set_age`, or `set_age_skipped` saying why none was given. With the `age`
  SingGuard context the frame's text says no face is visible and what the
  rest of the set shows. `/health` reports `set_age`.

### Changed

- The `identities` phase runs right after `faces` and `nsfw`, before
  `scenes`, `garments` and `guard`, so the set age is known before the guard
  pass. The ages themselves still arrive with the `faces` chunks.

## [0.6.1] - 2026-10-27

### Fixed

- Sharpness is measured with the image's long side at 1024px, so
  `blur_threshold` and `sharpness_ref` mean the same thing for every file
  size. The 3x3 edge kernel reads per-pixel contrast, which falls as
  resolution rises: a sharp 3200x4800 studio frame measured about 57 at
  native size, under the default threshold of 100, and was rejected as
  blurry. Rejected images skip face detection, so those frames also had no
  MiVOLO age and a quality score of 0. Images under 1024px measure as before.
  Scans of large files made before this release should be run again.

## [0.6.0] - 2026-10-27

### Added

- `ARGUS_CURATOR_SINGGUARD_CONTEXT` sends what the scan already knows about an
  image as SingGuard's request text, framed as recorded context rather than a
  request: `age` (each face's MiVOLO estimate, named as an estimate), `folder`
  (the image's folder under the scanned root), `caption` (an argus-lens
  `<stem>.txt` or `<stem>.xmp` sidecar). Off by default. Measure it on
  labelled images before relying on it: a wrong age leans the verdict the
  wrong way, and the under-18 rule does not read it.
- `ARGUS_CURATOR_SINGGUARD_THINKING` accepts `slow` and `cascade`, the
  argus-cull `singguard` service's modes, and `GuardResult.thinking` records
  the mode the server says it judged in. Other servers answer 400 to the new
  modes. `scripts/guard_eval.py --thinking` takes the same modes.
- `POST /scan/{scan_id}/cancel` stops the running scan. The scanner reads the
  flag before each image and between phases; the checks already with a model
  or a remote server finish, the ones still queued are dropped, and the stream
  ends with `event: cancelled`. The partial summary saved while it ran is
  deleted. `GET /scans` reports `active.cancelling` meanwhile.
- `POST /scans/purge` stops the running scan, if any, waits for it to let go,
  and deletes every cached scan. Returns `{cancelled, deleted}`. Source images
  are never touched.
- `scan_folder()` / `scan_items()` take `cancelled=`, a callable the scan
  polls; `argus_curator.runtime.ScanCancelled` is raised when it answers True.
  `ScanStore.delete_all()`.
- `GET /health` reports `scan_control: true`, so a client can ask for these
  routes instead of sniffing the version.

### Changed

- The `adult` and `adult-granular` policies say what passes under Safe: a
  single adult posing nude or in lingerie for the camera, and that consent
  cannot be read from an image, so a posed, camera-aware adult counts as
  consenting unless the image shows hidden recording, restraint, coercion or
  an unconscious person. Rule A's carve-out reads "by adults, alone or
  together" instead of "between adults", which SingGuard took to need two
  people. A solo adult in a professional set was flagged under A for lack of
  "adult interaction"; it now passes. The exceptions live under Safe because
  SingGuard reads a bullet under a risk heading as a match for that rule, even
  one that says "is not this rule". `adult-cautious` is unchanged.
- Requires `argus-cortex>=0.4.1` (and `argus-cortex[server]>=0.4.1` for the
  server). `GuardResult` comes from cortex, so its new `thinking` field is in
  the wire schema, and pinning the floor keeps the schema the same wherever
  curator is installed.

## [0.5.1] - 2026-10-27

### Changed

- Face detection and NudeNet run at the same time, straight after scoring
  and ahead of the Places365 and Fashionpedia passes: one is this process's
  CPU, the other a separate server, and a viewer needs both the MiVOLO ages
  and the exposed score to lift a blur safely. Their `progress` events
  interleave. Scenes and garments, which nothing waits on, follow, and
  SingGuard still comes after those.
- `pending_checks` on scan chunks now reports what is actually unfinished
  rather than inferring it from the phase name, and names `faces` while face
  detection is still running. Scan `progress` events carry the same list,
  and `GET /scans` reads it from them. A check that could not run is dropped
  from the list, so a client never waits for a result that will not come.

## [0.5.0] - 2026-10-27

### Added

- `POST /scan/folder/stream` sends `event: chunk` frames as batches of images
  finish, so a client can show results while NudeNet and SingGuard are still
  running.
  `GET /scans` lists the scans on disk and the one running, if any.
- Images with an under-18 MiVOLO age on any face and any reported exposed
  NudeNet detection are never kept (`minor-exposed`, checked before every
  other filter, whatever the request asks for). `GET /image?scan_id=` returns
  403 for them, and `POST /scan/{scan_id}/delete` (`rel_path` or up to 200
  `rel_paths`) deletes them from disk and the scan; it refuses the whole
  batch with 403 if any path is not one of them.
- A built-in `adult` SingGuard policy, picked per scan with
  `GuardConfig.policy` (`"guard": {"policy": "adult"}`, CLI
  `--guard-policy adult`). Its rule A covers only apparent minors,
  non-consent and sexual violence, and Safe names consensual adult nudity.
  Rules B to G keep their letters and areas but are written out as text, so
  their verdicts can differ from SingGuard's built-in wording. A request names
  a policy and never sends its text. `default` keeps today's behaviour: the policy file
  when one is set, else SingGuard's own rules. Results record `adult` as the
  policy name. `GET /detectors` lists the names in `singguard_policies`
  (empty without a SingGuard server).
- Two adult policy variants under evaluation, `adult-granular` (a letter per
  hard case) and `adult-cautious` (unclear age or consent counts as rule A),
  plus `scripts/guard_eval.py` and text probes in `eval/guard_probes.jsonl`
  to compare all policies against a SingGuard server.
- Policy comparison on real scans: `GuardConfig.variants` judges every image
  under extra policies too (`ImageResult.guard_variants`, by policy name, the
  official verdict included; CLI `--guard-variants`), and `guard_policies` on
  export and `/select` makes the SingGuard filters drop an image when any
  listed verdict fails or is missing (CLI `--guard-policies`). Keep reasons
  name the policy (`guard-unsafe:adult-granular`), and the CSV report has a
  `guard_variants` column.

### Changed

- Only one scan runs at a time: `POST /scan/folder` and its stream return 409 while another is
  running, as does deleting from a scan that is still running.
- MiVOLO age is recorded before identity clustering, so every face carries
  it.

## [0.4.0] - 2026-10-26

### Added

- `POST /scan/{scan_id}/select` runs the export filter (`decide_selection`) and
  returns `{rel_path, abs_path, keep_reason}` for every image without copying,
  symlinking, or moving. `GET /image` serves the original file under the same
  containment rules as `GET /thumb`. It refuses the same NSFW, SingGuard and
  garment-floor filters `/export` would.
- NudeNet nudity detection through a NudeNet server over HTTP
  (`ARGUS_CURATOR_NUDENET_URL`, plus `_TIMEOUT` / `_WORKERS`); curator never
  imports NudeNet (AGPL-3.0). Each image records its detections, the highest
  score per class and an exposed score over the five exposed-nudity classes
  (`ImageResult.nsfw`, plus `nsfw_exposed_score` / `nsfw_labels` CSV columns),
  as a new `nsfw` progress phase covering every image, quality rejects
  included. `GET /detectors` reports `nudenet`.
- NSFW export filters: `nsfw_threshold` on the exposed score and
  `nsfw_label_thresholds` per class (CLI `--nsfw-threshold`, `--nsfw-label`),
  with `nsfw_exposed_labels` choosing what the total covers. Thresholds are
  inclusive; unchecked images are excluded (`nsfw-unchecked`).
- `docker-compose.yaml` runs a NudeNet server (pinned by digest) next to
  curator and points it there unless `.env` says otherwise.
- SingGuard policy check through a SingGuard server over HTTP
  (`ARGUS_CURATOR_SINGGUARD_URL`, plus `_THINKING`, `_POLICY_FILE`, `_TIMEOUT`,
  `_WORKERS`), opt-out like NudeNet (`--guard/--no-guard`,
  `ScanRequest.guard`). Each image records the verdict, the rules that fired,
  the explanation, the model and the policy (`ImageResult.guard`, plus
  `guard_safe` / `guard_rules` CSV columns), as a new `guard` progress phase.
  Images over 1536 px are downscaled before sending. `GET /detectors` reports
  `singguard`.
- SingGuard export filters: `guard_unsafe` drops every unsafe verdict and
  `guard_rules` only the named rules, by letter or full name (CLI
  `--guard-unsafe`, `--guard-rules`). Unchecked images are excluded
  (`guard-unchecked`), as are unsafe verdicts naming no rule
  (`guard-rule-unknown`).
- Pushing a `v*` tag now creates the GitHub Release once PyPI has the
  version, with notes from this file's section for that version.

### Changed

- The NudeNet and SingGuard HTTP clients, NudeNet's classes, the
  exposed-nudity set and the `NsfwResult` / `GuardResult` types now come from
  `argus_cortex.moderation`, shared across the suite. Behaviour and the wire
  schema are unchanged.
- `argus-cortex>=0.4,<0.5` is now a base dependency (pydantic-only); the
  `[server]` extra moves to `argus-cortex[server]>=0.4,<0.5`.
- `GET /thumb` now serves only the extensions the scanner reads (`.jpg`,
  `.jpeg`, `.png`, `.webp`) and returns 400 for anything else.
- The NudeNet check is opt-out: whenever `ARGUS_CURATOR_NUDENET_URL` is set,
  scans run it unless they pass `--no-nsfw` / `"nsfw": {"enabled": false}`.
  Deployments that do not set the URL are unaffected; the bundled
  `docker compose` setup sets it, so set it empty there to opt out.

## [0.3.0] - 2026-10-25

### Added

- Face attributes from two optional models (`FaceConfig.fairface` /
  `FaceConfig.mivolo`, CLI `--fairface` / `--mivolo`; either implies face
  detection). FairFace gives race (7 classes), gender and age bracket, each
  with its softmax confidence; MiVOLO v2 gives age in years and gender, and
  supersedes FairFace for both when the two run together (`gender_source` /
  `age_source` record which). Recorded per face, on the primary face
  (`ImageResult.primary_face_attributes`, plus CSV report columns), and pooled
  per identity (`FaceCluster.attributes`: averaged probabilities, median age).
- Both run on onnxruntime. `argus-curator fairface-convert <weights.pt>` and
  `argus-curator mivolo-convert` (downloads `iitolstykh/mivolo_v2` pinned by
  commit and sha256) export the ONNX models once, behind the
  `[fairface-convert]` / `[mivolo-convert]` extras. Model paths come from
  `ARGUS_CURATOR_FAIRFACE_MODEL` / `ARGUS_CURATOR_MIVOLO_MODEL` (or
  `--fairface-model` / `--mivolo-model`), never from a request; the server
  refuses a scan asking for a model it lacks with 400 before scoring.
- Export filters on those attributes: `face_races`, `face_genders`,
  `face_ages`, `min_age_years` / `max_age_years` and `min_attribute_conf`
  (CLI `--race`, `--gender`, `--age`, `--min-age`, `--max-age`,
  `--min-attr-conf`), each exclusion with its own `keep_reason`.
- Places365 scene tags (`ScanRequest.scenes` / `SceneConfig`, CLI
  `--scenes`): each passing image gets its top-k scene categories with
  softmax probabilities and an indoor/outdoor call (`ImageResult.scene`, plus
  CSV columns), as a new `scenes` progress phase. Converted once with
  `argus-curator places365-convert` (`[places365-convert]` extra; ResNet-18 or
  ResNet-50, sha256-pinned since upstream serves plain HTTP); the model path is
  `ARGUS_CURATOR_PLACES365_MODEL` / `--places365-model`.
- Export filters `scenes`, `scene_environments` and `min_scene_conf` (CLI
  `--scene`, `--environment`, `--min-scene-conf`), and `diversify_scenes`
  (`--diversify-scenes`) to spread a `max_keep` selection across backdrops.
- Fashionpedia garment detection (`ScanRequest.garments` / `GarmentConfig`,
  CLI `--garments`): each passing image gets its garments (27 classes) with
  probability, box, a frame-edge "cut off" flag and a dominant colour
  (`ImageResult.garments`, plus a CSV column), as a new `garments` progress
  phase. Converted once with `argus-curator garments-convert`
  (`[garments-convert]` extra; `valentinafevu/yolos-fashionpedia` pinned by
  commit and sha256); the model path is `ARGUS_CURATOR_GARMENTS_MODEL` /
  `--garments-model`.
- Export filters `garments`, `exclude_garments`, `min_garment_conf` and
  `garments_uncropped` (CLI `--garment`, `--exclude-garment`,
  `--min-garment-conf`, `--uncropped`), and `diversify_outfits`
  (`--diversify-outfits`) to spread a `max_keep` selection across outfits.
- `GET /detectors` reports `fairface`, `mivolo`, `places365` and `garments`.

### Security

- Distrust the emSign ECC Root CA - G3 root
  ([discuss.python.org/t/109576](https://discuss.python.org/t/109576)). The
  Docker image disables it and the Comodo AAA cross-sign root in the system
  trust store, and upgrades to pip 26.2.2 from a sha256-pinned wheel, since
  older pip also trusts its own bundled roots. `certifi>=2026.10.19`, which
  drops the root, is required by the `[server]` extra (httpx verifies against
  it), `[dev]`, and the image.

## [0.2.1] - 2026-07-22

This section also covers 0.2.0 (2026-07-17): both were tagged without moving
their entries out of `[Unreleased]`, and 0.2.1 amended bullets 0.2.0 added,
so the two cannot be separated cleanly after the fact.

### Security

- Server scan/export endpoints now enforce path containment (issue #3):
  `POST /scan/folder`, `/scan/folder/stream` resolve `folder` under the
  configured source root (`--source-root`, `ARGUS_CURATOR_SCAN_ROOT` /
  `CURATOR_SOURCE_PATH`) and `/export`, `/export/stream` resolve `dest` under
  the export root (`--export-root`, `ARGUS_CURATOR_EXPORT_ROOT` /
  `CURATOR_EXPORT_PATH`), refusing traversal escapes and refusing outright
  when the root is unset. **Breaking**: request paths are now root-relative
  (absolute paths are tolerated only when already inside the root).
- `mode: "move"` exports are rejected with 403 unless the server is started
  with `--allow-move` / `CURATOR_ALLOW_MOVE=1`.
- `--cors` no longer means `allow_origins=["*"]` with credentials (which
  reflected any origin): it now allows only the localhost dev frontend.
  Additional origins via `--cors-origin` / `CURATOR_CORS_ORIGINS`; a
  credential-less wildcard is available behind `--cors-any`.
- `/health` reports `export_root` and `allow_move` alongside `source_root`.
- A cached scan is treated as data, not configuration: `/thumb?scan_id=` and
  `/export` re-check the scan's own folder against the source root instead of
  trusting `ScanSummary.folder` as a containment root. The scan store is shared
  with the CLI (which is unconstrained by design) and carries scans recorded
  before containment existed, so an arbitrary `folder` could otherwise read or
  export files the source root is meant to fence off. **Breaking**: `/export`
  now requires a source root, since it reads the scan's images by `abs_path`.
- A literal `"*"` in the CORS allow-list (`--cors-origin '*'`,
  `CURATOR_CORS_ORIGINS=*`) now takes the same credential-less path as
  `--cors-any`. Passed through verbatim it became `allow_origins=["*"]` with
  `allow_credentials=True`, which makes Starlette reflect any Origin — exactly
  the hole the allow-list closes.
- Malformed paths (e.g. an embedded NUL byte) are refused with 400 rather than
  raising `ValueError` out of `resolve()` as an unhandled 500 with a traceback.
- Cross-site state-changing requests are refused with 403. CORS is not a write
  boundary: `POST /upload` takes `multipart/form-data`, a CORS-safelisted
  content type, so browsers send it with **no preflight** — any page the user
  visited could drive it (the same-origin policy only stops that page from
  *reading* the reply) and, with no auth on a server usually bound to localhost
  or a LAN address, poison a dataset. Unsafe methods now require `Origin` to be
  absent (non-browser clients such as curl and the CLI), same-origin, or on the
  configured allow-list. **Note**: `--cors-any` grants anonymous *read* access
  from anywhere but never a cross-site write — name the origin with
  `--cors-origin` to allow writes from it. The guard itself now ships in
  `argus_cortex.server` (`WriteGuard` + `cross_site_refuse`) so the suite's
  servers share one implementation; the policy and its pure-ASGI,
  non-stream-buffering guarantee are unchanged.
- A literal `"*"` co-listed with a real origin
  (`CURATOR_CORS_ORIGINS=*,https://studio.example`) no longer revokes that
  origin's cross-site writes. The `"*"` still degrades to the credential-less
  read-only wildcard, but it is now dropped from the allow-list rather than
  collapsing it, so naming an origin — the one documented way to grant it
  writes — cannot silently do nothing.

### Fixed

- **Manifest 2.1** — `mode: "move"` manifests were unusable. Rows stamped
  `abs_path` with the *source*, which the transfer had already deleted by the
  time the manifest was written, so every row named a dead path (issue #9).
  argus-lens opens rows strictly by `abs_path` and never consults
  `exported_path`, so a `write_manifest: true` move export produced a manifest
  it could not read at all. `abs_path` is now the location the image can
  actually be read from — the destination under `move`, the still-present
  source under `copy`/`symlink`.
- `argus-curator scan --csv` crashed with a `TypeError` after the manifest-2.0
  change (`write_report` gained the `exported_paths` parameter but the CLI
  call site was not updated).
- `argus-curator scan --csv` with no `--copy-to`/`--move-to`/`--symlink-to` no
  longer runs a pointless self-export (`dest` == the source), which failed with
  `SameFileError` for every image — one spurious `export_transfer_failed`
  warning per file — and left the report's `exported_path` column empty.

### Added

- `POST /upload` server endpoint: multipart image upload into a folder under
  the configured source root (traversal-safe, non-image and duplicate-name
  files skipped, existing files never overwritten).
- `ExportResult.exported_paths`: the `rel_path -> exported_path` mapping for
  every successful transfer, so API callers get de-collided names even with
  `write_manifest: false`. The CSV report gains a matching `exported_path`
  column.
- `ExportResult.exported_abs_paths` and `ManifestRow.exported_abs_path`: the
  transferred files' **absolute** locations (issue #10). Joining `dest` onto
  `exported_path` made clients re-implement a layout rule only the curator
  knows — and with `preserve_structure: false` the de-collision suffix
  (`stem-<hash>.ext`) is not reproducible by a client at all. It also makes a
  manifest row self-contained: the manifest is often read somewhere other than
  the export root it was written into (argus-lens receives it as an upload) and
  no row names that root.
- `manifest_version` on `GET /health` and on `ExportResult` (issue #11).
  Consumers were compat-detecting by sniffing whether `exported_paths` was
  present, which cannot distinguish a pre-2.0 curator from a 2.x one that
  transferred nothing — both yield an empty mapping — and, since the field has
  a default, made the legacy branch unreachable and untestable. `/health` now
  declares the manifest contract separately from the package version, so a
  client can refuse an unknown major instead of guessing.

### Changed

- The `[server]` extra now depends on `argus-cortex[server]>=0.2.0,<0.3` for the
  shared write-guard. It is a *server* dependency, not a base one: a plain
  `pip install argus-curator` stays engine-only. The `<0.3` cap is deliberate —
  argus-cortex is pre-1.0, where a minor bump carries no compatibility
  guarantee, and this is security-critical middleware.
- The `[server]` extra's fastapi floor rises to `>=0.110.1`. fastapi 0.110.0
  pins `starlette<0.37` while `argus-cortex[server]` needs `starlette>=0.37`, so
  0.110.0 was never actually installable alongside it; the old `>=0.110` floor
  advertised support it did not have.
- The `[all]` extra is now composed from `[server,cli,faces]` instead of
  hand-duplicating their contents, so a new member of any of them cannot go
  missing from `all` (it had already drifted).
- `CURATOR_ALLOW_MOVE` parsing moved to `argus_cortex.server.env_flag`, shared
  with the other suite servers. Two behaviour changes: surrounding whitespace is
  now stripped, so an `env_file` line or secret file with a trailing newline
  (`CURATOR_ALLOW_MOVE=1 `) now *enables* destructive move-exports where it
  previously read as off; and a set-but-unrecognised value (`=enabled`) logs a
  warning instead of silently resolving off. **Check your deployment config if
  you relied on the old whitespace behaviour.**
- **Manifest 2.0** (breaking for consumers that derive destinations from
  `rel_path`): rows are written only for files whose transfer succeeded and
  carry `exported_path` — the real (possibly de-collided) path under the
  export root, which consumers must use. The row shape is now published in the
  wire schema as `ManifestRow`.
- `MANIFEST_VERSION` is `2.1`. The major stays `2`, so a consumer that accepts
  `2.x` (argus-forge's `SUPPORTED_MANIFEST_MAJORS`) keeps working unchanged and
  one that already required `exported_path` is unaffected.
- `ExportResult.selected_rel_paths` is documented as what the selection *chose*,
  not what landed: it is computed before the transfer loop, so it still lists
  files whose source vanished or whose transfer failed. `exported_paths` /
  `exported_abs_paths` are the record of what is on disk.

### Fixed

- Flattened exports (`preserve_structure: false`) no longer silently overwrite
  basename collisions: collision detection is case-insensitive and
  Unicode-normalised (safe on APFS/NTFS/exFAT destinations), generated names
  are checked against the whole plan (a clash with a pre-suffixed file extends
  the digest), and the export fails loudly if a unique name cannot be
  generated.

## [0.1.0] - 2026-07-02

Initial release — the curation stage of the Argus suite
([argus-quarry](https://github.com/smk762/argus-quarry) →
**argus-curator** →
[argus-lens](https://github.com/smk762/argus-lens)).

### Added

- Training-suitability scanning: hard filters (min short side, aspect ratio,
  blur), target-aware composite scoring, and per-image reject reasons.
- Near-duplicate detection via pHash clustering — keeps the highest-scoring
  representative and reports (never silently drops) the rest.
- Identity-aware face clustering (InsightFace, `[faces]` / `[gpu]` extras) with
  head-pose (yaw) capture and pose-balanced subset selection.
- Structure-preserving export (copy / symlink / move) with score threshold,
  diversity cap, face-cluster and pose filters, plus a per-image CSV report.
- Versioned JSONL handoff manifest carrying the shared `TargetProfile`, with an
  optional `caption_url` POST for a one-click curate→caption run against
  argus-lens.
- Wire-contract JSON Schema (`schema/curator-wire.schema.json`) published for
  consumer codegen, with a CI staleness check (`argus-curator schema --check`).
- FastAPI micro-server (`[server]` extra, :8101): `/health`, `/detectors`,
  `/folders`, `/scan/folder`, `/scan/{scan_id}` (paginated), `/thumb`,
  `/export`, and SSE streaming variants `/scan/folder/stream` and
  `/export/stream`.
- Typer CLI (`[cli]` extra): `scan`, `serve`, `detectors`, `schema`.
- On-disk scan cache keyed by `scan_id` (`CURATOR_CACHE_DIR`) powering
  pagination and export-by-id.
- Docker image (GHCR) and `docker compose` deployment.

[Unreleased]: https://github.com/smk762/argus-curator/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/smk762/argus-curator/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/smk762/argus-curator/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/smk762/argus-curator/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/smk762/argus-curator/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/smk762/argus-curator/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/smk762/argus-curator/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/smk762/argus-curator/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/smk762/argus-curator/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/smk762/argus-curator/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/smk762/argus-curator/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/smk762/argus-curator/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/smk762/argus-curator/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/smk762/argus-curator/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/smk762/argus-curator/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/smk762/argus-curator/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/smk762/argus-curator/releases/tag/v0.1.0
