# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.23.0] - 2026-11-23

**For operators running curator against argus-iris.** One dependency bump and a
simpler broker setup; nothing else breaks for callers of `/scan` or `/export`.

- **Requires argus-cortex `>=0.15,<0.16`** (was `>=0.14,<0.15`), for the
  zero-config GPU broker client (argus-cortex#204, argus-iris#47).
- **`MANIFEST_VERSION` stays 2.7 and `SPEECH_MANIFEST_VERSION` stays 1.0.**

### Changed

- **`ARGUS_GPU_BROKER_URL` and `ARGUS_GPU_BROKER_TOKEN` are all the broker
  needs (#241).** The URL alone selects argus-iris, and the tenant and its kinds
  come from `gpu-tenant.yaml`, which the image has at `/app/gpu-tenant.yaml`
  where cortex looks for it. `ARGUS_GPU_TENANT_MANIFEST` and `ARGUS_GPU_TENANT`
  are gone from compose and `.env.example`. An explicit `ARGUS_GPU_COORDINATOR`
  still wins, so a host `.env` that sets `lease` must drop it to use the broker.
- **The server registers with iris at startup** (`argus_cortex.gpu.aregistered`
  in its lifespan), keeps it registered, and signs off at shutdown, so the
  broker knows curator's kinds before its first lease. It never holds up
  startup. The CLI still registers with its first lease.
- **docker-compose.yaml no longer defaults `ARGUS_GPU_COORDINATOR` to `lease`**:
  that default would win over a broker URL. `.env.example` still sets `lease`;
  a host `.env` with neither it nor a broker URL now gates nothing.

## [0.22.0] - 2026-11-23

**For downstream callers (argus-lens, argus-forge, argus-proof, argus-mint, and
anything that drives `POST /export` or `argus-curator serve`).** Two behaviour
changes, and nothing else breaks:

- **`serve` binds `127.0.0.1` by default.** A request without an `Origin` still
  passes the write guard, so the old `0.0.0.0` default exposed scan, export and
  delete to the LAN. Pass `--host 0.0.0.0` (the Dockerfile does) when you mean
  to listen on all interfaces. It warns on a non-loopback bind.
- **Requires argus-cortex `>=0.14,<0.15`** (was `>=0.13,<0.14`). `schema` uses
  `argus_cortex.wire.sync_schema`, and consent hashing uses
  `argus_cortex.hashing.sha256_hex`.
- **`MANIFEST_VERSION` stays 2.7 and `SPEECH_MANIFEST_VERSION` stays 1.0.**
  Neither row shape changed. Speech consumers get clearer docs and a checked-in
  sample export (#233); the wire schema is regenerated via
  `argus_cortex.wire.sync_schema` (#236).

### Added

- **Speech manifest docs for consumers (#233).** `docs/speech.md` says to open
  clips by joining `exported_path` / `transcript_path` onto the consumer's view
  of the export folder (absolute paths are curator's filesystem, useless behind
  a container mount), and that a trainer's `<stem>.txt` wins if it is edited
  after export. [`samples/speech-export/`](samples/speech-export/) is a checked-in
  train + held-out export (empty WAV headers, no real voices) for consumers to
  test against.

### Changed

- **`schema` / `schema --check` use `argus_cortex.wire.sync_schema` (#236).**
  Same check as before; the committed `schema/curator-wire.schema.json` stays
  the source of truth for CI.
- Hashing goes through `argus_cortex.hashing.sha256_hex` where a local helper
  used to live.

### Fixed

Nightly ai-code-auditor packs on `dev` (#234, #237), plus small follow-ups:

- **Serve / CLI.** `serve` defaults to loopback (above). `scan` exits 2 when
  more than one of `--copy-to` / `--move-to` / `--symlink-to` is given. A
  `--face-reference` counts as in-scan only on the image's exact rel path.
- **Export.** Move mode unlinks a stale symlink at the destination before
  moving. Manifests (image and speech), transcript sidecars, CSV reports,
  `run_stats.json` and pseudonymised `<stem>.json` sidecars are written
  atomically (`O_EXCL|O_NOFOLLOW` temp, then `os.replace`), so a symlink at
  the destination or at the temp name is never followed. A failed seed copy
  leaves no partial sidecar. CSV report cells that look like a spreadsheet
  formula get a leading `'`, including a formula that sits behind whitespace.
  The speech report gets the same treatment. A named `selection` with `face_source` now applies the
  face-source gate (argus-quarry's `likeness_consent: false`), and a consented
  `face_source` with a named list is refused at export even without request
  validation. A copy or move whose source is itself a symlink, landing on that
  same entry, no longer unlinks the source.
- **Server / store.** The under-18 blocklist fails closed on an unparseable
  scan file (while that file sits in the cache, `/image` is 403 and `/thumb`
  blurs everything until it is fixed or purged), and keys its cache on
  device, inode, size, `st_mtime_ns` and `st_ctime_ns`. A read error fails
  that request closed and is not cached, so it does not stick after the file
  is readable again. One unreadable child no longer fails
  `/folders`. `/upload` creates each file with `O_EXCL|O_NOFOLLOW`.
  `POST /faces` refuses a chunked multipart body with no `Content-Length`
  (411) instead of spooling it uncapped. `ScanStore.load` and `list_meta`
  survive a concurrent delete.
- **Models.** `max_workers` is 1–64; `weight_*` fields are ≥ 0;
  `diversity_weight` is 0–1. The wire schema is regenerated.
- **Guard / guard_eval.** Untagged Ollama names compare as `:latest`.
  `beyond_noise` needs the two arms to disagree. An append after a cut-short
  cache line starts on a new line. An empty or single-run back-to-back cache
  check draws no false "miss" conclusion. A missing cortex policy text fails
  when that policy is requested, not on import.
- **Vision.** Body visible fraction is measured from the mask (dark pixels no
  longer count as masked). A NaN puppet `Retry-After` falls back to the
  default. A stats item whose `image` isn't a string is skipped.
- **Speech.** The Silero guard mask keeps the input's length. The hash-check
  cache also keys on device, inode and ctime. A GPU-lease failure in the CER
  phase fails the scan instead of rejecting every clip. An unmeasured SNR
  names the missing speech frames. Unlabelled clips no longer set the
  `balance_accents` cap.
- **Tests.** Manifest compatibility follows `$ref` chains and keywords beside
  a `$ref`, and walks `oneOf` / `allOf` / `prefixItems` / `not` / `if` / `then`
  / `else` / `contains`. Assertions that could never fail now can. Tests no
  longer hard-code LAN addresses (#235).

## [0.21.0] - 2026-11-21

**For downstream callers (argus-lens, argus-forge, argus-proof, argus-mint, and
anything that drives `POST /export`).** One behaviour change, and nothing else
breaks:

- **A consented voice export over HTTP now needs `speech.second_voice` on at
  scan time, or `POST /export` answers 400.** Under `speech.consent_record`
  the export checks every clip for a second voice, from windows the scan made
  and held in memory. Turn `second_voice` on in the scan's `SpeechConfig`
  (it needs `speakers` on, as a consented export already does). A server
  restart between the scan and the export also gives a 400, so rescan. The
  CLI's `--voice-consent` turns `second_voice` on automatically.
- **The default threshold is calibrated on synthetic audio only.**
  `speech.second_voice_max_distance` defaults to 0.40, set on synthetic
  voices; the A/B against real recordings is #228. Expect a consented export
  to keep fewer clips: clips flagged `consent:second-voice` or
  `consent:no-voice-windows` are left out.
- **`MANIFEST_VERSION` stays 2.7 and `SPEECH_MANIFEST_VERSION` stays 1.0.**
  Neither manifest row changed; the new `second_voice_distance` reading goes to
  `SelectRow` and `curation_report.csv`.
- No new extra, model, or GPU lease kind: the second-voice windows run under
  the existing `speakers` lease. argus-cortex stays `>=0.13,<0.14`.

### Added

- **A consented voice export refuses clips with a second voice in them
  (#217).** With `speech.second_voice` on (the CLI's `--voice-consent` turns it
  on), the scan embeds each clip's speech, as marked by the scan's VAD, in
  1.5 s windows every 0.75 s. They run in fixed-shape passes of 8 under the
  `speakers` lease, and are held in memory only. Under `speech.consent_record`,
  each window is measured against the centroid of the reference recordings'
  own windows. A clip whose farthest window is past
  `speech.second_voice_max_distance` (default 0.40) is left out as
  `consent:second-voice`. The distance is a plain number:
  `second_voice_distance` in `curation_report.csv` and on `SelectRow`. A
  consented export refuses (400) when the check can't run: no
  `second_voice` on the scan, a restart since the scan, or references with no
  windows. A clip with no windows is left out as `consent:no-voice-windows`.
  The default was calibrated on **synthetic** voices only. It reliably catches
  a second voice of 2 s or more, catches 92% at 1 s, and flags 13% of the
  subject's clean clips (docs/speech.md). The A/B against real recordings is
  #228. New CLI flag: `--second-voice-max-distance`. `SpeechManifestRow` is
  unchanged, so `SPEECH_MANIFEST_VERSION` stays 1.0.

## [0.20.0] - 2026-11-21

**For downstream consumers (argus-lens, argus-forge, argus-proof, argus-mint).**
Every change below is additive, so a 2.x image-manifest reader and a 1.x
speech-manifest reader keep working, but:

- **`MANIFEST_VERSION` 2.6 → 2.7.** Image manifest rows gain `pose_labels`
  (the primary person's `stance` / `arms` / `facing`) and `pose_cluster` (the
  scan-level body-pose cluster id), both null unless the scan ran the body-pose
  stage against argus-puppet. No keypoints are written.
- **`SPEECH_MANIFEST_VERSION` stays 1.0.** `SpeechManifestRow` is unchanged by
  the CLI export, Silero VAD, language ID and the voice-consent binding: their
  readings go to the scan rows and `curation_report.csv`, not the manifest. A
  consented voice export may now keep fewer clips, since a clip must match the
  consenting voice itself, not its cluster (#209).
- **New extra `[lid-convert]`** (torch, speechbrain, huggingface_hub, onnx) for
  the one-time `argus-curator lid-convert`; the language-ID check itself runs
  on onnxruntime under `[speech]`.
- **New optional models**, each needed only when asked for: Silero VAD v6.2.3
  (`argus-curator vad-fetch`, MIT, `ARGUS_CURATOR_VAD_MODEL`) and SpeechBrain
  VoxLingua107 ECAPA-TDNN (`argus-curator lid-convert`, Apache-2.0,
  `ARGUS_CURATOR_LID_MODEL`).
- **A new GPU lease kind, `language`** (912 MB in `gpu-tenant.yaml`). Silero
  runs on CPU and takes no lease.
- **Optional argus-puppet.** Setting `ARGUS_CURATOR_PUPPET_URL` turns the
  body-pose stage on for every scan, like NudeNet; a down puppet never fails a
  scan.
- argus-cortex stays `>=0.13,<0.14`.

### Added

- **Speech export from the CLI (#207).** `scan --modality speech` takes
  `--copy-to`, `--move-to` and `--symlink-to`, with the selection as flags
  (`--speaker-id`, `--accent`, `--language`, `--exclude-speaker-mismatch`,
  `--min-hours-per-speaker`, `--max-hours-per-speaker`,
  `--max-hours-per-accent`, `--balance-accents`, and the shared `--min-score`,
  `--keep-similar`, `--max-keep`, `--run-stats`), a speaker-disjoint split
  (`--held-out` or `--held-out-fraction`, `--split-method`, `--split-seed`,
  `--split-part`) and the provenance (`--corpus` and `--license`, or
  `--voice-consent` and `--consent-delivery`). It builds the same
  `ExportRequest` and runs it through `export_selection`, so it writes the
  `speech_manifest.jsonl` and transcripts `POST /export` writes. A missing or
  half-given provenance, an image filter, or a speech flag on an image scan is
  refused before a clip is read.
- **Silero VAD for speech scans (#204).** `SpeechConfig.vad: "silero"` (CLI
  `--vad silero`, model from `argus-curator vad-fetch`: Silero VAD v6.2.3,
  MIT, pinned by commit and sha256, `ARGUS_CURATOR_VAD_MODEL`) drives the
  speech ratio and the SNR split instead of the energy VAD, which stays the
  default. Rows record the VAD in `ClipResult.vad`, and a clip Silero couldn't
  read gets `vad_error` and no ratio or SNR (never an energy fallback). Silero
  runs on CPU alone, one thread per session, in fixed-shape 32 ms passes, and
  takes no GPU lease (it was ~10x slower per pass on CUDA). On 49 real takes
  mixed with noise its speech ratio held within 0.02 down to 0 dB SNR where
  the energy VAD's fell to 0, and its SNR tracked the mixed SNR within 0.5 dB
  (docs/speech.md). `GET /detectors` adds `silero_vad`. Speech manifest rows
  are unchanged.
- **Language-ID check of speech corpus labels (#208).**
  `speech.language_id` (CLI `--language-id`) classifies each passing clip with
  SpeechBrain's VoxLingua107 ECAPA-TDNN (Apache-2.0, 107 languages), and
  flags a clip whose corpus `language` label the model rules out
  (probability under `language_dispute_below`, default 0.001). Clips gain
  `language_predicted`, `language_confidence`, `language_label_confidence`,
  `language_mismatch` and `language_error`, and `ScanSummary` gains
  `language_mismatches`. The label is never changed, and nothing is
  rejected unless asked: `speech.exclude_language_mismatch` (CLI
  `--exclude-language-mismatch`) drops flagged clips at select and export
  (`language-mismatch`), and `reject_language_mismatch` turns a dispute into
  the `language` prefilter. `argus-curator lid-convert`
  (`[lid-convert]` extra, torch at conversion only) exports the pinned
  checkpoints to ONNX with a conversion record that is checked at load. It
  runs under the GPU lease as the kind `language`, in 3 fixed windows of 6 s
  (810 MB at peak on a 3090), and `GET /detectors` reports
  `language_id_model`. `curation_report.csv` adds `language_predicted`,
  `language_confidence` and `language_mismatch`. The speech manifest row is
  unchanged. Calibrated on 272 real clips in 14 languages (`docs/speech.md`).
  The accent half is left out, because no credible permissive model covers
  it (#223).
- **Body pose from argus-puppet (manifest 2.7, #221).** With
  `ARGUS_CURATOR_PUPPET_URL` set, a scan sends each passing image to
  argus-puppet's `POST /pose` after the faces (the primary face as the
  `face=x,y,w,h` hint, turned into the EXIF-upright frame;
  `X-Argus-Priority: batch`; a 503 retried after `Retry-After`, at most 3
  times), then clusters the scan with one inline `POST /stats`. Rows gain
  `pose_check` (`ok`, `unusable`, `no_person`, `failed`; null = not checked),
  `pose_labels` (the primary person's `stance` / `arms` / `facing`) and
  `pose_cluster` (the scan-level cluster id); `ScanSummary.pose` holds the
  cluster sizes and medoids, `largest_cluster_share`, `diversity`, the usable
  count and, when the scan was not clustered, why (`skipped`: over
  `ARGUS_CURATOR_PUPPET_MAX_SET` / `_MAX_JSON_BYTES`, refused, failed; never
  truncated). Clusters span the whole scan; a UI counts a selection's share of
  its most common cluster itself. On whenever the URL is set, like NudeNet:
  `pose.enabled` (`--body-pose/--no-body-pose`, `--puppet-url`) opts out or requires it,
  and `GET /detectors` reports `puppet`. It never fails a scan: a failed image
  keeps null, and after 8 failures with no success puppet is taken to be down.
  No keypoints are stored anywhere. Manifest rows (2.7) carry `pose_labels`
  and `pose_cluster`; the CSV report adds `pose_check`, `pose_stance`,
  `pose_arms`, `pose_facing` and `pose_cluster`. See `docs/body-pose.md`.

### Changed

- **A consented voice export binds by voice, not by cluster (#209).** Under
  `speech.consent_record`, a clip is kept only when its speaker embedding is
  within `speech.consent_max_distance` (cosine, default 0.25) of the centroid
  of the record's reference recordings, whatever speaker cluster it joined
  (`consent:other-speaker`; no embedding, `consent:no-embedding`). The
  distance is reported per clip (`consent_distance` in the CSV report, and
  `SelectRow.consent_distance`); the CLI sets the ceiling with
  `--voice-consent-max-distance`. Select and export refuse when no reference
  recording has an embedding held (a scan without `speakers`, or a restart
  since the scan), and never fall back to the clusters. The centroid is a
  voiceprint and is never stored or served. `SpeechManifestRow` and
  `SPEECH_MANIFEST_VERSION` (1.0) are unchanged. A clip with a second voice
  in it isn't refused yet: the cheap signal measured too weak
  ([#217](https://github.com/smk762/argus-curator/issues/217); see
  docs/speech.md).

## [0.19.0] - 2026-11-21

**For downstream consumers (argus-lens, argus-forge, argus-proof, argus-mint).**
Every manifest change below is additive, so a 2.x reader keeps working, but:

- **`MANIFEST_VERSION` 2.3 → 2.6.** 2.4 adds `split` on every row (`train` /
  `held_out`, or null when no split was asked for). 2.5 adds
  `target_profile.modality`, always `image`. 2.6 adds `consent_release_sha256`,
  `consent_record_sha256` and `provenance` (argus-cortex's `Provenance`) on rows
  of a consented face set.
- **Provenance names no one.** `provenance.subject` is argus-cortex 0.13's subject
  ref, `consent:<release_sha256>`, never the consenting person's name, and the
  default `source` no longer names who collected the release. The same holds
  for speech rows and for carried `<stem>.json` sidecars.
- **Every manifest is in rank order**, best first, each kept row with its
  1-based `rank`. Exports without `face_source` used to be in scan order.
  Consumers that ranked or split the set themselves (argus-mint) can take the
  order and `split` as given.
- **A new speech manifest.** A speech export writes `speech_manifest.jsonl`
  (`SpeechManifestRow`, `SPEECH_MANIFEST_VERSION` 1.0, snapshot
  `schema/speech-manifest-row.snapshot.json`) beside each clip's `<stem>.txt`
  transcript. It is a separate file with its own version; image exports and
  `manifest.jsonl` are unaffected. argus-forge reads it (argus-forge#58).
- **argus-cortex `>=0.13,<0.14`** (was `>=0.8,<0.9`); install the same cortex
  minor alongside.

### Added

- **`/scan/{id}/select` ranks, caps and splits (#183).** Select and export now
  run one `selection.plan_selection`, so they can't disagree. Kept rows come
  first, best first, each with its 1-based `rank`; `rank` picks the order
  (`"score"`, or `"identity"`: closest face first, by reference or cluster
  distance). `max_keep`, `diversity_weight`, `diversify_scenes` and
  `diversify_outfits` now apply to select too. An optional `split`
  (`held_out` or `held_out_fraction`, `group_by` `image` | `face_cluster`,
  `method` `spread` | `random`, `seed`) gives a deterministic train/held-out
  split in which no group spans both parts; select reports both parts, and an
  export transfers `split_part`. The group key is a table
  (`selection.SPLIT_GROUP_KEYS`) over a generic `split_held_out`, so other
  kinds of item can reuse it. This replaces argus-mint's own ranking and split.
- **Manifest 2.4: rows carry `split`** (`train` / `held_out`, or null). Additive.
- **Manifest 2.5: `target_profile` carries `modality`**, always `image`. Additive.
- **A caller-supplied identity reference (#130).** `faces.reference_embedding`
  takes a centroid instead of `faces.reference_paths`, and
  `faces.reference_embedding()` builds one from images outside the set (a
  consenting person's reference photos, #133); `faces.reference_distance()`
  compares any embedding with it. `--face-reference` accepts a file outside the
  scanned folder. The centroid is never stored: `ScanSummary` drops it from
  `faces_config`.
- **Consented face sets of a real person (#133).** A scan can read a consent
  record (`ScanRequest.consent`: `ConsentConfig` with `record`, a path under
  the scanned folder, and `max_distance`; CLI `--consent`,
  `--consent-max-distance`). It checks the record with argus-cortex's
  `consent_valid(use="avatar", likeness="face")`, and `ScanSummary.consent`
  (`ConsentCheck`) says whether the set is eligible and, if not, why. The
  record's reference images, found in the scan by sha256, become the identity
  reference, and each image gets `ImageResult.consent` (`ConsentBinding`:
  `bound`, `distance`, `reason`): an image whose primary face doesn't match,
  or that shows anyone else, is not bound. `face_source.consent` (CLI
  `--face-source-consent`) exports only bound images, re-reads and re-checks
  the record at select and export (so one revoked since the scan blocks the
  export), needs an adult set age, refuses the set when any bound face reads
  under 18, and refuses when nothing qualifies. Nothing about the consent
  outlives the scan: a purge removes it with the scan.
- **Face-source selections refuse argus-quarry's not-consented images.** Every
  `face_source` selection re-reads each image's `<stem>.json` and keeps out one
  marked `likeness_consent: false` (`face-source:not-consented`).
- **Manifest 2.6: consented face sets carry `consent_release_sha256`,
  `consent_record_sha256` and `provenance`** (argus-cortex's `Provenance`).
  Additive. The export never names the consenting person: `provenance.subject`
  is argus-cortex 0.13's subject ref, `consent:<release_sha256>`, applied with
  `pseudonymise` to both the default and a sidecar's provenance. The default
  `source` no longer names who collected the release. With `carry_sidecars`, a
  carried `<stem>.json` is a copy with its provenance pseudonymised, under every
  mode, and the source sidecar is never changed (under `move` it stays put).
- **Semantic novelty in selection (#188, item 1).** An opt-in scan phase,
  `ScanRequest.embeddings` (`EmbeddingConfig`: `enabled`, `device`; CLI
  `--embeddings`), runs the image tower of DataComp-1B's CLIP ViT-L/14 (MIT)
  under onnxruntime and the GPU lease (kind `embeddings`, batches of 8, its own
  capped 2 GB arena). `diversify_semantic` on Select/ExportRequest (CLI
  `--diversify-semantic`) adds a fourth novelty term to the `max_keep` cap: one
  minus the largest cosine to anything selected, with an image that has no
  embedding counted as a repeat. The vectors are never written: they live in
  memory for the scan, are dropped with it (or its rows), and are gone on a
  restart, after which `diversify_semantic` is refused (400). Convert the model
  once with `argus-curator embeddings-convert` (`[embeddings-convert]`).
  `ManifestRow` is unchanged.
- **Semantic near-duplicates (#196).** With embeddings on, `ScanConfig.semantic_duplicates`
  (CLI `--semantic-dupes`) also groups images whose CLIP embeddings reach
  `semantic_duplicate_cosine` (default 0.95, `--semantic-dupes-cosine`): mirrors,
  small crops and rescans that pHash misses. One leader grouping over both
  tests, so the keep-best rule and the `similar-to:` keep reason are unchanged,
  and a scan without it groups exactly as before. `ImageResult.duplicate_kind`
  (`phash`, `semantic` or `both`; also a CSV report column) says which test
  linked the image to `duplicate_of`. Asking without embeddings is refused.
  The vectors stay memory-only; a regroup after a restart keeps the recorded
  semantic links, and `ScanSummary.semantic_groups_stale` says when a dropped
  representative left one that could not be regrouped. `ManifestRow` is
  unchanged.
- **Likeness: a consistency measure for subjects without a face (#191).** A
  scan with `embeddings` on records each passing image's `likeness_distance`
  (`ImageResult`): the cosine distance from its CLIP embedding to the subject,
  which is the mean of `embeddings.reference_paths` (images in the scan; a
  rejected one is embedded from the scan's bytes), a caller-supplied
  `embeddings.reference_embedding` (request-only: the stored and served summary
  drops it, as it does a face `reference_embedding`), or else the set's own
  trimmed centroid. `ScanSummary.likeness_source` / `likeness_skipped` say
  which, or why none was measured. Only the distances persist, so select and
  export need no held vectors and survive a restart.
  - `rank: "likeness"` (closest first) and `max_likeness_distance` (keep
    reason `likeness-distance`) on `/scan/{id}/select` and `/export`, both 400
    on a scan that measured no distances.
  - `exclude_references` keeps the scan's reference images (face and likeness)
    out of the selection (keep reason `reference`), so argus-mint's anchors
    take no place in a cap or a split; 400 on a scan that named none.
  - `split.group_by: "similar_group"` keeps an image's near-duplicates in its
    part.
  - `ManifestRow` and `MANIFEST_VERSION` are unchanged.
- **Zero-shot profile fit and junk checks (#197).** With embeddings on,
  `ScanConfig.profile_fit` (CLI `--profile-fit`) scores how well each image
  reads as the kind the `TargetProfile` asks for, and `ScanConfig.junk_check`
  (`--junk-check`) the probability that it is a screenshot, collage, page of
  text or meme, against text prompts in the same CLIP space. The scores land on
  `ImageResult.zero_shot` (`profile_fit`, `junk`, and `junk_kind` once `junk`
  reaches `junk_threshold`, default 0.8; also CSV report columns) and feed the
  target bonus (`profile_fit_weight` 0.10 up, `junk_weight` 0.15 down; listed as
  `zero_shot_*` in `score_breakdown`). Neither rejects an image: the opt-in
  `PrefilterConfig.drop_junk` (`--drop-junk`) drops flagged junk as
  `prefilter:junk` before SingGuard. Each is off by default, and asking for any
  without embeddings is refused. The prompt vectors are computed offline by the
  pinned checkpoint's text tower (`embeddings-convert --prompts`) and shipped as
  `data/clip_prompts.json`, pinned by sha256, so no text model or tokenizer runs
  at scan time. Only the scores are stored, never the vectors. `ManifestRow` is
  unchanged.
- **Speech scans (#186).** `POST /scan/folder` (and its stream) and
  `argus-curator scan --modality speech` score a folder of WAV/FLAC clips when
  `target_profile.modality` is `speech`. `SpeechTargetProfile` (cortex's
  `TargetProfile` narrowed to speech) is taken only on that path: the image
  scanner, `ManifestRow` and an image scan still refuse a speech profile, and a
  speech request that turns on an image check is a 422. Transcripts come from a
  `.txt` sidecar, LJSpeech's `metadata.csv` or Common Voice TSVs (speaker,
  accent, locale and age carried verbatim). Each clip is read through
  `argus_cortex.limits.open_audio` and gets duration, sample rate, channels,
  clipping, leading/trailing silence, an SNR estimate, an energy-VAD speech
  ratio, its speaking rate, a 64-bit spectrogram fingerprint for
  near-duplicates and, with `speech.mos`, DNSMOS P.835 (pinned by commit and
  sha256; `argus-curator dnsmos-fetch`; GPU kind `mos` in `gpu-tenant.yaml`).
  Every threshold lives in `SpeechConfig` and every rejected clip names its
  check (`reject_reason`, `failed_checks`, `reject_detail`). A clip the
  corpus marks as under 18 is rejected undecoded. A `Transcriber` hook takes
  an STT engine for CER (argus-echo#4, #203); without one `max_cer` is
  refused. Rows are `ScanSummary.clips`; select and export take them through
  the speech export below (#187). New extra `[speech]`.
- **Speech export (#187).** `POST /scan/{id}/select` and `POST /export` take a
  speech scan, through `ExportRequest.speech` / `SelectRequest.speech`
  (`SpeechSelection`).
  - **Speakers:** `SpeechConfig.speakers` (CLI `--speakers`;
    `argus-curator speaker-fetch`) embeds each passing clip with WeSpeaker's
    ECAPA-TDNN (VoxCeleb2, CC BY 4.0; pinned by Hugging Face revision and
    sha256), ONNX under the GPU lease (kind `speakers` in `gpu-tenant.yaml`)
    with the capped arena, one fixed-shape batch of 4 windows of 3 s per clip.
    The clips are clustered into speakers (`ClipResult.speaker_cluster`,
    `ScanSummary.speaker_clusters`), and where the corpus names speakers the
    clips its labels and the clusters disagree on are flagged
    (`speaker_mismatch`). The embeddings stay in memory and go with the scan.
  - **Selection:** speaker, accent and language filters, hours per speaker
    (floor and cap) and per accent, accent balance, and a `max_keep` that takes
    speakers in turn. `SelectResult.speech` reports the hours by speaker,
    accent and split part.
  - **A speaker-disjoint split:** `split.group_by: "speaker"` (required on a
    speech scan). A corpus speaker label and a speaker cluster that share a
    clip are one group, so neither spans both parts; stratified by accent and
    gender. `selection.split_held_out` gains an optional `strata`.
  - **Export:** each clip with a `<stem>.txt` transcript beside it, and
    `speech_manifest.jsonl`: `SpeechManifestRow`, its own row shape and version
    (`SPEECH_MANIFEST_VERSION` 1.0, snapshot
    `schema/speech-manifest-row.snapshot.json`, checked by
    `make manifest-snapshot`). The image `ManifestRow` and `MANIFEST_VERSION`
    are unchanged.
  - **Provenance:** an export names the licensed corpus (`speech.corpus`,
    `speech.license`) or a cortex `ConsentRecord` for a real person's voice
    (`speech.consent_record`, checked for `training` of the `voice` at
    `speech.consent_delivery`, bound to its reference recordings' speaker
    clusters). A consented row's provenance names no one: `subject` is
    cortex's subject ref `consent:<release_sha256>` and `source` leaves out
    who collected the release. The corpus's `gender` label is carried with
    `speaker`, `accent`, `language` and `age`; age is never estimated.

### Changed

- **Requires argus-cortex 0.13 (`>=0.13,<0.14`, all pins; was `>=0.8,<0.9`)**
  for its consent subject ref (`subject_ref`, `pseudonymise`). Cortex's
  `TargetProfile` gained `modality` and speech categories; curator's
  `TargetProfile` is now a subclass narrowed to `modality="image"` and the
  image `TargetCategory`, so a speech profile is refused on the image path (a
  422 over HTTP, a `ValidationError` from `scan_folder`) instead of reaching
  the face-count rules, and the manifest's `target_category` enum stays
  image-only. A cortex image profile passed to `scan_folder` converts. EraX
  and the watermark client use cortex's `moderation._http.is_number` /
  `multipart_image` (moved from `nudenet._is_number` / `_multipart` in cortex
  0.11).
- **Kept images are always ranked.** Exports without `face_source` used to
  write the manifest in scan order (or the diversity pick's order with
  `max_keep`); every manifest is now in `rank` order.
- **`guard_eval.py` runs are repeatable and its reports weigh noise (#113).**
  Expectations moved out of the script into `eval/guard_cases.toml`: per
  bucket and policy, a correct verdict is `flag`, `pass` or `either`, an
  `[[override]]` carves out one case or a pattern (with a reason), and a
  stable held-out share keeps wording from being tuned on its own test set
  (`--split tune|heldout|all`, `tune` by default). Each `--repeat` is a whole
  pass in a shuffled order, interleaving cases and policies. `--cache FILE`
  keys every verdict by the bytes sent, policy text, context, mode, model
  digest and pass number, so a dead run resumes; the digest comes from
  `--model-digest`, Ollama's `/api/tags` or the guard server's `/health`, and
  a run stops if it changes between passes. Connection failures are retried,
  and a server that stays down stops the run. Text probes go through
  `check_text`, never beside a grey image; the `folder` context kind is
  refused; frames sent with no context are counted apart. A run with
  `--scan-json` first lists frames to review (anything `minor_exposed`
  catches, exposed readings in `safe`) and waits for `--preflight-ok`. A
  candidate whose rule letters skip one is refused. The report scores frames
  (a split counts as flagged), gives per-frame noise and per-subfolder tables,
  lists frames that keep splitting, marks each change between a candidate and
  its parent as within or beyond noise, and exits 1 when a candidate passes a
  `hard` frame its parent flagged. `--report ROWS` reprints it from a saved
  run. `--prompt-cache-check` measures Ollama's prompt cache straight against
  Ollama.

### Fixed

- **GPU memory stays near what curator's leases ask for (#189).** ONNX Runtime's
  default CUDA arena grows in power-of-two steps with no ceiling and keeps what
  it takes, and MiVOLO and FairFace ran every face in an image as one batch, so
  each new face count was a new, larger shape. One scan held ~20 GB while each
  lease asked for 512 MB, and its own next lease then waited on that memory.
  Every CUDA session (curator's own and InsightFace's) is now built with
  `arena_extend_strategy=kSameAsRequested` and a per-session `gpu_mem_limit`
  (`ARGUS_CURATOR_GPU_MEM_LIMIT_MB`, default 1024), and the attribute models
  run one face per pass (`runtime.ATTRIBUTE_BATCH`), so a crowd shot neither
  grows the arena nor fails against the cap. Measured on a 3090 over 600
  images, the faces check now peaks at ~1.5 GB (it was ~2.2 GB on those images)
  and runs about as fast. `runtime.VRAM_MB` and `gpu-tenant.yaml` carry the
  measured footprints (`faces` 1600 MB, `garments` 1268, `scenes` 656).
- **A lease reports `peak_vram_mb` on release.** It is the VRAM the process
  holds then (NVML), which is its high-water mark because the arena keeps what
  it reserves. NVML lists host PIDs, so in a container the figure is only
  found with `pid: host`. Otherwise nothing is reported, as before.
- **A rejected reference image is still a reference.** A `reference_paths`
  image the scan's filters rejected (often a studio portrait failing the blur
  check) used to give no distances; it is now embedded from the bytes the scan
  read, and its row stays rejected.

## [0.18.0] - 2026-11-19

### Added

- **A scan can pick its SingGuard mode (#180).** `guard.thinking` (CLI
  `--guard-thinking`) takes `fast`, `fast-slow`, `slow` or `cascade`; left out
  or null, the server's `ARGUS_CURATOR_SINGGUARD_THINKING` applies as before.
  `guard_config.thinking` records the mode the scan ran in, whichever set it,
  and `fell_back` keeps its meaning. `ARGUS_CURATOR_SINGGUARD_THINKING_ALLOWED`
  (comma-separated, default every mode) limits the choice: a mode outside it is
  refused with 422 naming the allowed ones, not downgraded. `GET /detectors`
  adds `singguard_thinking_modes` and `singguard_thinking_default`, and
  `GET /health` adds `guard_thinking: true`. The wire shape is unchanged; the
  `GuardConfig` description now says a request may set `thinking`.

### Changed

- **`caption_url` reaches only a configured host** (decided by the user, PR #179).
  The hand-off used to reach any public host, and an internal one listed in
  `CURATOR_CAPTION_ALLOWED_HOSTS`. Now only a listed host is sent the manifest,
  public or not (a bare `host` entry allows any port, `host:port` that port), and
  with the variable empty an export naming a `caption_url` is refused with 400
  naming it. The CLI and engine refuse the same hosts (the export still runs,
  uncaptioned). argus-studio sends no `caption_url`, so it is unaffected; a
  deployment handing off to a public lens must now list it.

- **argus-cortex 0.8** (ai-code-auditor#56). Every `argus-cortex` requirement,
  extras included, is now `>=0.8,<0.9`, and curator adopts its shared `net` and
  `limits` helpers.
- **Uploads are capped (`CURATOR_MAX_UPLOAD_MB`, default 50).** `POST /upload`
  and a multipart `POST /faces` read each file through `limits.read_upload`, and
  a raw `POST /faces` body through `limits.read_body` (`POST /faces` keeps its
  64 MiB ceiling). A larger file gets 413 (`POST /upload` saves none of the batch), a
  malformed `Content-Length` 400. A `CURATOR_MAX_UPLOAD_MB` that is not a
  positive number stops the server starting.
- **The export's `caption_url` hand-off uses argus-cortex 0.8's `net.fetch`**
  (`CURATOR_CAPTION_ALLOWED_HOSTS` is comma-separated `host` or `host:port`, the
  same shape as argus-lens's `LENS_URL_ALLOWED_HOSTS`; empty by default). An
  export whose `caption_url` names an unlisted host is refused with 400 before
  anything is transferred (see the first entry above). A POST is no longer sent
  on to a redirect, and the `caption_handoff_failed` log names the host, not the URL.
- **The bundled NudeNet is argus-retina's 640m image** (#156). `docker compose`
  ran the upstream `ghcr.io/notai-tech/nudenet` (320n); the `nudenet` service is
  now `ghcr.io/smk762/argus-retina-nudenet:0.1.0`, pinned by digest, the same
  image as the suite's shared server. NudeNet stays on by default. **640m scores
  higher than 320n**: on a 300-image sample, the 0.5 default dropped 153 images
  instead of 133, which is why 0.2 is now the recommended threshold (below); see
  `docs/nudity.md`. `.env.example` and the docs point at retina's shared ports
  (NudeNet :8130, EraX :8131, watermark :8132).
- **0.2 is the recommended NudeNet exposure threshold** (#156). The new
  `DEFAULT_NSFW_THRESHOLD` (0.2, any exposure NudeNet reports, darkroom's
  default) replaces 0.5 in the CLI scan summary's exposed count, the
  `--nsfw-threshold` help and `docs/nudity.md`. On 640m a 0.5 cutoff let
  through exposed images scored 0.27–0.49, and in a hand review 640m was right
  on 30 of the 38 images it and 320n disagreed on. Curator still filters
  nothing unless a request sets a threshold.

### Fixed

- **A negative `sharpness_ref` / `resolution_ref` is refused**
  (pack-argus-curator-730686a4ee). It inverted the scan's ranking; `ScanConfig`
  now requires both to be at least 0 (0 already turned the term off, and still
  does). The wire schema carries the new minimum.
- **One unreadable folder no longer breaks `GET /folders`**
  (pack-argus-curator-f82fcf9861). A child that can't be listed (`lost+found`,
  a dead mount) is left out and logged; its siblings are still listed.
- **A delete no longer resurrects rows removed while it ran**
  (pack-argus-curator-c80e146d60). `POST /scan/{id}/delete` saved the copy it
  loaded before unlinking; it now re-reads the scan and drops its rows from the
  current copy, under a lock it shares with a move export's drop.
- **Two saves of one scan can't corrupt it** (pack-argus-curator-af0f507d0f).
  Every save writes its own `<id>.<random>.json.tmp` before the atomic replace,
  so an overlapping save no longer moves another's half-written file into place.
- **A symlink export can't delete its source** (pack-argus-curator-38e6c3ff70).
  When the destination already was the source file (an export into the source
  folder, or through a parent symlink), `mode="symlink"` unlinked the original
  before linking to it. That file is now skipped, as `mode="copy"` skips it.
- **Carried sidecars no longer overwrite each other in a structured export**
  (pack-argus-curator-28f17d184f). With `preserve_structure` and
  `carry_sidecars`, `a/IMG.jpg` and `a/IMG.png` both carried to `a/IMG.txt`.
  Names that share a stem (or differ only in case or Unicode form) in one
  folder now get the hash suffix a flattened export gives them, and stay in
  their folder; `exported_path` in the manifest names the result.
- **`overlap` mode counts a young face's own person as young**
  (pack-argus-curator-7044e38d14). When a MiVOLO face age under 18 was matched
  to a person whose body read adult (or had no age), an exposure on that body
  counted as "only adults" and the image passed the minor-exposure guard. The
  younger reading now wins. The default `frame` rule was not affected.
- **`guard_policies` alone is refused on a scan that did not run SingGuard**
  (pack-argus-curator-67d5f2fc89), as `guard_unsafe` and `guard_rules` already were.
- **Images without a pHash no longer win the diversity term**
  (pack-argus-curator-705607f144). An unhashed candidate counted as the widest
  possible spread, so a capped selection filled with them; it now counts as a
  repeat, as an untagged scene or outfit does.
- **A face no longer loses its body to two weaker pairs**
  (pack-argus-curator-79d93c84fc). The face-to-body assignment was solved on
  every score and the `body_face_match` threshold applied after, so two
  sub-threshold cross pairs could outbid one valid match and leave that face
  face-only. Pairs at or under the threshold no longer weigh in the solve.
- **A body crop running past the image edge keeps its pixels**
  (pack-argus-curator-973d330c33). A face or person box that met the body only
  in its off-image part gave a negative slice stop, which blacked out most of
  the crop. Mask offsets are now clamped at both ends, and an empty one masks nothing.
- **A NudeNet detection goes to the person holding it**
  (pack-argus-curator-28e56cc0b1). The per-person `nsfw_labels` /
  `exposed_score` went to the highest IoU, which favours a small box the
  detection only grazes over a close-up person it lies wholly inside. It now
  goes to the box covering most of the detection (IoU breaks a tie). The
  minor-exposure guard never read this assignment: under `overlap` it tests each
  detection against every young box, so no image was let through by it.
- **`argus-curator scan` with SingGuard and a destination finishes its report**
  (pack-argus-curator-c7365a3eb7). The SingGuard summary line overwrote the
  export mode, so after the files were transferred the "Copied N images" line
  raised `KeyError` and the command exited 1 without reporting the manifest.
- **InsightFace's CUDA sessions are built under the GPU lease**
  (pack-argus-curator-00f89b68b4). `FaceAnalysis()` builds every onnxruntime
  session, so constructing it before taking the lease put the face models on
  the card while another tenant held it; only `prepare()` (which sets sizes)
  was leased. The pack is now fetched first, off the lease, and the app is
  built and prepared under it.
- **Garment boxes are plain floats** (pack-argus-curator-7d52f097b6). An edge
  the frame did not clip stayed the model's `np.float32`, which JSON cannot encode.
- **SingGuard's caption context finds a dotted name's sidecars**
  (pack-argus-curator-33c37e7c9b). For `2021.05.03.jpg` it read `2021.05.txt` /
  `2021.05.xmp` (a sibling's caption, or none) instead of `2021.05.03.txt` /
  `2021.05.03.xmp`.
- **The `MANIFEST_VERSION` check counts a new enum value as breaking**
  (pack-argus-curator-1993a6d0f8). It treated an added value as additive, but a
  consumer that validates the field rejects a row carrying it (argus-forge reads
  `target_profile` into cortex's `TargetProfile` `Literal`s), so it now needs a
  major bump, as a lost value already did.
- **The `MANIFEST_VERSION` check compares every use of a `$ref` the new row
  inlined** (pack-argus-curator-e3c65ac0e4). Its cycle guard keyed an inlined
  definition on the old name alone, so only the first use was compared and a
  change at a later one passed unseen.
- **A scan no longer follows symlinks out of its folder**
  (pack-argus-curator-aebf19d832). A file or link whose resolved path lies
  outside the scanned folder is skipped, and symlinked directories are not
  descended into.
- **The under-18 exposed block on `/image` can't be stepped around**
  (pack-argus-curator-1d6b28208d). It was decided from the raw `path` against
  the named scan only, so omitting `scan_id`, `./x.jpg`, `sub/../x.jpg` or a
  symlink served the original. `/image` now resolves the file first and refuses
  it (403) when any cached scan reads it as under-18 exposed, and `/thumb` blurs
  such a frame server-side instead of sending it unblurred.
- **`ARGUS_CURATOR_ERAX_URL` and `ARGUS_CURATOR_WATERMARK_URL` reach the
  container** (#156). The compose passed neither into `argus-curator`, so
  setting them in `.env` did nothing and EraX and the watermark detector stayed
  off. Both stay off unless `.env` sets a URL (argus-retina's :8131 and :8132).

## [0.17.0] - 2026-11-16

### Added

- **`MANIFEST_VERSION` is held to `ManifestRow`** (#168).
  `schema/manifest-row.snapshot.json` records the row for the current version.
  A test fails on a breaking change (a field removed, no longer always sent, or
  retyped; an enum value lost) under the same major, and on any row change
  without a version bump. `make manifest-snapshot` refreshes the record and
  refuses both of those.
- **A test refuses local copies of argus-cortex helpers** (#169). A curator
  module that defines a name argus-cortex also defines fails, unless the test's
  allow-list says why it isn't a copy.
- **CI runs argus-ci's suite rules** (#167): new violations of the suite's
  lint catalogue fail the build; existing ones are recorded in
  `.suite-rules-baseline.json`, which may only shrink.

### Changed

- **Requires `argus-cortex>=0.7.0,<0.8`** (#156). 0.7.0 only adds features. It brings
  `NudeNetClient.info()`, which reports the NudeNet model a server runs (`640m` /
  `320n`). Curator needs that to re-tune `--nsfw-threshold` when it moves to the
  shared 640m server.
- **Importing NudeNet fails lint** (#165). NudeNet is AGPL-3.0 and stays a
  separate server behind `NudeNetClient`; ruff's `TID251` banned-api now refuses
  `import nudenet` in `src/` and `tests/`.
- **`TargetProfile` is argus-cortex's** (#169). Curator kept its own copy of
  `argus_cortex.taxonomy.TargetProfile`; it now re-exports cortex's. The fields,
  types and defaults are identical, so the wire schema changes only in the
  description text. `wire_schema()` and `argus-curator schema` build on
  cortex's `argus_cortex.wire` helpers, so the output is byte-for-byte the same.
- CLAUDE.md lists what must never happen and the test, CI job or lint that
  enforces each rule (#165). Each rule has a check.

## [0.16.0] - 2026-11-16

### Added

- **The SingGuard mode a scan asked for is recorded (#152).** `guard_config.thinking`
  holds the mode from `ARGUS_CURATOR_SINGGUARD_THINKING`, and
  `guard_config.fell_back` is `true` when the server refused it and the checks ran
  in `fast`. Each verdict's `thinking` still says which mode judged it. Both are
  `null` when SingGuard did not run, and in older scans. A request's values are
  replaced. The CLI summary's SingGuard line shows the mode. Wire schema updated.
- **The argus-iris GPU broker (#155).** With `ARGUS_GPU_COORDINATOR=broker` and
  `ARGUS_GPU_BROKER_URL`, curator asks iris for each GPU lease, as a kind
  declared in the new `gpu-tenant.yaml` (`faces`, `scenes`, `garments`).
  Scans ask `batch` priority, `POST /faces` asks `interactive`, and `POST /faces`'s 30 s wait cap now
  covers the broker and its fallback. A lease iris refuses (`GpuLeaseRefused`)
  gets the same 503 as a timeout. Compose and `.env.example` carry the
  `ARGUS_GPU_BROKER_*` settings. Until iris ships leases, every lease falls back to
  `ARGUS_GPU_BROKER_FALLBACK` (`lease`). See `docs/gpu.md`.
- **`--trusted-host` / `CURATOR_TRUSTED_HOSTS`** on `serve`: DNS names whose
  same-host browser writes the write-guard allows (see below).

### Changed

- **The under-18 rule now covers the whole frame by default.** `minor-exposed`
  used to flag an image only when the exposure overlapped a young person's box
  (or no box at all). It now flags any image where MiVOLO reads a face or body
  as under 18 and the exposed score is at least 0.2, wherever the two are: the
  same rule as argus-darkroom. **More images are excluded**, for example a
  group photo with a child in it and an exposed adult. The rule stays a hard
  exclusion that no request can override. Set
  `ARGUS_CURATOR_MINOR_EXPOSED=overlap` (or `--minor-exposed overlap` on `scan`
  and `serve`) to restore the old rule. An unknown value is refused (exit
  status 2, and the server does not start), never a fallback. `GET /health`
  reports the rule in use as `minor_exposed`. The wider rule also widens what
  `POST /scan/{id}/delete` removes without `--allow-delete`, and what
  `GET /image` refuses to serve.
- **Requires `argus-cortex>=0.6.0,<0.7`** (#155). The `[gpu]` and `[server]` extras
  add cortex's `[remote,yaml]` extras for the broker client and `gpu-tenant.yaml`
  (`/health` builds the GPU coordinator, so a CPU server set to `broker` needs them).
- **A same-host write by DNS name is refused unless trusted.** Cortex 0.6's
  write-guard no longer trusts a browser write whose `Origin` matches the `Host`
  by name alone, because under DNS rebinding both carry the attacker's name. IP
  addresses and `localhost` still pass. A UI proxied onto curator under a DNS name
  (`http://morpheus:8101`) now gets 403 on writes until that name is in
  `CURATOR_TRUSTED_HOSTS` (or `--trusted-host`). Origins on the
  `--cors-origin` allow-list are unaffected.

## [0.15.1] - 2026-11-14

### Documentation

- **The README is split into `docs/` (#138).** The README is now the overview
  (what curator is, install, quick start, why not FiftyOne/fastdup, the shared
  `TargetProfile`) with links to one page per feature, so Studio and other UIs
  can import and deep-link them: moderation, scoring, HTTP API, Docker and the
  GPU, face attributes, face sources, scene tags, garments, nudity, policy
  check, prefilter, crops, sidecars and XMP.
- **`docs/manifest.md`**: a reference for `manifest.jsonl`, with every 2.3
  field, the changes in 1.0 to 2.3, the compatibility rule (refuse an unknown
  major) and an example row.
- **`docs/cli.md`**, generated from the Typer app (`make docs`; `make
  docs-check` fails when it is stale). It covers `fairface-convert` and
  `schema`, which the README never did.
- FairFace (`--fairface`, `--race`, `fairface-convert`,
  `ARGUS_CURATOR_FAIRFACE_MODEL`) is documented, as are
  `ARGUS_CURATOR_WATERMARK_TIMEOUT` and `ARGUS_CURATOR_WATERMARK_WORKERS`. The
  face-source page says that with a reference the identity comes from the
  reference distance, and the synthetic-face guide's multi-view command adds
  `--blur-threshold 0`.

### Fixed

- **The blur check measured the frame's brightness on low-detail images
  (#147).** PIL's `FIND_EDGES` answers the 1 px image border with a response
  that grows with brightness, and `_sharpness` took the variance over the whole
  output, so a flat bright frame scored ~205 and a heavily blurred photo with a
  bright background passed `blur_threshold`. Only the interior is measured now,
  so a flat image of any brightness scores 0. **`sharpness` values change**
  (the wire field on images and on faces, and `primary_face.sharpness` in the
  manifest): blurred and flat images fall to near 0, and sharp ones drop by the
  border's share (typically tens of points, more on bright images). The
  `blur_threshold` (100) and `sharpness_ref` (800) defaults are unchanged.

## [0.15.0] - 2026-11-14

### Added

- **Face-source measurements and preset for argus-mask (#130).** Every face now
  carries its head `roll`, its own `sharpness` (over the face box), `eye_open`
  and `mouth_open` (from the 68 landmarks), and `cluster_distance` to its
  identity's centroid. `FaceConfig.reference_paths` (`--face-reference`) names
  images in the scan whose faces are the reference; each face then gets a
  `reference_distance`, and `reference_skipped` on the summary says why none
  was measured. Embeddings are not stored. The 5 keypoints and 68 landmarks are
  kept in the scan cache only. `face_source` on export and select
  (`--face-source`) is a preset over these measurements: one face of one
  identity, frontal and level, large and sharp, eyes open, mouth closed, close
  to the identity, and an adult. It ranks the kept images best first, and each
  image it leaves out gets `face-source:<check>` as its keep reason. The CSV
  report gains pitch, roll and the face-quality columns.
- **Manifest 2.3.** Rows carry `primary_face` (box, pose, sharpness, eyes,
  mouth, distances, age, and landmarks with `landmarks: true` / `--landmarks`)
  and the scan's `set_age`. Additive, so a 2.x reader keeps working.
- **The `concept` target category (#134).** A style or visual-concept LoRA set:
  face count never lowers its score, and clean, high-resolution images get the
  composition bonus. `TargetStyle` and `TargetCategory` now come from
  `argus_cortex.taxonomy` instead of a local copy, so the categories can't drift
  from the rest of the suite. **Requires `argus-cortex>=0.4.7`**, the first
  release with `concept`.

- **A guide to synthetic source portraits for argus-mask**
  ([docs/synthetic-face-sets.md](docs/synthetic-face-sets.md)). It follows
  argus-mask's own guide step by step: generate with argus-darkroom, shortlist
  with curator, read argus-lens captions, then hand off. It also lists what no
  tool checks. Docs only.

### Changed

- **Requires `argus-cortex>=0.5.0,<0.6`.** 0.5.0 adds `argus_cortex.provenance`
  and `argus_cortex.consent`; curator uses neither yet, and nothing it does use
  changed. The wire schema is unchanged.
- **The wire schema says what the server always sends (#129).** In models that
  only go out (`ScanSummary`, `ImageResult`, `ExportResult`, `ManifestRow`,
  `SelectResult` and what they hold), a field with a default is now required,
  since every response carries it. Request models and the configs a request
  also carries are unchanged, as a client may still leave those fields out.
  `ExportResult.mode` is the `copy` / `symlink` / `move` enum rather than any
  string, and `WatermarkDetection.bbox` and `FaceCluster.representative_bbox`
  are 4 numbers, like the other boxes. The payloads themselves are unchanged,
  and so is the manifest version. Cortex's `NsfwResult`, `NsfwDetection` and
  `GuardResult` still list their defaulted fields as optional.

## [0.14.1] - 2026-11-06

### Changed

- **Requires `argus-cortex>=0.4.6`.** `adult-cinematic` now passes depicted crime
  without a film cue, and its letters B to G no longer match the other adult
  policies, so `--guard-rules B` picks a different rule under it, including on
  verdicts recorded earlier. The pin keeps every install on the same letters.

### Fixed

- **No undecided NudeNet score reaches a client.** With EraX on, a faces chunk
  could carry a row's NudeNet 0 before EraX had answered, so a viewer unblurred
  a frame EraX then flagged. Until a row is decided, chunks and the snapshots
  behind a running scan's `GET /scan/{id}` now say `nsfw: null`.
- **A streamed scan's first chunk no longer beats its first snapshot.** The
  scan is saved before its first chunk goes out, so `/thumb?scan_id=` cannot
  answer 404 for that chunk's rows. A snapshot that fails to save is tried again
  at the next chunk.
- **A streamed scan that fails no longer leaves its partial snapshot.** Only a
  cancel or a busy GPU dropped it; any other failure left it in `GET /scans`
  like a finished scan, open and exportable, with rows that may lack nudity or
  age results. If the snapshot cannot be removed, that is logged and the client
  still gets `event: error`. Deleting a scan also removes a half-written save.
- **A streamed scan cut off by a server kill or crash no longer stays listed.**
  Its snapshots carry a `<scan_id>.running` marker until the scan finishes, and
  the server removes any marked scan when it starts. Scans cached before this
  release have no marker and are kept, so a partial one left earlier stays until
  it is deleted or purged.
- **A scan asking for watermark detection is refused while the detector is
  down.** `/detectors` only says the URL is set, so a scan could start with the
  detector stopped or still loading and reject its frames as `crop:unchecked`.
  The scan routes now answer HTTP 503, and `--detect-watermarks` exits 2, when
  the detector's `GET /health` fails or returns 5xx. Any answer below 500 counts
  as up, as a deployer's server need not have `/health`.

## [0.14.0] - 2026-11-04

### Added

- **Crops.** `ScanRequest.crops` (`CropConfig`; CLI `--crops`) crops a boxed
  image in memory right after it is read, so every check judges the cropped
  frame, and export writes it cropped; the source is never changed. A box
  (`CropBox`, displayed pixels after EXIF orientation) comes from a
  `<image>.crop.json` sidecar or inline by `rel_path` (inline wins). A box
  that cannot be applied makes the image a `crop:invalid` / `crop:unreadable`
  / `crop:failed` reject that no export keeps. `ImageResult.crop`,
  `ScanSummary.crop_config`, `ExportResult.cropped`; `GET /health` reports
  `crops: true`, and `GET /thumb?scan_id=` shows the cropped frame.
- **Watermark detection.** `CropConfig.detect` (CLI `--detect-watermarks`,
  `--watermark-url`, `--max-cut`, `--keep-overlays`) sends each frame without
  a box by hand, upright, to a NudeNet-shaped detector at
  `ARGUS_CURATOR_WATERMARK_URL`, and cuts its hits (`min_conf` 0.4) away along
  the frame's edges, at most `max_cut` (0.15) of a side per edge. A hit no
  edge cut removes is a `crop:overlay` reject (`reject_overlay`, default on);
  an image the detector could not check is `crop:unchecked`.
  `ImageResult.watermarks` and `crop_source`; scan phase `watermarks`;
  `/detectors` reports `watermark`, `/health` `watermark_detect`.
- **Manifest 2.2.** Each row carries `crop`, the box baked into the exported
  file, or `null`.
- **`guard_eval.py` measures what a scan does.** `--candidate NAME=PARENT:FILE`
  tries a policy wording from a file, scored as its parent policy. `--scan-json`
  with `--context` sends each image case with the context and crop of its row
  in a saved scan, and a `--thinking cascade` run routes it as a scan would.
  Rows and the table record the pass that decided each verdict.

### Changed

- **A rejected frame stops at rejection.** It gets no face, age, scene or
  garment reading, so the under-18 rule could not fire on it; NudeNet, EraX and
  SingGuard now skip it too, and no export keeps it, by selection or otherwise.
  `include_rejected` is ignored and `--include-rejected` is gone. Rejected rows
  stay in `results` with their `reject_reason`, so a scan still says what it
  left out, with `nsfw` unset (a prefilter drop keeps the score it was dropped
  on): viewers must keep them hidden or blurred.
- `ScanConfig.min_short_side` cannot go below 512 (`MIN_SHORT_SIDE`). A scan
  cached with a lower value still loads, and reports 512.
- A scan that cropped any image exports only by `mode: "copy"` (HTTP 400, CLI
  exit 1, `ValueError` from `export_selection`).
- Under `ARGUS_CURATOR_SINGGUARD_THINKING=cascade`, every frame with an
  exposed reading of 0.2 or more (NudeNet, or EraX) is judged in `slow`,
  whatever its age; before, only the under-18 + exposure pairing skipped the
  fast screen. A fast pass is final, so exposed frames were passed without
  reasoning. Slower scans on nude sets; frames with no exposed reading keep
  the fast screen.
- Needs `pydantic>=2.7`, for the scan cache's `model_dump_json(context=...)`.
- `guard.check_images` judges every row it is given; the scanner now leaves out
  the images the prefilter dropped.
- Every streamed `chunk` carries `face_clusters`, even when empty.

### Fixed

- **`guard_eval.py --images` reads subfolders.** Images sorted into
  `<dir>/<expect>/<category>/` were skipped with no warning; case ids now carry
  the path under the expect folder, and a symlinked subfolder, which is not
  followed, is named on stderr. An image that disappears during a run is an
  error row, also named on stderr, instead of a crash. `--out` is written as
  verdicts land, so a run that dies keeps its results.
- An image the prefilter dropped no longer counts toward its identity:
  `face_clusters[].size` and the representative leave it out, also after a
  delete, and an identity with only dropped images is removed from
  `face_clusters`. Its rows and `set_age.cluster_id` keep naming that identity;
  pooled attributes and the set age still include dropped faces.
- **Identities and the set age follow a delete.** Removing images (a delete, or
  a move export) now pools `face_clusters[].attributes` again and re-reads
  `set_age` and `set_age_skipped` from the faces left. Each face keeps its raw
  readings in the scan's own cache file, so a deleted image or a purged scan
  leaves nothing behind. They are not part of the API or the wire schema. A
  scan cached before this keeps its pooled attributes; its set age is still
  re-read.
- **Shared sidecars under move.** Images that share a sidecar (`a.jpg`, `a.png`
  and `a.txt`) now all get it: it is copied to each but the last, which takes
  the original. If one of them fails to transfer, the original stays.
- **`run_stats.json` can no longer overwrite a carried sidecar.** With
  `carry_sidecars`, an image at the destination root with the stem `run_stats`,
  `manifest` or `curation_report` gets a hash suffix, even with
  `preserve_structure`; `exported_path` has the real name.
- **Failed move exports.** A server move export that fails after files moved
  drops their rows from the cached scan, and its error says how many left the
  source. A scan deleted during a move export is no longer written back.

## [0.13.0] - 2026-11-03

### Added

- **Export carries sidecars.** `ExportRequest.carry_sidecars` (CLI
  `--carry-sidecars`) transfers `<stem>.txt` (lens captions), `<file>.xmp`
  (lens, digiKam), `<stem>.xmp` (Lightroom) and `<stem>.json` beside each
  exported image with it, in the same mode, named after the destination file
  (de-collided names included). Under `mode="symlink"` the sidecars are
  copied, not linked, so a caption written into the export (argus-lens) never
  reaches the source library. Under `mode="move"` they are no longer left
  orphaned in the source folder. With `write_xmp`, curator's data is merged onto
  the carried sidecar, so a move leaves one merged sidecar and nothing behind.
  A failed sidecar is logged and counted and never undoes its image.
  `ExportResult.sidecars_carried` / `sidecars_failed` count them.
- **`run_stats.json`.** `ExportRequest.run_stats` (opaque client JSON, at most
  5 MB serialised, else 422; CLI `--run-stats FILE`) is written pretty-printed to
  `<dest>/run_stats.json` when the export produced a destination folder (a file
  transferred, or `write_manifest` on). `ExportResult.run_stats_path` names it.
- `GET /health` reports `export_sidecars: true`.

### Changed

- A server `mode="move"` export drops the moved images from the cached scan
  (as `POST /scan/{id}/delete` does), so their thumbnails no longer 404. A move
  export of the scan still running is refused with 409.

## [0.12.0] - 2026-11-02

### Added

- **Scan prefilter.** `ScanRequest.prefilter` (`PrefilterConfig`: `drop_nsfw`,
  `drop_no_face`, `drop_no_age`, all off by default; CLI `--drop-nsfw`,
  `--drop-no-face`, `--drop-no-age`) turns matching passing images into rejects
  right after identities, before scenes, garments and SingGuard, so none of
  those run on them. `reject_reason` is `prefilter:nsfw` (exposed reading of 0.2
  or more; an unscored image is kept), `prefilter:no-face` (an image whose face
  detection failed is kept) or `prefilter:no-age` (no MiVOLO, body-only or set
  age), the first that applies. The images stay in the results, and a streaming
  scan sends them in a `prefilter` chunk. Unlike a hard reject, a dropped image
  is never kept or exported, even with `include_rejected` and a score over
  `min_score` or when named in an export `selection`; its keep reason is its
  `reject_reason`. A rule without its check (`drop_nsfw` without NudeNet, a face
  rule without face detection, `drop_no_age` without MiVOLO) is refused with
  HTTP 400 (CLI exit 2). The summary records `prefilter_config`; `GET /health`
  reports `prefilter: true`.
- `ImageResult.face_error`: why face detection failed on an image (the
  detector raised, or it could not be decoded). With it set, `face_count` 0
  means unchecked, not faceless.
- `serve --allow-delete` (or `CURATOR_ALLOW_DELETE=1`, or `CURATOR_DEVMODE=1`)
  lets `POST /scan/{id}/delete` remove any image in a scan. `GET /health`
  reports `delete_any`, true only on such a server, so a client such as
  argus-cull can tell without sniffing the version.
- **`DELETE /scan/{id}` removes one cached scan.** Until now the only way to
  drop a scan was `POST /scans/purge`, which empties the whole cache. The
  running scan is refused with 409; `/health` reports `scan_delete: true`.

### Changed

- `POST /scan/{scan_id}/delete` takes any image in the scan, not only one
  MiVOLO reads as under 18 with an exposed score, on a server started with
  `--allow-delete` / `CURATOR_ALLOW_DELETE=1` / `CURATOR_DEVMODE=1`. Without
  it, other images are still a 403 and under-18 exposed ones stay deletable. A batch is still checked as a whole first (a path not in the scan is a 404
  and deletes nothing); if an unlink then fails part-way, the files already
  removed drop out of the cached scan and the 400 lists them, and a running scan is still a 409. The under-18 exposed
  rule still decides which thumbnails are blurred.

### Fixed

- **Tests no longer write scans into the real cache.** The CLI scan tests
  saved to `~/.cache/argus_curator/scans`, the directory a local curator
  serves, so its scan list filled with `/tmp/pytest-of-*` folders. The test
  fixture now points `CURATOR_CACHE_DIR` at a temporary directory.

## [0.11.1] - 2026-11-01

### Added

- **A new argus-cortex rebuilds the published image.** `release.yml` also runs on
  the `cortex-released` dispatch argus-cortex's release sends: only the image job runs,
  rebuilding the newest final `v*` tag without cache and republishing its tags, so the
  image picks up the cortex its pyproject range now resolves to.

### Fixed

- EraX drops a nipple hit only when its centre lies inside a NudeNet
  `MALE_BREAST_EXPOSED` box. A box that merely touched one was dropped, so a
  topless woman beside a bare-chested man passed the NSFW filters.
- The CLI scan summary no longer counts rows EraX filled after NudeNet failed
  as "NudeNet checked"; the export already treats them as unchecked. Their
  EraX hits still count as exposed.

## [0.11.0] - 2026-11-01

### Added

- Optional EraX-NSFW sidecar (`ARGUS_CURATOR_ERAX_URL`). When NudeNet returns
  no exposed score, EraX hits are mapped onto the same labels so filters,
  blur, pairing and cascade still work. `GET /detectors` reports `erax`.
  EraX runs beside NudeNet, not after it, and every image keeps its EraX
  result.
- `ImageResult.nudenet_failed`: NudeNet's check failed and the score is EraX's
  alone. The NudeNet filters exclude such a frame as `nsfw-unchecked` unless
  EraX reaches a threshold, since EraX has no buttocks class.

### Changed

- A malformed `ARGUS_CURATOR_ERAX_URL` is rejected before the scan starts
  (server 400, CLI exit 1) instead of after the whole NudeNet pass. The EraX
  pass reports an `erax` progress phase and stops on cancel.

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

[Unreleased]: https://github.com/smk762/argus-curator/compare/v0.23.0...HEAD
[0.23.0]: https://github.com/smk762/argus-curator/compare/v0.22.0...v0.23.0
[0.22.0]: https://github.com/smk762/argus-curator/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/smk762/argus-curator/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/smk762/argus-curator/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/smk762/argus-curator/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/smk762/argus-curator/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/smk762/argus-curator/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/smk762/argus-curator/compare/v0.15.1...v0.16.0
[0.15.1]: https://github.com/smk762/argus-curator/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/smk762/argus-curator/compare/v0.14.1...v0.15.0
[0.14.1]: https://github.com/smk762/argus-curator/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/smk762/argus-curator/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/smk762/argus-curator/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/smk762/argus-curator/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/smk762/argus-curator/compare/v0.11.0...v0.11.1
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
