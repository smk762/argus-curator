# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed

- `argus-curator scan --csv` crashed with a `TypeError` after the manifest-2.0
  change (`write_report` gained the `exported_paths` parameter but the CLI
  call site was not updated).

### Added

- `POST /upload` server endpoint: multipart image upload into a folder under
  the configured source root (traversal-safe, non-image and duplicate-name
  files skipped, existing files never overwritten).
- `ExportResult.exported_paths`: the `rel_path -> exported_path` mapping for
  every successful transfer, so API callers get de-collided names even with
  `write_manifest: false`. The CSV report gains a matching `exported_path`
  column.

### Changed

- **Manifest 2.0** (breaking for consumers that derive destinations from
  `rel_path`): rows are written only for files whose transfer succeeded and
  carry `exported_path` — the real (possibly de-collided) path under the
  export root, which consumers must use. The row shape is now published in the
  wire schema as `ManifestRow`.

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

[0.1.0]: https://github.com/smk762/argus-curator/releases/tag/v0.1.0
