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
