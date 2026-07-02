# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
