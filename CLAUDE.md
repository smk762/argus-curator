# CLAUDE.md — argus-curator

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

The **curation stage** of the Argus suite: it decides *which images, of whom, at what quality* belong in a LoRA training set, then exports the keepers plus a JSONL manifest that argus-lens captions verbatim.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire        curate/export    caption       configs        LoRA           validate
```

Curator and lens share one `TargetProfile`, so a manifest written here is consumed downstream with no remapping. Ships as an engine (Pillow/numpy/ImageHash/pydantic), an optional Typer CLI, and an optional FastAPI server on **:8101** (the `/curate` UI backend). Read [README.md](README.md) for the *why*.

## Layout

`src/argus_curator/`:

- `models.py` — Pydantic v2 wire types (`TargetProfile`, `ScanConfig`/`FaceConfig`, `ScanSummary`/`ImageResult`, `ExportRequest`/`ExportResult`, `ManifestRow`). **This is the API contract**: `WIRE_MODELS` + `wire_schema()` back `argus-curator schema` and the committed `schema/curator-wire.schema.json`. `MANIFEST_VERSION` (currently `"2.1"`) versions the JSONL handoff and moves *independently* of the package version.
- `scanner.py` — the training-suitability engine (ported from imogen). Phase-1 metrics (`_sharpness`, `_artifact_score`, pHash) + target-aware scoring (`_base_score`, `_target_bonus`, `_face_penalty`, `finalize_score`), near-duplicate clustering (`_mark_duplicates`/`_assign_clusters`), threaded orchestration in `scan_items`/`scan_folder`.
- `selection.py` — score-threshold + optional diversity cap (`decide_selection`, `select_diverse`). Nothing is silently dropped: every excluded image gets a `keep_reason`.
- `export.py` — structure-preserving transfer (copy/symlink/move), basename de-collision (`_plan_dest_paths`), `write_manifest`/`write_report`, optional `_post_manifest_to_lens` handoff.
- `faces.py` — optional InsightFace detect/embed/cluster into `face_<n>` identities + head-pose bucketing (`classify_pose`). All heavy imports deferred; `import faces` is always safe.
- `store.py` — `ScanStore`, a flat `<scan_id>.json` cache that makes paginated `GET /scan/{id}` and export-by-id possible.
- `cli.py` — Typer app (`scan`, `serve`, `detectors`, `schema`).
- `server/app.py` — `create_app()`: routes + containment + `argus_cortex` write-guard. Optional `[server]` extra.

## Commands

```bash
make dev       # uv venv + editable install with [dev,cli,server]
make test      # pytest --tb=short -q
make lint      # ruff check
make fmt       # ruff format + --fix
make schema    # regenerate schema/curator-wire.schema.json
make check     # lint + test + build
```

Run a single test: `uv run --no-sync pytest tests/test_export.py::test_name -q`.

## Conventions & gotchas

- **The wire schema is checked in CI.** If you touch a `WIRE_MODELS` type in `models.py`, run `argus-curator schema` and commit `schema/curator-wire.schema.json` — `tests/test_schema.py` and CI's `argus-curator schema --check` post-test both fail on drift.
- **`TargetProfile` is the shared moat.** It is consumed verbatim by argus-lens; keep the taxonomy (`target_style` / `target_backend` / `checkpoint` / `target_category`) in lockstep. Same for `ManifestRow` / `MANIFEST_VERSION` — bump the major only on a breaking row change; consumers (argus-forge, argus-lens) branch on it.
- **Manifest fidelity is load-bearing.** Rows exist only for files whose transfer *succeeded*; `exported_path`/`exported_abs_path` are the de-collided on-disk locations consumers must use instead of re-deriving from `rel_path`. Under `mode="move"` the source is gone, so `abs_path` is written as the destination (issue #9). Don't regress these.
- **Server containment — don't loosen it.** Request paths are untrusted: scan/thumb/upload resolve under `source_root`, exports under `export_root` (`_resolve_within` refuses traversal escapes; endpoints 400 when a root is unset). A cached `summary.folder` is re-checked against the source root (`_scan_root_or_400`) because the store is *data*. Destructive `mode="move"` is rejected unless `--allow-move` / `CURATOR_ALLOW_MOVE=1`. **The CLI is deliberately unconstrained** — it's your own shell.
- **CORS is not a write boundary.** The `WriteGuard` + `cross_site_refuse` from **argus-cortex** gate cross-site writes on Origin; the wildcard grants anonymous *reads* only. Reuse the shared cortex helpers (write-guard, `env_flag`) rather than re-implementing.
- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `src/argus_curator/_version.py` is generated (gitignored). Tag `vX.Y.Z` to release (PyPI via OIDC).
- `structlog` for logging; Pydantic v2 everywhere; `pytest asyncio_mode = auto`. Ruff line-length 120, `ignore = E501,B008`.
