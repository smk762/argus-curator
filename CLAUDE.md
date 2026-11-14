# CLAUDE.md — argus-curator

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

The **curation stage** of the Argus suite: it decides *which images, of whom, at what quality* belong in a LoRA training set, then exports the keepers plus a JSONL manifest that argus-lens captions verbatim.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire        curate/export    caption       configs        LoRA           validate
```

Curator and lens share one `TargetProfile`, so a manifest written here is consumed downstream with no remapping. Ships as an engine (Pillow/numpy/ImageHash/pydantic), an optional Typer CLI, and an optional FastAPI server on **:8101** (the `/curate` UI backend). Read [README.md](README.md) for the *why*.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Layout

`src/argus_curator/`:

- `models.py` — Pydantic v2 wire types (`TargetProfile`, `ScanConfig`/`FaceConfig`, `ScanSummary`/`ImageResult`, `ExportRequest`/`ExportResult`, `ManifestRow`). **This is the API contract**: `WIRE_MODELS` + `wire_schema()` back `argus-curator schema` and the committed `schema/curator-wire.schema.json`. `MANIFEST_VERSION` (currently `"2.3"`) versions the JSONL handoff and moves *independently* of the package version.
- `scanner.py` — the training-suitability engine (ported from imogen). Phase-1 metrics (`_sharpness`, `_artifact_score`, pHash) + target-aware scoring (`_base_score`, `_target_bonus`, `_face_penalty`, `finalize_score`), near-duplicate clustering (`_mark_duplicates`/`_assign_clusters`), threaded orchestration in `scan_items`/`scan_folder`.
- `prefilter.py` — opt-in `PrefilterConfig` rules (`prefilter:nsfw` / `no-face` / `no-age`) applied in `scan_items` right after identities; dropped rows become rejects that scenes, garments and SingGuard skip.
- `selection.py` — score-threshold + optional diversity cap (`decide_selection`, `select_diverse`). Nothing is silently dropped: every excluded image gets a `keep_reason`.
- `export.py` — structure-preserving transfer (copy/symlink/move), basename de-collision (`_plan_dest_paths`), `write_manifest`/`write_report`, optional `_post_manifest_to_lens` handoff.
- `faces.py` — optional InsightFace detect/embed/cluster into `face_<n>` identities + head-pose bucketing (`classify_pose`). All heavy imports deferred; `import faces` is always safe.
- `store.py` — `ScanStore`, a flat `<scan_id>.json` cache that makes paginated `GET /scan/{id}` and export-by-id possible. A `<scan_id>.running` marker sits beside a streamed scan's snapshots until it finishes; `create_app` drops any marked scan on start.
- `cli.py` — Typer app (`scan`, `serve`, `detectors`, `schema`).
- `server/app.py` — `create_app()`: routes + containment + `argus_cortex` write-guard. Optional `[server]` extra. `POST /faces` is memory-only (no source root); it is the age-check HTTP surface for darkroom.

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
- **Response-only models carry `model_config = RESPONSE_ONLY`**, so their defaulted fields are required in the schema (the server always sends them). Never put it on a model a request also carries (the `ScanRequest` configs, `TargetProfile`; `EraxConfig` is response-only): a client may omit those fields. `tests/test_schema.py` checks the set, and checks real `/scan`, `/select`, `/export` and manifest payloads carry every required field.
- **`TargetProfile` is the shared moat.** It is consumed verbatim by argus-lens; keep the taxonomy (`target_style` / `target_backend` / `checkpoint` / `target_category`) in lockstep. Same for `ManifestRow` / `MANIFEST_VERSION` — bump the major only on a breaking row change; consumers (argus-forge, argus-lens) branch on it.
- **Manifest fidelity is load-bearing.** Rows exist only for files whose transfer *succeeded*; `exported_path`/`exported_abs_path` are the de-collided on-disk locations consumers must use instead of re-deriving from `rel_path`. Under `mode="move"` the source is gone, so `abs_path` is written as the destination (issue #9). Don't regress these.
- **Server containment — don't loosen it.** Request paths are untrusted: scan/thumb/upload resolve under `source_root`, exports under `export_root` (`_resolve_within` refuses traversal escapes; endpoints 400 when a root is unset). A cached `summary.folder` is re-checked against the source root (`_scan_root_or_400`) because the store is *data*. Destructive `mode="move"` is rejected unless `--allow-move` / `CURATOR_ALLOW_MOVE=1`. Deleting a scanned source file that is not under-18 exposed is rejected (403) unless `--allow-delete` / `CURATOR_ALLOW_DELETE=1` / `CURATOR_DEVMODE=1`. **The CLI is deliberately unconstrained** — it's your own shell.
- **CORS is not a write boundary.** The `WriteGuard` + `cross_site_refuse` from **argus-cortex** gate cross-site writes on Origin; the wildcard grants anonymous *reads* only. Reuse the shared cortex helpers (write-guard, `env_flag`) rather than re-implementing.
- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `src/argus_curator/_version.py` is generated (gitignored). Tag `vX.Y.Z` to release (PyPI via OIDC).
- `structlog` for logging; Pydantic v2 everywhere; `pytest asyncio_mode = auto`. Ruff line-length 120, `ignore = E501,B008`.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.
