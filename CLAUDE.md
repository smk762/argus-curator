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

## Must never happen

Each rule names what enforces it. A rule marked **UNENFORCED** needs a check, not more prose: its linked issue tracks one. A rule about code shape (a banned import or API) can be a suite rule in argus-ci's catalogue (smk762/argus-ci#16) instead of a test here.

| Must never happen | Enforced by |
|---|---|
| A `WIRE_MODELS` change merged without regenerating `schema/curator-wire.schema.json` (`make schema`) | `tests/test_schema.py::test_committed_schema_is_current`; CI `argus-curator schema --check` |
| `model_config = RESPONSE_ONLY` on a model a request carries (the `ScanRequest` configs, `TargetProfile`): a client may omit those defaults | `tests/test_schema.py::test_only_models_no_request_carries_list_their_defaults_as_required` |
| A manifest row for a file whose transfer failed or was skipped | `tests/test_export.py::test_skipped_sources_are_excluded_from_manifest` |
| A manifest path consumers must re-derive: `exported_path`/`exported_abs_path` are the de-collided on-disk paths, and under `mode="move"` `abs_path` is the destination (#9) | `tests/test_export.py::test_exported_abs_paths_survive_flattened_de_collision`, `::test_move_manifest_abs_path_points_at_the_moved_file` |
| A breaking `ManifestRow` change without a `MANIFEST_VERSION` major bump (argus-lens and argus-forge branch on it) | **UNENFORCED**: #168 |
| A request path escaping `source_root`/`export_root` (scan, thumb, image, upload, export) | `tests/test_server.py::test_*_traversal_blocked` |
| A cached scan used from outside the source root: the store is data, so `summary.folder` is re-checked | `tests/test_server.py::test_*_rejects_scan_outside_source_root` |
| `mode="move"` over HTTP without `--allow-move` / `CURATOR_ALLOW_MOVE=1` | `tests/test_server.py::test_export_move_rejected_by_default` |
| Deleting a source file that isn't under-18 exposed without `--allow-delete` / `CURATOR_ALLOW_DELETE=1` / `CURATOR_DEVMODE=1` | `tests/test_server.py::test_delete_without_the_gate_takes_only_under_18_exposed` |
| A cross-site write allowed because CORS is open: CORS grants anonymous reads only, and the cortex write-guard gates writes on `Origin` | `tests/test_server.py::test_cors_any_does_not_grant_cross_site_writes` |
| A local copy of a shared argus-cortex helper (write-guard, `env_flag`, moderation clients, taxonomy) | **UNENFORCED**: #169 |
| An under-18 frame with any reported exposed score kept, whatever the request says | `tests/test_nsfw.py::test_under_18_with_a_reported_exposed_score_is_never_kept`, `::test_frame_rule_is_a_hard_exclusion` |
| A server that starts with an unknown `minor-exposed` mode (no fallback) | `tests/test_server.py::test_unknown_minor_exposed_mode_refuses_to_start` |
| A prefiltered image kept or exported | `tests/test_server.py::test_prefiltered_images_are_never_kept_or_exported` |
| Moderation filters applied to a scan that never ran the check | `tests/test_server.py::test_select_refuses_moderation_filters_on_unchecked_scan` |
| XMP sidecars writing an age | `tests/test_xmp.py::test_xmp_data_names_only_named_clusters_and_never_writes_age` |
| `POST /faces` (darkroom's age check) writing anything to disk | `tests/test_server.py::test_faces_returns_estimates_and_writes_nothing` |
| A config error echoing a URL or path | `tests/test_server.py::test_malformed_nudenet_url_is_refused_without_echoing_it` |
| Importing NudeNet (AGPL-3.0): it stays a separate server behind `NudeNetClient` | ruff `TID251` banned-api (`pyproject.toml`), run by `make lint` and CI |

## Facts the code doesn't show

- **The CLI is deliberately unconstrained**: it's your own shell. Containment and the move/delete gates are the server's.
- **`MANIFEST_VERSION` moves independently of the package version.** Bump the minor for an additive row change.
- **`TargetProfile` is shared with argus-lens**, which reads the manifest's profile verbatim. Its taxonomy (`TargetStyle`, `TargetCategory`) is imported from `argus_cortex.taxonomy`, so change it there, not here.
- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `_version.py` is generated. Tag `vX.Y.Z` to release (PyPI via OIDC).
- CI (argus-ci's reusable `python-ci.yml`) runs on pushes and PRs to `dev` and `main`: `ruff check`, `ruff format --check`, tests, build and `argus-curator schema --check`. `make check` covers most of that locally; also run `make fmt`.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.
