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

Each rule names what enforces it. A new rule nothing checks yet is marked **UNENFORCED** and links the issue that adds its check; it gets a check, not more prose. A rule about code shape (a banned import or API) can be a suite rule in argus-ci's catalogue (smk762/argus-ci#16) instead of a test here.

| Must never happen | Enforced by |
|---|---|
| A `WIRE_MODELS` change merged without regenerating `schema/curator-wire.schema.json` (`make schema`) | `tests/test_schema.py::test_committed_schema_is_current`; CI `argus-curator schema --check` |
| `model_config = RESPONSE_ONLY` on a model a request carries (the `ScanRequest` configs, `TargetProfile`): a client may omit those defaults | `tests/test_schema.py::test_only_models_no_request_carries_list_their_defaults_as_required` |
| A symlink export deleting the source file it was to link to (the destination already is the source) | `tests/test_export.py::test_symlink_export_into_the_source_folder_never_deletes_the_source` |
| A manifest row for a file whose transfer failed or was skipped | `tests/test_export.py::test_skipped_sources_are_excluded_from_manifest` |
| A manifest path consumers must re-derive: `exported_path`/`exported_abs_path` are the de-collided on-disk paths, and under `mode="move"` `abs_path` is the destination (#9) | `tests/test_export.py::test_exported_abs_paths_survive_flattened_de_collision`, `::test_move_manifest_abs_path_points_at_the_moved_file` |
| A breaking `ManifestRow` change without a `MANIFEST_VERSION` major bump, or any row change without a bump (argus-lens and argus-forge branch on it) | `tests/test_manifest_version.py::test_manifest_version_matches_the_recorded_row` against `schema/manifest-row.snapshot.json`; `make manifest-snapshot` refuses an unbumped or breaking change |
| A request path escaping `source_root`/`export_root` (scan, thumb, image, upload, export) | `tests/test_server.py::test_*_traversal_blocked` |
| A scan reading a file outside its folder through a symlink (a linked file or directory) | `tests/test_scanner.py::test_collect_images_skips_symlinks_that_leave_the_root` |
| A cached scan used from outside the source root: the store is data, so `summary.folder` is re-checked | `tests/test_server.py::test_*_rejects_scan_outside_source_root` |
| `mode="move"` over HTTP without `--allow-move` / `CURATOR_ALLOW_MOVE=1` | `tests/test_server.py::test_export_move_rejected_by_default` |
| Deleting a source file that isn't under-18 exposed without `--allow-delete` / `CURATOR_ALLOW_DELETE=1` / `CURATOR_DEVMODE=1` | `tests/test_server.py::test_delete_without_the_gate_takes_only_under_18_exposed` |
| A cross-site write allowed because CORS is open: CORS grants anonymous reads only, and the cortex write-guard gates writes on `Origin` | `tests/test_server.py::test_cors_any_does_not_grant_cross_site_writes` |
| A local copy of a shared argus-cortex helper (write-guard, `env_flag`, moderation clients, taxonomy, `TargetProfile`) | `tests/test_cortex_reuse.py::test_no_module_redefines_a_cortex_helper` (a module-level name cortex also defines fails unless `_ALLOWED` gives the reason) |
| An under-18 frame with any reported exposed score kept, whatever the request says | `tests/test_nsfw.py::test_under_18_with_a_reported_exposed_score_is_never_kept`, `::test_frame_rule_is_a_hard_exclusion` |
| `/image` serving, or `/thumb` sending unblurred, a file any cached scan reads as under-18 exposed, whatever the path's spelling, a symlink, or a missing `scan_id` | `tests/test_server.py::test_image_block_holds_for_any_spelling_of_the_path`, `::test_thumb_of_an_under_18_exposed_frame_is_blurred_by_the_server` |
| Under `overlap`, an exposure on a person whose matched face reads under 18 treated as an adult's because the body reads adult | `tests/test_bodies.py::test_overlap_rule_counts_a_young_face_on_a_person_whose_body_reads_adult` |
| InsightFace's face models built outside the GPU lease (`FaceAnalysis()` builds its sessions in the constructor, not in `prepare()`) | `tests/test_bodies.py::test_face_sessions_are_built_under_the_gpu_lease` |
| A server that starts with an unknown `minor-exposed` mode (no fallback) | `tests/test_server.py::test_unknown_minor_exposed_mode_refuses_to_start` |
| A prefiltered image kept or exported | `tests/test_server.py::test_prefiltered_images_are_never_kept_or_exported` |
| Moderation filters applied to a scan that never ran the check | `tests/test_server.py::test_select_refuses_moderation_filters_on_unchecked_scan` |
| XMP sidecars writing an age | `tests/test_xmp.py::test_xmp_data_names_only_named_clusters_and_never_writes_age` |
| `POST /faces` (darkroom's age check) writing anything to disk | `tests/test_server.py::test_faces_returns_estimates_and_writes_nothing` |
| An upload read past `CURATOR_MAX_UPLOAD_MB` (it goes through `argus_cortex.limits`) | `tests/test_paved_roads.py::test_upload_over_the_cap_is_413_and_saves_nothing`, `::test_faces_multipart_over_the_upload_cap_is_413` |
| The `caption_url` hand-off reaching a non-public host not listed in `CURATOR_CAPTION_ALLOWED_HOSTS`, or logging the full URL (it goes through `argus_cortex.net.fetch`) | `tests/test_paved_roads.py::test_caption_handoff_to_a_private_address_is_refused_without_an_allow_list`, `::test_export_refuses_a_private_caption_url_up_front`, `::test_caption_handoff_logs_the_host_never_the_url` |
| A config error echoing a URL or path | `tests/test_server.py::test_malformed_nudenet_url_is_refused_without_echoing_it` |
| Importing NudeNet (AGPL-3.0): it stays a separate server behind `NudeNetClient` | ruff `TID251` banned-api (`pyproject.toml`), run by `make lint` and CI |

## Facts the code doesn't show

- **The CLI is deliberately unconstrained**: it's your own shell. Containment and the move/delete gates are the server's.
- **`MANIFEST_VERSION` moves independently of the package version.** Bump the minor for an additive row change, the major for a breaking one, then `make manifest-snapshot`.
- **`TargetProfile` and its taxonomy are argus-cortex's** (`argus_cortex.taxonomy`), re-exported from `models.py`. Every stage reads the profile verbatim, so change it in cortex, not here.
- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `_version.py` is generated. Tag `vX.Y.Z` to release (PyPI via OIDC).
- CI (argus-ci's reusable `python-ci.yml`) runs on pushes and PRs to `dev` and `main`: `ruff check`, `ruff format --check`, tests, build and `argus-curator schema --check`. `make check` covers most of that locally; also run `make fmt`.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.
