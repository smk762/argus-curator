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
| `mode="move"` over HTTP without `--allow-move` / `CURATOR_ALLOW_MOVE=1` | `tests/test_server.py::test_export_move_rejected_by_default`, `tests/test_speech_export.py::test_speech_move_needs_the_gate_and_drops_the_moved_clips` |
| Deleting a source file that isn't under-18 exposed without `--allow-delete` / `CURATOR_ALLOW_DELETE=1` / `CURATOR_DEVMODE=1` | `tests/test_server.py::test_delete_without_the_gate_takes_only_under_18_exposed` |
| A cross-site write allowed because CORS is open: CORS grants anonymous reads only, and the cortex write-guard gates writes on `Origin` | `tests/test_server.py::test_cors_any_does_not_grant_cross_site_writes` |
| A local copy of a shared argus-cortex helper (write-guard, `env_flag`, moderation clients, taxonomy, `TargetProfile`) | `tests/test_cortex_reuse.py::test_no_module_redefines_a_cortex_helper` (a module-level name cortex also defines fails unless `_ALLOWED` gives the reason) |
| An under-18 frame with any reported exposed score kept, whatever the request says | `tests/test_nsfw.py::test_under_18_with_a_reported_exposed_score_is_never_kept`, `::test_frame_rule_is_a_hard_exclusion` |
| `/image` serving, or `/thumb` sending unblurred, a file any cached scan reads as under-18 exposed, whatever the path's spelling, a symlink, or a missing `scan_id` | `tests/test_server.py::test_image_block_holds_for_any_spelling_of_the_path`, `::test_thumb_of_an_under_18_exposed_frame_is_blurred_by_the_server` |
| Under `overlap`, an exposure on a person whose matched face reads under 18 treated as an adult's because the body reads adult | `tests/test_bodies.py::test_overlap_rule_counts_a_young_face_on_a_person_whose_body_reads_adult` |
| InsightFace's face models built outside the GPU lease (`FaceAnalysis()` builds its sessions in the constructor, not in `prepare()`) | `tests/test_bodies.py::test_face_sessions_are_built_under_the_gpu_lease` |
| A CUDA session (curator's own or InsightFace's) built without the capped arena (`kSameAsRequested` plus `gpu_mem_limit`), or an attribute model run on a whole crowd at once (#189: one scan held ~20 GB on 512 MB leases) | `tests/test_gpu_lease.py::test_curators_own_sessions_are_built_with_the_cap`, `::test_insightface_sessions_are_built_with_the_cap`, `::test_attribute_models_run_a_crowd_in_fixed_size_chunks`, `tests/test_embeddings.py::test_embedding_session_is_built_with_a_capped_arena_big_enough_for_its_weights` |
| A server that starts with an unknown `minor-exposed` mode (no fallback) | `tests/test_server.py::test_unknown_minor_exposed_mode_refuses_to_start` |
| A speech `TargetProfile` scored as an image one, a speech scan running an image check (faces, NudeNet, SingGuard, crops, consent, ...), or a speech category in the image manifest | `tests/test_speech.py::test_a_speech_profile_is_still_never_scored_as_an_image`, `::test_scan_request_routes_by_modality_and_refuses_image_checks_on_speech`; `tests/test_server.py::test_a_speech_target_profile_is_never_scored_as_an_image`; `tests/test_scanner.py::test_a_speech_target_profile_is_refused_before_scoring`; the manifest snapshot's image-only `target_category` enum |
| A `SpeechManifestRow` change without a `SPEECH_MANIFEST_VERSION` bump, or speech changing `ManifestRow` / `MANIFEST_VERSION` | `tests/test_manifest_version.py::test_speech_manifest_version_matches_the_recorded_row`, `::test_an_unbumped_speech_row_change_is_refused`, `::test_the_image_row_and_version_are_unchanged_by_speech` |
| A speaker in both train and held-out of a speech split, by corpus label or by cluster | `tests/test_speech_export.py::test_an_export_of_a_multi_speaker_multi_accent_corpus_has_no_speaker_in_both_splits`, `::test_label_and_cluster_join_into_one_split_group`, `::test_a_split_refuses_clips_with_no_speaker` |
| A clip of a speaker the corpus marks under 18 decoded, passed, kept or exported (a named selection included), whatever the transcript source | `tests/test_speech.py::test_a_speaker_the_corpus_marks_under_18_is_rejected_undecoded`, `tests/test_speech_export.py::test_an_under_18_speaker_is_never_exported_even_by_name` |
| A speech export with neither corpus + licence nor a consent record | `tests/test_speech_export.py::test_a_speech_export_must_name_where_its_clips_came_from`, `tests/test_speech.py::test_a_speech_scan_selects_and_exports_only_with_its_provenance` |
| A consented voice export carrying another speaker's clips, or from a record that doesn't cover voice training | `tests/test_speech_export.py::test_a_consented_voice_exports_only_the_consented_speaker`, `::test_a_voice_consent_record_that_does_not_cover_the_set_is_refused` |
| Audio decoded outside `argus_cortex.limits.open_audio`, or a file read past the byte cap `max_duration_s` implies | `tests/test_speech.py::test_a_long_file_is_never_read_past_its_cap` |
| A speech scan reading a clip, sidecar or corpus metadata file outside its folder through a symlink, or a speech `dest` or `consent_record` path escaping its root | `tests/test_speech.py::test_the_scan_reads_no_clip_outside_its_folder_through_a_symlink`, `::test_corpus_metadata_outside_the_folder_is_never_read`, `::test_a_speech_scan_folder_is_contained`; `tests/test_speech_export.py::test_speech_export_paths_are_contained` |
| A speech export into the source overwriting or deleting a source clip or its transcript sidecar | `tests/test_speech_export.py::test_symlink_export_into_the_source_folder_never_deletes_a_clip_or_its_transcript` |
| DNSMOS or the speaker model loaded unless it matches its pinned sha256, built outside the GPU lease or without the capped arena, or run on anything but fixed-size windows | `tests/test_speech.py::test_a_model_that_is_not_the_pinned_file_is_refused`, `::test_mos_loads_and_runs_under_the_gpu_lease_with_the_capped_arena`, `::test_mos_scores_each_passing_clip_in_fixed_size_windows`; `tests/test_speech_export.py::test_a_speaker_model_that_is_not_the_pinned_file_is_refused`, `::test_speaker_model_loads_and_runs_under_the_gpu_lease_with_the_capped_arena`; `tests/test_gpu_lease.py::test_gpu_tenant_manifest_declares_every_leased_model` |
| Speaker embeddings written to disk or outliving their scan | `tests/test_speech_export.py::test_speaker_embeddings_are_held_in_memory_and_purged_with_the_scan` |
| A prefiltered clip that doesn't name the check it failed | `tests/test_speech.py::test_each_prefilter_names_the_check_it_failed`, `::test_every_failed_check_is_listed_in_order` |
| A prefiltered image kept or exported | `tests/test_server.py::test_prefiltered_images_are_never_kept_or_exported` |
| A train/held-out split putting one group (identity, or any `SPLIT_GROUP_KEYS` key) in both parts | `tests/test_rank_split.py::test_no_group_lands_in_both_parts` |
| `/scan/{id}/select` and export keeping different rows, or the same rows in a different order, for the same fields (both run `selection.plan_selection`) | `tests/test_rank_split.py::test_select_and_export_keep_the_same_rows_in_the_same_order` |
| Moderation filters applied to a scan that never ran the check | `tests/test_server.py::test_select_refuses_moderation_filters_on_unchecked_scan` |
| XMP sidecars writing an age | `tests/test_xmp.py::test_xmp_data_names_only_named_clusters_and_never_writes_age` |
| A face embedding (a caller's `FaceConfig.reference_embedding`, or a consent record's centroid) written to the scan cache or served by `GET /scan/{id}`: only distances persist | `tests/test_face_source.py::test_a_scan_summary_never_holds_the_reference_embedding` |
| A semantic (CLIP) embedding written to disk or a response: they stay in memory and go with the scan and its rows | `tests/test_embeddings.py::test_embeddings_are_never_serialised`, `::test_store_holds_embeddings_in_memory_only`, `::test_store_drops_embeddings_with_the_scan_and_its_rows` |
| `diversify_semantic` silently ignored on a scan whose embeddings are gone or were never computed | `tests/test_embeddings.py::test_diversify_semantic_is_refused_without_embeddings`, `::test_select_route_uses_held_embeddings_and_refuses_without_them` |
| `semantic_duplicates` silently running pHash-only on a scan without embeddings | `tests/test_semantic_duplicates.py::test_a_scan_asking_for_semantic_duplicates_without_embeddings_is_refused`, `::test_scan_route_refuses_semantic_duplicates_without_embeddings`; `tests/test_cli.py::test_semantic_dupes_without_embeddings_fails_before_scanning` |
| A regroup after the embeddings are gone silently dropping the recorded semantic groups | `tests/test_semantic_duplicates.py::test_drop_results_without_embeddings_keeps_the_recorded_semantic_links` |
| Semantic dedupe changing a pHash-only scan's groups, or a `duplicate_of` reached through a chain rather than a direct match | `tests/test_semantic_duplicates.py::test_phash_only_grouping_is_unchanged_by_held_embeddings`, `::test_duplicate_of_is_always_a_direct_match_never_a_chain` |
| Semantic dedupe writing a vector to a row or the scan cache (only `duplicate_kind` and the group fields persist) | `tests/test_semantic_duplicates.py::test_the_scan_groups_semantic_duplicates_and_keeps_only_plain_values` |
| A consented face set exported after its record was revoked, expired, deleted or replaced since the scan (consent is re-checked at select and export) | `tests/test_consent.py::test_a_record_revoked_between_scan_and_export_blocks_the_export`, `::test_server_refuses_a_revoked_record_at_select_and_export`, `::test_an_expiry_passing_since_the_scan_blocks_the_export`, `::test_a_record_deleted_or_replaced_since_the_scan_blocks_the_export` |
| An image showing anyone but the consenting person, or whose primary face doesn't match the record's reference, in a consented set | `tests/test_consent.py::test_a_bystander_image_is_excluded`, `::test_a_mismatched_face_is_excluded` |
| A consented set exported with a bound face read under 18, an under-18 set age or no set age | `tests/test_consent.py::test_an_under_18_estimate_on_a_matched_face_refuses_the_set`, `::test_an_under_18_set_age_refuses_the_set`, `::test_a_consented_set_needs_a_set_age` |
| A consented set exported from a missing, invalid, expired, revoked or out-of-scope record, or when nothing qualifies | `tests/test_consent.py::test_a_missing_record_makes_the_set_ineligible`, `::test_an_expired_revoked_or_out_of_scope_record_makes_the_set_ineligible`, `::test_an_ineligible_set_is_refused_at_export`, `::test_an_export_where_nothing_qualifies_is_refused` |
| A consented selection made without the export-time consent gate | `tests/test_consent.py::test_a_consented_set_is_never_chosen_without_the_gate` |
| A face-source image argus-quarry marks `likeness_consent: false` kept | `tests/test_consent.py::test_quarry_images_marked_not_consented_are_refused`, `::test_a_quarry_mark_added_after_the_scan_is_read_at_export`, `::test_any_face_source_export_refuses_a_quarry_image` |
| A consent record (at scan, or as stored in a cached scan) or an image sidecar read from outside the scan folder over HTTP | `tests/test_consent.py::test_server_consent_record_traversal_blocked`, `::test_server_export_consent_record_traversal_blocked`, `::test_a_sidecar_outside_the_folder_is_refused` |
| Consent data outliving its scan, or the record's contents served or echoed | `tests/test_consent.py::test_nothing_about_consent_survives_a_purge`, `::test_server_scan_and_export_record_consent`, `::test_an_invalid_record_is_ineligible_without_echoing_its_contents` |
| `POST /faces` (darkroom's age check) writing anything to disk | `tests/test_server.py::test_faces_returns_estimates_and_writes_nothing` |
| An upload read past `CURATOR_MAX_UPLOAD_MB` (it goes through `argus_cortex.limits`) | `tests/test_paved_roads.py::test_upload_over_the_cap_is_413_and_saves_nothing`, `::test_faces_multipart_over_the_upload_cap_is_413` |
| The `caption_url` hand-off reaching any host (public or not) not listed in `CURATOR_CAPTION_ALLOWED_HOSTS`, or logging the full URL (it goes through `argus_cortex.net.fetch`) | `tests/test_paved_roads.py::test_caption_handoff_to_a_private_address_is_refused_without_an_allow_list`, `::test_export_refuses_a_caption_url_whose_host_is_not_configured`, `::test_an_empty_caption_host_config_names_the_variable_to_set`, `::test_caption_handoff_logs_the_host_never_the_url` |
| A config error echoing a URL or path | `tests/test_server.py::test_malformed_nudenet_url_is_refused_without_echoing_it` |
| Importing NudeNet (AGPL-3.0): it stays a separate server behind `NudeNetClient` | ruff `TID251` banned-api (`pyproject.toml`), run by `make lint` and CI |

## Facts the code doesn't show

- **The CLI is deliberately unconstrained**: it's your own shell. Containment and the move/delete gates are the server's.
- **`MANIFEST_VERSION` moves independently of the package version.** Bump the minor for an additive row change, the major for a breaking one, then `make manifest-snapshot`.
- **`TargetProfile` and its taxonomy are argus-cortex's** (`argus_cortex.taxonomy`). `models.py` subclasses it only to narrow `modality` to `image` and `target_category` to `TargetCategory` (curator curates images); every field and validator is cortex's. Every stage reads the profile verbatim, so change it in cortex, not here.
- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `_version.py` is generated. Tag `vX.Y.Z` to release (PyPI via OIDC).
- CI (argus-ci's reusable `python-ci.yml`) runs on pushes and PRs to `dev` and `main`: `ruff check`, `ruff format --check`, tests, build and `argus-curator schema --check`. `make check` covers most of that locally; also run `make fmt`.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.
