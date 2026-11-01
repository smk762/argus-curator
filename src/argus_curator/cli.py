"""argus-curator CLI — ``serve`` and ``scan`` (renamed curate_by_rating.py)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

try:
    import typer
    from typer import Argument, Option
except ImportError as _exc:  # pragma: no cover
    print("CLI requires: pip install argus-curator[cli]", file=sys.stderr)
    raise SystemExit(1) from _exc

app = typer.Typer(
    name="argus-curator",
    help="LoRA-native dataset curation: score, dedup, face-cluster, export.",
    no_args_is_help=True,
)


def _csv_list(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _face_names(value: str | None) -> dict[str, str] | None:
    """Parse ``face_1=Ada,face_2=Bo``; the names are validated by ExportRequest."""
    if not value:
        return None
    out: dict[str, str] = {}
    for pair in _csv_list(value) or []:
        cluster, sep, name = pair.partition("=")
        if not sep:
            raise ValueError(f"expected CLUSTER=name, got {pair!r}")
        out[cluster.strip()] = name.strip()
    return out


def _label_thresholds(value: str | None) -> dict[str, float] | None:
    """Parse ``LABEL=score,LABEL=score``; the labels are validated by ExportRequest."""
    if not value:
        return None
    out: dict[str, float] = {}
    for pair in _csv_list(value) or []:
        label, sep, score = pair.partition("=")
        if not sep or not label.strip():
            raise ValueError(f"expected LABEL=score, got {pair!r}")
        try:
            out[label.strip()] = float(score)
        except ValueError:
            raise ValueError(f"{pair!r}: {score!r} is not a number") from None
    return out


@app.command()
def scan(
    source: Path = Argument(..., help="Root folder to scan (recurses into sub-folders)"),
    min_score: float = Option(0.6, "--min-score", help="Score threshold for keepers (0..1)"),
    copy_to: Path | None = Option(None, "--copy-to", help="Copy keepers here (structure preserved)"),
    move_to: Path | None = Option(None, "--move-to", help="Move keepers here instead of copying"),
    symlink_to: Path | None = Option(None, "--symlink-to", help="Symlink keepers here"),
    include_rejected: bool = Option(
        False, "--include-rejected", help="Keep high-scoring images that failed hard filters"
    ),
    keep_similar: bool = Option(False, "--keep-similar", help="Keep non-representative near-duplicates"),
    no_cluster: bool = Option(False, "--no-cluster", help="Disable near-duplicate grouping"),
    cluster_distance: int = Option(10, "--cluster-distance", help="pHash Hamming distance for near-duplicates"),
    max_keep: int | None = Option(None, "--max-keep", help="Cap keepers to a diverse N"),
    diversity_weight: float = Option(0.40, "--diversity-weight", help="0=pure score, 1=pure spread"),
    min_short_side: int = Option(512, "--min-short-side", help="Minimum short-side pixels"),
    max_aspect_ratio: float = Option(3.0, "--max-aspect-ratio", help="Maximum long:short ratio"),
    blur_threshold: float = Option(100.0, "--blur-threshold", help="Minimum Laplacian variance"),
    max_workers: int = Option(4, "--max-workers", help="Parallel worker threads"),
    style: str = Option("photo", "--target-style", help="photo or anime"),
    target_backend: str = Option("sdxl", "--target-backend", help="Diffusion backend family"),
    checkpoint: str | None = Option(None, "--checkpoint", help="Target checkpoint"),
    category: str = Option("identity", "--target-category", help="identity|wardrobe|pose_composition|setting"),
    faces: bool = Option(False, "--faces", help="Run InsightFace detection + identity clustering"),
    faces_model: str = Option("buffalo_l", "--faces-model", help="InsightFace model name"),
    min_det_score: float = Option(0.5, "--min-det-score", help="Minimum face detection score"),
    cluster_eps: float = Option(0.5, "--cluster-eps", help="Cosine-distance threshold for face clustering"),
    device: str = Option("auto", "--device", help="auto|cpu|cuda for face detection"),
    fairface: bool = Option(
        False, "--fairface", help="Predict race/gender/age bracket per face with FairFace (implies --faces)"
    ),
    mivolo: bool = Option(
        False, "--mivolo", help="Predict age in years + gender per face with MiVOLO (implies --faces)"
    ),
    fairface_model: Path | None = Option(
        None, "--fairface-model", help="Converted FairFace ONNX model (ARGUS_CURATOR_FAIRFACE_MODEL)"
    ),
    set_age: bool = Option(
        False,
        "--set-age",
        help="Give frames with no face the set's median MiVOLO age when one identity holds the scan (implies --mivolo)",
    ),
    mivolo_body: bool = Option(
        False,
        "--mivolo-body",
        help="Also give MiVOLO each face's body, from InsightFace's person detector (implies --mivolo; under evaluation)",
    ),
    mivolo_model: Path | None = Option(
        None, "--mivolo-model", help="Converted MiVOLO ONNX model (ARGUS_CURATOR_MIVOLO_MODEL)"
    ),
    scenes: bool = Option(False, "--scenes", help="Tag each image's scene + indoor/outdoor with Places365"),
    scene_top_k: int = Option(5, "--scene-top-k", help="Scene categories recorded per image"),
    places365_model: Path | None = Option(
        None, "--places365-model", help="Converted Places365 ONNX model (ARGUS_CURATOR_PLACES365_MODEL)"
    ),
    garments: bool = Option(False, "--garments", help="Detect garments (type, colour, cut-off) with Fashionpedia"),
    garments_model: Path | None = Option(
        None, "--garments-model", help="Converted garment-detector ONNX model (ARGUS_CURATOR_GARMENTS_MODEL)"
    ),
    face_clusters: str | None = Option(None, "--face-clusters", help="Comma-separated cluster ids to export"),
    face_poses: str | None = Option(
        None, "--pose", help="Export only these primary-face poses (comma-separated: frontal,three_quarter,profile)"
    ),
    face_races: str | None = Option(
        None,
        "--race",
        help="Export only these primary-face races (comma-separated: white,black,latino_hispanic,"
        "east_asian,southeast_asian,indian,middle_eastern)",
    ),
    face_genders: str | None = Option(None, "--gender", help="Export only these primary-face genders (male,female)"),
    face_ages: str | None = Option(
        None, "--age", help="Export only these primary-face age brackets (comma-separated, e.g. 20-29,30-39)"
    ),
    min_age_years: float | None = Option(
        None, "--min-age", help="Export only primary faces at least this old (MiVOLO)"
    ),
    max_age_years: float | None = Option(None, "--max-age", help="Export only primary faces at most this old (MiVOLO)"),
    min_attribute_conf: float = Option(
        0.0, "--min-attr-conf", help="Minimum confidence for a filtered attribute to count as a match (0..1)"
    ),
    scene_filter: str | None = Option(
        None,
        "--scene",
        help="Export only images showing one of these Places365 scenes (comma-separated, e.g. beach,coast)",
    ),
    environment: str | None = Option(
        None, "--environment", help="Export only indoor or outdoor images (indoor,outdoor)"
    ),
    min_scene_conf: float = Option(
        0.1, "--min-scene-conf", help="Minimum probability for a scene / environment to count as a match"
    ),
    diversify_scenes: bool = Option(
        False, "--diversify-scenes", help="With --max-keep, also spread the selection across scenes"
    ),
    garment_filter: str | None = Option(
        None, "--garment", help="Export only images wearing one of these garments (comma-separated, e.g. dress,skirt)"
    ),
    exclude_garments: str | None = Option(
        None, "--exclude-garment", help="Drop images showing any of these garments (e.g. bag_wallet,umbrella)"
    ),
    min_garment_conf: float = Option(
        0.5, "--min-garment-conf", help="Minimum probability for a garment to count (filters + outfit diversity)"
    ),
    garments_uncropped: bool = Option(
        False, "--uncropped", help="With --garment, the garment must not be cut off by the frame edge"
    ),
    diversify_outfits: bool = Option(
        False, "--diversify-outfits", help="With --max-keep, also spread the selection across outfits"
    ),
    nsfw: bool | None = Option(
        None,
        "--nsfw/--no-nsfw",
        help="NudeNet nudity check (default: on when a NudeNet URL is configured; --no-nsfw opts out)",
    ),
    nudenet_url: str | None = Option(
        None, "--nudenet-url", help="NudeNet server origin, e.g. http://nudenet:8080 (ARGUS_CURATOR_NUDENET_URL)"
    ),
    erax_url: str | None = Option(
        None, "--erax-url", help="EraX-NSFW server origin, e.g. http://erax:8080 (ARGUS_CURATOR_ERAX_URL)"
    ),
    nsfw_threshold: float | None = Option(
        None, "--nsfw-threshold", help="Drop images whose exposed-nudity score is at or above this (0.2..1)"
    ),
    nsfw_label_thresholds: str | None = Option(
        None,
        "--nsfw-label",
        help="Per-class thresholds, comma-separated LABEL=score (e.g. FEMALE_BREAST_EXPOSED=0.5,BELLY_EXPOSED=0.8)",
    ),
    nsfw_exposed_labels: str | None = Option(
        None, "--nsfw-exposed-labels", help="NudeNet classes --nsfw-threshold is taken over (default: the 5 exposed)"
    ),
    content_ratings: str | None = Option(
        None,
        "--content-ratings",
        help="Keep only these NudeNet content ratings, comma-separated: safe, suggestive, explicit",
    ),
    guard: bool | None = Option(
        None,
        "--guard/--no-guard",
        help="SingGuard policy check (default: on when a SingGuard URL is configured; --no-guard opts out)",
    ),
    singguard_url: str | None = Option(
        None, "--singguard-url", help="SingGuard server origin, e.g. http://guard:9150 (ARGUS_CURATOR_SINGGUARD_URL)"
    ),
    guard_policy: str = Option(
        "default",
        "--guard-policy",
        help="SingGuard policy: default (the policy file, else SingGuard's rules) or adult (allows adult nudity)",
    ),
    guard_variants: str | None = Option(
        None,
        "--guard-variants",
        help="Also judge every image under these policies, comma-separated (one more SingGuard pass each)",
    ),
    guard_policies: str | None = Option(
        None,
        "--guard-policies",
        help="Which policies' verdicts --guard-unsafe/--guard-rules read, comma-separated; any failing drops (default: the official one)",
    ),
    guard_unsafe: bool = Option(False, "--guard-unsafe", help="Drop images SingGuard judged unsafe, under any rule"),
    guard_rules: str | None = Option(
        None, "--guard-rules", help="Drop images where one of these SingGuard rules fired, comma-separated (e.g. A,C)"
    ),
    xmp: bool = Option(
        False, "--xmp", help="Write an XMP sidecar beside each exported file (face regions, names, content rating)"
    ),
    xmp_sources: bool = Option(
        False,
        "--xmp-sources",
        help=(
            "Merge face regions, names and content rating into sidecars beside every "
            "scanned original (including rejected ones; independent of --xmp / --copy-to). "
            "Writes into the scanned folder"
        ),
    ),
    face_names: str | None = Option(
        None, "--face-names", help="Person names for the sidecars by cluster, comma-separated (face_1=Ada,face_2=Bo)"
    ),
    csv_out: Path | None = Option(None, "--csv", help="Write the per-image CSV report here"),
    json_out: Path | None = Option(None, "--json", help="Write the full ScanSummary JSON here"),
    verbose: bool = Option(False, "--verbose", "-v", help="Print per-image details"),
) -> None:
    """Scan a folder, rank for training suitability, and optionally export keepers."""
    from pydantic import ValidationError

    from argus_curator.export import export_selection
    from argus_curator.models import (
        ExportRequest,
        FaceConfig,
        GarmentConfig,
        GuardConfig,
        NsfwConfig,
        ScanConfig,
        SceneConfig,
        TargetProfile,
    )
    from argus_curator.scanner import scan_folder as _scan
    from argus_curator.store import ScanStore

    if not source.is_dir():
        typer.echo(f"Error: not a directory: {source}", err=True)
        raise typer.Exit(1)
    import os

    from argus_curator import erax as erax_mod
    from argus_curator import fairface as fairface_mod
    from argus_curator import garments as garments_mod
    from argus_curator import guard as guard_mod
    from argus_curator import mivolo as mivolo_mod
    from argus_curator import nsfw as nsfw_mod
    from argus_curator import scenes as scenes_mod

    for model_file, env in (
        (fairface_model, fairface_mod.MODEL_ENV),
        (mivolo_model, mivolo_mod.MODEL_ENV),
        (places365_model, scenes_mod.MODEL_ENV),
        (garments_model, garments_mod.MODEL_ENV),
        (nudenet_url, nsfw_mod.URL_ENV),
        (erax_url, erax_mod.URL_ENV),
        (singguard_url, guard_mod.URL_ENV),
    ):
        if model_file is not None:
            os.environ[env] = str(model_file)

    profile = TargetProfile(
        target_style=style,
        target_backend=target_backend,
        checkpoint=checkpoint,
        target_category=category,
    )
    cfg = ScanConfig(
        min_short_side=min_short_side,
        max_aspect_ratio=max_aspect_ratio,
        blur_threshold=blur_threshold,
        cluster_distance=-1 if no_cluster else cluster_distance,
        diversity_weight=diversity_weight,
        max_workers=max_workers,
    )
    faces_cfg = FaceConfig(
        enabled=faces,
        model=faces_model,
        min_det_score=min_det_score,
        cluster_eps=cluster_eps,
        device=device,
        fairface=fairface,
        mivolo=mivolo,
        set_age=set_age,
        mivolo_body=mivolo_body,
    )

    scene_cfg = SceneConfig(enabled=scenes, top_k=scene_top_k, device=device)
    # Record garments at least as low as the export floor asks for: a scan
    # keeps nothing below GarmentConfig.min_conf, so a lower floor would match
    # nothing extra.
    garment_cfg = GarmentConfig(
        enabled=garments, device=device, min_conf=min(GarmentConfig().min_conf, min_garment_conf)
    )

    # Everything that can be wrong with the request fails here, before the
    # scan rather than after it: unknown labels, inverted ranges, missing models.
    dest = move_to or copy_to or symlink_to
    mode = "move" if move_to else ("symlink" if symlink_to else "copy")
    try:
        label_thresholds = _label_thresholds(nsfw_label_thresholds)
    except ValueError as exc:
        typer.echo(f"Error: --nsfw-label: {exc}", err=True)
        raise typer.Exit(2) from exc
    try:
        names = _face_names(face_names)
    except ValueError as exc:
        typer.echo(f"Error: --face-names: {exc}", err=True)
        raise typer.Exit(2) from exc
    if xmp and dest is None:
        typer.echo("Error: --xmp writes beside exported files; add --copy-to, --move-to or --symlink-to", err=True)
        raise typer.Exit(2)
    try:
        export_req = ExportRequest(
            dest=str(dest) if dest else str(source),
            mode=mode,
            min_score=min_score,
            include_rejected=include_rejected,
            keep_similar=keep_similar,
            max_keep=max_keep,
            face_clusters=[c.strip() for c in face_clusters.split(",")] if face_clusters else None,
            face_poses=_csv_list(face_poses),
            face_races=_csv_list(face_races),
            face_genders=_csv_list(face_genders),
            face_ages=_csv_list(face_ages),
            min_age_years=min_age_years,
            max_age_years=max_age_years,
            min_attribute_conf=min_attribute_conf,
            scenes=_csv_list(scene_filter),
            scene_environments=_csv_list(environment),
            min_scene_conf=min_scene_conf,
            diversify_scenes=diversify_scenes,
            garments=_csv_list(garment_filter),
            exclude_garments=_csv_list(exclude_garments),
            min_garment_conf=min_garment_conf,
            garments_uncropped=garments_uncropped,
            diversify_outfits=diversify_outfits,
            nsfw_threshold=nsfw_threshold,
            nsfw_label_thresholds=label_thresholds,
            content_ratings=_csv_list(content_ratings),
            guard_unsafe=guard_unsafe,
            guard_rules=_csv_list(guard_rules),
            guard_policies=_csv_list(guard_policies),
            **({"nsfw_exposed_labels": _csv_list(nsfw_exposed_labels)} if nsfw_exposed_labels else {}),
            write_manifest=dest is not None,
            write_xmp=xmp,
            face_names=names,
        )
    except ValidationError as exc:
        for err in exc.errors():
            field = ".".join(str(x) for x in err["loc"]) or "request"
            typer.echo(f"Error: {field}: {err['msg']}", err=True)
        raise typer.Exit(2) from exc

    from argus_curator.attributes import fairface_available, mivolo_available

    needed = (
        (fairface, fairface_available, "FairFace", fairface_mod.MODEL_ENV, "fairface-convert <weights.pt>"),
        (mivolo, mivolo_available, "MiVOLO", mivolo_mod.MODEL_ENV, "mivolo-convert"),
        (scenes, scenes_mod.scenes_available, "Places365", scenes_mod.MODEL_ENV, "places365-convert"),
        (garments, garments_mod.garments_available, "garment-detector", garments_mod.MODEL_ENV, "garments-convert"),
    )
    for wanted, available, name, env, command in needed:
        if wanted and not available():
            typer.echo(f"Error: no {name} model (run `argus-curator {command}` or set {env})", err=True)
            raise typer.Exit(1)

    nsfw_cfg = NsfwConfig(enabled=nsfw)
    nsfw_on = nsfw_mod.resolve_enabled(nsfw_cfg)
    if nsfw_on:
        try:
            nudenet_endpoint = nsfw_mod.endpoint()
            if erax_mod.resolve_enabled(nsfw_cfg):
                erax_mod.endpoint()
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if nudenet_endpoint is None:
            typer.echo(f"Error: --nsfw needs a NudeNet server (--nudenet-url or {nsfw_mod.URL_ENV})", err=True)
            raise typer.Exit(1)
    elif nsfw_threshold is not None or label_thresholds or content_ratings:
        why = "--no-nsfw was given" if nsfw is False else f"no NudeNet server is configured ({nsfw_mod.URL_ENV})"
        typer.echo(f"Error: NSFW thresholds and content ratings need the NudeNet check, but {why}", err=True)
        raise typer.Exit(1)

    variant_names = _csv_list(guard_variants) or []
    for name in (guard_policy, *variant_names):
        if name not in guard_mod.POLICIES:
            typer.echo(f"Error: unknown SingGuard policy {name!r} (one of {', '.join(guard_mod.POLICIES)})", err=True)
            raise typer.Exit(1)
    guard_cfg = GuardConfig(enabled=guard, policy=guard_policy, variants=variant_names)
    ran = [guard_cfg.policy, *guard_cfg.variants]
    if not_run := [name for name in export_req.guard_policies or [] if name not in ran]:
        typer.echo(f"Error: --guard-policies {', '.join(not_run)} not judged (add them to --guard-variants)", err=True)
        raise typer.Exit(1)
    if guard_mod.resolve_enabled(guard_cfg):
        try:
            guard_mod.endpoint()
            guard_mod.thinking()
            for name in (guard_cfg.policy, *guard_cfg.variants):
                guard_mod.policy(name)
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if not guard_mod.configured():
            typer.echo(f"Error: --guard needs a SingGuard server (--singguard-url or {guard_mod.URL_ENV})", err=True)
            raise typer.Exit(1)
    elif export_req.guard_unsafe or export_req.guard_rules:
        why = "--no-guard was given" if guard is False else f"no SingGuard server is configured ({guard_mod.URL_ENV})"
        typer.echo(f"Error: SingGuard filters need the SingGuard check, but {why}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {source} (recursive) ...")
    summary = _scan(
        source,
        profile,
        cfg,
        faces_cfg,
        scene_cfg=scene_cfg,
        garment_cfg=garment_cfg,
        nsfw_cfg=nsfw_cfg,
        guard_cfg=guard_cfg,
    )

    typer.echo("=" * 56)
    typer.echo(f"  Scanned:               {summary.total}")
    typer.echo(f"  Passed hard filters:   {summary.passed}")
    typer.echo(f"  Similar clusters (>1): {summary.similar_clusters}  ({summary.duplicates} non-representative)")
    if summary.face_clusters:
        typer.echo(f"  Face identities:       {len(summary.face_clusters)}")
    pose_counts: dict[str, int] = {}
    for r in summary.results:
        if r.primary_face_pose:
            pose_counts[r.primary_face_pose] = pose_counts.get(r.primary_face_pose, 0) + 1
    if pose_counts:
        dist = "  ".join(f"{p}={n}" for p, n in sorted(pose_counts.items()))
        typer.echo(f"  Primary-face pose:     {dist}")
    for name in ("race", "gender", "age"):
        counts: dict[str, int] = {}
        for r in summary.results:
            label = getattr(r.primary_face_attributes, name, None)
            if label:
                counts[label] = counts.get(label, 0) + 1
        if counts:
            dist = "  ".join(f"{k}={n}" for k, n in sorted(counts.items()))
            typer.echo(f"  Primary-face {name + ':':<9}{dist}")
    env_counts: dict[str, int] = {}
    scene_counts: dict[str, int] = {}
    for r in summary.results:
        if r.scene:
            env_counts[r.scene.environment] = env_counts.get(r.scene.environment, 0) + 1
            if r.scene.categories:
                top = r.scene.categories[0].label
                scene_counts[top] = scene_counts.get(top, 0) + 1
    if env_counts:
        typer.echo(f"  Environment:           {'  '.join(f'{k}={n}' for k, n in sorted(env_counts.items()))}")
        common = sorted(scene_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        typer.echo(f"  Top scenes:            {'  '.join(f'{k}={n}' for k, n in common)}")
    garment_counts: dict[str, int] = {}
    for r in summary.results:
        for label in {g.label for g in r.garments or [] if g.conf >= min_garment_conf}:
            garment_counts[label] = garment_counts.get(label, 0) + 1
    if garment_counts:
        common = sorted(garment_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        typer.echo(f"  Garments (images):     {'  '.join(f'{k}={n}' for k, n in common)}")
    if summary.nsfw_config.enabled:
        # A row EraX filled after NudeNet failed was not checked by NudeNet,
        # but its EraX hits still count as exposed.
        scored = [r for r in summary.results if r.nsfw is not None]
        checked = [r for r in scored if not r.nudenet_failed]
        # Count against the threshold the export will apply, over its classes.
        at = export_req.nsfw_threshold if export_req.nsfw_threshold is not None else 0.5
        exposed = sum(1 for r in scored if nsfw_mod.exposed_score(r.nsfw.labels, export_req.nsfw_exposed_labels) >= at)
        unchecked = summary.total - len(checked)
        tail = f"  ({unchecked} could not be checked)" if unchecked else ""
        typer.echo(f"  NudeNet:               {len(checked)} checked, {exposed} with exposed score >= {at:g}{tail}")
    if summary.guard_config.enabled:
        judged = [r.guard for r in summary.results if r.guard is not None]
        by_rule = Counter(guard_mod.rule_letter(rule) for g in judged for rule in g.rules)
        rules = f" ({', '.join(f'{k}={n}' for k, n in sorted(by_rule.items()))})" if by_rule else ""
        unchecked = summary.total - len(judged)
        tail = f"  ({unchecked} could not be checked)" if unchecked else ""
        unsafe = sum(1 for g in judged if not g.safe)
        typer.echo(f"  SingGuard:             {len(judged)} checked, {unsafe} unsafe{rules}{tail}")
    typer.echo("=" * 56)
    if summary.reject_reasons:
        typer.echo("\nRejection reasons:")
        for reason, count in sorted(summary.reject_reasons.items(), key=lambda x: -x[1]):
            typer.echo(f"  {count:4d}  {reason}")

    if verbose:
        typer.echo("\nPer-image (grouped by cluster, then score):")
        for r in sorted(summary.results, key=lambda x: (x.similar_group, -x.score)):
            tag = f"g{r.similar_group}" + ("*" if r.is_representative and r.group_size > 1 else "")
            if r.primary_face_cluster:
                pose = f" {r.primary_face_pose}" if r.primary_face_pose else ""
                a = r.primary_face_attributes
                age = f"{a.age_years:.0f}y" if a and a.age_years is not None else (a.age if a else None)
                attrs = " ".join(str(v) for v in (a.race, a.gender, age) if v) if a else ""
                attrs = f" {attrs}" if attrs else ""
                face = f" [{r.primary_face_cluster}{pose}{attrs}]"
            else:
                face = ""
            scene = ""
            if r.scene and r.scene.categories:
                scene = f" <{r.scene.categories[0].label}, {r.scene.environment}>"
            worn = sorted({g.label for g in r.garments or [] if g.conf >= min_garment_conf})
            if worn:
                scene += f" {{{', '.join(worn)}}}"
            if r.nsfw is not None and r.nsfw.exposed_score > 0:
                scene += f" nsfw={r.nsfw.exposed_score:.2f}"
            if r.guard is not None and not r.guard.safe:
                scene += f" guard=unsafe[{','.join(guard_mod.rule_letter(rule) for rule in r.guard.rules)}]"
            typer.echo(f"  {r.score:.3f} {tag:>5}  {r.rel_path}{face}{scene}")

    # Persist so the result is reusable by the server / export-by-id.
    ScanStore().save(summary)

    if xmp_sources:
        from argus_curator.export import write_source_sidecars

        written = write_source_sidecars(summary.results, export_req.face_names)
        typer.echo(f"\nXMP sidecars merged beside {written} scanned originals")

    if json_out:
        json_out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"\nJSON summary written -> {json_out}")

    if dest is not None or csv_out is not None:
        req = export_req.model_copy(update={"scan_id": summary.scan_id})
        # With no --copy-to/--move-to/--symlink-to there is nowhere to transfer
        # to: exporting would copy every file onto itself (SameFileError per
        # image) for a report-only run, so skip it and report no exported paths.
        result = export_selection(summary, req) if dest is not None else None
        if csv_out:
            from argus_curator.export import write_report
            from argus_curator.selection import decide_selection

            selected, keep_reason = decide_selection(summary.results, req, summary.config.diversity_weight)
            exported_paths = result.exported_paths if result is not None else {}
            write_report(summary.results, keep_reason, {r.rel_path for r in selected}, exported_paths, csv_out)
            typer.echo(f"CSV report written -> {csv_out}")
        if result is not None:
            verb = {"copy": "Copied", "move": "Moved", "symlink": "Symlinked"}[mode]
            typer.echo(f"\n{verb} {result.copied} images -> {result.dest}")
            if result.manifest_path:
                typer.echo(f"Manifest -> {result.manifest_path}")
            if xmp:
                typer.echo(f"XMP sidecars -> {result.xmp_written} exported files")

    typer.echo(f"\nDone. scan_id={summary.scan_id}")


@app.command()
def serve(
    port: int = Option(8101, "--port", "-p", help="Port to listen on"),
    host: str = Option("0.0.0.0", "--host", help="Host to bind to"),
    cors: bool = Option(False, "--cors", help="Enable CORS for the localhost dev frontend (:3000)"),
    cors_origin: list[str] = Option(
        [], "--cors-origin", help="Allowed CORS origin (repeatable; implies CORS; or CURATOR_CORS_ORIGINS)"
    ),
    cors_any: bool = Option(
        False, "--cors-any", help="Allow ANY origin, credential-less (public demos only; implies CORS)"
    ),
    source_root: str | None = Option(
        None, "--source-root", help="Root that scan/thumb/upload paths resolve under (ARGUS_CURATOR_SCAN_ROOT)"
    ),
    export_root: str | None = Option(
        None, "--export-root", help="Root that export destinations resolve under (ARGUS_CURATOR_EXPORT_ROOT)"
    ),
    allow_move: bool = Option(False, "--allow-move", help='Permit destructive mode="move" exports (default: rejected)'),
) -> None:
    """Start the argus-curator micro-server (FastAPI) on :8101.

    Scan folders and export destinations from request bodies are resolved
    relative to --source-root / --export-root; the endpoints refuse when their
    root is not configured.
    """
    try:
        import uvicorn
    except ImportError as _exc:  # pragma: no cover
        typer.echo("Server requires: pip install argus-curator[server]", err=True)
        raise typer.Exit(1) from _exc

    from argus_curator.server import create_app

    application = create_app(
        cors=cors,
        cors_origins=cors_origin or None,
        cors_allow_any=cors_any,
        source_root=source_root,
        export_root=export_root,
        allow_move=allow_move or None,  # None -> fall back to CURATOR_ALLOW_MOVE
    )
    uvicorn.run(application, host=host, port=port)


@app.command()
def detectors() -> None:
    """Report which optional detectors/backends are available."""
    from argus_curator.server.app import _detectors

    for name, ok in _detectors().items():
        marker = "+" if ok else "-"
        typer.echo(f"  [{marker}] {name}")


@app.command("fairface-convert")
def fairface_convert(
    weights: Path = Argument(..., help="Official FairFace 7-race weights (fairface_alldata_20191111.pt)"),
    output: Path | None = Option(None, "--output", "-o", help="Where to write the ONNX model (default: model cache)"),
) -> None:
    """Convert the FairFace PyTorch weights to the ONNX model --fairface runs.

    Download the weights from the FairFace repo (github.com/dchen236/FairFace,
    CC BY 4.0). Needs: pip install argus-curator[fairface-convert]
    """
    from argus_curator.attributes import AttributesUnavailable
    from argus_curator.fairface import export_onnx, model_path

    if not weights.is_file():
        typer.echo(f"Error: not a file: {weights}", err=True)
        raise typer.Exit(1)
    try:
        written = export_onnx(weights, output or model_path())
    except AttributesUnavailable as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote FairFace ONNX model -> {written}")


@app.command("mivolo-convert")
def mivolo_convert(
    weights: Path | None = Option(
        None, "--weights", help="Local model.safetensors (default: download the pinned iitolstykh/mivolo_v2)"
    ),
    output: Path | None = Option(None, "--output", "-o", help="Where to write the ONNX model (default: model cache)"),
) -> None:
    """Fetch, verify and convert MiVOLO v2 to the ONNX model --mivolo runs.

    Weights: huggingface.co/iitolstykh/mivolo_v2 (Apache-2.0), pinned by commit
    and sha256. Needs: pip install argus-curator[mivolo-convert]
    """
    from argus_curator.attributes import AttributesUnavailable
    from argus_curator.mivolo import export_onnx, fetch_weights, model_path

    try:
        source = weights or fetch_weights()
        written = export_onnx(source, output or model_path())
    except AttributesUnavailable as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote MiVOLO ONNX model -> {written}")


@app.command("places365-convert")
def places365_convert(
    arch: str = Option("resnet18", "--arch", help="resnet18 (default, 45 MB) or resnet50 (97 MB, ~1 pt better top-1)"),
    weights: Path | None = Option(
        None, "--weights", help="Local <arch>_places365.pth.tar (default: download it; sha256-checked)"
    ),
    output: Path | None = Option(None, "--output", "-o", help="Where to write the ONNX model (default: model cache)"),
) -> None:
    """Fetch, verify and convert a Places365-CNN to the ONNX model --scenes runs.

    Weights: places2.csail.mit.edu (CC BY), pinned by sha256 since upstream
    serves them over plain HTTP. Needs: pip install argus-curator[places365-convert]
    """
    from argus_curator.runtime import ModelUnavailable
    from argus_curator.scenes import WEIGHTS, export_onnx, fetch_weights, model_path

    if arch not in WEIGHTS:
        typer.echo(f"Error: --arch must be one of {', '.join(WEIGHTS)}", err=True)
        raise typer.Exit(1)
    try:
        source = weights or fetch_weights(arch)
        written = export_onnx(source, arch, output or model_path())
    except ModelUnavailable as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote Places365 ({arch}) ONNX model -> {written}")


@app.command("garments-convert")
def garments_convert(
    checkpoint: Path | None = Option(
        None, "--checkpoint", help="Local dir with config.json + pytorch_model.bin (default: download the pinned one)"
    ),
    output: Path | None = Option(None, "--output", "-o", help="Where to write the ONNX model (default: model cache)"),
) -> None:
    """Fetch, verify and convert the Fashionpedia garment detector for --garments.

    Weights: huggingface.co/valentinafevu/yolos-fashionpedia (MIT; trained on
    Fashionpedia, CC BY 4.0), pinned by commit and sha256.
    Needs: pip install argus-curator[garments-convert]
    """
    from argus_curator.garments import export_onnx, fetch_weights, model_path
    from argus_curator.runtime import ModelUnavailable

    try:
        source = checkpoint or fetch_weights()
        written = export_onnx(source, output or model_path())
    except ModelUnavailable as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote garment-detector ONNX model -> {written}")


DEFAULT_SCHEMA_PATH = Path("schema/curator-wire.schema.json")


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against.

    Run without flags to (re)write the committed schema; run with --check in CI
    to fail if the models have drifted from the committed artifact.
    """
    import json

    from argus_curator.models import wire_schema

    rendered = json.dumps(wire_schema(), indent=2, sort_keys=True) + "\n"

    if check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != rendered:
            typer.echo(f"{output} is stale — run `argus-curator schema` and commit the result.", err=True)
            raise typer.Exit(1)
        typer.echo(f"{output} is up to date.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(f"Wrote wire schema -> {output}")


if __name__ == "__main__":
    app()
