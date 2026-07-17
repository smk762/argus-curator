"""argus-curator CLI — ``serve`` and ``scan`` (renamed curate_by_rating.py)."""

from __future__ import annotations

import sys
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
    face_clusters: str | None = Option(None, "--face-clusters", help="Comma-separated cluster ids to export"),
    face_poses: str | None = Option(
        None, "--pose", help="Export only these primary-face poses (comma-separated: frontal,three_quarter,profile)"
    ),
    csv_out: Path | None = Option(None, "--csv", help="Write the per-image CSV report here"),
    json_out: Path | None = Option(None, "--json", help="Write the full ScanSummary JSON here"),
    verbose: bool = Option(False, "--verbose", "-v", help="Print per-image details"),
) -> None:
    """Scan a folder, rank for training suitability, and optionally export keepers."""
    from argus_curator.export import export_selection
    from argus_curator.models import ExportRequest, FaceConfig, ScanConfig, TargetProfile
    from argus_curator.scanner import scan_folder as _scan
    from argus_curator.store import ScanStore

    if not source.is_dir():
        typer.echo(f"Error: not a directory: {source}", err=True)
        raise typer.Exit(1)

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
    )

    typer.echo(f"Scanning {source} (recursive) ...")
    summary = _scan(source, profile, cfg, faces_cfg)

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
                face = f" [{r.primary_face_cluster}{pose}]"
            else:
                face = ""
            typer.echo(f"  {r.score:.3f} {tag:>5}  {r.rel_path}{face}")

    # Persist so the result is reusable by the server / export-by-id.
    ScanStore().save(summary)

    if json_out:
        json_out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"\nJSON summary written -> {json_out}")

    dest = move_to or copy_to or symlink_to
    if dest is not None or csv_out is not None:
        mode = "move" if move_to else ("symlink" if symlink_to else "copy")
        req = ExportRequest(
            scan_id=summary.scan_id,
            dest=str(dest) if dest else str(source),
            mode=mode,
            min_score=min_score,
            include_rejected=include_rejected,
            keep_similar=keep_similar,
            max_keep=max_keep,
            face_clusters=[c.strip() for c in face_clusters.split(",")] if face_clusters else None,
            face_poses=[p.strip() for p in face_poses.split(",")] if face_poses else None,
            write_manifest=dest is not None,
        )
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
