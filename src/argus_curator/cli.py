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
    typer.echo("=" * 56)
    if summary.reject_reasons:
        typer.echo("\nRejection reasons:")
        for reason, count in sorted(summary.reject_reasons.items(), key=lambda x: -x[1]):
            typer.echo(f"  {count:4d}  {reason}")

    if verbose:
        typer.echo("\nPer-image (grouped by cluster, then score):")
        for r in sorted(summary.results, key=lambda x: (x.similar_group, -x.score)):
            tag = f"g{r.similar_group}" + ("*" if r.is_representative and r.group_size > 1 else "")
            face = f" [{r.primary_face_cluster}]" if r.primary_face_cluster else ""
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
            write_manifest=dest is not None,
        )
        result = export_selection(summary, req)
        if csv_out:
            from argus_curator.export import write_report
            from argus_curator.selection import decide_selection

            selected, keep_reason = decide_selection(summary.results, req, summary.config.diversity_weight)
            write_report(summary.results, keep_reason, {r.rel_path for r in selected}, csv_out)
            typer.echo(f"CSV report written -> {csv_out}")
        if dest is not None:
            verb = {"copy": "Copied", "move": "Moved", "symlink": "Symlinked"}[mode]
            typer.echo(f"\n{verb} {result.copied} images -> {result.dest}")
            if result.manifest_path:
                typer.echo(f"Manifest -> {result.manifest_path}")

    typer.echo(f"\nDone. scan_id={summary.scan_id}")


@app.command()
def serve(
    port: int = Option(8101, "--port", "-p", help="Port to listen on"),
    host: str = Option("0.0.0.0", "--host", help="Host to bind to"),
    cors: bool = Option(False, "--cors", help="Enable CORS (allow all origins)"),
    source_root: str | None = Option(None, "--source-root", help="Default mount root for /thumb"),
) -> None:
    """Start the argus-curator micro-server (FastAPI) on :8101."""
    try:
        import uvicorn
    except ImportError as _exc:  # pragma: no cover
        typer.echo("Server requires: pip install argus-curator[server]", err=True)
        raise typer.Exit(1) from _exc

    from argus_curator.server import create_app

    application = create_app(cors=cors, source_root=source_root)
    uvicorn.run(application, host=host, port=port)


@app.command()
def detectors() -> None:
    """Report which optional detectors/backends are available."""
    from argus_curator.server.app import _detectors

    for name, ok in _detectors().items():
        marker = "+" if ok else "-"
        typer.echo(f"  [{marker}] {name}")


if __name__ == "__main__":
    app()
