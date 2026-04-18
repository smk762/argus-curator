"""CLI entry point for argus-curator."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="argus-curator",
        description="Dataset curation for LoRA training.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── scan ──────────────────────────────────────────────────────────────
    scan_p = sub.add_parser("scan", help="Scan a folder and print a curation summary.")
    scan_p.add_argument("folder", help="Path to image folder.")
    scan_p.add_argument("--objective", default="identity",
                        choices=["identity", "style", "wardrobe", "concept"])
    scan_p.add_argument("--target-style", default="photo", choices=["photo", "anime"])
    scan_p.add_argument("--target-count", type=int, default=None)
    scan_p.add_argument("--top-percent", type=float, default=80.0)
    scan_p.add_argument("--diversity-weight", type=float, default=0.40)
    scan_p.add_argument("--blur-threshold", type=float, default=100.0)
    scan_p.add_argument("--min-short-side", type=int, default=512)
    scan_p.add_argument("--no-clip", action="store_true",
                        help="Disable CLIP embedding (faster, less accurate diversity).")
    scan_p.add_argument("--dino", action="store_true",
                        help="Enable DINOv2 embeddings alongside CLIP.")
    scan_p.add_argument("--yolo", action="store_true", help="Enable YOLO person detection.")
    scan_p.add_argument("--mtcnn", action="store_true", help="Enable MTCNN face detection.")
    scan_p.add_argument("--output", default="-",
                        help="Output JSON path. '-' for stdout.")
    scan_p.add_argument("--no-preset", action="store_true",
                        help="Disable objective preset (use raw defaults).")

    # ── export ────────────────────────────────────────────────────────────
    export_p = sub.add_parser("export", help="Copy/move selected images from a previous scan.")
    export_p.add_argument("scan_result", help="Path to scan result JSON.")
    export_p.add_argument("target_folder", help="Destination folder.")
    export_p.add_argument("--move", action="store_true",
                          help="Move files instead of copying.")

    # ── serve ─────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the FastAPI HTTP server.")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8101)
    serve_p.add_argument(
        "--cors",
        action="store_true",
        help="Enable CORS. If omitted, ARGUS_CURATOR_CORS controls CORS (see server env docs).",
    )

    args = parser.parse_args()

    if args.command == "scan":
        _cmd_scan(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "serve":
        _cmd_serve(args)


def _cmd_scan(args: argparse.Namespace) -> None:
    from argus_curator.types import CurateConfig
    from argus_curator.scanner import scan_folder

    cfg = CurateConfig(
        objective=args.objective,
        target_style=args.target_style,
    )
    if not args.no_preset:
        from argus_curator.presets import apply_preset
        apply_preset(cfg)

    cfg.filters.blur_threshold = args.blur_threshold
    cfg.filters.min_short_side = args.min_short_side
    cfg.embeddings.use_clip = not args.no_clip
    cfg.embeddings.use_dino = args.dino
    cfg.detectors.use_yolo = args.yolo
    cfg.detectors.use_mtcnn = args.mtcnn
    cfg.selection.top_percent = args.top_percent
    cfg.selection.diversity_weight = args.diversity_weight
    if args.target_count:
        cfg.selection.target_count = args.target_count

    summary = scan_folder(args.folder, cfg)
    data = summary.to_dict()

    if args.output == "-":
        json.dump(data, sys.stdout, indent=2)
        print()
    else:
        import pathlib
        pathlib.Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"Results written to {args.output}")
        print(f"Total: {data['total']}  Selected: {data['selected']}  "
              f"Rejected: {data['rejected_filters']}  Duplicates: {data['duplicates_removed']}")


def _cmd_export(args: argparse.Namespace) -> None:
    import pathlib
    import shutil

    scan_data = json.loads(pathlib.Path(args.scan_result).read_text())
    target = pathlib.Path(args.target_folder)
    target.mkdir(parents=True, exist_ok=True)

    results = scan_data.get("results", [])
    selected = [r for r in results if r.get("selected")]

    moved, failed = 0, 0
    op = "Moving" if args.move else "Copying"
    print(f"{op} {len(selected)} selected images to {target}…")

    for r in selected:
        src_str: str = r.get("source", "")
        if src_str.startswith("local:"):
            src = pathlib.Path(src_str[len("local:"):])
            rel_name = str(r.get("name") or src.name).replace("\\", "/")
            dest = target.joinpath(*pathlib.Path(rel_name).parts)
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if args.move:
                    shutil.move(str(src), str(dest))
                else:
                    shutil.copy2(str(src), str(dest))
                moved += 1
            except Exception as exc:
                print(f"  FAIL {src.name}: {exc}", file=sys.stderr)
                failed += 1

    print(f"Done. {moved} {'moved' if args.move else 'copied'}, {failed} failed.")


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install 'argus-curator[server]'", file=sys.stderr)
        sys.exit(1)

    from argus_curator.server import create_app
    app = create_app(cors=True if args.cors else None)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
