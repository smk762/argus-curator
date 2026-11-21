"""argus-curator CLI — ``serve`` and ``scan`` (renamed curate_by_rating.py)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from argus_cortex.taxonomy import SPEECH_CATEGORIES, TARGET_CATEGORIES

if TYPE_CHECKING:
    from pydantic import ValidationError

    from argus_curator.models import ExportRequest, ScanSummary

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


_MINOR_EXPOSED_HELP = (
    "Under-18 rule: frame (default; a young age and any exposure in the frame) or overlap "
    "(the exposure must also sit on a young person's box). Or ARGUS_CURATOR_MINOR_EXPOSED"
)


def _apply_minor_exposed(value: str | None) -> None:
    """Set ``ARGUS_CURATOR_MINOR_EXPOSED`` from *value*, then check it; exit 2 on an unknown mode."""
    import os

    from argus_curator.selection import MINOR_EXPOSED_ENV, MinorExposedConfigError, minor_exposed_mode

    if value is not None:
        os.environ[MINOR_EXPOSED_ENV] = value
    try:
        minor_exposed_mode()
    except MinorExposedConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _invalid(exc: ValidationError) -> typer.Exit:
    """Print each field of a pydantic ``ValidationError``; the exit to raise after it."""
    for err in exc.errors():
        field = ".".join(str(x) for x in err["loc"]) or "request"
        typer.echo(f"Error: {field}: {err['msg']}", err=True)
    return typer.Exit(2)


def _outside_references(source: Path, refs: list[str]) -> list[Path] | None:
    """The reference files when any of *refs* is not an image in *source*, else None.

    References in the scanned folder stay rel paths (``FaceConfig.reference_paths``).
    One outside it (a portrait or consent photo kept apart from the set) makes all of
    them files, read here and averaged into ``FaceConfig.reference_embedding``.
    """

    def in_scan(ref: str) -> bool:
        return not Path(ref).is_absolute() and (source / ref).is_file()

    if all(in_scan(ref) for ref in refs):
        return None
    files = [source / ref if in_scan(ref) else Path(ref) for ref in refs]
    for f in files:
        if not f.is_file():
            typer.echo(f"Error: --face-reference {f}: no such file", err=True)
            raise typer.Exit(2)
    return files


@app.command()
def scan(
    source: Path = Argument(..., help="Root folder to scan (recurses into sub-folders)"),
    min_score: float = Option(0.6, "--min-score", help="Score threshold for keepers (0..1)"),
    copy_to: Path | None = Option(None, "--copy-to", help="Copy keepers here (structure preserved)"),
    move_to: Path | None = Option(None, "--move-to", help="Move keepers here instead of copying"),
    symlink_to: Path | None = Option(None, "--symlink-to", help="Symlink keepers here"),
    keep_similar: bool = Option(False, "--keep-similar", help="Keep non-representative near-duplicates"),
    no_cluster: bool = Option(False, "--no-cluster", help="Disable near-duplicate grouping"),
    cluster_distance: int = Option(10, "--cluster-distance", help="pHash Hamming distance for near-duplicates"),
    semantic_dupes: bool = Option(
        False, "--semantic-dupes", help="Also group near-duplicates by CLIP embedding similarity (needs --embeddings)"
    ),
    semantic_dupes_cosine: float = Option(
        0.95, "--semantic-dupes-cosine", help="With --semantic-dupes, the cosine similarity that makes a duplicate"
    ),
    profile_fit: bool = Option(
        False, "--profile-fit", help="Score each image's zero-shot CLIP fit to the target profile (needs --embeddings)"
    ),
    junk_check: bool = Option(
        False,
        "--junk-check",
        help="Score each image's zero-shot CLIP junk probability: screenshot, collage, text, meme (needs --embeddings)",
    ),
    junk_threshold: float = Option(
        0.8, "--junk-threshold", help="The junk probability that names a junk kind (and that --drop-junk drops at)"
    ),
    max_keep: int | None = Option(None, "--max-keep", help="Cap keepers to a diverse N"),
    diversity_weight: float = Option(0.40, "--diversity-weight", help="0=pure score, 1=pure spread"),
    min_short_side: int = Option(512, "--min-short-side", help="Minimum short-side pixels (512 at least)"),
    max_aspect_ratio: float = Option(3.0, "--max-aspect-ratio", help="Maximum long:short ratio"),
    blur_threshold: float = Option(100.0, "--blur-threshold", help="Minimum Laplacian variance"),
    max_workers: int = Option(4, "--max-workers", help="Parallel worker threads"),
    style: str = Option("photo", "--target-style", help="photo or anime"),
    target_backend: str | None = Option(
        None, "--target-backend", help="Diffusion backend family (default sdxl), or a TTS family for speech"
    ),
    checkpoint: str | None = Option(None, "--checkpoint", help="Target checkpoint"),
    category: str | None = Option(
        None,
        "--target-category",
        help=f"{'|'.join(TARGET_CATEGORIES)} (default identity); for speech {'|'.join(SPEECH_CATEGORIES)} (default voice)",
    ),
    modality: str = Option(
        "image", "--modality", help="image, or speech to scan the folder's WAV/FLAC clips and transcripts instead"
    ),
    transcripts: str | None = Option(
        None, "--transcripts", help="Speech: auto (default), sidecar, ljspeech or common_voice"
    ),
    mos: bool = Option(False, "--mos", help="Speech: score each passing clip with DNSMOS (see dnsmos-fetch)"),
    dnsmos_model: Path | None = Option(
        None, "--dnsmos-model", help="Speech: DNSMOS ONNX model (ARGUS_CURATOR_DNSMOS_MODEL)"
    ),
    speech_speakers: bool = Option(
        False, "--speakers", help="Speech: embed and cluster each passing clip's speaker (see speaker-fetch)"
    ),
    speaker_model: Path | None = Option(
        None, "--speaker-model", help="Speech: speaker-embedding ONNX model (ARGUS_CURATOR_SPEAKER_MODEL)"
    ),
    speech_config: Path | None = Option(
        None, "--speech-config", help="Speech: a JSON SpeechConfig with the thresholds (the flags above override it)"
    ),
    speaker_ids: str | None = Option(
        None, "--speaker-id", help="Speech export: keep only these speakers (comma-separated speaker_id)"
    ),
    accents: str | None = Option(
        None, "--accent", help="Speech export: keep only these accents (comma-separated corpus labels, exact)"
    ),
    languages: str | None = Option(
        None, "--language", help="Speech export: keep only these languages (comma-separated corpus labels, exact)"
    ),
    exclude_speaker_mismatch: bool = Option(
        False,
        "--exclude-speaker-mismatch",
        help="Speech export: drop clips whose corpus speaker label the clustering disputes (needs --speakers)",
    ),
    min_hours_per_speaker: float | None = Option(
        None, "--min-hours-per-speaker", help="Speech export: drop a speaker with fewer hours of kept audio"
    ),
    max_hours_per_speaker: float | None = Option(
        None, "--max-hours-per-speaker", help="Speech export: cap each speaker's hours, best clips first"
    ),
    max_hours_per_accent: float | None = Option(
        None, "--max-hours-per-accent", help="Speech export: cap each accent's hours, taking its speakers in turn"
    ),
    balance_accents: bool = Option(
        False, "--balance-accents", help="Speech export: cap every accent at the smallest accent's hours"
    ),
    corpus: str | None = Option(
        None, "--corpus", help="Speech export: the licensed corpus every clip comes from (with --license)"
    ),
    corpus_license: str | None = Option(
        None, "--license", help="Speech export: the corpus's licence, e.g. CC0-1.0 (with --corpus)"
    ),
    voice_consent: str | None = Option(
        None,
        "--voice-consent",
        help="Speech export: a real person's voice consent record (argus-cortex ConsentRecord JSON), "
        "relative to the scanned folder or absolute, instead of --corpus (with --consent-delivery; needs --speakers)",
    ),
    consent_delivery: str | None = Option(
        None, "--consent-delivery", help="Speech export: the voice delivery --voice-consent must cover, e.g. neutral"
    ),
    held_out: int | None = Option(
        None, "--held-out", help="Speech export: hold out about this many clips, by whole speakers (or a fraction)"
    ),
    held_out_fraction: float | None = Option(
        None, "--held-out-fraction", help="Speech export: hold out about this fraction of the clips, by whole speakers"
    ),
    split_method: str = Option(
        "spread",
        "--split-method",
        help="Speech export: spread (held-out speakers evenly through the ranking) or random",
    ),
    split_seed: int = Option(0, "--split-seed", help="Speech export: the seed --split-method random shuffles with"),
    split_part: str = Option(
        "train", "--split-part", help="Speech export: which part of the split to export, train or held_out"
    ),
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
    embeddings: bool = Option(
        False, "--embeddings", help="Compute CLIP image embeddings (memory only) for --diversify-semantic"
    ),
    embeddings_model: Path | None = Option(
        None, "--embeddings-model", help="Converted CLIP image-tower ONNX model (ARGUS_CURATOR_EMBEDDINGS_MODEL)"
    ),
    face_clusters: str | None = Option(None, "--face-clusters", help="Comma-separated cluster ids to export"),
    face_reference: list[str] = Option(
        [],
        "--face-reference",
        help="An image whose face is the identity reference: a path relative to the scanned folder, "
        "or any image file outside it; repeatable (several are averaged). "
        "Each face then gets a reference_distance (implies --faces)",
    ),
    face_source: bool = Option(
        False,
        "--face-source",
        help="Export only face-source images (one identity, frontal and level, large, sharp, eyes open, "
        "mouth closed, adult; needs --mivolo), ranked best first",
    ),
    consent: str | None = Option(
        None,
        "--consent",
        help="A real person's consent record (argus-cortex ConsentRecord JSON), relative to the scanned folder "
        "or absolute: bind the set to its reference images (implies --mivolo --set-age)",
    ),
    consent_max_distance: float = Option(
        0.4, "--consent-max-distance", help="Ceiling on a face's distance from the record's reference to be bound"
    ),
    face_source_consent: bool = Option(
        False,
        "--face-source-consent",
        help="Export a consented face set: --face-source over the images bound to --consent's record, re-checked "
        "at export, adult only; refused when none qualifies (implies --face-source)",
    ),
    landmarks: bool = Option(False, "--landmarks", help="Write the primary face's landmarks into the manifest"),
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
    diversify_semantic: bool = Option(
        False,
        "--diversify-semantic",
        help="With --max-keep, also spread the selection semantically (CLIP embeddings; needs --embeddings)",
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
    watermark_url: str | None = Option(
        None,
        "--watermark-url",
        help="Watermark detector origin, e.g. http://watermark:8080 (ARGUS_CURATOR_WATERMARK_URL)",
    ),
    nsfw_threshold: float | None = Option(
        None,
        "--nsfw-threshold",
        help="Drop images whose exposed-nudity score is at or above this (0.2..1; 0.2, any reported exposure, is recommended)",
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
    body_pose: bool | None = Option(
        None,
        "--body-pose/--no-body-pose",
        help="Body pose from argus-puppet (default: on when a puppet URL is configured; --no-body-pose opts out)",
    ),
    puppet_url: str | None = Option(
        None, "--puppet-url", help="argus-puppet server origin, e.g. http://puppet:8220 (ARGUS_CURATOR_PUPPET_URL)"
    ),
    guard_policy: str = Option(
        "default",
        "--guard-policy",
        help="SingGuard policy: default (the policy file, else SingGuard's rules) or adult (allows adult nudity)",
    ),
    guard_thinking: str | None = Option(
        None,
        "--guard-thinking",
        help="SingGuard mode: fast, fast-slow, slow or cascade (default: ARGUS_CURATOR_SINGGUARD_THINKING, else cascade)",
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
    drop_nsfw: bool = Option(
        False,
        "--drop-nsfw",
        help="Reject images NudeNet reads as exposed (>= 0.2) before scenes, garments and SingGuard",
    ),
    drop_no_face: bool = Option(
        False, "--drop-no-face", help="Reject images with no face before scenes, garments and SingGuard (needs --faces)"
    ),
    drop_no_age: bool = Option(
        False,
        "--drop-no-age",
        help="Reject images with no MiVOLO, body-only or set age before scenes, garments and SingGuard (needs --mivolo)",
    ),
    drop_junk: bool = Option(
        False,
        "--drop-junk",
        help="Reject images whose zero-shot junk probability reaches --junk-threshold, before SingGuard "
        "(needs --embeddings)",
    ),
    minor_exposed: str | None = Option(None, "--minor-exposed", help=_MINOR_EXPOSED_HELP),
    crops: bool = Option(
        False,
        "--crops",
        help=(
            "Crop each image that has a <image>.crop.json sidecar ({left, top, right, bottom}, displayed pixels) "
            "in memory before every check, and export it cropped (--copy-to only). Sources are never changed."
        ),
    ),
    detect_watermarks: bool = Option(
        False,
        "--detect-watermarks",
        help=(
            "Find watermarks with the detector (--watermark-url) and cut them away at the frame's edges; "
            "implies --crops. A watermark no edge cut removes rejects the image (crop:overlay)."
        ),
    ),
    max_cut: float = Option(0.15, "--max-cut", help="Most of a side one edge cut may take (--detect-watermarks)"),
    keep_overlays: bool = Option(
        False, "--keep-overlays", help="Keep images whose watermark cannot be cut away, uncropped, instead of rejecting"
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
    carry_sidecars: bool = Option(
        False,
        "--carry-sidecars",
        help="Transfer each exported image's <stem>.txt / <file>.xmp / <stem>.xmp / <stem>.json with it, in the same mode",
    ),
    run_stats: Path | None = Option(
        None, "--run-stats", help="JSON file copied (pretty-printed) to <dest>/run_stats.json with the export"
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
        DEFAULT_NSFW_THRESHOLD,
        ConsentConfig,
        CropConfig,
        EmbeddingConfig,
        ExportRequest,
        FaceConfig,
        FaceSourceFilter,
        GarmentConfig,
        GuardConfig,
        NsfwConfig,
        PoseConfig,
        PrefilterConfig,
        ScanConfig,
        SceneConfig,
        TargetProfile,
    )
    from argus_curator.prefilter import missing as prefilter_missing
    from argus_curator.scanner import scan_folder as _scan
    from argus_curator.store import ScanStore

    _apply_minor_exposed(minor_exposed)
    if not source.is_dir():
        typer.echo(f"Error: not a directory: {source}", err=True)
        raise typer.Exit(1)
    if modality not in ("image", "speech"):
        typer.echo("Error: --modality must be image or speech", err=True)
        raise typer.Exit(2)
    speech_export_flags = {
        "--speaker-id": speaker_ids,
        "--accent": accents,
        "--language": languages,
        "--exclude-speaker-mismatch": exclude_speaker_mismatch,
        "--min-hours-per-speaker": min_hours_per_speaker is not None,
        "--max-hours-per-speaker": max_hours_per_speaker is not None,
        "--max-hours-per-accent": max_hours_per_accent is not None,
        "--balance-accents": balance_accents,
        "--corpus": corpus,
        "--license": corpus_license,
        "--voice-consent": voice_consent,
        "--consent-delivery": consent_delivery,
        "--held-out": held_out is not None,
        "--held-out-fraction": held_out_fraction is not None,
        "--split-method": split_method != "spread",
        "--split-seed": split_seed != 0,
        "--split-part": split_part != "train",
    }
    if modality == "speech":
        image_only = {
            "--csv": csv_out,
            "--face-clusters": face_clusters,
            "--pose/--race/--gender/--age": face_poses or face_races or face_genders or face_ages,
            "--min-age/--max-age": min_age_years is not None or max_age_years is not None,
            "--min-attr-conf": min_attribute_conf != 0.0,
            "--scene/--environment": scene_filter or environment,
            "--garment/--exclude-garment/--uncropped": garment_filter or exclude_garments or garments_uncropped,
            "--diversify-*": diversify_scenes or diversify_outfits or diversify_semantic,
            "--diversity-weight": diversity_weight != 0.40,
            "--nsfw-threshold/--nsfw-label/--content-ratings": (
                nsfw_threshold is not None or nsfw_label_thresholds or nsfw_exposed_labels or content_ratings
            ),
            "--guard-unsafe/--guard-rules/--guard-policies": guard_unsafe or guard_rules or guard_policies,
            "--landmarks": landmarks,
            "--face-names": face_names,
            "--carry-sidecars": carry_sidecars,
            "--faces": faces or fairface or mivolo or set_age or mivolo_body or bool(face_reference),
            "--scenes": scenes,
            "--garments": garments,
            "--embeddings": embeddings,
            "--crops": crops or detect_watermarks,
            "--guard": guard,
            "--drop-*": drop_nsfw or drop_no_face or drop_no_age or drop_junk,
            "--profile-fit/--junk-check": profile_fit or junk_check,
            "--xmp": xmp or xmp_sources,
            "--consent": consent or face_source_consent or face_source,
        }
        if asked := [flag for flag, value in image_only.items() if value]:
            typer.echo(
                f"Error: a speech scan runs no image checks or image filters; drop {', '.join(asked)}",
                err=True,
            )
            raise typer.Exit(2)
        # Every flag is checked before a clip is read; the export itself runs after the scan.
        speech_req = _speech_export_request(
            source,
            dest=move_to or copy_to or symlink_to,
            mode="move" if move_to else ("symlink" if symlink_to else "copy"),
            flags=speech_export_flags,
            min_score=min_score,
            keep_similar=keep_similar,
            max_keep=max_keep,
            speaker_ids=speaker_ids,
            accents=accents,
            languages=languages,
            exclude_speaker_mismatch=exclude_speaker_mismatch,
            min_hours_per_speaker=min_hours_per_speaker,
            max_hours_per_speaker=max_hours_per_speaker,
            max_hours_per_accent=max_hours_per_accent,
            balance_accents=balance_accents,
            corpus=corpus,
            corpus_license=corpus_license,
            voice_consent=voice_consent,
            consent_delivery=consent_delivery,
            speakers_clustered=speech_speakers,
            held_out=held_out,
            held_out_fraction=held_out_fraction,
            split_method=split_method,
            split_seed=split_seed,
            split_part=split_part,
            run_stats=run_stats,
        )
        summary = _scan_speech(
            source,
            style=style,
            target_backend=target_backend,
            checkpoint=checkpoint,
            category=category or "voice",
            transcripts=transcripts,
            mos=mos,
            dnsmos_model=dnsmos_model,
            speakers=speech_speakers,
            speaker_model=speaker_model,
            speech_config=speech_config,
            device=device,
            max_workers=max_workers,
            no_cluster=no_cluster,
            json_out=json_out,
            verbose=verbose,
        )
        if speech_req is not None:
            _export_speech(summary, speech_req)
        return
    if transcripts or mos or dnsmos_model or speech_config or speech_speakers or speaker_model:
        typer.echo(
            "Error: --transcripts, --mos, --dnsmos-model, --speakers, --speaker-model and --speech-config "
            "need --modality speech",
            err=True,
        )
        raise typer.Exit(2)
    if asked := [flag for flag, value in speech_export_flags.items() if value]:
        typer.echo(f"Error: {', '.join(asked)}: speech export options; they need --modality speech", err=True)
        raise typer.Exit(2)
    import os

    from argus_curator import bodypose as bodypose_mod
    from argus_curator import embeddings as embeddings_mod
    from argus_curator import erax as erax_mod
    from argus_curator import fairface as fairface_mod
    from argus_curator import garments as garments_mod
    from argus_curator import guard as guard_mod
    from argus_curator import mivolo as mivolo_mod
    from argus_curator import nsfw as nsfw_mod
    from argus_curator import scenes as scenes_mod
    from argus_curator import watermarks as watermarks_mod

    for model_file, env in (
        (fairface_model, fairface_mod.MODEL_ENV),
        (mivolo_model, mivolo_mod.MODEL_ENV),
        (places365_model, scenes_mod.MODEL_ENV),
        (garments_model, garments_mod.MODEL_ENV),
        (embeddings_model, embeddings_mod.MODEL_ENV),
        (nudenet_url, nsfw_mod.URL_ENV),
        (erax_url, erax_mod.URL_ENV),
        (watermark_url, watermarks_mod.URL_ENV),
        (singguard_url, guard_mod.URL_ENV),
        (puppet_url, bodypose_mod.URL_ENV),
    ):
        if model_file is not None:
            os.environ[env] = str(model_file)

    # CropConfig validates max_cut whether or not detection is on, so check it first, not as a traceback.
    if not 0.0 < max_cut < 0.5:
        typer.echo("Error: --max-cut must be between 0 and 0.5", err=True)
        raise typer.Exit(2)
    if detect_watermarks:
        from argus_curator.runtime import ModelUnavailable

        if not watermarks_mod.configured():
            typer.echo(
                f"Error: --detect-watermarks needs a detector: --watermark-url or {watermarks_mod.URL_ENV}", err=True
            )
            raise typer.Exit(2)
        try:
            detector = watermarks_mod.client()
        except ModelUnavailable as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from exc
        if detector is not None and (why := detector.unreachable()) is not None:
            typer.echo(f"Error: --detect-watermarks: the detector {why}", err=True)
            raise typer.Exit(2)
    if bodypose_mod.resolve_enabled(PoseConfig(enabled=body_pose)):
        from argus_curator.runtime import ModelUnavailable

        try:
            puppet = bodypose_mod.client()
        except ModelUnavailable as exc:  # names the variable, never the URL
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from exc
        if puppet is None:
            typer.echo(f"Error: --body-pose needs a server: --puppet-url or {bodypose_mod.URL_ENV}", err=True)
            raise typer.Exit(2)

    try:
        profile = TargetProfile(
            target_style=style,
            target_backend=target_backend or "sdxl",
            checkpoint=checkpoint,
            target_category=category or "identity",
        )
    except ValidationError as exc:
        raise _invalid(exc) from exc
    try:
        cfg = ScanConfig(
            min_short_side=min_short_side,
            max_aspect_ratio=max_aspect_ratio,
            blur_threshold=blur_threshold,
            cluster_distance=-1 if no_cluster else cluster_distance,
            semantic_duplicates=semantic_dupes,
            semantic_duplicate_cosine=semantic_dupes_cosine,
            profile_fit=profile_fit,
            junk_check=junk_check,
            junk_threshold=junk_threshold,
            diversity_weight=diversity_weight,
            max_workers=max_workers,
        )
    except ValidationError as exc:
        raise _invalid(exc) from exc
    outside_refs = _outside_references(source, face_reference) if face_reference else None
    faces_cfg = FaceConfig(
        # An outside reference is embedded just before the scan, once every
        # other check has passed; it needs face detection all the same.
        enabled=faces or outside_refs is not None,
        model=faces_model,
        min_det_score=min_det_score,
        cluster_eps=cluster_eps,
        device=device,
        fairface=fairface,
        mivolo=mivolo,
        set_age=set_age,
        mivolo_body=mivolo_body,
        reference_paths=face_reference if outside_refs is None else [],
    )
    if diversify_semantic and not embeddings:
        typer.echo("Error: --diversify-semantic needs --embeddings", err=True)
        raise typer.Exit(2)
    if semantic_dupes and not embeddings:
        typer.echo("Error: --semantic-dupes needs --embeddings", err=True)
        raise typer.Exit(2)
    zero_shot = [
        flag
        for flag, on in (("--profile-fit", profile_fit), ("--junk-check", junk_check), ("--drop-junk", drop_junk))
        if on
    ]
    if zero_shot and not embeddings:
        typer.echo(f"Error: {zero_shot[0]} needs --embeddings", err=True)
        raise typer.Exit(2)
    face_source = face_source or face_source_consent
    if face_source_consent and consent is None:
        typer.echo("Error: --face-source-consent needs a consent record (--consent)", err=True)
        raise typer.Exit(2)
    try:
        consent_cfg = ConsentConfig(record=consent, max_distance=consent_max_distance)
    except ValidationError as exc:
        raise _invalid(exc) from exc
    if consent is not None:
        from argus_curator.consent import faces_for_consent

        if outside_refs is not None:
            typer.echo("Error: --consent takes its reference from the record; drop --face-reference", err=True)
            raise typer.Exit(2)
        try:
            faces_cfg = faces_for_consent(faces_cfg)
        except ValueError as exc:
            typer.echo(f"Error: --consent: {exc}", err=True)
            raise typer.Exit(2) from exc
    if face_source and not faces_cfg.mivolo:
        typer.echo("Error: --face-source checks for an adult age, which needs --mivolo", err=True)
        raise typer.Exit(2)

    prefilter_cfg = PrefilterConfig(
        drop_nsfw=drop_nsfw, drop_no_face=drop_no_face, drop_no_age=drop_no_age, drop_junk=drop_junk
    )
    nsfw_cfg = NsfwConfig(enabled=nsfw)
    nsfw_on = nsfw_mod.resolve_enabled(nsfw_cfg)
    if (gap := prefilter_missing(prefilter_cfg, faces_cfg, nsfw_on)) is not None:
        rule, setting = gap
        if rule == "drop_no_age":
            needs = "--mivolo"  # which turns face detection on too
        else:
            needs = {
                "faces.enabled": "--faces",
                "nsfw": f"NudeNet (--nudenet-url or {nsfw_mod.URL_ENV}, without --no-nsfw)",
            }[setting]
        typer.echo(f"Error: --{rule.replace('_', '-')} needs {needs}", err=True)
        raise typer.Exit(2)

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
    crops = crops or detect_watermarks
    if crops and dest is not None and mode != "copy":
        typer.echo("Error: --crops writes cropped files; export with --copy-to", err=True)
        raise typer.Exit(1)
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
    for flag, on in (("--carry-sidecars", carry_sidecars), ("--run-stats", run_stats is not None)):
        if on and dest is None:
            typer.echo(f"Error: {flag} needs an export; add --copy-to, --move-to or --symlink-to", err=True)
            raise typer.Exit(2)
    stats = _read_run_stats(run_stats)
    try:
        export_req = ExportRequest(
            dest=str(dest) if dest else str(source),
            mode=mode,
            min_score=min_score,
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
            diversify_semantic=diversify_semantic,
            nsfw_threshold=nsfw_threshold,
            nsfw_label_thresholds=label_thresholds,
            content_ratings=_csv_list(content_ratings),
            guard_unsafe=guard_unsafe,
            guard_rules=_csv_list(guard_rules),
            guard_policies=_csv_list(guard_policies),
            **({"nsfw_exposed_labels": _csv_list(nsfw_exposed_labels)} if nsfw_exposed_labels else {}),
            face_source=FaceSourceFilter(consent=face_source_consent) if face_source else None,
            write_manifest=dest is not None,
            landmarks=landmarks,
            write_xmp=xmp,
            face_names=names,
            carry_sidecars=carry_sidecars,
            run_stats=stats,
        )
    except ValidationError as exc:
        raise _invalid(exc) from exc

    from argus_curator.attributes import fairface_available, mivolo_available

    needed = (
        (fairface, fairface_available, "FairFace", fairface_mod.MODEL_ENV, "fairface-convert <weights.pt>"),
        (faces_cfg.mivolo, mivolo_available, "MiVOLO", mivolo_mod.MODEL_ENV, "mivolo-convert"),
        (scenes, scenes_mod.scenes_available, "Places365", scenes_mod.MODEL_ENV, "places365-convert"),
        (garments, garments_mod.garments_available, "garment-detector", garments_mod.MODEL_ENV, "garments-convert"),
        (
            embeddings,
            embeddings_mod.embeddings_available,
            "CLIP embedding",
            embeddings_mod.MODEL_ENV,
            "embeddings-convert",
        ),
    )
    for wanted, available, name, env, command in needed:
        if wanted and not available():
            typer.echo(f"Error: no {name} model (run `argus-curator {command}` or set {env})", err=True)
            raise typer.Exit(1)

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
    if guard_thinking is not None and guard_thinking not in guard_mod.THINKING_MODES:
        typer.echo(
            f"Error: unknown SingGuard mode {guard_thinking!r} (one of {', '.join(guard_mod.THINKING_MODES)})", err=True
        )
        raise typer.Exit(1)
    guard_cfg = GuardConfig(enabled=guard, policy=guard_policy, variants=variant_names, thinking=guard_thinking)
    ran = [guard_cfg.policy, *guard_cfg.variants]
    if not_run := [name for name in export_req.guard_policies or [] if name not in ran]:
        typer.echo(f"Error: --guard-policies {', '.join(not_run)} not judged (add them to --guard-variants)", err=True)
        raise typer.Exit(1)
    if guard_mod.resolve_enabled(guard_cfg):
        try:
            guard_mod.endpoint()
            guard_mod.thinking(guard_cfg)
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

    if outside_refs is not None:
        from argus_curator.faces import FacesUnavailable, reference_embedding

        try:
            centroid = reference_embedding([f.read_bytes() for f in outside_refs], faces_cfg)
        except (FacesUnavailable, ValueError, OSError) as exc:
            typer.echo(f"Error: --face-reference: {exc}", err=True)
            raise typer.Exit(2) from exc
        faces_cfg = faces_cfg.model_copy(update={"reference_embedding": centroid})

    typer.echo(f"Scanning {source} (recursive) ...")
    summary = _scan(
        source,
        profile,
        cfg,
        faces_cfg,
        scene_cfg=scene_cfg,
        garment_cfg=garment_cfg,
        embed_cfg=EmbeddingConfig(enabled=embeddings, device=device),
        nsfw_cfg=nsfw_cfg,
        guard_cfg=guard_cfg,
        prefilter_cfg=prefilter_cfg,
        crop_cfg=CropConfig(enabled=crops, detect=detect_watermarks, max_cut=max_cut, reject_overlay=not keep_overlays),
        pose_cfg=PoseConfig(enabled=body_pose),
        consent_cfg=consent_cfg,
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
        at = export_req.nsfw_threshold if export_req.nsfw_threshold is not None else DEFAULT_NSFW_THRESHOLD
        exposed = sum(1 for r in scored if nsfw_mod.exposed_score(r.nsfw.labels, export_req.nsfw_exposed_labels) >= at)
        # A rejected frame is not sent, so its missing score is no failure.
        unchecked = sum(1 for r in summary.results if r.passed and (r.nsfw is None or r.nudenet_failed))
        tail = f"  ({unchecked} could not be checked)" if unchecked else ""
        typer.echo(f"  NudeNet:               {len(checked)} checked, {exposed} with exposed score >= {at:g}{tail}")
    if summary.guard_config.enabled:
        judged = [r.guard for r in summary.results if r.guard is not None]
        by_rule = Counter(guard_mod.rule_letter(rule) for g in judged for rule in g.rules)
        rules = f" ({', '.join(f'{k}={n}' for k, n in sorted(by_rule.items()))})" if by_rule else ""
        unchecked = sum(1 for r in summary.results if r.passed and r.guard is None)
        tail = f"  ({unchecked} could not be checked)" if unchecked else ""
        unsafe = sum(1 for g in judged if not g.safe)
        gc = summary.guard_config
        # Not `mode`: that is the export's copy/move/symlink, read again after the transfer.
        thinking = f", {gc.thinking}" + (" (refused, ran fast)" if gc.fell_back else "") if gc.thinking else ""
        typer.echo(f"  SingGuard:             {len(judged)} checked, {unsafe} unsafe{rules}{thinking}{tail}")
    typer.echo("=" * 56)
    if summary.consent is not None:
        check = summary.consent
        if check.eligible:
            bound = sum(1 for r in summary.results if r.consent is not None and r.consent.bound)
            refs = len(check.references)
            typer.echo(f"  Consent:               eligible; {bound} image(s) bound ({refs} reference image(s))")
        else:
            typer.echo(f"  Consent:               INELIGIBLE ({check.reason}): {check.detail}")
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
        from argus_curator.selection import face_source_problem

        # Only knowable once the scan ran: a --face-reference that gave no
        # distances would let the identity check fall back to the largest
        # cluster, so another person could pass as the reference's.
        if (problem := face_source_problem(req, summary)) is not None:
            typer.echo(f"Error: {problem}", err=True)
            raise typer.Exit(2)
        # With no --copy-to/--move-to/--symlink-to there is nowhere to transfer
        # to: exporting would copy every file onto itself (SameFileError per
        # image) for a report-only run, so skip it and report no exported paths.
        from argus_curator.consent import ConsentRefused, face_source_gate

        try:
            result = export_selection(summary, req) if dest is not None else None
            gate = face_source_gate(req, summary) if csv_out else None
        except ConsentRefused as exc:
            typer.echo(f"Error: consent refused: {exc}", err=True)
            raise typer.Exit(2) from exc
        if csv_out:
            from argus_curator.export import write_report
            from argus_curator.selection import decide_selection, scan_references

            selected, keep_reason = decide_selection(
                summary.results,
                req,
                summary.config.diversity_weight,
                set_age=summary.set_age,
                embeddings=summary.embeddings,
                gate=gate,
                references=scan_references(summary),
            )
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
            if carry_sidecars:
                failed = f" ({result.sidecars_failed} failed)" if result.sidecars_failed else ""
                typer.echo(f"Sidecars carried -> {result.sidecars_carried}{failed}")
            if result.run_stats_path:
                typer.echo(f"Run stats -> {result.run_stats_path}")

    typer.echo(f"\nDone. scan_id={summary.scan_id}")


def _read_run_stats(run_stats: Path | None) -> dict | None:
    """The ``--run-stats`` file's JSON object, or None without one; exit 2 if it does not read as one."""
    if run_stats is None:
        return None
    import json

    try:
        stats = json.loads(run_stats.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: --run-stats: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(stats, dict):
        typer.echo("Error: --run-stats: the file must hold a JSON object", err=True)
        raise typer.Exit(2)
    return stats


_SPEECH_PROVENANCE_HELP = (
    "a speech export names where its clips came from: --corpus and --license for a licensed corpus, "
    "or --voice-consent and --consent-delivery for a consented voice"
)


def _speech_export_request(
    source: Path,
    *,
    dest: Path | None,
    mode: str,
    flags: dict[str, object],
    min_score: float,
    keep_similar: bool,
    max_keep: int | None,
    speaker_ids: str | None,
    accents: str | None,
    languages: str | None,
    exclude_speaker_mismatch: bool,
    min_hours_per_speaker: float | None,
    max_hours_per_speaker: float | None,
    max_hours_per_accent: float | None,
    balance_accents: bool,
    corpus: str | None,
    corpus_license: str | None,
    voice_consent: str | None,
    consent_delivery: str | None,
    speakers_clustered: bool,
    held_out: int | None,
    held_out_fraction: float | None,
    split_method: str,
    split_seed: int,
    split_part: str,
    run_stats: Path | None,
) -> ExportRequest | None:
    """The speech export ``scan --modality speech`` asks for (#207), checked before the scan; None without one.

    It is an ordinary :class:`ExportRequest` with ``speech`` set, for
    :func:`argus_curator.export.export_selection` (which routes it to
    :func:`argus_curator.speech_export.export_speech`, as ``POST /export``
    does), so the selection, the refusals and what is written are the
    library's. This only turns the flags into it, and names a missing or
    half-given provenance by its flags, before a clip is read.
    """
    from pydantic import ValidationError

    from argus_curator.models import ExportRequest, SpeechSelection, SplitConfig
    from argus_curator.speech_export import provenance_problem

    if dest is None:
        asked = [flag for flag, value in flags.items() if value]
        if run_stats is not None:
            asked.append("--run-stats")
        if asked:
            typer.echo(
                f"Error: {', '.join(asked)}: speech export options; add --copy-to, --move-to or --symlink-to",
                err=True,
            )
            raise typer.Exit(2)
        return None
    if (corpus is None) != (corpus_license is None):
        typer.echo("Error: --corpus and --license go together: name the corpus and its licence", err=True)
        raise typer.Exit(2)
    if (voice_consent is None) != (consent_delivery is None):
        typer.echo("Error: --voice-consent needs --consent-delivery (the voice delivery it covers)", err=True)
        raise typer.Exit(2)
    if corpus is not None and voice_consent is not None:
        typer.echo(
            "Error: a speech export is from a licensed corpus (--corpus) or under --voice-consent, not both", err=True
        )
        raise typer.Exit(2)
    if corpus is None and voice_consent is None:
        typer.echo(f"Error: {_SPEECH_PROVENANCE_HELP}", err=True)
        raise typer.Exit(2)
    if voice_consent is not None:
        # The export reads and checks it again after the scan: this only spares a scan
        # whose export could not start. Relative to the scanned folder, as --consent's.
        if not (source / voice_consent).is_file():
            typer.echo(f"Error: --voice-consent {voice_consent}: no such file", err=True)
            raise typer.Exit(2)
        if not speakers_clustered:
            typer.echo("Error: --voice-consent binds the set by speaker clusters; add --speakers", err=True)
            raise typer.Exit(2)
    if exclude_speaker_mismatch and not speakers_clustered:
        typer.echo("Error: --exclude-speaker-mismatch needs a scan that clusters speakers; add --speakers", err=True)
        raise typer.Exit(2)
    splitting = held_out is not None or held_out_fraction is not None
    if not splitting and (asked := [f for f in ("--split-method", "--split-seed", "--split-part") if flags[f]]):
        typer.echo(f"Error: {', '.join(asked)} need a split: --held-out or --held-out-fraction", err=True)
        raise typer.Exit(2)
    if held_out is not None and held_out_fraction is not None:
        typer.echo("Error: a split takes --held-out or --held-out-fraction, not both", err=True)
        raise typer.Exit(2)
    try:
        speech = SpeechSelection(
            speakers=_csv_list(speaker_ids),
            accents=_csv_list(accents),
            languages=_csv_list(languages),
            exclude_speaker_mismatch=exclude_speaker_mismatch,
            min_hours_per_speaker=min_hours_per_speaker,
            max_hours_per_speaker=max_hours_per_speaker,
            max_hours_per_accent=max_hours_per_accent,
            balance_accents=balance_accents,
            corpus=corpus,
            license=corpus_license,
            consent_record=voice_consent,
            consent_delivery=consent_delivery,
        )
        split = (
            SplitConfig(
                held_out=held_out,
                held_out_fraction=held_out_fraction,
                group_by="speaker",
                method=split_method,
                seed=split_seed,
            )
            if splitting
            else None
        )
        req = ExportRequest(
            dest=str(dest),
            mode=mode,
            min_score=min_score,
            keep_similar=keep_similar,
            max_keep=max_keep,
            split=split,
            split_part=split_part,
            speech=speech,
            write_manifest=True,
            run_stats=_read_run_stats(run_stats),
        )
    except ValidationError as exc:
        raise _invalid(exc) from exc
    if provenance_problem(req) is not None:  # the library's own check, should the flags above drift from it
        typer.echo(f"Error: {_SPEECH_PROVENANCE_HELP}", err=True)
        raise typer.Exit(2)
    return req


def _export_speech(summary: ScanSummary, req: ExportRequest) -> None:
    """Run the speech export *req* on the scan *summary* through ``export_selection``, as ``POST /export`` does."""
    from argus_curator.consent import ConsentRefused
    from argus_curator.export import export_selection

    req = req.model_copy(update={"scan_id": summary.scan_id})
    try:
        result = export_selection(summary, req)
    except ConsentRefused as exc:
        typer.echo(f"Error: consent refused: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ValueError as exc:  # SplitError among them; each raised before anything is transferred
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    verb = {"copy": "Copied", "move": "Moved", "symlink": "Symlinked"}[req.mode]
    part = f" ({req.split_part} part of the split)" if req.split is not None else ""
    skipped = f", {result.skipped} skipped" if result.skipped else ""
    typer.echo("")
    typer.echo(f"{verb} {result.copied} clips with their transcripts{part}{skipped} -> {result.dest}")
    if result.manifest_path:
        typer.echo(f"Speech manifest -> {result.manifest_path}")
    if result.run_stats_path:
        typer.echo(f"Run stats -> {result.run_stats_path}")


def _scan_speech(
    source: Path,
    *,
    style: str,
    target_backend: str | None,
    checkpoint: str | None,
    category: str,
    transcripts: str | None,
    mos: bool,
    dnsmos_model: Path | None,
    speakers: bool,
    speaker_model: Path | None,
    speech_config: Path | None,
    device: str,
    max_workers: int,
    no_cluster: bool,
    json_out: Path | None,
    verbose: bool,
) -> ScanSummary:
    """``scan --modality speech``: score the folder's clips (#186), save the scan to the store and return it."""
    import json
    import os

    from pydantic import ValidationError

    from argus_curator import dnsmos as dnsmos_mod
    from argus_curator.models import SpeechConfig, SpeechTargetProfile
    from argus_curator.runtime import ModelUnavailable
    from argus_curator.speech import scan_speech_folder
    from argus_curator.store import ScanStore

    if dnsmos_model is not None:
        os.environ[dnsmos_mod.MODEL_ENV] = str(dnsmos_model)
    if speaker_model is not None:
        from argus_curator import speakers as speakers_mod

        os.environ[speakers_mod.MODEL_ENV] = str(speaker_model)
    try:
        base = json.loads(speech_config.read_text(encoding="utf-8")) if speech_config is not None else {}
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: --speech-config: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(base, dict):
        typer.echo("Error: --speech-config must hold a JSON object", err=True)
        raise typer.Exit(2)
    overrides: dict[str, object] = {"max_workers": max_workers}
    if transcripts is not None:
        overrides["transcripts"] = transcripts
    if mos:
        overrides.update(mos=True, mos_device=device)
    if speakers:
        overrides.update(speakers=True, speaker_device=device)
    if no_cluster:
        overrides["duplicate_distance"] = -1
    try:
        cfg = SpeechConfig.model_validate({**base, **overrides})
        profile = SpeechTargetProfile(
            target_style=style, target_backend=target_backend, checkpoint=checkpoint, target_category=category
        )
    except ValidationError as exc:
        raise _invalid(exc) from exc

    typer.echo(f"Scanning speech clips in {source} (recursive) ...")
    try:
        summary = scan_speech_folder(source, profile, cfg)
    except ModelUnavailable as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("=" * 56)
    typer.echo(f"  Clips scanned:         {summary.total}")
    typer.echo(f"  Passed:                {summary.passed}")
    typer.echo(f"  Similar clusters (>1): {summary.similar_clusters}  ({summary.duplicates} non-representative)")
    kept = [c for c in summary.clips if c.passed and not c.is_duplicate]
    hours = sum(c.duration_s or 0.0 for c in kept) / 3600
    typer.echo(f"  Usable audio:          {hours:.2f} h in {len(kept)} clips")
    if summary.speaker_clusters:
        typer.echo(
            f"  Speakers (clustered):  {len(summary.speaker_clusters)}  "
            f"({summary.speaker_mismatches} clips' corpus labels disagree)"
        )
    scored = sorted(c.mos.ovrl for c in summary.clips if c.mos is not None)
    if scored:
        typer.echo(f"  DNSMOS OVRL (median):  {scored[len(scored) // 2]:.2f} over {len(scored)} clips")
    typer.echo("=" * 56)
    if summary.reject_reasons:
        typer.echo("")
        typer.echo("Rejection reasons:")
        for reason, count in sorted(summary.reject_reasons.items(), key=lambda x: -x[1]):
            typer.echo(f"  {count:4d}  {reason}")
    if verbose:
        typer.echo("")
        typer.echo("Per-clip (grouped by cluster, then score):")
        for c in sorted(summary.clips, key=lambda x: (x.similar_group, -x.score)):
            tag = f"g{c.similar_group}" + ("*" if c.is_representative and c.group_size > 1 else "")
            why = f"  [{c.reject_reason}: {c.reject_detail}]" if c.reject_reason else ""
            snr = f" snr={c.snr_db:.1f}" if c.snr_db is not None else ""
            mos_text = f" mos={c.mos.ovrl:.2f}" if c.mos is not None else ""
            typer.echo(f"  {c.score:.3f} {tag:>5}  {c.rel_path}{snr}{mos_text}{why}")

    ScanStore().save(summary)
    if json_out:
        json_out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        typer.echo("")
        typer.echo(f"JSON summary written -> {json_out}")
    typer.echo(f"Scan saved as {summary.scan_id}")
    return summary


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
    trusted_host: list[str] = Option(
        [],
        "--trusted-host",
        help="DNS name this server is reached by, whose same-host writes are allowed (repeatable; or "
        "CURATOR_TRUSTED_HOSTS). IP literals and localhost need none.",
    ),
    source_root: str | None = Option(
        None, "--source-root", help="Root that scan/thumb/upload paths resolve under (ARGUS_CURATOR_SCAN_ROOT)"
    ),
    export_root: str | None = Option(
        None, "--export-root", help="Root that export destinations resolve under (ARGUS_CURATOR_EXPORT_ROOT)"
    ),
    allow_move: bool = Option(False, "--allow-move", help='Permit destructive mode="move" exports (default: rejected)'),
    allow_delete: bool = Option(
        False,
        "--allow-delete",
        help="Let POST /scan/{id}/delete remove any scanned image (CURATOR_ALLOW_DELETE / CURATOR_DEVMODE; "
        "default: only under-18 exposed images)",
    ),
    minor_exposed: str | None = Option(None, "--minor-exposed", help=_MINOR_EXPOSED_HELP),
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

    _apply_minor_exposed(minor_exposed)
    from argus_curator.server import create_app

    application = create_app(
        cors=cors,
        cors_origins=cors_origin or None,
        cors_allow_any=cors_any,
        trusted_hosts=trusted_host or None,
        source_root=source_root,
        export_root=export_root,
        allow_move=allow_move or None,  # None -> fall back to CURATOR_ALLOW_MOVE
        allow_delete=allow_delete or None,  # None -> CURATOR_ALLOW_DELETE / CURATOR_DEVMODE
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


@app.command("embeddings-convert")
def embeddings_convert(
    checkpoint: Path | None = Option(
        None, "--checkpoint", help="Local dir with config.json + pytorch_model.bin (default: download the pinned one)"
    ),
    output: Path | None = Option(None, "--output", "-o", help="Where to write the ONNX model (default: model cache)"),
    prompts: bool = Option(
        False,
        "--prompts",
        help="Instead, embed the zero-shot prompts with the text tower and write the pinned prompt file "
        "(default: the package's data/clip_prompts.json; a maintainer step)",
    ),
) -> None:
    """Fetch, verify and convert the CLIP image tower for --embeddings.

    Weights: huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K (MIT;
    DataComp-1B CLIP ViT-L/14), pinned by commit and sha256. Only the image
    tower and its projection are exported (~1.2 GB fp32). With --prompts, the
    text tower embeds the zero-shot prompts into the pinned prompt file the
    package ships (a maintainer step; scans never run the text tower).
    Needs: pip install argus-curator[embeddings-convert]
    """
    from argus_curator.embeddings import export_onnx, fetch_weights, model_path
    from argus_curator.runtime import ModelUnavailable

    if prompts:
        from argus_curator import zeroshot

        try:
            source = checkpoint or fetch_weights(zeroshot.PROMPT_HF_FILES)
            written, digest = zeroshot.export_prompts(source, output or zeroshot.default_prompts_path())
        except ModelUnavailable as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Wrote CLIP prompt vectors -> {written}\nsha256 {digest} (set zeroshot.PROMPTS_SHA256 to it)")
        return
    try:
        source = checkpoint or fetch_weights()
        written = export_onnx(source, output or model_path())
    except ModelUnavailable as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote CLIP image-tower ONNX model -> {written}")


DEFAULT_SCHEMA_PATH = Path("schema/curator-wire.schema.json")


@app.command("dnsmos-fetch")
def dnsmos_fetch(
    output: Path | None = Option(None, "--output", "-o", help="Where to write the model (default: model cache)"),
) -> None:
    """Fetch the DNSMOS P.835 model a speech scan's --mos runs.

    microsoft/DNS-Challenge (code MIT, other content CC BY 4.0), pinned by
    commit and sha256; 1.2 MB, runs on onnxruntime (argus-curator[speech]).
    """
    from argus_curator.dnsmos import fetch_model
    from argus_curator.runtime import ModelUnavailable

    try:
        written = fetch_model(output)
    except (ModelUnavailable, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"DNSMOS model in place -> {written}")


@app.command("speaker-fetch")
def speaker_fetch(
    output: Path | None = Option(None, "--output", "-o", help="Where to write the model (default: model cache)"),
) -> None:
    """Fetch the speaker-embedding model a speech scan's --speakers runs.

    WeSpeaker's ECAPA-TDNN (VoxCeleb2, large-margin fine-tuned; code
    Apache-2.0, model CC BY 4.0), pinned by Hugging Face revision and
    sha256; 25 MB, runs on onnxruntime (argus-curator[speech]).
    """
    from argus_curator.runtime import ModelUnavailable
    from argus_curator.speakers import fetch_model

    try:
        written = fetch_model(output)
    except (ModelUnavailable, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Speaker model in place -> {written}")


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against.

    Run without flags to (re)write the committed schema; run with --check in CI
    to fail if the models have drifted from the committed artifact.
    """
    from argus_curator.models import render_wire_schema

    rendered = render_wire_schema()

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
