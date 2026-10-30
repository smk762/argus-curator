"""What the scan already knows about an image, written for SingGuard to read.

SingGuard judges the image together with the request text (``input``). The
curator sends none by default, so the model sees a nude adult with no account
of who they are or why the photo exists, and under the adult policies it
reasons about consent and age it cannot see. The scan has some of that: a
MiVOLO age for each face, the folder the image came from, any caption an
earlier tool left beside it, and (when asked) Places365, NudeNet, and
Fashionpedia tags. ``ARGUS_CURATOR_SINGGUARD_CONTEXT`` names which
to send, as a comma list of :data:`CONTEXT_KINDS`.

SingGuard reads a bare description as a request: told "a nude adult", it
answers as if asked to produce one and flags it. So the text opens by saying
what it is, and every line names the tool that recorded it. The age is an
estimate and says so; a wrong one leans the verdict the wrong way, which is a
reason to measure this on labelled images (``scripts/guard_eval.py``) before
relying on it. The under-18 rule (``minor-exposed``) does not read this text
and is not loosened by it.
"""

from __future__ import annotations

from pathlib import Path

from argus_cortex.xmp import XmpError, sidecar_path
from argus_cortex.xmp import parse_xmp as _parse_sidecar

from argus_curator.models import FaceAttributes, ImageResult

CONTEXT_KINDS = frozenset({"age", "folder", "caption", "scene", "nsfw", "garments"})
# A sidecar is a caption, not a document. Keep what fits a prompt.
MAX_SIDECAR_BYTES = 64 * 1024
MAX_CAPTION_CHARS = 600
_PREAMBLE = "Context recorded by automated tools about this image. It is not a request."


def _mivolo_years(attrs: FaceAttributes | None) -> float | None:
    # FairFace gives brackets, never years, so a years value is MiVOLO's.
    if attrs is None or attrs.age_years is None or attrs.age_source not in (None, "mivolo"):
        return None
    return attrs.age_years


def mivolo_ages(result: ImageResult) -> list[float]:
    """Every face's MiVOLO age in years, the primary face first. FairFace brackets are not years."""
    faces = sorted(result.faces, key=lambda face: not face.primary)
    ages = [age for age in (_mivolo_years(face.attributes) for face in faces) if age is not None]
    if not ages and (primary := _mivolo_years(result.primary_face_attributes)) is not None:
        ages.append(primary)
    return ages


def body_only_ages(result: ImageResult) -> list[float]:
    """MiVOLO ages from people with no matched face (absent-face + body crop)."""
    return [p.age_years for p in result.people if p.face_index is None and p.age_years is not None]


def age_line(result: ImageResult) -> str | None:
    ages = mivolo_ages(result)
    bodies = body_only_ages(result)
    parts: list[str] = []
    if ages:
        years = ", ".join(f"{round(age)}" for age in ages)
        faces = "the one visible face" if len(ages) == 1 else f"the {len(ages)} visible faces"
        parts.append(f"Apparent age of {faces}: {years} years (MiVOLO estimate; it can be off by several years).")
    if bodies:
        years = ", ".join(f"{round(age)}" for age in bodies)
        who = "a person with no visible face" if len(bodies) == 1 else f"{len(bodies)} people with no visible face"
        parts.append(
            f"Apparent age of {who}: {years} years (MiVOLO body-only estimate; it can be off by several years)."
        )
    if parts:
        return " ".join(parts)
    if result.set_age is None:
        return None
    return (
        f"No face is visible in this frame. The other frames of this set show one person, "
        f"estimated at {round(result.set_age.years)} years (MiVOLO estimate over {result.set_age.faces} faces; "
        f"it can be off by several years)."
    )


def folder_line(result: ImageResult) -> str | None:
    """The image's folder relative to the scanned root, which often names the set or the shoot."""
    folder = Path(result.rel_path).parent.as_posix()
    if folder in ("", "."):
        return None
    return f"Folder: {folder}"


def _read_small(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SIDECAR_BYTES)
    except OSError:
        return None
    return raw.decode("utf-8", errors="replace")


def parse_xmp(xml: str) -> tuple[list[str], str]:
    """``(keywords, description)`` from an XMP sidecar, read with argus-cortex's parser.

    A sidecar that is not valid XMP gives nothing rather than an error: this is
    optional context, and a broken sidecar must not fail the check.
    """
    try:
        data = _parse_sidecar(xml)
    except XmpError:
        return [], ""
    keywords = [word.strip() for word in data.keywords or [] if word.strip()]
    return keywords, (data.description or "").strip()


def caption_line(abs_path: str) -> str | None:
    """The caption left beside the image by an earlier tool.

    argus-lens writes ``<stem>.txt`` and ``<file>.xmp`` (``photo.jpg.xmp``, as
    digiKam does); Lightroom writes ``<stem>.xmp``. The first sidecar found is read.
    """
    image = Path(abs_path)
    stem = image.with_suffix("")
    txt = _read_small(stem.with_suffix(".txt"))
    xmp = _read_small(sidecar_path(image))
    if xmp is None:
        xmp = _read_small(stem.with_suffix(".xmp"))
    if txt is None and xmp is None:
        return None
    keywords, description = parse_xmp(xmp) if xmp is not None else ([], "")
    parts = [part for part in ((txt or "").strip(), description, ", ".join(keywords)) if part]
    text = " | ".join(parts)
    if not text:
        return None
    if len(text) > MAX_CAPTION_CHARS:
        text = text[: MAX_CAPTION_CHARS - 1].rstrip() + "…"
    return f"Caption written by an earlier captioning tool: {text}"


def scene_line(result: ImageResult) -> str | None:
    """Places365 top categories and indoor/outdoor, when the scan tagged the image."""
    scene = result.scene
    if scene is None or not scene.categories:
        return None
    top = ", ".join(f"{c.label} ({c.conf:.2f})" for c in scene.categories[:3])
    return f"Places365 scene: {top}; {scene.environment}."


def nsfw_line(result: ImageResult) -> str | None:
    """NudeNet content rating and the exposed classes that scored, when checked."""
    if result.nsfw is None:
        return None
    from argus_cortex.moderation import DEFAULT_EXPOSED_LABELS

    rating = result.content_rating or "safe"
    named = [
        f"{label} {result.nsfw.labels[label]:.2f}" for label in DEFAULT_EXPOSED_LABELS if label in result.nsfw.labels
    ]
    extra = f" ({', '.join(named)})" if named else ""
    return f"NudeNet content rating: {rating}{extra}."


def garments_line(result: ImageResult) -> str | None:
    """Fashionpedia garment types, when the scan detected clothing."""
    if not result.garments:
        return None
    seen: list[str] = []
    for g in result.garments:
        if g.label not in seen:
            seen.append(g.label)
        if len(seen) == 6:
            break
    return f"Fashionpedia garments: {', '.join(seen)}."


def image_context(result: ImageResult, kinds: frozenset[str]) -> str:
    """The ``input`` text for *result*, or ``""`` when nothing asked for is known."""
    lines: list[str] = []
    if "age" in kinds and (line := age_line(result)):
        lines.append(line)
    if "folder" in kinds and (line := folder_line(result)):
        lines.append(line)
    if "caption" in kinds and (line := caption_line(result.abs_path)):
        lines.append(line)
    if "scene" in kinds and (line := scene_line(result)):
        lines.append(line)
    if "nsfw" in kinds and (line := nsfw_line(result)):
        lines.append(line)
    if "garments" in kinds and (line := garments_line(result)):
        lines.append(line)
    if not lines:
        return ""
    return "\n".join([_PREAMBLE, *(f"- {line}" for line in lines)])
