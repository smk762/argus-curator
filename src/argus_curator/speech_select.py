"""Selection for a speech scan (#187): filters, hours caps, accent balance and a speaker-disjoint split.

The speech counterpart of :func:`argus_curator.selection.plan_selection`, and
like it the one path ``POST /scan/{id}/select`` and a speech export share, so
a dry run and an export with the same fields keep the same clips in the same
parts. Every clip left out carries a ``keep_reason``.

A speech split is **speaker-disjoint**: zero-shot TTS is judged on speakers the
model never heard, so a split that only holds out clips measures memorisation.
Clips are grouped by :func:`speaker_groups`, which joins every clip sharing a
corpus speaker label *or* a speaker cluster, so neither reading of a speaker
can straddle the parts even where the two disagree. The groups go to
:func:`argus_curator.splitting.split_held_out` (the one split curator has),
stratified by each group's accent and sex where the corpus labels them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

from argus_curator.models import (
    ClipResult,
    ExportRequest,
    ScanSummary,
    SpeechSelection,
    SpeechSelectStats,
    SplitPart,
)
from argus_curator.speech_corpus import under_18_label
from argus_curator.splitting import SplitError, split_held_out

# ExportRequest fields that filter, rank or annotate images. A speech request
# that sets any of them away from its default is refused rather than ignored.
IMAGE_ONLY_FIELDS: tuple[str, ...] = (
    "rank",
    "max_likeness_distance",
    "exclude_references",
    "diversity_weight",
    "face_clusters",
    "face_poses",
    "face_races",
    "face_genders",
    "face_ages",
    "min_age_years",
    "max_age_years",
    "min_attribute_conf",
    "scenes",
    "scene_environments",
    "min_scene_conf",
    "diversify_scenes",
    "garments",
    "exclude_garments",
    "min_garment_conf",
    "garments_uncropped",
    "diversify_outfits",
    "diversify_semantic",
    "nsfw_threshold",
    "nsfw_label_thresholds",
    "nsfw_exposed_labels",
    "content_ratings",
    "guard_unsafe",
    "guard_rules",
    "guard_policies",
    "face_source",
    "landmarks",
    "caption_url",
    "write_xmp",
    "face_names",
    "carry_sidecars",
)

_SECONDS_PER_HOUR = 3600.0


def speech_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req* cannot select from the speech scan *summary*, else ``None``."""
    defaults = ExportRequest.model_fields
    asked = [
        name
        for name in IMAGE_ONLY_FIELDS
        if getattr(req, name) != defaults[name].get_default(call_default_factory=True)
    ]
    if asked:
        return (
            f"{', '.join(asked)}: image options; a speech scan selects by speech, min_score, keep_similar, "
            "max_keep and split"
        )
    if req.split is not None and req.split.group_by != "speaker":
        return 'a speech split is speaker-disjoint: set split.group_by to "speaker"'
    speech = req.speech or SpeechSelection()
    if speech.exclude_speaker_mismatch and not summary.speaker_clusters:
        return "speech.exclude_speaker_mismatch needs a scan that clustered speakers; rescan with speech.speakers on"
    if speech.exclude_language_mismatch and not (summary.speech_config and summary.speech_config.language_id):
        return "speech.exclude_language_mismatch needs a scan that ran language ID; rescan with speech.language_id on"
    return None


def hard_exclusion(clip: ClipResult) -> str | None:
    """Why no request can keep *clip* (a named selection included), or ``None``.

    A rejected clip by its own ``reject_reason`` (``corpus_under_18`` among
    them), then a passed clip whose corpus age label reads under 18 anyway (a
    scan cached before that check, say).
    """
    if not clip.passed:
        return clip.reject_reason or "rejected"
    if under_18_label(clip.age):
        return "corpus_under_18"
    return None


def speaker_id(clip: ClipResult) -> str | None:
    """The clip's speaker: the corpus's label when it has one, else its speaker cluster."""
    return clip.speaker or clip.speaker_cluster


def speaker_groups(clips: Sequence[ClipResult]) -> dict[str, str | None]:
    """Each clip's split group: clips sharing a corpus label or a speaker cluster, transitively.

    A clip with neither has no group (``None``). Groups are named by their
    first clip's speaker id, so the name is stable for a given clip order.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nodes: dict[str, list[str]] = {}
    for c in clips:
        keys = [f"label:{c.speaker}"] if c.speaker else []
        if c.speaker_cluster:
            keys.append(f"cluster:{c.speaker_cluster}")
        for k in keys:
            parent.setdefault(k, k)
        for k in keys[1:]:
            union(keys[0], k)
        nodes[c.rel_path] = keys
    name: dict[str, str] = {}
    out: dict[str, str | None] = {}
    for c in clips:
        keys = nodes[c.rel_path]
        if not keys:
            out[c.rel_path] = None
            continue
        root = find(keys[0])
        out[c.rel_path] = name.setdefault(root, speaker_id(c) or root)
    return out


def _stratum(members: Sequence[ClipResult]) -> tuple[str, str]:
    """A speaker group's (accent, gender): the most common labelled pair among its clips, "" for unknown."""
    counts = Counter((c.accent or "", c.gender or "") for c in members)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _hours(seconds: float) -> float:
    return round(seconds / _SECONDS_PER_HOUR, 4)


def _seconds(clip: ClipResult) -> float:
    return float(clip.duration_s or 0.0)


def _best_first(clips: Sequence[ClipResult]) -> list[ClipResult]:
    return sorted(clips, key=lambda c: (-c.score, c.rel_path))


def _by_speaker(clips: Sequence[ClipResult]) -> dict[str, list[ClipResult]]:
    """Clips by speaker id, best first; a clip with no speaker is a speaker of its own."""
    out: dict[str, list[ClipResult]] = {}
    for c in _best_first(clips):
        out.setdefault(speaker_id(c) or f"clip:{c.rel_path}", []).append(c)
    return out


def _round_robin(queues: Sequence[list[ClipResult]], fits) -> list[ClipResult]:
    """Take clips from each queue in turn (best first) while *fits(clip)* allows."""
    taken: list[ClipResult] = []
    heads = [0] * len(queues)
    live = [i for i, q in enumerate(queues) if q]
    while live:
        still: list[int] = []
        for i in live:
            clip = queues[i][heads[i]]
            heads[i] += 1
            if fits(clip):
                taken.append(clip)
            if heads[i] < len(queues[i]):
                still.append(i)
        live = still
    return taken


@dataclass(frozen=True)
class SpeechPlan:
    """What a speech selection keeps: ``chosen`` best first, why the rest are out, the split and the hours."""

    chosen: list[ClipResult]
    keep_reason: dict[str, str]
    split: dict[str, SplitPart] = field(default_factory=dict)
    stats: SpeechSelectStats | None = None


def stats(chosen: Sequence[ClipResult], split: Mapping[str, SplitPart]) -> SpeechSelectStats:
    """Hours kept, by speaker, by accent and (with a split) by part."""
    by_speaker: Counter[str] = Counter()
    by_accent: Counter[str] = Counter()
    by_part: Counter[str] = Counter()
    speakers_by_part: dict[str, set[str]] = {}
    for c in chosen:
        who = speaker_id(c) or ""
        by_speaker[who] += _seconds(c)
        by_accent[c.accent or ""] += _seconds(c)
        if c.rel_path in split:
            part = split[c.rel_path]
            by_part[part] += _seconds(c)
            speakers_by_part.setdefault(part, set()).add(who)
    return SpeechSelectStats(
        hours=_hours(sum(_seconds(c) for c in chosen)),
        hours_by_speaker={k: _hours(v) for k, v in sorted(by_speaker.items())},
        hours_by_accent={k: _hours(v) for k, v in sorted(by_accent.items())},
        hours_by_split={k: _hours(v) for k, v in sorted(by_part.items())},
        speakers_by_split={k: len(v) for k, v in sorted(speakers_by_part.items())},
    )


def decide(
    clips: Sequence[ClipResult], req: ExportRequest, excluded: Mapping[str, str] | None = None
) -> tuple[list[ClipResult], dict[str, str]]:
    """The clips *req* keeps, best first, and a keep reason for every clip (empty when kept).

    *excluded* maps a rel_path to a reason the caller's disk checks gave (the
    consent gate), applied after the hard exclusions.
    """
    speech = req.speech or SpeechSelection()
    excluded = excluded or {}
    keep_reason: dict[str, str] = {}
    speakers = set(speech.speakers) if speech.speakers else None
    accents = set(speech.accents) if speech.accents else None
    languages = set(speech.languages) if speech.languages else None

    eligible: list[ClipResult] = []
    for c in clips:
        if (hard := hard_exclusion(c)) is not None:
            keep_reason[c.rel_path] = hard
        elif c.rel_path in excluded:
            keep_reason[c.rel_path] = excluded[c.rel_path]
        elif not c.transcript:
            keep_reason[c.rel_path] = "no-transcript"
        elif c.score < req.min_score:
            keep_reason[c.rel_path] = "below-score"
        elif c.is_duplicate and not req.keep_similar:
            keep_reason[c.rel_path] = f"similar-to:{c.duplicate_of}"
        elif speakers is not None and speaker_id(c) not in speakers:
            keep_reason[c.rel_path] = "speaker-excluded"
        elif accents is not None and c.accent not in accents:
            keep_reason[c.rel_path] = "accent-excluded"
        elif languages is not None and c.language not in languages:
            keep_reason[c.rel_path] = "language-excluded"
        elif speech.exclude_speaker_mismatch and c.speaker_mismatch:
            keep_reason[c.rel_path] = "speaker-mismatch"
        elif speech.exclude_language_mismatch and c.language_mismatch:
            keep_reason[c.rel_path] = "language-mismatch"
        else:
            eligible.append(c)

    # Per speaker: a floor on their hours, then a cap, best clips first.
    kept: list[ClipResult] = []
    for clips_of in _by_speaker(eligible).values():
        total = sum(_seconds(c) for c in clips_of)
        if speech.min_hours_per_speaker is not None and total < speech.min_hours_per_speaker * _SECONDS_PER_HOUR:
            keep_reason.update(dict.fromkeys((c.rel_path for c in clips_of), "speaker-under-min-hours"))
            continue
        used = 0.0
        for c in clips_of:
            cap = speech.max_hours_per_speaker
            if cap is not None and used + _seconds(c) > cap * _SECONDS_PER_HOUR:
                keep_reason[c.rel_path] = "speaker-cap"
                continue
            used += _seconds(c)
            kept.append(c)

    # Per accent: a cap (the smallest accent's hours, with balance_accents),
    # taken from the accent's speakers in turn. Clips with no accent label are
    # capped like an accent but never set the balance: a few unlabelled clips
    # would otherwise cut every real accent down to their hours.
    by_accent: dict[str, list[ClipResult]] = {}
    for c in kept:
        by_accent.setdefault(c.accent or "", []).append(c)
    caps = [speech.max_hours_per_accent * _SECONDS_PER_HOUR] if speech.max_hours_per_accent is not None else []
    labelled = [members for accent, members in by_accent.items() if accent]
    if speech.balance_accents and labelled:
        caps.append(min(sum(_seconds(c) for c in members) for members in labelled))
    if caps:
        cap = min(caps)
        balanced: list[ClipResult] = []
        for members in by_accent.values():
            used = 0.0

            def fits(c: ClipResult) -> bool:
                nonlocal used
                if used + _seconds(c) > cap + 1e-9:
                    return False
                used += _seconds(c)
                return True

            take = _round_robin(list(_by_speaker(members).values()), fits)
            taken = {c.rel_path for c in take}
            keep_reason.update({c.rel_path: "accent-cap" for c in members if c.rel_path not in taken})
            balanced.extend(take)
        kept = balanced

    # max_keep: speakers in turn, each giving its best remaining clip, so the cap keeps them even.
    if req.max_keep is not None and len(kept) > req.max_keep:
        room = req.max_keep

        def has_room(_c: ClipResult) -> bool:
            nonlocal room
            if room <= 0:
                return False
            room -= 1
            return True

        queues = sorted(_by_speaker(kept).values(), key=lambda q: (-q[0].score, q[0].rel_path))
        take = _round_robin(queues, has_room)
        taken = {c.rel_path for c in take}
        keep_reason.update({c.rel_path: "max-keep" for c in kept if c.rel_path not in taken})
        kept = take

    chosen = _best_first(kept)
    for c in chosen:
        keep_reason[c.rel_path] = ""
    return chosen, keep_reason


def split_speakers(chosen: Sequence[ClipResult], req: ExportRequest) -> dict[str, SplitPart]:
    """Each chosen clip's part of ``req.split``: speaker-disjoint, stratified by accent and sex.

    Raises :class:`SplitError` when a clip has no speaker (neither a corpus
    label nor a cluster), since it could be anyone: the scan must cluster
    speakers, or the corpus must name them.
    """
    assert req.split is not None
    groups = speaker_groups(chosen)
    unknown = [c.rel_path for c in chosen if groups[c.rel_path] is None]
    if unknown:
        raise SplitError(
            f"{len(unknown)} kept clip(s) have no speaker (e.g. {unknown[0]}); a speaker-disjoint split needs "
            "the corpus's speaker labels or a scan with speech.speakers on"
        )
    members: dict[Hashable, list[ClipResult]] = {}
    for c in chosen:
        members.setdefault(groups[c.rel_path], []).append(c)
    stratum = {key: _stratum(group) for key, group in members.items()}
    keys = [groups[c.rel_path] for c in chosen]
    parts = split_held_out(keys, req.split, strata=[stratum[k] for k in keys])
    return {c.rel_path: part for c, part in zip(chosen, parts, strict=True)}


def plan_speech_selection(
    clips: Sequence[ClipResult], req: ExportRequest, excluded: Mapping[str, str] | None = None
) -> SpeechPlan:
    """The filter, caps, ranking and split for *req* on a speech scan's *clips*.

    Raises :class:`SplitError` when ``req.split`` cannot be made.
    """
    chosen, keep_reason = decide(clips, req, excluded)
    split = split_speakers(chosen, req) if req.split is not None else {}
    return SpeechPlan(chosen=chosen, keep_reason=keep_reason, split=split, stats=stats(chosen, split))
