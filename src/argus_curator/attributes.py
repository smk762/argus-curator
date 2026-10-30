"""Face attributes from FairFace and MiVOLO, merged into one record per face.

Two optional models, each switched on in :class:`FaceConfig`:

- FairFace (``fairface``): race (7), gender, age bracket — each a softmax
  probability.
- MiVOLO (``mivolo``): age in years and gender — the stronger age/gender model.

When both run, MiVOLO supersedes FairFace for gender and age (FairFace still
alone supplies race), and the age bracket is derived from MiVOLO's years so
bracket filters behave the same whichever model ran. ``gender_source`` /
``age_source`` record which model each value came from.

Model files are process configuration (``ARGUS_CURATOR_FAIRFACE_MODEL`` /
``ARGUS_CURATOR_MIVOLO_MODEL``), never request fields; see :mod:`argus_curator.runtime`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from argus_curator.models import FACE_AGES, FACE_GENDERS, FACE_RACES, FaceAge, FaceAttributes
from argus_curator.runtime import ModelUnavailable, onnxruntime_available

# Lower bound (years) of each FairFace bracket, in FACE_AGES order.
_BRACKET_FLOORS = (0, 3, 10, 20, 30, 40, 50, 60, 70)

# Kept as the name callers catch; the runtime raises the shared type.
AttributesUnavailable = ModelUnavailable


def fairface_available() -> bool:
    from argus_curator import fairface

    return onnxruntime_available() and fairface.model_path().is_file()


def mivolo_available() -> bool:
    from argus_curator import mivolo

    return onnxruntime_available() and mivolo.model_path().is_file()


def age_bracket(years: float) -> FaceAge:
    """The FairFace bracket a regression age falls in."""
    idx = 0
    for i, floor in enumerate(_BRACKET_FLOORS):
        if years >= floor:
            idx = i
    return FACE_AGES[idx]


@dataclass
class RawAttributes:
    """One face's model outputs before labelling (kept to pool per cluster)."""

    fairface: np.ndarray | None = None  # (18,) per-head probabilities
    mivolo_gender: np.ndarray | None = None  # (2,) [male, female]
    mivolo_age: float | None = None
    mivolo_body: bool = False  # the age was read with a body crop

    def empty(self) -> bool:
        return self.fairface is None and self.mivolo_gender is None


def to_attributes(raw: RawAttributes) -> FaceAttributes | None:
    """Top label + confidence per attribute, MiVOLO winning gender and age."""
    if raw.empty():
        return None
    out = FaceAttributes()
    if raw.fairface is not None:
        race, gender, age = raw.fairface[:7], raw.fairface[7:9], raw.fairface[9:18]
        ri, gi, ai = int(race.argmax()), int(gender.argmax()), int(age.argmax())
        out.race, out.race_conf = FACE_RACES[ri], round(float(race[ri]), 4)
        out.gender, out.gender_conf, out.gender_source = FACE_GENDERS[gi], round(float(gender[gi]), 4), "fairface"
        out.age, out.age_conf, out.age_source = FACE_AGES[ai], round(float(age[ai]), 4), "fairface"
    if raw.mivolo_gender is not None:
        gi = int(raw.mivolo_gender.argmax())
        out.gender, out.gender_conf = FACE_GENDERS[gi], round(float(raw.mivolo_gender[gi]), 4)
        out.gender_source = "mivolo"
    if raw.mivolo_age is not None:
        out.age_years = round(raw.mivolo_age, 1)
        # A regression has no probability; the bracket is derived, not scored.
        out.age, out.age_conf, out.age_source = age_bracket(raw.mivolo_age), None, "mivolo"
        out.age_body = raw.mivolo_body
    return out


def pool(raws: list[RawAttributes]) -> FaceAttributes | None:
    """One identity's attributes from all its faces.

    Probabilities are averaged before taking the top label and ages take the
    median, so one odd frame cannot flip or drag an identity's label.
    """
    ff = [r.fairface for r in raws if r.fairface is not None]
    mg = [r.mivolo_gender for r in raws if r.mivolo_gender is not None]
    ma = [r.mivolo_age for r in raws if r.mivolo_age is not None]
    pooled = to_attributes(
        RawAttributes(
            fairface=np.mean(np.vstack(ff), axis=0) if ff else None,
            mivolo_gender=np.mean(np.vstack(mg), axis=0) if mg else None,
            mivolo_age=statistics.median(ma) if ma else None,
        )
    )
    if pooled is not None:
        pooled.age_body = None  # a median over faces, read with a body or not
    return pooled
