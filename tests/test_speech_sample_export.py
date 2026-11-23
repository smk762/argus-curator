"""The checked-in sample speech export (#233): rows parse, files match, speakers stay disjoint."""

from __future__ import annotations

import json
from pathlib import Path

from argus_curator.models import SpeechManifestRow

_SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "speech-export"
_TRAIN = _SAMPLE / "train"
_HELD = _SAMPLE / "held_out"


def _rows(folder: Path) -> list[SpeechManifestRow]:
    return [
        SpeechManifestRow.model_validate(json.loads(line))
        for line in (folder / "speech_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_sample_speech_export_rows_parse_and_match_their_files() -> None:
    assert _SAMPLE.is_dir(), "samples/speech-export is missing; restore it for consumers (#233)"
    for folder in (_TRAIN, _HELD):
        rows = _rows(folder)
        assert rows, f"{folder.name}/speech_manifest.jsonl is empty"
        for row in rows:
            clip = folder / row.exported_path
            text = folder / row.transcript_path
            assert clip.is_file() and clip.stat().st_size > 0
            assert text.is_file()
            assert text.read_text(encoding="utf-8").strip() == row.transcript.strip()
            assert text.resolve().parent == clip.resolve().parent
            assert text.stem == clip.stem
            # Placeholders: absolute paths name a fictional curator layout, not this checkout.
            assert not Path(row.exported_abs_path).is_file()
            assert not Path(row.transcript_abs_path).is_file()


def test_the_sample_train_and_held_out_are_speaker_disjoint() -> None:
    train, held = _rows(_TRAIN), _rows(_HELD)
    assert {r.split for r in train} == {"train"}
    assert {r.split for r in held} == {"held_out"}

    def ids(rows: list[SpeechManifestRow]) -> set[str]:
        out: set[str] = set()
        for r in rows:
            for value in (r.speaker_id, r.speaker_label, r.speaker_cluster):
                if value is not None:
                    out.add(value)
        return out

    assert ids(train) & ids(held) == set()
    # The train set includes a de-collided basename so consumers can't rebuild paths from rel_path.
    assert any(r.exported_path != Path(r.rel_path).name for r in train)
