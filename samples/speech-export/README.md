# Sample speech export

A tiny `speech_manifest.jsonl` export for consumers (argus-forge and anyone
else reading curator's speech rows). The WAVs are an empty header only: no
recorded voice. The transcripts are short placeholder lines.

```
samples/speech-export/
  train/          # three train clips, two speakers; one de-collided basename
  held_out/       # one held-out clip of a third speaker
```

Each folder is what `POST /export` (or `scan --modality speech --copy-to`)
writes for one `split_part`. Use them together when you need a
speaker-disjoint train / held-out pair.

## How to open it

Join `exported_path` and `transcript_path` onto the folder you can see:

```python
from pathlib import Path
import json

export = Path("samples/speech-export/train")
for line in (export / "speech_manifest.jsonl").read_text().splitlines():
    row = json.loads(line)
    clip = export / row["exported_path"]
    text = (export / row["transcript_path"]).read_text().strip()
    assert clip.is_file() and text == row["transcript"].strip()
```

`exported_abs_path` / `transcript_abs_path` in these rows name a fictional
curator layout (`/sample/export/...`). They are only for a consumer that
shares curator's filesystem; behind a container mount, ignore them and join
the relative paths as above. See [Speech scans](../../docs/speech.md#for-consumers).

## Contract

- File: `speech_manifest.jsonl` (never `manifest.jsonl`).
- Row shape: `SpeechManifestRow`, `speech_manifest_version` **1.1**.
- Schema: `schema/speech-manifest-row.snapshot.json` and the live model in
  `argus_curator.models`.
- The `<stem>.txt` beside each clip is what a trainer reads; if someone edits
  it after export, that file wins over the row's `transcript`.
