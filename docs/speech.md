# Speech scans

A voice training set needs the decisions an image set does (what is usable,
what repeats, what is unsafe to train on), measured on audio. A speech scan
scores a folder of WAV/FLAC clips and their transcripts. It decides and
records; it never rewrites a clip, and it writes no manifest rows yet (speech
selection, speakers and the speaker-disjoint split are
[#187](https://github.com/smk762/argus-curator/issues/187)).

```bash
pip install "argus-curator[speech]"           # soundfile (WAV/FLAC) + onnxruntime for DNSMOS
argus-curator scan ./LJSpeech-1.1 --modality speech -v
argus-curator dnsmos-fetch                    # once, for --mos
argus-curator scan ./corpus --modality speech --mos --speech-config thresholds.json
```

Over HTTP it is the same `POST /scan/folder` (or `/scan/folder/stream`), with a
speech profile:

```json
{"folder": "corpus", "target_profile": {"modality": "speech", "target_category": "voice"},
 "speech": {"max_duration_s": 15, "min_snr_db": 25, "mos": true, "min_mos": 3.0}}
```

The profile is argus-cortex's `TargetProfile` with `modality: "speech"` and a
speech category (`voice`, `accent`, `language`, `speaking_style`). The request
is refused (422) if it also turns on an image check (`faces`, `scenes`,
`garments`, `embeddings`, `nsfw`, `guard`, `crops`, `prefilter`). The summary
keeps its rows in `clips` (`ClipResult`), with `results` empty, and the stream's
`chunk` events carry `clips` with `modality: "speech"`. `GET /scan/{id}`
paginates the clips. Select and export refuse a speech scan (400). An image
scan, the image scanner and `ManifestRow` still refuse a speech profile.

## Transcripts

Read from inside the scanned folder only (a symlink out of it is ignored):

| `speech.transcripts` | Where | Notes |
|---|---|---|
| `sidecar` | `<clip>.txt` beside the clip | Up to 64 KB. |
| `ljspeech` | `metadata.csv` at the top | `id\|text\|normalized`; the normalized text wins. `id` is the clip's stem. |
| `common_voice` | `validated.tsv`, `train.tsv`, `dev.tsv`, `test.tsv` at the top | `path` matched by stem (convert Common Voice's MP3 to WAV/FLAC first); `client_id`, `accents`, `locale` and `age` are carried as `speaker`, `accent`, `language`, `age`. |
| `auto` (default) | a sidecar, else LJSpeech, else Common Voice | |

Common Voice's labels are read whatever the layout, so a sidecar can't hide
them. A clip whose corpus age label is under 18 (Common Voice's `teens` covers
13-19 and counts as a whole) is rejected as `corpus_under_18` before it is
decoded, and no setting turns that off. Nothing estimates age from a voice.

The Blizzard 2027 layout isn't read yet: its data isn't published
([#202](https://github.com/smk762/argus-curator/issues/202)).

## Checks

Each clip is read through `argus_cortex.limits.open_audio` (WAV/FLAC only,
size, channels, rate and duration checked before any sample is decoded). Every
threshold in `SpeechConfig` is a prefilter. A clip past one is rejected,
`reject_reason` names the first check it failed, `failed_checks` lists every
one, and `reject_detail` says what was measured. `null` turns a threshold off.

| Check | Measured | Default threshold |
|---|---|---|
| `unreadable` | not a WAV/FLAC clip the scan can decode (rates 8-96 kHz, at most 2 channels) | always |
| `duration` | length | `min_duration_s` 1, `max_duration_s` 20 (at most 120) |
| `sample_rate` | rate | `min_sample_rate` 16000 |
| `channels` | channels | `max_channels` 1 |
| `silent` | peak under -80 dBFS | always |
| `clipping` | share of samples at full scale | `max_clipping_ratio` 0.001 |
| `leading_silence`, `trailing_silence` | time before the first and after the last 30 ms frame within `silence_db` (-40) of the loudest | 1 s each |
| `transcript` | no transcript found | `require_transcript` true |
| `speaking_rate` | letters and digits per second of the clip less its edge silence | 4-30 chars/s |
| `snr` | speech frames' power over the rest (energy VAD) | off (`min_snr_db`) |
| `speech_ratio` | share of 10 ms frames the energy VAD marks as speech (a model VAD is [#204](https://github.com/smk762/argus-curator/issues/204)) | off (`min_speech_ratio`) |
| `mos` | DNSMOS OVRL, 1-5 (with `mos: true`) | off (`min_mos`); with it set, a clip DNSMOS couldn't score is rejected |
| `cer` | CER of the transcript against an STT pass | off (`max_cer`); needs an STT engine, see below |

The score (0-1) weights the measured terms (`weight_mos` 0.5 on (OVRL-1)/4,
`weight_snr` 0.3 on SNR/`snr_ref_db`, `weight_speech_ratio` 0.2) and
renormalises over the ones measured.

**Near-duplicates.** Each clip gets a 64-bit spectrogram fingerprint: log
energies in 32 bands (100-3800 Hz) of the clip at 8 kHz, cut to its loud part,
resampled to 32 steps, with the 8x8 lowest DCT coefficients against their median
(pHash for a spectrogram). Clips within `duplicate_distance` bits (6; -1
disables) whose lengths are within 10% group together, and the best-scoring
one is kept as the representative. A re-encode, resample or gain change of a
clip lands within a few bits. A different take of the same line doesn't.

## DNSMOS

With `mos: true`, each clip that passed the checks above is scored by
Microsoft's DNSMOS P.835 (`sig_bak_ovr.onnx`, DNS-Challenge, code MIT and other
content CC BY 4.0): speech (`sig`), background (`bak`) and overall (`ovrl`).
`argus-curator dnsmos-fetch` downloads it (1.2 MB), pinned by commit
`82f1b17e` and sha256 `269fbebd...6edd`, and a file that doesn't hash to that is
refused at load. Set `ARGUS_CURATOR_DNSMOS_MODEL` to keep it elsewhere.

The model reads 9.01 s of 16 kHz audio. A clip is resampled, tiled up to
that length, and scored over at most 8 windows (1 s hops, thinned evenly), one
fixed-shape window per pass, so the capped CUDA arena never grows with a clip's
length. On CUDA it loads and runs under the host GPU lease as the kind `mos`
(about 270 MB loaded and 650 MB at peak on a 3090; `gpu-tenant.yaml`).

## CER and argus-echo

`speech.Transcriber` is the hook for a speech-to-text pass: mono samples and a
rate in, text out. Given one, `scan_speech_folder(..., transcriber=...)`
records `stt_transcript` and `cer` on every passing clip with a transcript, and
`max_cer` rejects clips past it. Curator ships no STT model. It will call
argus-echo's engine once that exists
([argus-echo#4](https://github.com/smk762/argus-echo/issues/4),
[#203](https://github.com/smk762/argus-curator/issues/203)). Until then a scan
that sets `max_cer` is refused.
