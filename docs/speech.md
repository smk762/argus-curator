# Speech scans

A voice training set needs the decisions an image set does (what is usable,
what repeats, what is unsafe to train on), measured on audio. A speech scan
scores a folder of WAV/FLAC clips and their transcripts, can cluster them into
speakers, and exports the keepers with a transcript beside each clip and a
speech manifest ([below](#speakers)). It never rewrites a clip.

```bash
pip install "argus-curator[speech]"           # soundfile (WAV/FLAC) + onnxruntime for DNSMOS
argus-curator scan ./LJSpeech-1.1 --modality speech -v
argus-curator dnsmos-fetch                    # once, for --mos
argus-curator speaker-fetch                   # once, for --speakers
argus-curator vad-fetch                       # once, for --vad silero
argus-curator scan ./corpus --modality speech --mos --speakers --speech-config thresholds.json
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
paginates the clips. An image scan, the image scanner and `ManifestRow` still
refuse a speech profile.

## Transcripts

Read from inside the scanned folder only (a symlink out of it is ignored):

| `speech.transcripts` | Where | Notes |
|---|---|---|
| `sidecar` | `<clip>.txt` beside the clip | Up to 64 KB. |
| `ljspeech` | `metadata.csv` at the top | `id\|text\|normalized`; the normalized text wins. `id` is the clip's stem. |
| `common_voice` | `validated.tsv`, `train.tsv`, `dev.tsv`, `test.tsv` at the top | `path` matched by stem (convert Common Voice's MP3 to WAV/FLAC first); `client_id`, `accents`, `locale`, `age` and `gender` are carried as `speaker`, `accent`, `language`, `age`, `gender`. |
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
| `snr` | speech frames' power over the rest (the `vad`'s mask, [below](#silero-vad)) | off (`min_snr_db`) |
| `speech_ratio` | share of 10 ms frames the `vad` marks as speech: the energy VAD (default) or Silero | off (`min_speech_ratio`) |
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

## Silero VAD

The default VAD is the energy VAD: a 30 ms frame is speech when it sits 10 dB
over the clip's 10th-percentile frame and within `silence_db` of its loudest,
with gaps under ~200 ms closed. It needs no model, but loudness is all it
reads, so noise hides speech from it and loud non-speech counts as speech.

With `vad: "silero"` (CLI `--vad silero`), the speech ratio and the SNR read
[Silero VAD](https://github.com/snakers4/silero-vad) instead (Silero Team,
MIT). `argus-curator vad-fetch` downloads its `silero_vad.onnx` (2.3 MB) from
release `v6.2.3`, pinned by commit `5cd79456` and sha256 `1a153a22...88e3`; a
file that doesn't hash to that is refused at load. Set
`ARGUS_CURATOR_VAD_MODEL` to keep it elsewhere. Every row records the VAD that
produced its numbers in `ClipResult.vad` (`energy` or `silero`). If Silero
can't read a clip, `vad_error` says why and `speech_ratio` and `snr_db` stay
null: the energy VAD is never swapped in, and with `min_snr_db` or
`min_speech_ratio` set the clip is rejected.

- **Input:** the clip is resampled to 16 kHz and read in 32 ms windows (512
  samples, with the previous window's last 64 before them). Each pass is one
  window, batch 1, carrying the model's state to the next, so every pass has
  the same shape.
- **Mask:** Silero's defaults. A window starts speech at probability 0.5 and
  speech ends below 0.35. Gaps under 100 ms are closed, and speech runs under
  250 ms are dropped. Each 10 ms scan frame takes the window its centre falls
  in.
- **SNR split:** the speech side is the mask. The noise side leaves out every
  window Silero gives 0.35 or more (a word too short to count as speech still
  isn't noise) and 50 ms either side of them. Without that guard band, the
  decay at the end of each word counted as noise, and clean clips read about
  20 dB low.
- **CPU only, no GPU lease.** Silero runs on onnxruntime's CPU provider alone,
  with one intra-op thread per session (the scan's `max_workers` clip threads
  are the parallelism). It never touches the GPU, so it takes no lease, holds
  no CUDA arena and has no `gpu-tenant.yaml` kind. One pass is launch-bound
  on CUDA: on a 3090 (onnxruntime-gpu 1.32) a pass took 1.6-2.0 ms, against
  0.12-0.24 ms on one CPU thread. Each pass needs the last one's state, so
  nothing can be batched to win that back. In a scan it costs about 5 ms per
  second of audio (~0.1 s for a 20 s clip), against ~0.05 ms for the energy
  VAD.

**Calibration.** 49 real takes (studio-style recordings, 24 kHz mono, 2-23 s,
770 s in all, read up to 20 s each) were measured clean. Each was then mixed
with white noise, pink noise and a non-speech "music" bed (a sustained chord
with tremolo and percussive clicks) at a known SNR, set over the clean
clip's speech frames. The music bed and pink noise were also measured alone,
at speech level. The medians, with the 10th-90th percentile for SNR:

| Condition | Speech ratio, energy | Speech ratio, Silero | SNR dB, energy | SNR dB, Silero |
|---|---|---|---|---|
| clean | 0.80 | 0.80 | 44.2 [39.9, 52.6] | 42.4 [33.6, 60.1] |
| white noise 30 dB | 0.77 | 0.80 | 28.8 [28.3, 29.5] | 29.7 [28.7, 30.0] |
| white noise 20 dB | 0.73 | 0.80 | 18.8 [18.2, 19.2] | 19.9 [19.8, 20.0] |
| white noise 10 dB | 0.48 | 0.80 | 7.6 [6.9, 8.5] | 9.9 [9.8, 10.1] |
| white noise 5 dB | 0.13 | 0.80 | 4.5 [3.6, 5.8] | 4.9 [4.8, 5.1] |
| white noise 0 dB | 0.00 | 0.81 | null on 42 of 49 | -0.1 [-0.4, 0.1] |
| pink noise 20 dB | 0.74 | 0.80 | 19.4 [18.9, 20.0] | 19.9 [19.3, 20.6] |
| pink noise 10 dB | 0.61 | 0.80 | 8.6 [7.8, 9.5] | 10.1 [9.2, 10.8] |
| pink noise 0 dB | 0.03 | 0.81 | 4.0 [2.9, 5.0] | 0.1 [-1.4, 1.9] |
| music bed 20 dB | 0.75 | 0.81 | 19.2 [18.2, 21.3] | 19.8 [18.8, 21.6] |
| music bed 10 dB | 0.67 | 0.79 | 9.3 [8.3, 11.9] | 9.8 [8.6, 11.7] |
| music bed 0 dB | 0.34 | 0.78 | 2.8 [2.0, 4.8] | 0.1 [-2.9, 2.6] |
| music bed alone | 0.46 | 0.00 | 5.5 [5.2, 5.6] | null on all 49 |
| pink noise alone | 0.00 | 0.00 | 6.2 on 32 of 49 | null on all 49 |

- On clean clips the two agree on the speech ratio: the median per-clip
  difference is 0.01, and the largest is 0.07.
- Under noise, Silero's speech ratio holds within 0.02 of the clean clip's
  down to 0 dB. The energy VAD's falls (0.48 at 10 dB white noise, 0.13 at
  5 dB), so `min_speech_ratio` on the energy VAD rejects noisy clips for
  noise, not for lacking speech.
- Silero's median SNR is within 0.5 dB of the mixed SNR from 30 dB down to
  0 dB. The energy VAD's reads 1-2.5 dB low at 10-20 dB, and at 0 dB it reads
  3-7 dB high, or nothing.
- Loud non-speech alone: the energy VAD calls 46% of the music bed speech, and
  gives pink noise alone an SNR. Silero finds no speech in either, so neither
  gets an SNR.
- The two VADs' SNRs aren't interchangeable at the top end. On takes padded
  with digital silence, Silero's noise side is all zero samples and its SNR
  reads the 100 dB cap, while the energy VAD's noise side takes in a word's
  onset and reads 49-58 dB. Set `min_snr_db` for the VAD you run. On 2 of the
  49 clean takes Silero read under 30 dB (16 and 27 dB) where the energy VAD
  read 52 and 31. On the first, a 60 ms click at the start of the take falls
  outside Silero's speech and counts as noise.

Those mixed SNRs were set over Silero's speech frames on the clean clip,
but the two VADs mark the same share of a clean clip (0.80 each), so the
reference barely favours either. The takes were copied to scratch for the
run and deleted afterwards. No audio is committed.

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

## Speakers

With `speech.speakers: true` (CLI `--speakers`), each clip that passed gets a
speaker embedding, and the clips are clustered into speakers, the counterpart
of face clustering.

- **Model:** WeSpeaker's ECAPA-TDNN (`ECAPA_TDNN_GLOB_c512-ASTP-emb192`,
  large-margin fine-tuned, trained on VoxCeleb2 dev; WeSpeaker's code is
  Apache-2.0 and its VoxCeleb models CC BY 4.0). `argus-curator speaker-fetch`
  downloads WeSpeaker's own ONNX export (25 MB) from Hugging Face
  (`Wespeaker/wespeaker-ecapa-tdnn512-LM`), pinned by revision `a2f3dcb1` and
  sha256 `d71b85d9...10b0`; a file that doesn't hash to that is refused at
  load. Set `ARGUS_CURATOR_SPEAKER_MODEL` to keep it elsewhere.
- **Input:** 80-bin Kaldi filterbanks (25 ms frames every 10 ms, Hamming
  window, 16 kHz), mean-normalised over the clip, computed in numpy. A clip is
  cut into 4 windows of 3 s spread evenly across it (a short clip is tiled up
  to 3 s first), run as one fixed-shape batch, and its embedding is the mean of
  the windows', unit length. On CUDA it loads and runs under the host GPU
  lease as the kind `speakers` (298 MB loaded and 420 MB at peak on a 3090;
  `gpu-tenant.yaml`).
- **Clustering:** each clip joins the nearest speaker whose centroid is within
  `speaker_threshold` (cosine distance, 0.5), or starts one; then every clip
  is reassigned to its nearest centroid a few times. It is linear in clips
  times speakers, so a corpus of hundreds of thousands of clips clusters
  without a pairwise matrix. The threshold is loose on purpose: two speakers
  merged keeps the split speaker-disjoint, one speaker split in two could put
  them in both parts. `ClipResult.speaker_cluster` (`spk_1` is the largest),
  `speaker_distance` (to its centroid) and `speaker_error` record it;
  `ScanSummary.speaker_clusters` lists the speakers.
- **Corpus labels:** where the corpus names speakers (Common Voice's
  `client_id`), a cluster holding more than one label, or a label spread over
  clusters, is flagged: `SpeakerCluster.mismatch`, and `speaker_mismatch` on
  each clip whose label is the minority in its cluster or lives mostly in
  another. `ScanSummary.speaker_mismatches` counts them.
- **Memory only:** the embeddings stay in memory (`ScanSummary.embeddings`),
  held by the scan store per scan and dropped with it, as CLIP embeddings are.
  A centroid is a voiceprint, so none is stored.

Checking the accent and language labels with a model is
[#208](https://github.com/smk762/argus-curator/issues/208).

## Select and export

`POST /scan/{id}/select` and `POST /export` take a speech scan. They read
`min_score`, `keep_similar`, `max_keep`, `split` / `split_part`, the transfer
fields and `speech` (`SpeechSelection`); any image filter set on a speech scan
is refused (400), and `speech` on an image scan too.

```json
{"scan_id": "...", "dest": "blizzard-train", "mode": "symlink",
 "split": {"held_out_fraction": 0.1, "group_by": "speaker"}, "split_part": "train",
 "speech": {"corpus": "Common Voice 17 pt", "license": "CC0-1.0",
            "max_hours_per_speaker": 2, "balance_accents": true}}
```

The CLI exports a speech scan the same way
([#207](https://github.com/smk762/argus-curator/issues/207)): `scan
--modality speech` with `--copy-to`, `--move-to` or `--symlink-to` builds the
same `ExportRequest` and runs it through `export_selection` once the scan is
saved, so it writes what `POST /export` writes. The request above is:

```bash
argus-curator scan corpus/ --modality speech --symlink-to blizzard-train \
  --corpus "Common Voice 17 pt" --license CC0-1.0 \
  --max-hours-per-speaker 2 --balance-accents \
  --held-out-fraction 0.1 --split-part train
```

| CLI flag | `ExportRequest` field |
|---|---|
| `--speaker-id`, `--accent`, `--language` (comma-separated) | `speech.speakers`, `speech.accents`, `speech.languages` |
| `--exclude-speaker-mismatch` | `speech.exclude_speaker_mismatch` (needs `--speakers`) |
| `--min-hours-per-speaker`, `--max-hours-per-speaker`, `--max-hours-per-accent`, `--balance-accents` | the `speech` hours caps |
| `--min-score`, `--keep-similar`, `--max-keep`, `--run-stats` | as for images |
| `--held-out` or `--held-out-fraction`, `--split-method`, `--split-seed`, `--split-part` | `split` (`group_by` is always `speaker`), `split_part` |
| `--corpus` and `--license` | `speech.corpus`, `speech.license` |
| `--voice-consent` and `--consent-delivery` | `speech.consent_record` (relative to the scanned folder, or absolute; needs `--speakers`), `speech.consent_delivery` |

The flags are checked before a clip is read: no provenance, half of one, both
kinds, an image filter (`--pose`, `--garment`, `--nsfw-threshold`,
`--carry-sidecars`, ...) or a speech export flag without a destination is
refused (exit 2). What can only be known from the scan (a split with too few
speakers, a consent record that doesn't cover the set) is refused after it,
before anything is transferred, and the scan stays saved. As with images, the
CLI has no move gate: it is your own shell.

**Selection.** A clip is kept when it passed, has a transcript, scores at
least `min_score` and isn't a near-duplicate (unless `keep_similar`). Then:

| `speech` field | Effect | Keep reason |
|---|---|---|
| `speakers`, `accents`, `languages` | keep only these (`speaker_id`, the corpus labels) | `speaker-excluded`, `accent-excluded`, `language-excluded` |
| `exclude_speaker_mismatch` | drop clips whose corpus label the clustering disputes (needs `speakers` on the scan) | `speaker-mismatch` |
| `min_hours_per_speaker` | drop a speaker with fewer hours | `speaker-under-min-hours` |
| `max_hours_per_speaker` | cap each speaker, best clips first | `speaker-cap` |
| `max_hours_per_accent`, `balance_accents` | cap each accent (at the smallest accent's hours with `balance_accents`), taking clips from its speakers in turn | `accent-cap` |

`max_keep` then takes speakers in turn, each giving its best remaining clip
(`max-keep`), so a cap doesn't hand the set to a few voices. A clip whose
corpus age label reads under 18 is never kept, whatever the request; it is
already rejected at scan (`corpus_under_18`). Select returns each row's
`speaker_id` and `SelectResult.speech`: hours kept, by speaker, by accent, and
with a split hours and speakers per part.

**The split is speaker-disjoint.** Zero-shot TTS is judged on speakers the
model never heard, so `split.group_by` must be `"speaker"`. Clips sharing a
corpus speaker label *or* a speaker cluster form one group, transitively, so
neither reading of a speaker spans both parts even where they disagree. A
clip with neither (no corpus label, and a scan without `speakers`) refuses
the split: it could be anyone. Groups are stratified by their most common
accent and gender label: held-out speakers are shared across those strata in
proportion to how many speakers each has, and a stratum's share never takes
its last speaker. The split is curator's one `split_held_out` (#183), so
it is deterministic and `held_out` / `held_out_fraction` count clips. The
held-out part becomes proof's reference set: export each part to its own
`dest`.

**Provenance.** An export names where its clips came from, or is refused (400):

- a licensed corpus: `speech.corpus` and `speech.license`, written on every
  row, with `provenance` `{kind: "stock", source: corpus, license}`;
- a real person's voice outside one: `speech.consent_record` (an argus-cortex
  `ConsentRecord`, a path under the scanned folder) and
  `speech.consent_delivery`. It is read and checked at select and export for
  `training` of the subject's `voice` at that delivery (revoked, expired, or a
  use or delivery it doesn't cover is refused with cortex's reason). At least
  one of its `reference_recordings` must be a passing clip in the scan, by
  sha256, and the scan must have clustered speakers: only clips in those
  recordings' clusters are kept (`consent:other-speaker`), and rows carry
  `consent_release_sha256`, `consent_record_sha256` and a `consented`
  provenance that names no one: its `subject` is argus-cortex's subject ref
  `consent:<release_sha256>`, never the record's name, and its `source`
  doesn't name who collected the release. `speaker_label` and the transcript
  are your corpus's own text and are carried as they are. Binding by distance to the reference recordings is
  [#209](https://github.com/smk762/argus-curator/issues/209).

Age is never estimated from a voice. A clip's corpus age label is carried as
`age`, and a corpus that gives none leaves it null.

**What is written.** Under `dest`, each clip by the same rules as images
(`copy`, `symlink`, `move` with the server's move gate; a symlink export into
the source never deletes the source), named by the same de-collision plan,
by stem. Beside each clip, `<stem>.txt` holds its transcript: written fresh
(never a link to the source's sidecar, so editing it never reaches the
source), before the clip, and removed if the clip's transfer fails. Then
`speech_manifest.jsonl` (one `SpeechManifestRow` per exported clip, below),
`curation_report.csv` (every clip and its keep reason) and `run_stats.json`
when asked. A move takes the moved clips out of the cached scan. Under
`move` a source sidecar transcript stays in the source folder.

## The speech manifest

`speech_manifest.jsonl` is its own file with its own row shape and version,
never mixed with image rows: `SpeechManifestRow`, version
`SPEECH_MANIFEST_VERSION` **1.0** (`GET /health`'s
`speech_manifest_version`, and every `ExportResult`). It moves independently
of the image `MANIFEST_VERSION` under the same rule (the major for a breaking
change, the minor for an additive one), checked against
`schema/speech-manifest-row.snapshot.json` by `make manifest-snapshot`. Every
key is always present; a field with no value is `null`.

| Field | Type | Meaning |
|---|---|---|
| `speech_manifest_version` | string | `"1.0"`. Refuse a major you don't know. |
| `modality` | string | always `speech` |
| `rel_path` | string | the clip's path in the scanned folder |
| `abs_path` | string | where the clip can be read (under `move`, the destination) |
| `exported_path`, `exported_abs_path` | string | the clip as written under the export root (de-collided) |
| `transcript_path`, `transcript_abs_path` | string | its `<stem>.txt` beside it |
| `target_profile` | object | the scan's `SpeechTargetProfile` (`modality: speech`, a speech category, a TTS `target_backend`) |
| `transcript` | string | the clip's transcript (what the `.txt` holds) |
| `duration_s`, `sample_rate`, `channels` | number | as decoded |
| `speaker_id` | string or null | the corpus speaker label, else the speaker cluster |
| `speaker_label`, `speaker_cluster` | string or null | each on its own |
| `speaker_mismatch` | boolean | the clustering disputes the corpus label |
| `accent`, `language`, `gender`, `age` | string or null | the corpus labels, verbatim; `age` null when unknown, never estimated |
| `score`, `similar_group` | number | as scanned |
| `split` | string or null | `train` or `held_out` with a split, else null |
| `corpus`, `license` | string | where every clip came from |
| `provenance` | object | argus-cortex's `Provenance` (`stock` for a corpus, `consented` under a record, whose `subject` is the subject ref `consent:<release_sha256>`, never a name) |
| `consent_release_sha256`, `consent_record_sha256` | string or null | a consented voice only |

argus-forge reading this file is
[argus-forge#58](https://github.com/smk762/argus-forge/issues/58) (for its
XTTS-v2 emitter, [argus-forge#56](https://github.com/smk762/argus-forge/issues/56)).
