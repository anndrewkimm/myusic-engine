# Phase-3 feature and model card

## Intended use

This extractor produces reproducible descriptors and music embeddings from audio that is owned,
licensed, public-domain, or available under a compatible Creative Commons license. Its purpose is
audio similarity, clustering experiments, and inputs to a personal preference ranker.

It is a clean-room approximation of useful music-information-retrieval concepts. It does not copy,
query, or claim to reconstruct Spotify's private feature implementation. Values use custom,
versioned names so they cannot be confused with official Spotify outputs.

## Input and provenance boundary

Each JSON Lines manifest item requires:

- `track_id`: stable recording identity;
- `audio_path`: absolute path or a path relative to the manifest;
- `rights_basis`: `owned`, `licensed`, `public_domain`, or `creative_commons`.

Audio is decoded to floating point, mixed to mono, and resampled with a rational polyphase filter.
Every output retains its feature name, source, extractor/model version, analyzed seconds, and
confidence. Objective and embedding confidence remain separate from identity-match confidence.

## Objective DSP feature set v1

Objective DSP v0.2.0 analyzes mono 44.1 kHz audio with a 4,096-sample Hann window and 1,024-sample
hop. Rhythm analysis uses a 2,048-sample frame and 512-sample hop. Configuration changes that alter
values require a new `source_version`.

| Feature | Definition | Main limitation |
|---|---|---|
| `tempo_bpm_estimate_v1` | Robust-gated onset-envelope beat tracker estimate with a minimum tracked-beat requirement | Half/double tempo and weak-rhythm ambiguity |
| `beat_strength_v1` | Mean beat-synchronous gated onset strength relative to active-onset peaks | Not a calibrated danceability score |
| `onset_rate_hz_v1` | Robust-gated spectral onsets per analyzed second | Sensitive to percussion density and mastering |
| `key_estimate_v1`, `mode_estimate_v1` | A440-referenced mean chroma matched against rotated major/minor key profiles | Detuning, ambiguous keys, modulations, and relative major/minor |
| `key_strength_v1` | Key-profile correlation weighted by chroma concentration | Heuristic confidence, not probability |
| `integrated_loudness_lufs_v1` | BS.1770-style gated integrated loudness via `pyloudnorm` | Short or unusual signals may not yield a value |
| `dynamic_range_db_v1` | 95th minus 10th percentile frame RMS in dB | Not album-level loudness range |
| `bass_energy_ratio_150hz_v1`, `bass_energy_ratio_250hz_v1` | Power below the cutoff divided by total spectral power | Mix-level bass amount, not isolated bass stem |
| `spectral_centroid_hz_v1` | Power-weighted mean frequency | Brightness proxy affected by arrangement and mastering |
| `spectral_rolloff_85_hz_v1` | Frequency containing 85% of mean spectral power | Summary statistic only |
| `spectral_flatness_v1` | Mean frame geometric/arithmetic power ratio | Noise-likeness, not a full timbre representation |
| `chroma_mean_v1` | L1-normalized mean 12-bin pitch-class energy | Discards chord order and sections |
| `mfcc_mean_v1`, `mfcc_std_v1` | Mean and standard deviation of 20 MFCCs | Discards temporal structure |

Coverage confidence reaches 1.0 at 20 seconds. Tempo and key observations additionally reduce
confidence according to periodicity and tonal strength. The rhythm frontend suppresses
frequency-local leakage, gates against both an absolute floor and a median-absolute-deviation noise
floor, and requires at least four tracked beats. Unmeasurable tempo or loudness is omitted; silence
is rejected instead of receiving invented values.

## Discogs-EffNet embedding v1

The pinned model is MTG's `discogs-effnet-bsdynamic-1.onnx`, trained to predict 400 Discogs music
styles. The model file is 18,027,718 bytes and must match SHA-256
`a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c`.

Preprocessing reproduces the published Essentia MusiCNN frontend:

1. Resample mono audio to 16 kHz.
2. Compute magnitude spectra with a 512-sample Hann frame, 256-sample hop, and centered zero padding.
3. Project to 96 Slaney-normalized mel bands from 0–8 kHz.
4. Apply `log10(1 + 10000 * mel)` compression.
5. Form 128-frame patches every 62 frames (about one prediction per second).
6. Infer one 1,280-dimensional vector per patch with ONNX Runtime.
7. Retain window vectors, take their arithmetic mean, then L2-normalize the track vector.

The aggregate record selector is:

```text
discogs_effnet_embedding_v1
  @ mtg_essentia_onnx
  : discogs-effnet-bsdynamic-1+musicnn-preprocess-v1+mean-l2-v1
```

Embedding confidence is a coverage reliability score capped at 0.95 after 30 seconds. It is not a
probability that two songs are similar. The model was trained on editorial style metadata, so it may
overweight genre/production cues, underrepresent rare traditions, and reflect Discogs taxonomy and
dataset biases.

MTG distributes its models under CC BY-NC-SA 4.0 and offers proprietary licensing separately. The
download command requires explicit acknowledgment. Review compatibility before any public or
commercial use. Model and Essentia documentation:

- https://essentia.upf.edu/models.html
- https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json
- https://github.com/MTG/essentia

## Learned audio score pack v1

Six pinned Essentia binary classifier heads reuse the raw 1,280-dimensional window embeddings. This
adds only small feed-forward models rather than decoding audio or running the base network again.

| Feature | Positive class | Interpretation |
|---|---|---|
| `danceable_score_v1` | danceable | Learned danceable/not-danceable Softmax score |
| `acoustic_score_v1` | acoustic | Learned acoustic/non-acoustic Softmax score |
| `instrumental_score_v1` | instrumental | Learned instrumental/voice Softmax score |
| `happy_score_v1` | happy | Learned happy/non-happy Softmax score |
| `aggressive_score_v1` | aggressive | Learned aggressive/non-aggressive Softmax score |
| `relaxed_score_v1` | relaxed | Learned relaxed/non-relaxed Softmax score |

For each feature, the track value is the arithmetic mean of the positive-class score across all
windows. The model SHA-256, class ordering, and positive-class index are pinned in code. The output
is explicitly a score, not a calibrated probability and not a Spotify value. Feature confidence
combines coverage, mean classification margin, and cross-window agreement. Window scores are saved
beside embeddings when `--window-output-dir` is used so later calibration can change without
re-extracting audio.

Run `download-feature-head-models --accept-noncommercial-license`, then pass
`--feature-head-model-dir artifacts/models/feature-heads` to `analyze-audio`. Exact source-task
sizes, reported validation metrics, Spotify concept boundaries, and the fine-tuning plan are in
[Spotify-like feature research](spotify-like-feature-research.md).

## Validation completed

Automated synthetic tests currently verify:

- known click tempo near 120 BPM;
- monotonic low-frequency ratios for bass versus treble tones;
- gain sensitivity for LUFS but gain invariance for spectral energy ratios;
- pinned chroma/MFCC and embedding dimensions;
- deterministic 128 × 96 model patches;
- valid mean pooling, nonzero checks, and unit-norm output;
- manifest rights declarations, decoding, mono mixing, and resampling;
- end-to-end CLI output without requiring private data;
- real ONNX Runtime inference using the pinned model on a generated signal.
- verified ONNX inference through all six pinned classifier heads;
- rhythm abstention on a stationary tone that previously produced a false tempo.

## Required before phase-3 exit

1. Run on a permitted real-music corpus such as a compatible FMA subset or owned files.
2. Add time-stretch, pitch-shift, bass-EQ, compression, silence, and codec transformation tests.
3. Create human-reviewed similar/dissimilar pairs, including rhythm-only and timbre-only matches.
4. Measure Recall@K / nDCG@K on those pairs and compare embeddings against handcrafted features.
5. Inspect nearest neighbors across genres and underrepresented music for obvious failure modes.
6. Decide whether robust or section-aware pooling improves retrieval over the mean baseline.
7. Collect independent labels and calibrate learned scores with artist-grouped splits.
8. Add scalar energy, speechiness, liveness, meter, or valence analogues only when each has a labeled
   target definition, baseline, cross-validation protocol, calibration curve, and separate model
   version.
