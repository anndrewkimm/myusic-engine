# Spotify-like audio feature research and clean-room proxy plan

## Conclusion

Spotify publishes useful descriptions of its audio-feature concepts, but not the training data,
labels, model architecture, calibration procedure, or exact signal-processing implementation.
Therefore this project can build independently measured analogues, but it cannot honestly claim to
reproduce Spotify's values. Every analogue uses a custom name, source, model version, analyzed
coverage, and confidence.

The deprecated Spotify Audio Features endpoint also carries an explicit policy notice that Spotify
Content may not be used to train or otherwise be ingested into an ML/AI model. The safe project
boundary is therefore:

- Spotify exports provide private listening behavior only;
- owned, licensed, public-domain, or compatible Creative Commons audio provides signal inputs;
- independent public models or independently collected labels provide learned targets;
- Spotify audio-feature outputs are never used as distillation labels.

Primary reference: [Spotify's Audio Features endpoint](https://developer.spotify.com/documentation/web-api/reference/get-audio-features).

## Feature coverage map

| Published Spotify concept | Current clean-room analogue | Interpretation | Current gap |
|---|---|---|---|
| acousticness | `acoustic_score_v1` | Mean acoustic-class Softmax score | Not calibrated to Spotify's scale |
| danceability | `danceable_score_v1`, tempo, beat strength, onset rate | Learned binary danceability plus transparent rhythm evidence | Binary-head score, not Spotify's continuous target |
| energy | LUFS, dynamic range, onset rate, flatness, centroid, bass ratios | Published contributing measurements remain separate | No defensible scalar until labels exist |
| instrumentalness | `instrumental_score_v1` | Mean instrumental-class Softmax score | Voice/instrumental classifier is not Spotify's vocal-content definition |
| key and mode | `key_estimate_v1`, `mode_estimate_v1`, `key_strength_v1` | Chroma/profile estimate with abstention confidence | Relative-key ambiguity and no parity calibration |
| liveness | none | — | Needs an audience/live-performance corpus and benchmark |
| loudness | `integrated_loudness_lufs_v1` | BS.1770-style gated integrated loudness | LUFS is explicit and reproducible, but not guaranteed to equal Spotify's dB value |
| speechiness | none | — | Needs speech/music/rap section labels; vocal presence is not speechiness |
| tempo | `tempo_bpm_estimate_v1` | Onset-envelope beat-tracker estimate | Half/double-tempo ambiguity remains |
| time signature | none | — | Meter inference is unreliable without a dedicated model and labeled evaluation |
| valence | `happy_score_v1`, `aggressive_score_v1`, `relaxed_score_v1` | Separate, inspectable mood axes | These must not be collapsed into “valence” without labels |

This separation is useful for data science: a later model can learn whether loudness, rhythmic
regularity, timbre, and learned mood scores predict this user's behavior instead of accepting one
opaque vendor number as ground truth.

## Learned score pack v1

Six official Essentia ONNX classifier heads now operate on each retained 1,280-dimensional
Discogs-EffNet window. The implementation validates the model hash and tensor interface, selects the
declared positive class, retains every window score, and emits the arithmetic mean as the track
score. These scores are Softmax outputs, not calibrated probabilities.

| Output | Positive class | Training items reported by MTG | Reported 5-fold normalized accuracy |
|---|---:|---:|---:|
| `danceable_score_v1` | danceable | 306 | 0.97 |
| `acoustic_score_v1` | acoustic | 321 | 0.95 |
| `instrumental_score_v1` | instrumental | 1,000 | 0.96 |
| `happy_score_v1` | happy | 302 | 0.87 |
| `aggressive_score_v1` | aggressive | 280 | 0.98 |
| `relaxed_score_v1` | relaxed | 446 | 0.91 |

Those figures are the authors' cross-validation results on small in-house collections. They are
evidence that the heads learned their source tasks, not proof of performance on this project's
catalog or of agreement with Spotify. Model metadata is available in Essentia's official
[classification-head index](https://essentia.upf.edu/models/classification-heads/), including the
[danceability](https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.json),
[acoustic](https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.json),
and [voice/instrumental](https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json)
model cards.

Feature confidence is deliberately separate from the score. It combines audio coverage, average
binary-classification margin, and agreement across windows. A sectional track can consequently have
a valid middle score with lower confidence. The private window NPZ retains both embeddings and all
six score trajectories for later section-aware pooling and calibration.

## Rhythm correction in objective DSP v0.2.0

Controlled probes found that normalizing a tiny onset envelope could turn spectral leakage from a
stationary tone into a high-confidence tempo. The revised extractor now uses frequency-local maximum
suppression, an absolute onset floor, a median-absolute-deviation noise floor, a minimum spacing
between onset peaks, and a minimum of four tracked beats. It abstains when those conditions are not
met. Synthetic regression tests verify both a known 120 BPM click track and a stationary tone.

An additional 16-second controlled probe produced:

| Input | Estimated tempo | Onset rate | Result |
|---:|---:|---:|---|
| 60 BPM clicks | 60.09 BPM | 1.000/s | correct |
| 90 BPM clicks | 90.67 BPM | 1.500/s | correct |
| 120 BPM clicks | 120.19 BPM | 1.938/s | correct |
| 150 BPM clicks | 74.90 BPM | 2.438/s | half-tempo alias |
| stationary tone | omitted | 0.000/s | correct abstention |

The 150 BPM case is retained as a known failure rather than tuned away on one synthetic example.
The onset rate preserves evidence for a later multi-candidate tempo model, while the model card
continues to disclose half/double-tempo ambiguity.

## Calibration and fine-tuning plan

Fine-tuning before lawful labeled audio exists would optimize against synthetic artifacts or an
unverifiable vendor target. The next defensible sequence is:

1. Assemble a permitted evaluation corpus and freeze a track/artist-grouped split before modeling.
2. Collect human labels with explicit questions: binary danceable, acoustic versus produced,
   vocal-content fraction, and pairwise mood judgments.
3. Evaluate untouched heads first using ROC-AUC/PR-AUC, Brier score, expected calibration error, and
   artist-grouped confidence intervals.
4. Compare against simple DSP/logistic-regression baselines and report ablations. A pretrained head
   only earns its complexity if it beats them out of sample.
5. Calibrate with Platt scaling or isotonic regression on a calibration split; never fit calibration
   on the held-out test set.
6. Fine-tune the embedding network only if frozen-head and shallow-model baselines plateau and the
   labeled set is large enough. Preserve a frozen baseline to quantify whether fine-tuning helps.
7. Run gain, codec, silence-padding, bass-EQ, dynamic-compression, pitch-shift, and time-stretch
   stress tests, then inspect nearest-neighbor failures by genre and source.

The underlying representation is documented in the ISMIR paper
[Music Representation Learning Based on Editorial Metadata from Discogs](https://archives.ismir.net/ismir2022/paper/000099.pdf).
The downloadable MTG models use CC BY-NC-SA 4.0; commercial use needs a separate licensing review.
