# SynSync-Bench

[![ci](https://github.com/habib-analyst/synsync-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/habib-analyst/synsync-bench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **controlled audio-visual manipulation benchmark** with **exact ground-truth sync offsets** (known by construction, never estimated), plus a **classical + learned fusion detector** for deepfake triage.

**The problem it solves.** Deepfake detectors are usually evaluated within-dataset, where they silently overfit to one manipulation family. SynSync-Bench generates A/V manipulations programmatically — desync, clock drift, re-encode chains, frame-rate conversion, lip-sync jitter — so ground truth is exact, and evaluates detectors on **held-out manipulation families** (the honest analog of cross-dataset testing).

**The finding.** A cheap logistic fusion of two classical signal families — A/V desync lag and spectral compression artifacts — beats either family alone, including on manipulation types never seen in training.

```
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ synthetic    │    │  feature extraction │    │  logistic fusion │
│ talker       │───▶│  • desync lag/drift │───▶│  8 features      │──▶ manipulated?
│ (ffmpeg A/V  │    │  • spectral: blocki-│    │  P(manipulated)  │
│  manipula-   │    │    ness, HF-ratio,  │    │                  │
│  tions)      │    │    blur, noise      │    │                  │
└──────────────�ulcorner    └─────────────────────┘    └──────────────────┘
```

## Quickstart (< 5 min)

```bash
pip install -r requirements.txt   # needs ffmpeg on PATH
# 1. generate the benchmark (default: 600 clips, ~25 min CPU)
synsync-bench generate --out bench/
# 2. extract features
synsync-bench features bench/
# 3. train fusion + run all eval protocols
synsync-bench evaluate bench/ --out results/
# single-clip triage
synsync-bench analyze clip.mp4 clip.wav
```

## Benchmark design

| Manipulation | Params | Ground truth | Label |
|---|---|---|---|
| `clean` | — | offset 0 | real |
| `desync` | ±40, ±80, ±160, ±320 ms | exact offset (sample-exact shift) | manipulated |
| `drift` | ±15, ±30, ±60 ms/s | exact drift rate (resampling) | manipulated |
| `reencode` | 100k / 300k bitrate | codec artifacts, sync intact | manipulated |
| `fps_convert` | 25→30→25 round-trip | judder, sync intact | manipulated |
| `lip_jitter` | 3 jittered mouth segments | visual-only | manipulated |

Source material is a synthetic "talker": mouth openness follows a speech-like amplitude contour that also modulates the audio — strongly correlated A/V like real speech, fully deterministic per seed. Every clip ships in `manifest.jsonl` with its exact parameters.

## Features (8, all CPU)

- **Desync (4):** median |lag|, mean confidence, drift slope, bad-window fraction — windowed normalized cross-correlation of audio-energy vs mouth-openness envelopes.
- **Spectral (4):** blockiness (8×8 boundary discontinuity), high-frequency DCT energy ratio, blur (Laplacian variance), noise residual.

## Results (600-clip benchmark, measured)

| Protocol | desync rule | spectral-only | **fusion** |
|---|---|---|---|
| In-family 5-fold CV (AUC) | 0.64 | 0.62 | **0.98** |
| Cross-family A→B (AUC) | 0.59 | 0.60 | **0.84** |
| Cross-family B→A (AUC) | 0.68 | 0.66 | **0.92** |

Cross-family = train on {desync, reencode}, test on {drift, fps-convert, lip-jitter} and vice versa — manipulation families the model never saw. Fusion beats the best single signal by **+24pp and +26pp AUC** cross-family (success bar was +3pp).

Offset sensitivity (fusion, trained without any desync clips): 94% detection at 40 ms → 100% at ≥80 ms.

## Layout

```
synsync/          source renderer, manipulations, features, fusion, CLI
tests/            determinism + pipeline tests (pytest)
scripts/          extract_features.py helper
```

## Limits & honesty

- Synthetic talkers, not real faces — a stepping stone, not a replacement for FaceForensics++/Celeb-DF evaluation (access forms pending).
- Fusion is logistic regression on handcrafted features: light and auditable by design (triage tool), not SOTA-chasing.

## Citation

```bibtex
@software{rehman2026synsync,
  author  = {Habib Ur Rehman},
  title   = {SynSync-Bench: controlled A/V manipulation benchmark with exact ground truth},
  year    = {2026},
  url     = {https://github.com/habib-analyst/synsync-bench}
}
```

## Roadmap

- [ ] FaceForensics++ / Celeb-DF cross-dataset evaluation (access pending)
- [ ] Learned visual encoder (tiny CNN) as a third signal family
- [ ] Real-time streaming triage mode
