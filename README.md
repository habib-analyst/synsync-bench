[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/synsync-bench.svg)]

# SynSync-Bench

Controlled A/V manipulation benchmark with exact ground-truth sync offsets.

Deepfake detectors are usually evaluated within-dataset, where they silently overfit to one manipulation family. SynSync-Bench generates A/V manipulations programmatically — desync, clock drift, re-encode chains, frame-rate conversion, lip-sync jitter — so ground truth is exact by construction, and evaluates detectors on **held-out manipulation families** (the honest analog of cross-dataset testing).

## Quickstart

```bash
git clone https://github.com/habib-analyst/synsync-bench.git && cd synsync-bench
pip install -r requirements.txt   # needs ffmpeg on PATH

synsync-bench generate --out bench/       # build the benchmark
# plan: 600 clips
# manifest: bench/manifest.jsonl

synsync-bench features bench/             # extract 8 features per clip
# done: 600 new feature rows

synsync-bench evaluate bench/ --out results/   # train fusion + run all eval protocols
# prints per-protocol AUCs, then: full results: results/results.json
```

Single-clip triage dumps the 8 features as JSON: `synsync-bench analyze clip.mp4 clip.wav`.

## Measured results

600-clip benchmark (100 clean, 200 desync ±40–320 ms, 120 drift ±15–60 ms/s, 80 re-encode, 40 fps-convert, 60 lip-jitter):

| protocol | desync rule | spectral-only | **fusion** |
|---|---|---|---|
| in-family 5-fold CV (AUC) | 0.64 | 0.62 | **0.98** |
| cross-family A→B (AUC) | 0.59 | 0.60 | **0.84** |
| cross-family B→A (AUC) | 0.68 | 0.66 | **0.92** |

Cross-family = train on {desync, reencode}, test on {drift, fps-convert, lip-jitter} and vice versa — manipulation families the model never saw. Fusion beats the best single signal by **+24pp / +26pp AUC** cross-family (the success bar was +3pp). Offset sensitivity, fusion trained with zero desync clips: 94% detection at 40 ms → 100% at ≥80 ms. Per-family detection: desync .995, drift 1.0, re-encode .91, fps 1.0, lip-jitter 1.0; clean false-positive rate .18.

## How it works

```
synthetic talker ──▶ feature extraction ──▶ logistic fusion ──▶ P(manipulated)
(ffmpeg A/V; mouth      • 4 desync: lag,       8 features,
 openness follows a      confidence, drift,    CPU-only,
 speech-like envelope    bad-window fraction   auditable by
 that modulates audio)  • 4 spectral: blocki-  design
                          ness, HF-ratio,
                          blur, noise
```

The source material is a synthetic "talker": mouth openness follows a speech-like amplitude contour that also modulates the audio — strongly correlated A/V like real speech, fully deterministic per seed. Every clip ships in `manifest.jsonl` with its exact parameters. Fusion is logistic regression on handcrafted features: light and auditable by design — a triage tool, not SOTA-chasing.

Limits, stated plainly: synthetic talkers, not real faces — a stepping stone, not a replacement for FaceForensics++/Celeb-DF evaluation (access forms pending).

## Citation

```bibtex
@software{rehman2026synsync,
  author  = {Habib Ur Rehman},
  title   = {SynSync-Bench: controlled A/V manipulation benchmark with exact ground truth},
  year    = {2026},
  url     = {https://github.com/habib-analyst/synsync-bench}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
