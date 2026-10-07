"""Feature extraction for the fusion detector.

Two classical signal families, both CPU-only:

1. DESYNC features — windowed normalized cross-correlation between the
   audio-energy envelope and the visual motion-energy envelope
   (same principle as the ``desync-detect`` package, reimplemented here
   self-contained): median |lag|, mean confidence, drift slope, bad-window
   fraction.
2. SPECTRAL features — per-frame compression-artifact statistics sampled
   over the clip: blockiness (8x8 boundary discontinuity), high-frequency
   DCT energy ratio, blur (Laplacian variance), sensor-noise residual.

Output: a fixed 8-dim feature vector per clip.
"""

from __future__ import annotations

import os
import wave

import cv2
import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.stats import theilslopes

SAMPLE_RATE = 16000
DESYNC_TOL_MS = 150.0
MAX_LAG_MS = 2000.0

FEATURE_NAMES = [
    "desync_median_abs_lag_ms",
    "desync_mean_confidence",
    "desync_drift_slope_ms_per_s",
    "desync_bad_frac",
    "spec_blockiness",
    "spec_hf_ratio",
    "spec_blur",
    "spec_noise",
]


# ---------------------------------------------------------------------------
# envelopes
# ---------------------------------------------------------------------------

def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sw, sr, ch = w.getsampwidth(), w.getframerate(), w.getnchannels()
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError("unsupported wav width")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def _audio_envelope(wav_path: str, fps: float, n_frames: int) -> np.ndarray:
    samples, sr = _read_wav_mono(wav_path)
    env = np.zeros(n_frames)
    for i in range(n_frames):
        a = int(round(i / fps * sr))
        b = max(a + 1, int(round((i + 1) / fps * sr)))
        if a >= len(samples):
            break
        seg = samples[a:min(b, len(samples))]
        env[i] = float(np.sqrt(np.mean(seg ** 2)))
    k = np.ones(3) / 3
    return np.convolve(env, k, mode="same")


def _motion_envelope(video_path: str, max_frames: int | None = None):
    """Mouth-openness envelope: mean darkness in the mouth ROI per frame.

    The synthetic talker's mouth is a dark ellipse whose AREA tracks the
    speech amplitude contour, so mean darkness in a fixed mouth ROI is
    proportional to the contour itself (not its derivative, as frame
    differencing would give). Returns (envelope, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    # mouth ROI: center (64, 84), generous box around the ellipse
    x0, x1, y0, y1 = 44, 84, 72, 104
    env, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi = gray[y0:y1, x0:x1]
        env.append(float(np.mean(200.0 - roi)))   # darkness ~ mouth area
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()
    return np.asarray(env), fps


# ---------------------------------------------------------------------------
# desync features
# ---------------------------------------------------------------------------

def _znorm(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s


def _window_lag(a: np.ndarray, m: np.ndarray, fps: float):
    n = len(a)
    lags = correlation_lags(n, len(m), mode="full")
    lim = int(round(MAX_LAG_MS / 1000 * fps))
    sel = np.abs(lags) <= lim
    if not np.any(sel):
        return 0.0, 0.0
    corr = np.abs(correlate(_znorm(a), _znorm(m), mode="full")) / n
    masked = corr.copy()
    masked[~sel] = -1
    pos = int(np.argmax(masked))
    peak = float(masked[pos])
    # confidence: peak prominence above floor
    lo = max(0, pos - 5)
    hi = min(len(masked), pos + 6)
    floor_m = np.ones_like(masked, bool)
    floor_m[lo:hi] = False
    floor_m &= sel
    floor = float(np.mean(masked[floor_m])) if np.any(floor_m) else 0.0
    conf = float(np.clip((peak - floor) / (1 - floor + 1e-9), 0, 1))
    return float(lags[pos]) / fps * 1000.0, conf


def desync_features(video_path: str, audio_path: str) -> dict[str, float]:
    motion, fps = _motion_envelope(video_path)
    audio = _audio_envelope(audio_path, fps, len(motion))
    n = min(len(audio), len(motion))
    audio, motion = audio[:n], motion[:n]
    # 2 s windows / 1 s step: a 6 s clip (150 frames @25 fps) yields 5 windows.
    # (The 5 s windows of the desync-detect triage tool assume longer clips.)
    win = int(2.0 * fps)
    step = int(1.0 * fps)
    lags, confs = [], []
    s = 0
    while s + win <= n:
        lag, conf = _window_lag(audio[s:s + win], motion[s:s + win], fps)
        lags.append(lag)
        confs.append(conf)
        s += step
    lags = np.array(lags)
    confs = np.array(confs)
    if len(lags) < 2:
        return {"desync_median_abs_lag_ms": 0.0, "desync_mean_confidence": 0.0,
                "desync_drift_slope_ms_per_s": 0.0, "desync_bad_frac": 0.0}
    ok = confs >= 0.35
    valid_lags = lags[ok] if np.any(ok) else lags
    centers = np.arange(len(lags)) * (step / fps)
    slope = float(theilslopes(lags, centers)[0]) if len(lags) >= 4 else 0.0
    return {
        "desync_median_abs_lag_ms": float(np.median(np.abs(valid_lags))),
        "desync_mean_confidence": float(np.mean(confs)),
        "desync_drift_slope_ms_per_s": slope,
        "desync_bad_frac": float(np.mean(np.abs(valid_lags) > DESYNC_TOL_MS)),
    }


# ---------------------------------------------------------------------------
# spectral / compression-artifact features
# ---------------------------------------------------------------------------

def _blockiness(gray: np.ndarray) -> float:
    """Mean |diff| across 8x8 block boundaries minus interior diffs."""
    h, w = gray.shape
    dh = np.abs(np.diff(gray.astype(np.float32), axis=1))
    bnd = dh[:, 7::8].mean() if w > 8 else 0.0     # x = 8k-1 -> 8k boundary
    intr = dh[:, 3::8].mean() if w > 8 else 0.0
    dv = np.abs(np.diff(gray.astype(np.float32), axis=0))
    bnd_v = dv[7::8, :].mean() if h > 8 else 0.0
    intr_v = dv[3::8, :].mean() if h > 8 else 0.0
    return float((bnd - intr + bnd_v - intr_v) / 2.0)


def _hf_ratio(gray: np.ndarray) -> float:
    """High-frequency DCT energy / total AC energy over 8x8 blocks."""
    h, w = gray.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    g = gray[:h8, :w8].astype(np.float32) - 128.0
    blocks = g.reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    from scipy.fft import dctn
    dcts = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    ac = dcts.copy()
    ac[:, 0, 0] = 0
    tot = float(np.sum(ac ** 2)) + 1e-9
    yy, xx = np.mgrid[0:8, 0:8]
    hf = float(np.sum(ac[:, (yy + xx) >= 8] ** 2))
    return hf / tot


def spectral_features(video_path: str, n_sample: int = 12) -> dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 1), n_sample).astype(int)
    vals = {"block": [], "hf": [], "blur": [], "noise": []}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals["block"].append(_blockiness(gray))
        vals["hf"].append(_hf_ratio(gray))
        vals["blur"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        med = cv2.medianBlur(gray, 3)
        vals["noise"].append(float(np.median(np.abs(
            gray.astype(np.float32) - med.astype(np.float32)))))
    cap.release()
    return {
        "spec_blockiness": float(np.mean(vals["block"])) if vals["block"] else 0.0,
        "spec_hf_ratio": float(np.mean(vals["hf"])) if vals["hf"] else 0.0,
        "spec_blur": float(np.mean(vals["blur"])) if vals["blur"] else 0.0,
        "spec_noise": float(np.mean(vals["noise"])) if vals["noise"] else 0.0,
    }


def extract_all(video_path: str, audio_path: str) -> dict[str, float]:
    feats = desync_features(video_path, audio_path)
    feats.update(spectral_features(video_path))
    return feats


def to_vector(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)


def extract_benchmark(bench_dir: str) -> int:
    """Extract features for every clip in manifest.jsonl -> features.jsonl.

    Resume-safe: skips clip_ids already present. Returns # new rows.
    """
    import json
    manifest = os.path.join(bench_dir, "manifest.jsonl")
    out_path = os.path.join(bench_dir, "features.jsonl")
    clips_dir = os.path.join(bench_dir, "clips")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["clip_id"])

    n = 0
    with open(manifest) as f, open(out_path, "a") as out:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["clip_id"]
            if cid in done:
                continue
            video = os.path.join(clips_dir, f"{cid}.mp4")
            audio = os.path.join(clips_dir, f"{cid}.wav")
            feats = extract_all(video, audio)
            out.write(json.dumps({
                "clip_id": cid,
                "manipulation": rec["manipulation"],
                "label": rec["label"],
                "offset_ms": rec.get("offset_ms", 0.0),
                "features": feats,
            }) + "\n")
            out.flush()
            n += 1
    return n
