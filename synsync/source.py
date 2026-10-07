"""Synthetic talking-head A/V source renderer (fully offline, deterministic).

Renders a simple animated "talker": an ellipse head with a mouth whose
opening follows a speech-like amplitude contour, paired with audio that is
a harmonic complex amplitude-modulated by the SAME contour. The result has
strongly correlated audio/visual energy envelopes — like real speech —
which is exactly what the desync detector keys on.

Determinism: every stochastic choice derives from ``np.random.default_rng(seed)``.
"""

from __future__ import annotations

import math
import os
import wave

import cv2
import numpy as np

SAMPLE_RATE = 16000
FPS = 25.0
SIZE = 128                 # frame is SIZE x SIZE
DURATION = 6.0             # seconds per clip
F0 = 120.0                # voice fundamental (Hz)


def speech_contour(duration: float, seed: int) -> np.ndarray:
    """Speech-like amplitude contour at video frame rate.

    Syllables: random onsets every 120-280 ms, each a raised-cosine burst
    120-250 ms long. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * FPS))
    contour = np.zeros(n)
    t = 0.25
    while t < duration - 0.3:
        syl_dur = float(rng.uniform(0.12, 0.25))
        amp = float(rng.uniform(0.6, 1.0))
        i0 = int(t * FPS)
        i1 = min(n, int((t + syl_dur) * FPS))
        m = i1 - i0
        if m > 1:
            burst = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / (m - 1))
            contour[i0:i1] = np.maximum(contour[i0:i1], amp * burst)
        t += float(rng.uniform(0.12, 0.28))
    return contour


def _render_frame(contour_val: float) -> np.ndarray:
    """One 128x128 frame: head ellipse + eyes + mouth opened by contour_val."""
    img = np.full((SIZE, SIZE, 3), 90, dtype=np.uint8)          # background
    cx, cy = SIZE // 2, SIZE // 2
    cv2.ellipse(img, (cx, cy), (38, 48), 0, 0, 360, (210, 180, 150), -1)  # head
    cv2.ellipse(img, (cx - 15, cy - 12), (6, 8), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(img, (cx + 15, cy - 12), (6, 8), 0, 0, 360, (30, 30, 30), -1)
    mouth_h = int(2 + 14 * contour_val)                          # mouth opens
    cv2.ellipse(img, (cx, cy + 20), (12, mouth_h), 0, 0, 360, (40, 20, 20), -1)
    return img


def render_video(path: str, contour: np.ndarray, fps: float = FPS) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (SIZE, SIZE))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    for c in contour:
        vw.write(_render_frame(float(c)))
    vw.release()


def render_video_jittered(path: str, contour: np.ndarray, seed: int,
                          fps: float = FPS) -> list[dict]:
    """Render with lip-sync jitter: 3 segments where the mouth is frozen at a
    wrong phase (shifted contour). Returns the jitter segment specs."""
    rng = np.random.default_rng(seed + 999)
    n = len(contour)
    jittered = contour.copy()
    segments = []
    for _ in range(3):
        s = int(rng.integers(0, n - 25))
        e = s + int(rng.integers(10, 25))
        shift = int(rng.choice([-1, 1]) * rng.integers(3, 8))   # ±120..320 ms
        src = np.clip(np.arange(s, e) + shift, 0, n - 1)
        jittered[s:e] = contour[src]
        segments.append({"start_s": round(s / fps, 3), "end_s": round(e / fps, 3),
                         "shift_ms": int(shift / fps * 1000)})
    render_video(path, jittered, fps)
    return segments


def render_audio(path: str, contour: np.ndarray, sample_rate: int = SAMPLE_RATE,
                 fps: float = FPS) -> None:
    """Harmonic complex (f0 + 4 harmonics) AM-modulated by the contour."""
    n = int(DURATION * sample_rate)
    t = np.arange(n) / sample_rate
    # upsample contour to audio rate
    frame_idx = np.clip((t * fps).astype(int), 0, len(contour) - 1)
    amp = contour[frame_idx]
    sig = np.zeros(n)
    for h in range(1, 6):
        sig += (0.5 / h) * np.sin(2 * math.pi * F0 * h * t)
    sig *= (0.15 + 0.85 * amp)          # AM: quiet between syllables
    sig *= 0.5
    pcm = np.clip(sig, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def render_source(out_dir: str, seed: int) -> dict[str, str]:
    """Render one clean source pair. Returns {video, audio} paths."""
    os.makedirs(out_dir, exist_ok=True)
    contour = speech_contour(DURATION, seed)
    v_path = os.path.join(out_dir, f"src_{seed:04d}.mp4")
    a_path = os.path.join(out_dir, f"src_{seed:04d}.wav")
    render_video(v_path, contour)
    render_audio(a_path, contour)
    return {"video": v_path, "audio": a_path, "seed": seed}
