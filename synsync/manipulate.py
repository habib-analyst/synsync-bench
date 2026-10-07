"""Manipulations with EXACT ground truth, applied to rendered source pairs.

Each function takes a clean (video, audio) source pair and writes a muxed
mp4 clip plus a ground-truth record. Manipulations:

- clean        : pristine remux (label 0 = real)
- desync       : constant A/V offset ±40..±320 ms (numpy shift, sample-exact)
- drift        : linear clock drift via audio resampling (exact rate)
- reencode     : H.264 re-encode at low bitrate (codec artifacts, sync intact)
- fps_convert  : 25 -> 30 -> 25 fps round-trip (judder, sync intact)
- lip_jitter   : mouth frozen at wrong phase in 3 segments (visual-only)

Ground truth is exact BY CONSTRUCTION (no estimation involved).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
from scipy.signal import resample

from .source import DURATION, FPS, SAMPLE_RATE, render_audio, render_video_jittered, speech_contour

MANIPULATIONS = ["clean", "desync", "drift", "reencode", "fps_convert", "lip_jitter"]


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")


def _read_wav_mono(path: str) -> np.ndarray:
    import wave
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sw = w.getsampwidth()
    if sw == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    raise ValueError("expected 16-bit wav")


def _write_wav_mono(path: str, sig: np.ndarray) -> None:
    import wave
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _mux(video: str, audio: str, out: str, vcodec_args: list[str] | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio,
           "-c:a", "aac", "-b:a", "64k", "-shortest"]
    cmd += vcodec_args if vcodec_args else ["-c:v", "libx264", "-crf", "18",
                                            "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    cmd += [out]
    _run(cmd)


def _extract_audio(mp4: str, wav_out: str) -> None:
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
          "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1", wav_out])


def apply_manipulation(src_video: str, src_audio: str, seed: int,
                       manipulation: str, params: dict, work_dir: str,
                       clip_id: str) -> dict:
    """Apply one manipulation; return the ground-truth record (without paths)."""
    os.makedirs(work_dir, exist_ok=True)
    out_mp4 = os.path.join(work_dir, f"{clip_id}.mp4")
    gt: dict = {"clip_id": clip_id, "seed": seed, "manipulation": manipulation,
                "params": params, "label": 0, "offset_ms": 0.0,
                "drift_ms_per_s": 0.0}

    if manipulation == "clean":
        _mux(src_video, src_audio, out_mp4)

    elif manipulation == "desync":
        offset_ms = float(params["offset_ms"])
        sig = _read_wav_mono(src_audio)
        shift = int(round(offset_ms / 1000.0 * SAMPLE_RATE))
        n = len(sig)
        out = np.zeros(n)
        if shift >= 0:      # audio delayed -> audio lags video
            out[shift:] = sig[:n - shift]
        else:
            out[:n + shift] = sig[-shift:]
        tmp = os.path.join(work_dir, f"{clip_id}.wav")
        _write_wav_mono(tmp, out)
        _mux(src_video, tmp, out_mp4)
        gt["offset_ms"] = offset_ms
        gt["label"] = 1

    elif manipulation == "drift":
        rate = float(params["drift_ms_per_s"])      # lag growth, ms per s
        sig = _read_wav_mono(src_audio)
        # resample so audio plays slower (rate>0 => lag grows): new_len = n/(1-rate/1000)
        factor = 1.0 - rate / 1000.0
        new_len = int(round(len(sig) / factor))
        drifted = resample(sig, new_len)[:len(sig)]
        if len(drifted) < len(sig):                 # pad if short
            drifted = np.pad(drifted, (0, len(sig) - len(drifted)))
        tmp = os.path.join(work_dir, f"{clip_id}.wav")
        _write_wav_mono(tmp, drifted)
        _mux(src_video, tmp, out_mp4)
        gt["drift_ms_per_s"] = rate
        gt["label"] = 1

    elif manipulation == "reencode":
        bitrate = params["bitrate"]
        tmp = os.path.join(work_dir, f"{clip_id}_re.mp4")
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_video,
              "-c:v", "libx264", "-b:v", bitrate, "-preset", "veryfast",
              "-pix_fmt", "yuv420p", tmp])
        _mux(tmp, src_audio, out_mp4)
        gt["label"] = 1

    elif manipulation == "fps_convert":
        t1 = os.path.join(work_dir, f"{clip_id}_30.mp4")
        t2 = os.path.join(work_dir, f"{clip_id}_back.mp4")
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_video,
              "-r", "30", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
              "-pix_fmt", "yuv420p", t1])
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", t1,
              "-r", "25", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
              "-pix_fmt", "yuv420p", t2])
        _mux(t2, src_audio, out_mp4)
        gt["label"] = 1

    elif manipulation == "lip_jitter":
        contour = speech_contour(DURATION, seed)
        jvid = os.path.join(work_dir, f"{clip_id}_jit.mp4")
        segments = render_video_jittered(jvid, contour, seed)
        _mux(jvid, src_audio, out_mp4)
        gt["params"] = {"segments": segments}
        gt["label"] = 1

    else:
        raise ValueError(f"unknown manipulation: {manipulation}")

    wav_out = os.path.join(work_dir, f"{clip_id}.wav")
    _extract_audio(out_mp4, wav_out)
    gt["video"] = out_mp4
    gt["audio"] = wav_out
    return gt


# ---------------------------------------------------------------------------
# benchmark assembly
# ---------------------------------------------------------------------------

DESYNC_OFFSETS = [-320, -160, -80, -40, 40, 80, 160, 320]     # ms
DRIFT_RATES = [-60, -30, -15, 15, 30, 60]                      # ms/s
REENCODE_BITRATES = ["100k", "300k"]


def build_plan(n_clean=100, n_desync_each=25, n_drift_each=20,
               n_reencode_each=40, n_fps=40, n_jitter=60) -> list[dict]:
    """Deterministic clip plan: list of {seed, manipulation, params}."""
    plan = []
    seed = 0
    for _ in range(n_clean):
        plan.append({"seed": seed, "manipulation": "clean", "params": {}})
        seed += 1
    for off in DESYNC_OFFSETS:
        for _ in range(n_desync_each):
            plan.append({"seed": seed, "manipulation": "desync",
                         "params": {"offset_ms": off}})
            seed += 1
    for rate in DRIFT_RATES:
        for _ in range(n_drift_each):
            plan.append({"seed": seed, "manipulation": "drift",
                         "params": {"drift_ms_per_s": rate}})
            seed += 1
    for br in REENCODE_BITRATES:
        for _ in range(n_reencode_each):
            plan.append({"seed": seed, "manipulation": "reencode",
                         "params": {"bitrate": br}})
            seed += 1
    for _ in range(n_fps):
        plan.append({"seed": seed, "manipulation": "fps_convert", "params": {}})
        seed += 1
    for _ in range(n_jitter):
        plan.append({"seed": seed, "manipulation": "lip_jitter", "params": {}})
        seed += 1
    return plan


def build_benchmark(out_dir: str, plan: list[dict] | None = None,
                    cache_sources: bool = True) -> str:
    """Build the full benchmark. Returns path to manifest.jsonl.

    Sources are rendered once per seed and cached under ``out_dir/sources``.
    """
    from .source import render_source
    plan = plan if plan is not None else build_plan()
    clips_dir = os.path.join(out_dir, "clips")
    src_dir = os.path.join(out_dir, "sources")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    # resume support: skip clips already in the manifest
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["clip_id"])

    src_cache: dict[int, dict] = {}
    with open(manifest_path, "a") as mf:
        for i, spec in enumerate(plan):
            clip_id = f"clip_{i:05d}"
            if clip_id in done:
                continue
            seed = spec["seed"]
            if seed not in src_cache:
                v_flat = os.path.join(src_dir, f"src_{seed:04d}.mp4")
                a_flat = os.path.join(src_dir, f"src_{seed:04d}.wav")
                if cache_sources and os.path.exists(v_flat) and os.path.exists(a_flat):
                    src_cache[seed] = {"video": v_flat, "audio": a_flat}
                else:
                    # render_source writes src_<seed>.mp4/.wav directly in src_dir
                    src_cache[seed] = render_source(src_dir, seed)
            src = src_cache[seed]
            gt = apply_manipulation(src["video"], src["audio"], seed,
                                    spec["manipulation"], spec["params"],
                                    clips_dir, clip_id)
            rec = {k: v for k, v in gt.items() if k in (
                "clip_id", "seed", "manipulation", "params", "label",
                "offset_ms", "drift_ms_per_s")}
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
    return manifest_path
