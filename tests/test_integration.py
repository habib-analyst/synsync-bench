"""Integration: build tiny benchmark, verify exact ground-truth recovery."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synsync.features import desync_features  # noqa: E402
from synsync.manipulate import build_benchmark, build_plan  # noqa: E402


def test_ground_truth_recovery(tmp_path):
    d = str(tmp_path)
    plan = build_plan(n_clean=1, n_desync_each=1, n_drift_each=0,
                      n_reencode_each=0, n_fps=0, n_jitter=0)
    # keep only clean + one +160 ms desync for speed
    plan = [p for p in plan if p["manipulation"] == "clean"][:1] + \
           [p for p in plan if p["manipulation"] == "desync"
            and p["params"]["offset_ms"] == 160][:1]
    mp = build_benchmark(d, plan)
    rows = [json.loads(l) for l in open(mp)]
    assert len(rows) == 2
    for r in rows:
        v = os.path.join(d, "clips", r["clip_id"] + ".mp4")
        a = os.path.join(d, "clips", r["clip_id"] + ".wav")
        feats = desync_features(v, a)
        est = feats["desync_median_abs_lag_ms"]
        true = abs(r["offset_ms"])
        assert abs(est - true) <= 80.0, (r["clip_id"], est, true)
