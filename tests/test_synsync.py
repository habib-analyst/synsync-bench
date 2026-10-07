"""Tests: determinism, manifest schema, feature shapes, fusion determinism."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synsync"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synsync import source  # noqa: E402
from synsync.features import FEATURE_NAMES, extract_all, to_vector  # noqa: E402
from synsync.fusion import desync_rule_predict  # noqa: E402
from synsync.manipulate import build_plan  # noqa: E402


def test_contour_deterministic():
    a = source.speech_contour(6.0, seed=42)
    b = source.speech_contour(6.0, seed=42)
    assert np.array_equal(a, b)
    c = source.speech_contour(6.0, seed=43)
    assert not np.array_equal(a, c)


def test_plan_schema_and_counts():
    plan = build_plan(n_clean=4, n_desync_each=1, n_drift_each=1,
                      n_reencode_each=1, n_fps=1, n_jitter=1)
    assert len(plan) == 4 + 8 + 6 + 2 + 1 + 1
    seeds = [p["seed"] for p in plan]
    assert len(set(seeds)) == len(seeds), "seeds must be unique"
    for p in plan:
        assert p["manipulation"] in (
            "clean", "desync", "drift", "reencode", "fps_convert", "lip_jitter")


def test_feature_vector_shape(tmp_path):
    d = str(tmp_path)
    src = source.render_source(d, seed=0)
    feats = extract_all(src["video"], src["audio"])
    v = to_vector(feats)
    assert v.shape == (8,)
    assert list(feats.keys()) == FEATURE_NAMES
    assert np.all(np.isfinite(v))


def test_desync_rule_clean_is_negative(tmp_path):
    d = str(tmp_path)
    src = source.render_source(d, seed=1)
    from synsync.features import extract_all, to_vector
    v = to_vector(extract_all(src["video"], src["audio"])).reshape(1, -1)
    # clean clip: desync rule should not fire (lag < 150 ms)
    assert desync_rule_predict(v)[0] == 0


def test_fusion_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] > 0).astype(int)
    from synsync.fusion import evaluate_in_family
    r1 = evaluate_in_family(X, y)
    r2 = evaluate_in_family(X, y)
    assert r1 == r2
