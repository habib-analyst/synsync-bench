"""Fusion detector: dataset assembly, baselines, logistic fusion, evaluation.

Detection task: label 1 = manipulated clip, label 0 = pristine.

Models:
- desync_rule   : manipulated iff median |lag| > 150 ms (single-signal baseline)
- spectral_lr   : logistic regression on the 4 spectral features (baseline)
- fusion_lr     : logistic regression on all 8 features (the fusion claim)

Evaluation protocols:
- in_family : stratified 5-fold CV over all clips
- cross_family: train on {clean, desync, reencode} families, test on
  {drift, fps_convert, lip_jitter} (+ held-out clean) and vice versa —
  the honest analog of cross-dataset generalization.
- offset_curve: detection rate vs |offset| for the desync family.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, to_vector

SPEC_IDX = [4, 5, 6, 7]
FAMILY_OF = {
    "clean": "clean", "desync": "desync", "drift": "drift",
    "reencode": "reencode", "fps_convert": "fps_convert",
    "lip_jitter": "lip_jitter",
}


def load_dataset(features_path: str):
    """features.jsonl rows: {clip_id, manipulation, label, features{...}}."""
    rows = []
    with open(features_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    X = np.stack([to_vector(r["features"]) for r in rows])
    y = np.array([r["label"] for r in rows], dtype=int)
    fam = np.array([FAMILY_OF[r["manipulation"]] for r in rows])
    off = np.array([abs(float(r.get("offset_ms", 0.0))) for r in rows])
    return X, y, fam, off, rows


def desync_rule_predict(X: np.ndarray) -> np.ndarray:
    return (X[:, 0] > 150.0).astype(int)


def _fit_lr(Xtr: np.ndarray, ytr: np.ndarray, cols) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler().fit(Xtr[:, cols])
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    clf.fit(sc.transform(Xtr[:, cols]), ytr)
    return sc, clf


def _scores(sc, clf, Xte: np.ndarray, cols) -> np.ndarray:
    return clf.predict_proba(sc.transform(Xte[:, cols]))[:, 1]


def evaluate_in_family(X, y, seed=0) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {"desync_rule": [], "spectral_lr": [], "fusion_lr": []}
    for tr, te in skf.split(X, y):
        out["desync_rule"].append({
            "acc": float(accuracy_score(y[te], desync_rule_predict(X[te]))),
            "auc": float(roc_auc_score(y[te], desync_rule_predict(X[te])))})
        sc, clf = _fit_lr(X[tr], y[tr], SPEC_IDX)
        p = _scores(sc, clf, X[te], SPEC_IDX)
        out["spectral_lr"].append({"acc": float(accuracy_score(y[te], p > 0.5)),
                                   "auc": float(roc_auc_score(y[te], p))})
        sc, clf = _fit_lr(X[tr], y[tr], list(range(X.shape[1])))
        p = _scores(sc, clf, X[te], list(range(X.shape[1])))
        out["fusion_lr"].append({"acc": float(accuracy_score(y[te], p > 0.5)),
                                 "auc": float(roc_auc_score(y[te], p))})
    return {k: {"acc_mean": float(np.mean([r["acc"] for r in v])),
                "acc_std": float(np.std([r["acc"] for r in v])),
                "auc_mean": float(np.mean([r["auc"] for r in v])),
                "auc_std": float(np.std([r["auc"] for r in v]))}
            for k, v in out.items()}


def evaluate_cross_family(X, y, fam) -> dict:
    """Train on family set A, test on disjoint family set B (+ clean held out).

    Split 1: train {desync, reencode} -> test {drift, fps_convert, lip_jitter}
    Split 2: train {drift, fps_convert, lip_jitter} -> test {desync, reencode}
    Clean clips are split 50/50 between train and test in both splits.
    """
    rng = np.random.default_rng(0)
    clean_idx = np.flatnonzero(fam == "clean")
    rng.shuffle(clean_idx)
    half = len(clean_idx) // 2
    clean_tr, clean_te = clean_idx[:half], clean_idx[half:]
    splits = {
        "A->B": (["desync", "reencode"], ["drift", "fps_convert", "lip_jitter"]),
        "B->A": (["drift", "fps_convert", "lip_jitter"], ["desync", "reencode"]),
    }
    results = {}
    for name, (train_fams, test_fams) in splits.items():
        tr = np.concatenate([clean_tr] + [np.flatnonzero(fam == f) for f in train_fams])
        te = np.concatenate([clean_te] + [np.flatnonzero(fam == f) for f in test_fams])
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        r_rule = desync_rule_predict(Xte)
        sc, clf = _fit_lr(Xtr, ytr, SPEC_IDX)
        p_spec = _scores(sc, clf, Xte, SPEC_IDX)
        sc2, clf2 = _fit_lr(Xtr, ytr, list(range(X.shape[1])))
        p_fus = _scores(sc2, clf2, Xte, list(range(X.shape[1])))
        results[name] = {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "desync_rule": {"acc": float(accuracy_score(yte, r_rule)),
                            "auc": float(roc_auc_score(yte, r_rule))},
            "spectral_lr": {"acc": float(accuracy_score(yte, p_spec > 0.5)),
                            "auc": float(roc_auc_score(yte, p_spec))},
            "fusion_lr": {"acc": float(accuracy_score(yte, p_fus > 0.5)),
                          "auc": float(roc_auc_score(yte, p_fus))},
        }
    return results


def offset_sensitivity(X, y, fam, off, X_all_cols=True) -> list[dict]:
    """Detection rate of the fusion model vs |offset| on the desync family.

    Fusion model trained on all NON-desync clips (so the curve is honest).
    """
    curve = []
    mask_tr = fam != "desync"
    sc, clf = _fit_lr(X[mask_tr], y[mask_tr], list(range(X.shape[1])))
    for o in sorted(set(off[fam == "desync"].tolist())):
        m = (fam == "desync") & (off == o)
        p = _scores(sc, clf, X[m], list(range(X.shape[1])))
        curve.append({"offset_ms": float(o), "n": int(m.sum()),
                      "detect_rate": float(np.mean(p > 0.5))})
    return curve


def run_all(features_path: str, out_dir: str) -> dict:
    X, y, fam, off, _ = load_dataset(features_path)
    results = {
        "n_clips": int(len(y)),
        "n_manipulated": int(y.sum()),
        "in_family_cv": evaluate_in_family(X, y),
        "cross_family": evaluate_cross_family(X, y, fam),
        "offset_curve": offset_sensitivity(X, y, fam, off),
        "feature_names": FEATURE_NAMES,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
