"""CLI: synsync-bench generate | features | evaluate | analyze."""

from __future__ import annotations

import argparse
import json
import os
import sys


def cmd_generate(args) -> None:
    from synsync.manipulate import build_benchmark, build_plan
    plan = build_plan(n_clean=args.clean, n_desync_each=args.desync_each,
                      n_drift_each=args.drift_each,
                      n_reencode_each=args.reencode_each,
                      n_fps=args.fps, n_jitter=args.jitter)
    print(f"plan: {len(plan)} clips")
    mp = build_benchmark(args.out, plan)
    print(f"manifest: {mp}")


def cmd_features(args) -> None:
    from synsync.features import extract_benchmark
    n = extract_benchmark(args.bench)
    print(f"done: {n} new feature rows")


def cmd_evaluate(args) -> None:
    from synsync.fusion import run_all
    res = run_all(os.path.join(args.bench, "features.jsonl"), args.out)
    print(json.dumps(res, indent=1)[:3000])
    print(f"\nfull results: {os.path.join(args.out, 'results.json')}")


def cmd_analyze(args) -> None:
    from synsync.features import extract_all
    feats = extract_all(args.video, args.audio)
    print(json.dumps(feats, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="synsync-bench",
                                description="Controlled A/V manipulation benchmark + fusion detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build the benchmark")
    g.add_argument("--out", required=True)
    g.add_argument("--clean", type=int, default=100)
    g.add_argument("--desync-each", type=int, default=25)
    g.add_argument("--drift-each", type=int, default=20)
    g.add_argument("--reencode-each", type=int, default=40)
    g.add_argument("--fps", type=int, default=40)
    g.add_argument("--jitter", type=int, default=60)
    g.set_defaults(fn=cmd_generate)

    f = sub.add_parser("features", help="extract features for all clips")
    f.add_argument("bench")
    f.set_defaults(fn=cmd_features)

    e = sub.add_parser("evaluate", help="train fusion + run eval protocols")
    e.add_argument("bench")
    e.add_argument("--out", default="results")
    e.set_defaults(fn=cmd_evaluate)

    a = sub.add_parser("analyze", help="feature dump for one clip")
    a.add_argument("video")
    a.add_argument("audio")
    a.set_defaults(fn=cmd_analyze)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
