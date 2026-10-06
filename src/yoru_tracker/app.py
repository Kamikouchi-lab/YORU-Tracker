# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""``yoru-tracker`` / ``python -m yoru_tracker``.

With no arguments the GUI opens.  The subcommands do the same work headless::

    yoru-tracker track VIDEO --model best.pt --out results/
    yoru-tracker track results/VIDEO_detections.csv --config my_tracker.yaml
    yoru-tracker batch videos/ --model best.pt --out results/
    yoru-tracker bench
    yoru-tracker config > tracker.yaml

Imports are kept inside the subcommands, so ``--help`` answers without
loading OpenCV, torch or DearPyGui.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _version() -> str:
    from yoru_tracker import __version__

    return __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yoru-tracker",
        description="YORU Tracker - multi-animal tracking built on YORU's detectors.",
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_gui = sub.add_parser("gui", help="Open the YORU Tracker window (the default).")
    p_gui.add_argument("--exit-after", type=int, default=None, metavar="FRAMES",
                       help=argparse.SUPPRESS)  # smoke tests: render N frames, then quit
    p_gui.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    p_gui.add_argument("--view", default=None, choices=("home", "video", "realtime", "batch"),
                       help="Open this screen instead of the start screen.")
    p_gui.set_defaults(func=_cmd_gui)

    def tracker_args(p):
        p.add_argument("--config", default=None,
                       help="Tracker configuration (YAML or JSON); default settings otherwise.")
        p.add_argument("--mode", default=None,
                       help="Override the tracker mode (lite, baseline, ...).")
        p.add_argument("--include-predicted", action="store_true",
                       help="Also write rows for lost tracks' predicted positions.")

    def detector_args(p, required):
        p.add_argument("--model", required=required, help="Detector model (any YORU backend).")
        p.add_argument("--backend", default="auto", help="Detector backend (default: auto).")
        p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
        p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
        p.add_argument("--exclude-class", type=int, action="append", default=[],
                       metavar="ID", help="Class ID not to track (repeatable).")

    p_track = sub.add_parser(
        "track", help="Track one video, or re-track a saved detections CSV.")
    p_track.add_argument("input", help="A video file, or a detections CSV.")
    p_track.add_argument("--out", default=None, help="Output folder (default: beside the input).")
    p_track.add_argument("--render", action="store_true",
                         help="Also write the video with the tracks drawn on it.")
    detector_args(p_track, required=False)
    tracker_args(p_track)
    p_track.set_defaults(func=_cmd_track)

    p_batch = sub.add_parser("batch", help="Track every video in a folder.")
    p_batch.add_argument("folder")
    p_batch.add_argument("--out", required=True, help="Output folder.")
    p_batch.add_argument("--recursive", action="store_true")
    detector_args(p_batch, required=True)
    tracker_args(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    p_bench = sub.add_parser("bench", help="Score trackers on the synthetic scenarios.")
    p_bench.add_argument("--scenario", action="append", default=None,
                         help="Scenario to run (repeatable; default: all).")
    p_bench.add_argument("--tracker", action="append", default=None,
                         help="Tracker mode to score (repeatable; default: lite and baseline).")
    p_bench.add_argument("--config", default=None,
                         help="Score this configuration as 'custom' as well.")
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--json", default=None, help="Also write the rows as JSON here.")
    p_bench.set_defaults(func=_cmd_bench)

    p_cfg = sub.add_parser("config", help="Print the default tracker configuration (YAML).")
    p_cfg.set_defaults(func=_cmd_config)

    parser.set_defaults(func=_cmd_gui, command="gui", exit_after=None, screenshot=None,
                        view=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from yoru_tracker.logs import log_exception, setup_logging

    setup_logging()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log_exception(f"yoru-tracker {args.command} failed", exc)
        print(f"[yoru-tracker] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _tracker_config(args):
    from dataclasses import replace

    from yoru_tracker.core.config import TrackerConfig

    config = TrackerConfig.load(args.config) if args.config else TrackerConfig()
    if args.mode:
        config = replace(config, mode=args.mode).validate()
    return config


def _detector_config(args):
    from yoru_tracker.runtime.detection import DetectorConfig

    return DetectorConfig(model_path=str(args.model), backend=args.backend,
                          conf_thresh=args.conf, iou_thresh=args.iou,
                          exclude_classes=tuple(args.exclude_class))


def _cmd_gui(args) -> int:
    from yoru_tracker.gui.main_window import run

    return run(exit_after=args.exit_after, screenshot=args.screenshot, view=args.view)


def _cmd_track(args) -> int:
    from yoru_tracker.core.registry import create_tracker
    from yoru_tracker.export.csv_export import write_tracks_csv
    from yoru_tracker.export.metadata import build_metadata, write_metadata
    from yoru_tracker.runtime.frames import track_frames

    config = _tracker_config(args)
    source = Path(args.input)
    out_dir = Path(args.out) if args.out else source.parent

    if source.suffix.lower() == ".csv":
        from yoru_tracker.runtime.detections_file import load_detections

        layout, frames = load_detections(source)
        tracker = create_tracker(config)
        results = track_frames(tracker, frames)
        stem = source.stem.removesuffix("_detections").removesuffix("_detect")
        csv_path = out_dir / f"{stem}_tracks.csv"
        rows = write_tracks_csv(csv_path, results, include_predicted=args.include_predicted)
        ids = sorted({t.track_id for r in results for t in r.tracked})
        write_metadata(out_dir / f"{stem}_tracks.json", build_metadata(
            tracker,
            source={"kind": "detections", "path": str(source), "layout": layout,
                    "frames": len(frames)},
            outputs={"tracks_csv": csv_path.name, "rows": rows,
                     "include_predicted": args.include_predicted},
            summary={"frames": len(results), "track_ids": len(ids)},
        ))
        print(f"{len(results)} frames, {len(ids)} track IDs -> {csv_path}")
        return 0

    if not args.model:
        raise SystemExit("A video needs --model (or give a detections CSV instead).")
    from yoru_tracker.runtime.detection import YoruDetector
    from yoru_tracker.runtime.video import detect_and_track, export_run, output_paths, render_video

    det_config = _detector_config(args)
    detector = YoruDetector.load(det_config)
    last = [-1]

    def progress(run):
        pct = int(100 * run.processed / max(1, run.info.frame_count))
        if pct // 10 > last[0] // 10:
            last[0] = pct
            print(f"  {pct:3d}% ({run.processed}/{run.info.frame_count} frames)", flush=True)

    run = detect_and_track(source, detector, config, detector_settings=det_config.to_dict(),
                           on_frame=progress)
    written = export_run(run, out_dir, include_predicted=args.include_predicted)
    if args.render:
        written["video"] = render_video(run, output_paths(out_dir, source.stem)["video"])
    summary = run.summary()
    print(f"{summary['frames']} frames, {summary['track_ids']} track IDs; "
          f"detector {summary['detector_ms_per_frame']:.1f} ms/frame, "
          f"tracker {summary['tracker_ms_per_frame']:.3f} ms/frame")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_batch(args) -> int:
    from yoru_tracker.runtime.batch import BatchItem, failure_report, find_videos, run_batch
    from yoru_tracker.runtime.detection import YoruDetector

    config = _tracker_config(args)
    det_config = _detector_config(args)
    videos = find_videos(args.folder, recursive=args.recursive)
    if not videos:
        raise SystemExit(f"No videos found in {args.folder}")
    detector = YoruDetector.load(det_config)

    def report(item):
        if item.status in ("done", "failed", "stopped"):
            print(f"[{item.status:>7}] {item.name}  frames={item.frames} ids={item.track_ids}"
                  + (f"  {item.error}" if item.error else ""), flush=True)

    items = run_batch([BatchItem(v) for v in videos], detector, config, args.out,
                      detector_settings=det_config.to_dict(),
                      include_predicted=args.include_predicted, on_update=report)
    print(failure_report(items))
    return 1 if any(i.status == "failed" for i in items) else 0


def _cmd_bench(args) -> int:
    from yoru_tracker.core.config import TrackerConfig
    from yoru_tracker.evaluation.benchmark import format_table, run_benchmark

    modes = args.tracker or ["lite", "baseline"]
    configs = {m: TrackerConfig(mode=m) for m in modes}
    if args.config:
        configs["custom"] = TrackerConfig.load(args.config)
    rows = run_benchmark(configs, scenarios=args.scenario, seed=args.seed)
    print(format_table(rows))
    if args.json:
        Path(args.json).write_text(json.dumps([r.to_dict() for r in rows], indent=2),
                                   encoding="utf-8")
    return 0


def _cmd_config(args) -> int:
    from yoru_tracker.core.config import TrackerConfig

    sys.stdout.write(TrackerConfig().to_yaml())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
