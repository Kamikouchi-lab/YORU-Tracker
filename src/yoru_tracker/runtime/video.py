# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Tracking a video file, offline.

The detector runs once per frame and its output is kept.  Tracking runs over
the kept detections -- during the first pass as each frame arrives, and again
from the start whenever the settings change -- so tuning the tracker on a
long video takes seconds, not another detector pass.  Results always come
from one sequential pass over every frame; seeking in the GUI only chooses
what is shown, never what is tracked.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.registry import create_tracker
from yoru_tracker.core.tracker_base import TrackerBase
from yoru_tracker.core.types import TrackingResult
from yoru_tracker.drawing.overlays import OverlayOptions, draw_tracking, trails_from_results
from yoru_tracker.export.csv_export import write_tracks_csv
from yoru_tracker.export.metadata import build_metadata, write_metadata
from yoru_tracker.runtime.detection import Detector
from yoru_tracker.runtime.detections_file import write_detections_csv
from yoru_tracker.runtime.frames import FrameDetections, track_frames
from yoru_tracker.runtime.sources import VideoFileSource, VideoInfo

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    frames: int = 0
    detector_seconds: float = 0.0
    tracker_seconds: float = 0.0

    @property
    def detector_ms(self) -> float:
        return 1000.0 * self.detector_seconds / self.frames if self.frames else 0.0

    @property
    def tracker_ms(self) -> float:
        return 1000.0 * self.tracker_seconds / self.frames if self.frames else 0.0


@dataclass
class VideoTracking:
    """Detections and tracking results for one video, frame by frame.

    ``frames[i]`` and ``results[i]`` belong to video frame ``i``.  Both lists
    only grow while a pass is running, so a reader on another thread may use
    any index below their current length.

    *frame_ms* is each frame's presentation time as the file stamps it,
    recorded by the sequential pass, so that a seek can be checked against
    it (see :meth:`VideoFileSource.frame_at`); ``None`` when the file's
    times are not usable (missing, or not increasing).
    """

    info: VideoInfo
    tracker_config: TrackerConfig
    detector_settings: Optional[dict] = None
    frames: List[FrameDetections] = field(default_factory=list)
    results: List[TrackingResult] = field(default_factory=list)
    stats: RunStats = field(default_factory=RunStats)
    complete: bool = False
    tracker: Optional[TrackerBase] = None
    frame_ms: Optional[List[float]] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return min(len(self.frames), len(self.results))

    def track_ids(self) -> List[int]:
        return sorted({t.track_id for r in self.results for t in r.tracked})

    def shown_index(self, index: int) -> Optional[int]:
        """Which result to show on video frame *index*.

        Its own -- or, on a frame the detector never ran on, the latest one
        before it, as a live display was showing then.  ``None`` before the
        first observed frame.
        """
        for i in range(min(index, self.processed - 1), -1, -1):
            if self.frames[i].observed:
                return i
        return None

    def summary(self) -> Dict:
        return {
            "frames": self.processed,
            "complete": self.complete,
            "track_ids": len(self.track_ids()),
            "detector_ms_per_frame": round(self.stats.detector_ms, 3),
            "tracker_ms_per_frame": round(self.stats.tracker_ms, 4),
        }


def detect_and_track(
    video_path,
    detector: Detector,
    tracker_config: TrackerConfig,
    *,
    v_flip: bool = False,
    h_flip: bool = False,
    detector_settings: Optional[dict] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_frame: Optional[Callable[[VideoTracking], None]] = None,
    into: Optional[VideoTracking] = None,
) -> VideoTracking:
    """Run the detector and the tracker over every frame of *video_path*.

    *into* lets a caller (the GUI) hold the :class:`VideoTracking` before the
    pass starts and watch it fill.  Stopping early leaves a valid, shorter
    result with ``complete`` false.
    """
    tracker = create_tracker(tracker_config)
    with VideoFileSource(video_path, v_flip=v_flip, h_flip=h_flip) as source:
        run = into or VideoTracking(source.info, tracker_config)
        run.info = source.info
        run.tracker_config = tracker_config
        run.detector_settings = detector_settings
        run.tracker = tracker
        run.frame_ms = []
        fps = source.info.fps
        while True:
            if should_stop is not None and should_stop():
                break
            index, image = source.read()
            if image is None:
                run.complete = True
                break
            ms = source.position_ms()
            if run.frame_ms is not None:
                if ms is None or (run.frame_ms and ms <= run.frame_ms[-1]):
                    run.frame_ms = None          # no times to check seeks against
                else:
                    run.frame_ms.append(ms)
            t0 = time.perf_counter()
            detections = tuple(detector.detect(image))
            t1 = time.perf_counter()
            frame = FrameDetections(index, index / fps, detections)
            result = tracker.update(frame.detections, frame.frame_id, frame.timestamp, image)
            t2 = time.perf_counter()
            run.stats.frames += 1
            run.stats.detector_seconds += t1 - t0
            run.stats.tracker_seconds += t2 - t1
            run.frames.append(frame)
            run.results.append(result)
            if on_frame is not None:
                on_frame(run)
    logger.info("Tracked %s: %s", Path(str(video_path)).name, run.summary())
    return run


def retrack(run: VideoTracking, tracker_config: TrackerConfig) -> VideoTracking:
    """Track *run*'s stored detections again with *tracker_config*.

    Returns a new :class:`VideoTracking` sharing the detections; the
    detector is not run.
    """
    tracker = create_tracker(tracker_config)
    t0 = time.perf_counter()
    results = track_frames(tracker, run.frames)
    elapsed = time.perf_counter() - t0
    stats = RunStats(len(results), run.stats.detector_seconds, elapsed)
    return VideoTracking(
        info=run.info,
        tracker_config=tracker_config,
        detector_settings=run.detector_settings,
        frames=run.frames,
        results=results,
        stats=stats,
        complete=run.complete,
        tracker=tracker,
        frame_ms=run.frame_ms,
    )


def output_paths(out_dir, stem: str) -> Dict[str, Path]:
    out = Path(out_dir)
    return {
        "tracks_csv": out / f"{stem}_tracks.csv",
        "metadata": out / f"{stem}_tracks.json",
        "detections_csv": out / f"{stem}_detections.csv",
        "video": out / f"{stem}_tracked.mp4",
    }


def export_run(run: VideoTracking, out_dir, *, include_predicted: bool = False,
               save_detections: bool = True, stem: Optional[str] = None,
               source: Optional[dict] = None) -> Dict[str, Path]:
    """Write the tracks CSV, its metadata and (optionally) the detections."""
    stem = stem or Path(run.info.path).stem
    paths = output_paths(out_dir, stem)
    written = {}
    rows = write_tracks_csv(paths["tracks_csv"], run.results[:run.processed],
                            include_predicted=include_predicted)
    written["tracks_csv"] = paths["tracks_csv"]
    if save_detections:
        write_detections_csv(paths["detections_csv"], run.frames[:run.processed])
        written["detections_csv"] = paths["detections_csv"]
    tracker = run.tracker or create_tracker(run.tracker_config)
    metadata = build_metadata(
        tracker,
        detector=run.detector_settings,
        source=source or {
            "kind": "video",
            "path": str(run.info.path),
            "fps": run.info.fps,
            "frame_count": run.info.frame_count,
            "width": run.info.width,
            "height": run.info.height,
        },
        outputs={k: str(v.name) for k, v in written.items()} | {
            "include_predicted": include_predicted, "rows": rows,
        },
        summary=run.summary(),
    )
    written["metadata"] = write_metadata(paths["metadata"], metadata)
    return written


def render_video(run: VideoTracking, out_path, options: OverlayOptions = OverlayOptions(),
                 *, v_flip: bool = False, h_flip: bool = False,
                 should_stop: Optional[Callable[[], bool]] = None) -> Path:
    """Write the video with the tracking overlay drawn on every frame."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = run.info
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
                             (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")
    try:
        with VideoFileSource(info.path, v_flip=v_flip, h_flip=h_flip) as source:
            shown = None  # run.shown_index(i), kept up as we go
            for i in range(run.processed):
                if should_stop is not None and should_stop():
                    break
                _, image = source.read()
                if image is None:
                    break
                if run.frames[i].observed:
                    shown = i
                if shown is not None:
                    trails = trails_from_results(run.results, shown, options.trail_length)
                    draw_tracking(image, run.results[shown], options, trails=trails,
                                  detections=run.frames[shown].detections)
                writer.write(image)
    finally:
        writer.release()
    return out_path


class VideoJob:
    """:func:`detect_and_track` on a worker thread, for the GUI."""

    def __init__(self, video_path, detector_factory: Callable[[], Detector],
                 tracker_config: TrackerConfig, *, v_flip=False, h_flip=False,
                 detector_settings: Optional[dict] = None):
        self.video_path = str(video_path)
        self._detector_factory = detector_factory
        self.tracker_config = tracker_config
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.detector_settings = detector_settings
        with VideoFileSource(self.video_path) as probe:
            info = probe.info
        self.run = VideoTracking(info, tracker_config, detector_settings)
        self.error: Optional[BaseException] = None
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._work, name="yoru-tracker-video",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _work(self) -> None:
        try:
            self.phase = "loading model"
            detector = self._detector_factory()
            self.phase = "tracking"
            detect_and_track(
                self.video_path, detector, self.tracker_config,
                v_flip=self.v_flip, h_flip=self.h_flip,
                detector_settings=self.detector_settings,
                should_stop=self._stop.is_set, into=self.run,
            )
            self.phase = "done" if self.run.complete else "stopped"
        except BaseException as exc:  # reported to the GUI, which logs it
            self.error = exc
            self.phase = "failed"
