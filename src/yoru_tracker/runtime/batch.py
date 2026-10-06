# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Many videos, one detector, one tracker configuration.

The model is loaded once.  Each video gets a fresh tracker, so IDs never
carry over from one file to the next, and its own outputs (see
:func:`yoru_tracker.runtime.video.export_run`).  A video that fails is
recorded as failed with its error and the batch moves on; nothing about a
failure is swallowed -- :func:`failure_report` lists every one.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.runtime.detection import Detector
from yoru_tracker.runtime.sources import VIDEO_EXTENSIONS
from yoru_tracker.runtime.video import detect_and_track, export_run

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    path: str
    status: str = "pending"  # pending, running, done, stopped, failed
    frames: int = 0
    total_frames: int = 0
    track_ids: int = 0
    error: str = ""
    detail: str = ""
    outputs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name


def find_videos(folder, extensions=VIDEO_EXTENSIONS, recursive: bool = False) -> List[str]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    pattern = "**/*" if recursive else "*"
    exts = {e.lower() for e in extensions}
    return sorted(str(p) for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in exts)


def run_batch(
    items: Iterable[BatchItem],
    detector: Detector,
    tracker_config: TrackerConfig,
    out_dir,
    *,
    detector_settings: Optional[dict] = None,
    include_predicted: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
    on_update: Optional[Callable[[BatchItem], None]] = None,
) -> List[BatchItem]:
    items = list(items)
    tracker_config.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        if should_stop is not None and should_stop():
            break
        item.status = "running"
        if on_update:
            on_update(item)

        def progress(run, item=item):
            item.frames = run.processed
            item.total_frames = run.info.frame_count
            if on_update:
                on_update(item)

        try:
            run = detect_and_track(
                item.path, detector, tracker_config,
                detector_settings=detector_settings,
                should_stop=should_stop, on_frame=progress,
            )
            item.outputs = {k: str(v) for k, v in export_run(
                run, out_dir, include_predicted=include_predicted).items()}
            item.frames = run.processed
            item.track_ids = len(run.track_ids())
            item.status = "done" if run.complete else "stopped"
        except Exception as exc:
            item.status = "failed"
            item.error = f"{type(exc).__name__}: {exc}"
            item.detail = traceback.format_exc()
            logger.error("Batch: %s failed: %s", item.path, item.error, exc_info=exc)
        if on_update:
            on_update(item)
    return items


def failure_report(items: Iterable[BatchItem]) -> str:
    failed = [i for i in items if i.status == "failed"]
    if not failed:
        return "No failures."
    lines = [f"{len(failed)} file(s) failed:"]
    for item in failed:
        lines.append(f"- {item.path}: {item.error}")
    return "\n".join(lines)


class BatchJob:
    """:func:`run_batch` on a worker thread, for the GUI."""

    def __init__(self, paths: Iterable[str], detector_factory: Callable[[], Detector],
                 tracker_config: TrackerConfig, out_dir, *,
                 detector_settings: Optional[dict] = None,
                 include_predicted: bool = False):
        self.items = [BatchItem(str(p)) for p in paths]
        self._detector_factory = detector_factory
        self.tracker_config = tracker_config
        self.out_dir = out_dir
        self.detector_settings = detector_settings
        self.include_predicted = include_predicted
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
        self._thread = threading.Thread(target=self._work, name="yoru-tracker-batch",
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
            self.phase = "running"
            run_batch(self.items, detector, self.tracker_config, self.out_dir,
                      detector_settings=self.detector_settings,
                      include_predicted=self.include_predicted,
                      should_stop=self._stop.is_set)
            self.phase = "stopped" if self._stop.is_set() else "done"
        except BaseException as exc:
            self.error = exc
            self.phase = "failed"
