# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Live tracking: camera -> detector -> tracker -> display.

Two threads in one process.  The capture thread reads frames as fast as the
source delivers them and keeps only the newest; the processing thread takes
the newest frame it has not seen, detects, tracks and publishes a
:class:`TrackingSnapshot`.  When the detector is slower than the camera the
frames in between are dropped (and counted) rather than queued, so the
display never falls further and further behind.

Detector and tracker share the processing thread on purpose: a Lite update
takes well under a millisecond, and a process boundary would cost more than
that in serialisation.  The stages stay separate in the code, so moving
tracking to its own thread or process later is a change here only.

Every snapshot carries its frame's capture, detection and tracking times; the
GUI adds when it was drawn.  Their differences are the latency figures in
:class:`LiveStats`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.registry import create_tracker
from yoru_tracker.core.types import Detection, TrackingResult
from yoru_tracker.export.csv_export import TrackCsvWriter
from yoru_tracker.export.metadata import build_metadata, write_metadata
from yoru_tracker.runtime.detection import Detector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    captured_at: float
    image: object


@dataclass(frozen=True)
class TrackingSnapshot:
    """One processed frame, as published to the display.

    *generation* changes whenever the tracker is reset, so a consumer can
    tell results from before a reset apart from those after it.
    """

    frame_id: int
    captured_at: float
    detect_started_at: float
    detected_at: float
    tracked_at: float
    generation: int
    image: object
    detections: Tuple[Detection, ...]
    result: TrackingResult

    @property
    def wait_ms(self) -> float:
        """From capture until the detector picked the frame up."""
        return 1000.0 * (self.detect_started_at - self.captured_at)

    @property
    def detector_ms(self) -> float:
        return 1000.0 * (self.detected_at - self.detect_started_at)

    @property
    def tracker_ms(self) -> float:
        return 1000.0 * (self.tracked_at - self.detected_at)


class _Rate:
    """Events per second over a sliding window."""

    def __init__(self, window: float = 2.0):
        self.window = window
        self._times = deque()

    def tick(self, now: float) -> None:
        self._times.append(now)
        while self._times and now - self._times[0] > self.window:
            self._times.popleft()

    def value(self, now: Optional[float] = None) -> float:
        now = time.perf_counter() if now is None else now
        while self._times and now - self._times[0] > self.window:
            self._times.popleft()
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class _Mean:
    def __init__(self, size: int = 120):
        self._values = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._values.append(value)

    def value(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


@dataclass(frozen=True)
class LiveStats:
    capture_fps: float
    processed_fps: float
    wait_ms: float
    detector_ms: float
    tracker_ms: float
    draw_ms: float
    end_to_end_ms: float
    dropped_frames: int
    processed_frames: int
    recording_rows: int


class RealtimeTracking:
    """Run the live pipeline on background threads.

    *source_factory* opens the frame source on the capture thread (cameras do
    not like being opened on one thread and read on another) and must return
    an object with ``read()`` and ``close()``.  *detector_factory* loads the
    model on the processing thread.  Errors in either thread stop the
    pipeline and are kept in :attr:`error` for the caller to report.
    *on_result*, if given, is called on the processing thread with every
    snapshot, in order -- the display only ever sees the newest.
    """

    def __init__(self, source_factory: Callable[[], object],
                 detector_factory: Callable[[], Detector],
                 tracker_config: TrackerConfig, *,
                 detector_settings: Optional[dict] = None,
                 source_description: str = "",
                 on_result: Optional[Callable[["TrackingSnapshot"], None]] = None):
        self._source_factory = source_factory
        self._on_result = on_result
        self._detector_factory = detector_factory
        self.tracker_config = tracker_config.validate()
        self.detector_settings = detector_settings
        self.source_description = source_description
        self.tracker = create_tracker(self.tracker_config)
        self.error: Optional[BaseException] = None
        self.phase = "idle"

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_frame: Optional[FrameSnapshot] = None
        self._latest: Optional[TrackingSnapshot] = None
        self._generation = 0
        self._reset_requested = False
        self._reset_config: Optional[TrackerConfig] = None
        self._threads = []
        self._t0 = time.perf_counter()

        self._capture_rate = _Rate()
        self._processed_rate = _Rate()
        self._wait_ms = _Mean()
        self._detector_ms = _Mean()
        self._tracker_ms = _Mean()
        self._draw_ms = _Mean()
        self._end_to_end_ms = _Mean()
        self._dropped = 0
        self._processed = 0
        self._last_processed_id = 0

        self._recorder: Optional[TrackCsvWriter] = None
        self._recording_meta = None

    # -- control --------------------------------------------------------

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @property
    def generation(self) -> int:
        return self._generation

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self.phase = "starting"
        self._t0 = time.perf_counter()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="yoru-tracker-capture", daemon=True),
            threading.Thread(target=self._process_loop, name="yoru-tracker-process", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            self._frame_ready.notify_all()
        for thread in self._threads:
            thread.join(timeout)
        self.stop_recording()
        if self.phase not in ("failed",):
            self.phase = "stopped"

    def reset_tracker(self, config: Optional[TrackerConfig] = None) -> None:
        """Drop every track; IDs restart at 0.  Takes effect on the next frame.

        With *config*, the tracker is rebuilt with those settings -- the only
        moment new settings reach a live run, since a reset is the one point
        where changing them cannot alter IDs already handed out.
        """
        if config is not None:
            config = config.validate()
        with self._lock:
            self._reset_requested = True
            self._reset_config = config
        logger.info("Live tracker reset requested by the user%s",
                    f" (new settings, mode {config.mode})" if config else "")

    def latest(self) -> Optional[TrackingSnapshot]:
        with self._lock:
            return self._latest

    def latest_frame(self) -> Optional[FrameSnapshot]:
        with self._lock:
            return self._latest_frame

    def report_drawn(self, snapshot: TrackingSnapshot, draw_started: float) -> None:
        """Called by the display after drawing *snapshot*."""
        now = time.perf_counter()
        self._draw_ms.add(1000.0 * (now - draw_started))
        self._end_to_end_ms.add(1000.0 * (now - snapshot.captured_at))

    def stats(self) -> LiveStats:
        return LiveStats(
            capture_fps=self._capture_rate.value(),
            processed_fps=self._processed_rate.value(),
            wait_ms=self._wait_ms.value(),
            detector_ms=self._detector_ms.value(),
            tracker_ms=self._tracker_ms.value(),
            draw_ms=self._draw_ms.value(),
            end_to_end_ms=self._end_to_end_ms.value(),
            dropped_frames=self._dropped,
            processed_frames=self._processed,
            recording_rows=self._recorder.rows if self._recorder else 0,
        )

    # -- recording ------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    def start_recording(self, csv_path, *, include_predicted: bool = False) -> None:
        with self._lock:
            if self._recorder is not None:
                return
            self._recorder = TrackCsvWriter(csv_path, include_predicted=include_predicted)
            self._recording_meta = {
                "csv": str(csv_path),
                "started_generation": self._generation,
                "include_predicted": include_predicted,
            }
        logger.info("Recording tracks to %s", csv_path)

    def stop_recording(self) -> None:
        with self._lock:
            recorder, self._recorder = self._recorder, None
            meta, self._recording_meta = self._recording_meta, None
        if recorder is None:
            return
        recorder.close()
        metadata = build_metadata(
            self.tracker,
            detector=self.detector_settings,
            source={"kind": "live", "description": self.source_description},
            outputs={"tracks_csv": recorder.path.name, "rows": recorder.rows,
                     "include_predicted": meta["include_predicted"]},
            summary={"processed_frames": self._processed, "dropped_frames": self._dropped,
                     "tracker_resets": self._generation - meta["started_generation"]},
        )
        write_metadata(recorder.path.with_suffix(".json"), metadata)
        logger.info("Recording stopped: %d rows in %s", recorder.rows, recorder.path)

    # -- threads --------------------------------------------------------

    def _fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = exc
        self.phase = "failed"
        logger.error("Live tracking stopped: %s", exc, exc_info=exc)
        self._stop.set()
        with self._lock:
            self._frame_ready.notify_all()

    def _capture_loop(self) -> None:
        source = None
        try:
            source = self._source_factory()
            frame_id = 0
            while not self._stop.is_set():
                image = source.read()
                now = time.perf_counter()
                frame_id += 1
                with self._lock:
                    self._latest_frame = FrameSnapshot(frame_id, now, image)
                    self._frame_ready.notify_all()
                self._capture_rate.tick(now)
        except BaseException as exc:
            self._fail(exc)
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    logger.exception("Closing the frame source failed")

    def _process_loop(self) -> None:
        try:
            self.phase = "loading model"
            detector = self._detector_factory()
            self.phase = "running"
            while not self._stop.is_set():
                with self._lock:
                    while not self._stop.is_set() and (
                        self._latest_frame is None
                        or self._latest_frame.frame_id == self._last_processed_id
                    ):
                        self._frame_ready.wait(0.1)
                    if self._stop.is_set():
                        break
                    frame = self._latest_frame
                    if self._reset_requested:
                        self._reset_requested = False
                        if self._reset_config is not None:
                            self.tracker = create_tracker(self._reset_config)
                            self.tracker_config = self._reset_config
                            self._reset_config = None
                        else:
                            self.tracker.reset()
                        self._generation += 1
                if self._last_processed_id:
                    self._dropped += max(0, frame.frame_id - self._last_processed_id - 1)
                self._last_processed_id = frame.frame_id

                started_at = time.perf_counter()
                detections = tuple(detector.detect(frame.image))
                detected_at = time.perf_counter()
                # Timestamps are seconds since start, as in YORU's recordings.
                result = self.tracker.update(detections, frame.frame_id,
                                             frame.captured_at - self._t0, frame.image)
                tracked_at = time.perf_counter()
                snapshot = TrackingSnapshot(
                    frame_id=frame.frame_id,
                    captured_at=frame.captured_at,
                    detect_started_at=started_at,
                    detected_at=detected_at,
                    tracked_at=tracked_at,
                    generation=self._generation,
                    image=frame.image,
                    detections=detections,
                    result=result,
                )
                with self._lock:
                    self._latest = snapshot
                    # Under the lock: stop_recording() may close the file.
                    if self._recorder is not None:
                        self._recorder.write(result)
                if self._on_result is not None:
                    self._on_result(snapshot)
                self._processed += 1
                self._processed_rate.tick(tracked_at)
                self._wait_ms.add(snapshot.wait_ms)
                self._detector_ms.add(snapshot.detector_ms)
                self._tracker_ms.add(snapshot.tracker_ms)
        except BaseException as exc:
            self._fail(exc)
