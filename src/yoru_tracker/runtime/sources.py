# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Where frames come from: a video file or a live camera."""

from __future__ import annotations

import bisect
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg")

#: Frame rate taken for a file that states none and whose frames carry no
#: usable times either -- so that times are on some scale, and said to be.
ASSUMED_FPS = 30.0

#: A frame at most this far ahead is reached by reading on, which is exact,
#: rather than by seeking, which decodes from a keyframe before it anyway.
READ_AHEAD = 32


def frame_of(times: Sequence[float], ms: Optional[float]) -> Optional[int]:
    """Which frame of a sequential pass has presentation time *ms*.

    *times* are the times that pass recorded, frame by frame, increasing.
    ``len(times)`` for a time past every recorded one; ``None`` when it
    matches no frame.
    """
    if ms is None or not math.isfinite(ms) or not times:
        return None
    n = len(times)
    i = bisect.bisect_left(times, ms, 0, n)
    best = min((j for j in (i - 1, i) if 0 <= j < n), key=lambda j: abs(times[j] - ms))
    gaps = [times[k] - times[k - 1] for k in (best, best + 1) if 0 < k < n]
    if abs(times[best] - ms) < (min(gaps) / 2 if gaps else 0.5):
        return best
    return n if ms > times[n - 1] else None


def _flip(frame, v_flip: bool, h_flip: bool):
    if v_flip and h_flip:
        return cv2.flip(frame, -1)
    if v_flip:
        return cv2.flip(frame, 0)
    if h_flip:
        return cv2.flip(frame, 1)
    return frame


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    #: Where *fps* came from: ``"file"``, the rate the file states;
    #: ``"timestamps"``, measured from its first frames' presentation times
    #: when it states none; ``"assumed"``, :data:`ASSUMED_FPS` when it has
    #: neither -- every time in seconds then rests on that guess.
    fps_source: str = "file"

    @property
    def fps_note(self) -> str:
        """What a user should know about *fps*; empty when the file stated it."""
        if self.fps_source == "timestamps":
            return "the file states no frame rate; measured from its timestamps"
        if self.fps_source == "assumed":
            return (f"the file states no frame rate and has no timestamps; "
                    f"{ASSUMED_FPS:g} fps assumed, so times in seconds are a guess")
        return ""


class VideoFileSource:
    """Frames of a video file, by index.

    ``read()`` continues from the current position; ``seek()`` moves it.
    Frame indices start at 0 and are what the tracker is given as frame IDs,
    so a result can always be put back on the frame it came from.
    """

    def __init__(self, path, *, v_flip: bool = False, h_flip: bool = False):
        self.path = str(path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self.v_flip = v_flip
        self.h_flip = h_flip
        self._next = 0
        fps, fps_source = self._cap.get(cv2.CAP_PROP_FPS) or 0.0, "file"
        if not (math.isfinite(fps) and fps > 0):
            measured = self._fps_from_times()
            fps, fps_source = (measured, "timestamps") if measured else (ASSUMED_FPS, "assumed")
        self.info = VideoInfo(
            path=self.path,
            fps=fps,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps_source=fps_source,
        )
        if fps_source != "file":
            logger.warning("%s: %s (%.3f fps)", self.path, self.info.fps_note, fps)

    def _fps_from_times(self, frames: int = 25) -> Optional[float]:
        """The frame rate the first frames' presentation times show, if any."""
        times = []
        while len(times) < frames and self._cap.grab():
            ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms is None or not math.isfinite(ms):
                break
            times.append(ms)
        self._reopen()                  # back at frame 0, exactly
        steps = sorted(b - a for a, b in zip(times, times[1:]))
        step = steps[len(steps) // 2] if steps else 0.0   # the median: robust to a stray one
        return 1000.0 / step if step > 0 else None

    def read(self) -> Tuple[Optional[int], Optional[object]]:
        """``(index, frame)``, or ``(None, None)`` at the end."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None
        index = self._next
        self._next += 1
        return index, _flip(frame, self.v_flip, self.h_flip)

    def position_ms(self) -> Optional[float]:
        """Presentation time of the frame last read, as the file stamps it."""
        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        return float(ms) if ms is not None and math.isfinite(ms) else None

    def seek(self, index: int) -> None:
        index = max(0, int(index))
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index

    def frame_at(self, index: int, times: Optional[Sequence[float]] = None):
        """The frame at *index* -- the one reading from the start numbers so --
        or ``None`` past the end.

        A seek lands where the file's index says, and for some files
        (variable frame rate, edit lists, broken timestamps) that is not the
        frame sequential reading numbers *index*: the picture would not be
        the one the results belong to.  So a frame a little ahead is read
        on to instead, and with *times* -- every frame's presentation time,
        recorded by a sequential pass -- a seek is checked and corrected.
        """
        index = max(0, int(index))
        if self._next <= index <= self._next + READ_AHEAD:
            return self._read_on_to(index)
        if not times or index >= len(times):
            self.seek(index)
            _, frame = self.read()
            return frame
        aim = index
        for _ in range(4):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, aim)
            if not self._cap.grab():
                break
            at = frame_of(times, self._cap.get(cv2.CAP_PROP_POS_MSEC))
            if at is None:              # a time the pass never saw: trust the seek
                at = aim
            if at <= index:
                self._next = at + 1     # the frame grabbed is frame `at`
                if at == index:
                    ok, frame = self._cap.retrieve()
                    return _flip(frame, self.v_flip, self.h_flip) if ok else None
                return self._read_on_to(index)
            aim = max(0, aim - 2 * (at - index) - 1)   # landed past it: aim further back
        # Reading from the start is slow but exact.
        self._reopen()
        return self._read_on_to(index)

    def _read_on_to(self, index: int):
        """Read on from the current position to *index*; that frame, or None."""
        while self._next < index:
            if not self._cap.grab():
                return None
            self._next += 1
        _, frame = self.read()
        return frame

    def _reopen(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(self.path)
        self._next = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CameraSource:
    """A live camera, opened through YORU's ``open_camera``."""

    def __init__(self, camera_id: int = 0, width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None, *,
                 v_flip: bool = False, h_flip: bool = False):
        from yoru.libs.camera import open_camera

        self.camera_id = camera_id
        self._cap = open_camera(camera_id, width, height, fps)
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.description = f"camera {camera_id}"

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Camera {self.camera_id} returned no frame; check the connection")
        return _flip(frame, self.v_flip, self.h_flip)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class PacedVideoSource:
    """A video file played in real time, as if it were a camera.

    For trying the live pipeline without a camera: frames are delivered no
    faster than the file's frame rate, and the file loops at the end.
    """

    def __init__(self, path, *, v_flip: bool = False, h_flip: bool = False, loop: bool = True):
        self._video = VideoFileSource(path, v_flip=v_flip, h_flip=h_flip)
        self._interval = 1.0 / self._video.info.fps
        self._due = None
        self._loop = loop
        self.description = f"video {Path(path).name} (real time)"

    def read(self):
        now = time.perf_counter()
        if self._due is not None and now < self._due:
            time.sleep(self._due - now)
        self._due = max(now, self._due or now) + self._interval
        _, frame = self._video.read()
        if frame is None and self._loop:
            self._video.seek(0)
            _, frame = self._video.read()
        if frame is None:
            raise EOFError("End of video")
        return frame

    def close(self) -> None:
        self._video.close()
