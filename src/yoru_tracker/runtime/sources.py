# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Where frames come from: a video file or a live camera."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg")


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
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.info = VideoInfo(
            path=self.path,
            fps=fps if fps > 0 else 30.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
        self._next = 0

    def read(self) -> Tuple[Optional[int], Optional[object]]:
        """``(index, frame)``, or ``(None, None)`` at the end."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None
        index = self._next
        self._next += 1
        return index, _flip(frame, self.v_flip, self.h_flip)

    def seek(self, index: int) -> None:
        index = max(0, int(index))
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index

    def frame_at(self, index: int):
        """The frame at *index*, or ``None`` past the end."""
        self.seek(index)
        _, frame = self.read()
        return frame

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
