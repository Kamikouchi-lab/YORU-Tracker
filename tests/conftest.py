from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from yoru_tracker.core.types import Detection


def det(cx, cy, w=40.0, h=16.0, angle=0.0, conf=0.9, cls=0, name="fly"):
    """A detection centred at ``(cx, cy)``; fly-sized and upright by default."""
    return Detection.from_obb(cx, cy, w, h, angle, confidence=conf, class_id=cls, class_name=name)


@pytest.fixture
def make_det():
    return det


class BlobDetector:
    """Finds bright rectangles in a frame: a detector with no model, for runtime tests."""

    names = {0: "blob"}

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in sorted(contours, key=lambda c: tuple(cv2.boundingRect(c)[:2])):
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 20:
                continue
            out.append(Detection.from_xyxy(x, y, x + w, y + h, confidence=0.9,
                                           class_id=0, class_name="blob"))
        return out


@pytest.fixture
def blob_detector():
    return BlobDetector()


def write_video(path: Path, frames: int = 40, size=(160, 120), fps: float = 20.0,
                blobs=None) -> Path:
    """A small video of white rectangles moving on black.

    *blobs* is a list of functions ``t -> (cx, cy)``; by default two blobs
    crossing horizontally at different heights.
    """
    if blobs is None:
        blobs = [lambda t: (20 + 3 * t, 35), lambda t: (140 - 3 * t, 85)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    for t in range(frames):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        for f in blobs:
            cx, cy = f(t)
            cv2.rectangle(img, (int(cx - 8), int(cy - 4)), (int(cx + 8), int(cy + 4)),
                          (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture
def video_file(tmp_path):
    return write_video(tmp_path / "blobs.avi")


def walk(n, start, velocity):
    """Positions of a straight walk, one per frame."""
    return [(start[0] + velocity[0] * t, start[1] + velocity[1] * t) for t in range(n)]
