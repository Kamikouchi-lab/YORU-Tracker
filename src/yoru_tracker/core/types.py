# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The values that cross the tracker API: detections in, tracks out.

Everything here is a frozen dataclass of plain numbers and strings, so a
result can be compared, hashed, pickled and written to JSON without knowing
which tracker produced it.  No Kalman filter, history buffer or detector
object is ever reachable from these types.

A box is carried as YORU carries it, ``(cx, cy, w, h, angle)`` with *angle* in
radians (see ``yoru.libs.obb``).  An upright box simply has angle 0.  A
rectangle's angle is only defined modulo pi, so it is an *axis*, not a
heading: nothing here tells an animal's head from its tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from yoru.libs.detector_base import DETECTION_COLUMNS, obb_of
from yoru.libs.obb import aabb_to_obb, obb_to_aabb

Box = Tuple[float, float, float, float, float]

__all__ = [
    "Box",
    "Detection",
    "EventKind",
    "Track",
    "TrackEvent",
    "TrackState",
    "TrackedDetection",
    "TrackingResult",
]


def _finite(*values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Detection:
    """One detector output, normalised and independent of the detector family.

    ``x1..y2`` is the upright box around the object; ``cx..angle`` is the box
    itself, rotated for an OBB model and identical to the upright one
    otherwise.  Build one with :meth:`from_yoru` (a YORU detector dict),
    :meth:`from_row` (a row in YORU's ``DETECTION_COLUMNS`` order),
    :meth:`from_obb` or :meth:`from_xyxy`.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    angle: float

    # -- construction ---------------------------------------------------

    @classmethod
    def from_obb(cls, cx, cy, w, h, angle=0.0, *, confidence=1.0, class_id=0,
                 class_name="") -> "Detection":
        box = (float(cx), float(cy), float(w), float(h), float(angle))
        x1, y1, x2, y2 = obb_to_aabb(box)
        return cls(x1, y1, x2, y2, float(confidence), int(class_id), str(class_name), *box)

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2, *, confidence=1.0, class_id=0,
                  class_name="") -> "Detection":
        box = aabb_to_obb(x1, y1, x2, y2)
        return cls(float(x1), float(y1), float(x2), float(y2), float(confidence),
                   int(class_id), str(class_name), *box)

    @classmethod
    def from_yoru(cls, detection: Mapping[str, Any]) -> "Detection":
        """From one dict of ``DetectorBase.detect()``, OBB or not."""
        cx, cy, w, h, angle = obb_of(detection)
        return cls(
            float(detection["x1"]), float(detection["y1"]),
            float(detection["x2"]), float(detection["y2"]),
            float(detection["conf"]), int(detection["class_id"]),
            str(detection["class_name"]),
            cx, cy, w, h, angle,
        )

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "Detection":
        """From a row in YORU's ``DETECTION_COLUMNS`` order (``total_time`` ignored)."""
        if len(row) < len(DETECTION_COLUMNS):
            raise ValueError(
                f"a detection row has {len(DETECTION_COLUMNS)} columns, got {len(row)}"
            )
        x1, y1, x2, y2, conf, cls_id, cls_name, _t, cx, cy, w, h, angle = row[:13]
        return cls(float(x1), float(y1), float(x2), float(y2), float(conf),
                   int(float(cls_id)), str(cls_name),
                   float(cx), float(cy), float(w), float(h), float(angle))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})

    # -- views ----------------------------------------------------------

    @property
    def box(self) -> Box:
        return (self.cx, self.cy, self.w, self.h, self.angle)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def is_valid(self) -> bool:
        """Finite numbers and a box with positive extent."""
        return (
            _finite(self.x1, self.y1, self.x2, self.y2, self.cx, self.cy,
                    self.w, self.h, self.angle, self.confidence)
            and self.w > 0 and self.h > 0
        )

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class TrackState(str, Enum):
    """Where a track is in its life.

    ``TENTATIVE``
        Seen, but not often enough yet to be given an ID.
    ``ACTIVE``
        Has an ID and was matched to a detection this frame.
    ``LOST``
        Has an ID, was not matched this frame, and is still kept for recovery;
        its box is a prediction.
    ``OCCLUDED``
        Reserved for trackers that detect occlusion explicitly.  The Lite
        tracker never reports it.

    A retired track is simply gone; there is no terminal state.
    """

    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    OCCLUDED = "occluded"


@dataclass(frozen=True)
class TrackedDetection:
    """One confirmed track in one frame.

    *box* is where the track is this frame: the matched detection's box, or,
    when *predicted* is true, the motion model's estimate while the track is
    unmatched.  *detection* is the matched detection, ``None`` when predicted.
    *association_cost* is the cost of the match that produced this row, for
    debugging; ``None`` for a new or predicted track.
    """

    track_id: int
    track_state: TrackState
    box: Box
    predicted: bool
    detection: Optional[Detection]
    class_id: int
    class_name: str
    track_confidence: float
    velocity: Tuple[float, float]
    hits: int
    age: int
    missed_frames: int
    association_cost: Optional[float] = None

    @property
    def cx(self) -> float:
        return self.box[0]

    @property
    def cy(self) -> float:
        return self.box[1]

    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        return obb_to_aabb(self.box)

    @property
    def confidence(self) -> Optional[float]:
        """The detector's confidence, ``None`` for a predicted box."""
        return None if self.detection is None else self.detection.confidence

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "track_state": self.track_state.value,
            "box": list(self.box),
            "predicted": self.predicted,
            "detection": None if self.detection is None else self.detection.to_dict(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_confidence": self.track_confidence,
            "velocity": list(self.velocity),
            "hits": self.hits,
            "age": self.age,
            "missed_frames": self.missed_frames,
            "association_cost": self.association_cost,
        }


@dataclass(frozen=True)
class Track:
    """A read-only summary of one track, for inspection.

    Unlike :class:`TrackedDetection` this is not tied to a frame's output; it
    is what ``TrackerBase.tracks()`` reports about every track the tracker is
    holding, tentative ones included (their *track_id* is ``None``).
    """

    track_id: Optional[int]
    track_state: TrackState
    class_id: int
    class_name: str
    box: Box
    velocity: Tuple[float, float]
    hits: int
    age: int
    missed_frames: int
    track_confidence: float
    first_frame_id: int
    last_seen_frame_id: int


class EventKind(str, Enum):
    CREATED = "created"        # a confirmed track was given an ID
    LOST = "lost"              # a confirmed track missed its first frame
    RECOVERED = "recovered"    # a lost track was matched again
    RETIRED = "retired"        # a track exceeded max_age and was removed
    RESET = "reset"            # every track was dropped on request


@dataclass(frozen=True)
class TrackEvent:
    kind: EventKind
    frame_id: int
    track_id: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "frame_id": self.frame_id,
                "track_id": self.track_id, "detail": self.detail}


@dataclass(frozen=True)
class TrackingResult:
    """Everything a tracker decided about one frame.

    *tracked* holds one row per confirmed track, sorted by ID.  *unassigned*
    are valid detections not (yet) part of a confirmed track -- a newcomer
    still tentative, for instance.  *rejected* are detections the tracker
    refused as invalid (non-finite numbers, empty box).

    Timing is deliberately not part of a result: the same detections in the
    same order must give an identical result, and a wall-clock number would
    make every two runs differ.
    """

    frame_id: int
    timestamp: Optional[float]
    tracked: Tuple[TrackedDetection, ...] = ()
    unassigned: Tuple[Detection, ...] = ()
    rejected: Tuple[Detection, ...] = ()
    events: Tuple[TrackEvent, ...] = field(default=())

    @property
    def active_ids(self) -> Tuple[int, ...]:
        return tuple(t.track_id for t in self.tracked if not t.predicted)

    @property
    def lost_ids(self) -> Tuple[int, ...]:
        return tuple(t.track_id for t in self.tracked if t.predicted)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "tracked": [t.to_dict() for t in self.tracked],
            "unassigned": [d.to_dict() for d in self.unassigned],
            "rejected": [d.to_dict() for d in self.rejected],
            "events": [e.to_dict() for e in self.events],
        }
