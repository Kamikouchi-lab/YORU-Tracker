# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The tracker API: deliberately two methods wide.

A tracker is fed one frame's detections at a time, in frame order, and
returns what it decided about that frame.  Where the detections came from --
a camera, a video, a CSV written last week -- is not its business: the same
ordered detections must produce the same result from any source.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar, Iterable, Optional, Tuple

from yoru_tracker.core.config import TrackerConfig

if TYPE_CHECKING:  # pragma: no cover
    from yoru_tracker.core.capabilities import TrackerInfo
    from yoru_tracker.core.types import Detection, Track, TrackingResult

#: Version of the contract below.  A tracker plugin declares the version it
#: was written against (``api_version``); the registry refuses one that does
#: not match instead of running it and hoping.
TRACKER_API_VERSION = 1


class TrackerBase(abc.ABC):
    """Base class of every tracker, built in or plugged in.

    Subclasses set :attr:`info` and implement :meth:`reset` and
    :meth:`update`.  The configuration is validated here, once, so no tracker
    ever runs on a configuration that would be rejected on load.
    """

    info: ClassVar["TrackerInfo"]
    api_version: ClassVar[int] = TRACKER_API_VERSION

    def __init__(self, config: Optional[TrackerConfig] = None):
        self._config = (config or TrackerConfig()).validate()

    @property
    def config(self) -> TrackerConfig:
        return self._config

    @abc.abstractmethod
    def reset(self) -> None:
        """Drop every track and start numbering IDs from zero again."""

    @abc.abstractmethod
    def update(
        self,
        detections: Iterable["Detection"],
        frame_id: int,
        timestamp: Optional[float] = None,
        frame=None,
    ) -> "TrackingResult":
        """Advance by one frame.

        *frame_id* must increase from call to call; a gap means frames were
        skipped (dropped by a live camera, say) and is taken into account by
        the motion model.  *timestamp* is carried into the result.  *frame* is
        the image, for trackers whose :attr:`info` says they require it; the
        Lite tracker ignores it.
        """

    def tracks(self) -> Tuple["Track", ...]:
        """Every track currently held, for inspection.  Optional."""
        return ()
