# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""What a tracker can do, declared rather than discovered by trying."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrackerInfo:
    """Capabilities of one tracker implementation.

    The GUI reads these instead of special-casing tracker names: a tracker
    that is not *realtime_capable* is not offered for live tracking, one that
    *requires_frame* is handed the image on every update, and so on.
    """

    name: str
    version: str
    api_version: int
    description: str = ""
    realtime_capable: bool = True
    requires_frame: bool = False
    requires_gpu: bool = False
    supports_obb: bool = True
    supports_variable_population: bool = True
    deterministic: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
