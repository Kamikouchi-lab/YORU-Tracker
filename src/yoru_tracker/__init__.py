# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YORU Tracker: identity, trajectories and tracking on top of YORU's detectors.

The public tracking API lives in :mod:`yoru_tracker.core`; importing this
package does not load OpenCV, a detector or the GUI.
"""

from yoru_tracker.core.tracker_base import TRACKER_API_VERSION

__version__ = "0.1.0"  # keep in step with pyproject.toml

__all__ = ["TRACKER_API_VERSION", "__version__"]
