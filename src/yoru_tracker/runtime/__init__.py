# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Running a tracker over frames from a camera, a video, or stored detections.

Submodules import OpenCV and, when a model is loaded, the detector stack;
import them directly (``yoru_tracker.runtime.video`` and so on).
"""
