# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Motion models for a track's centre.

The state is ``[x, y, vx, vy]`` in pixels and pixels per frame.  Time is
counted in frames, from frame IDs, not from wall-clock timestamps: a live
camera's timestamps jitter, and a model driven by them would make the same
detections track differently live and from the recorded video.  A gap in
frame IDs (frames the detector never saw) is a longer step.

Box width, height and angle are deliberately not in the state; the track takes
them from its latest detection.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_I4 = np.eye(4)

#: Squared Mahalanobis distance beyond which an observation is not one the
#: motion model expects: the 99% point of chi-square with two degrees of
#: freedom.  On the benchmark every animal found again after a detector miss
#: stays below 1.2; an animal that jumped is over 200.
IMPLAUSIBLE = 9.21


class ConstantVelocityModel:
    """Kalman filter with constant-velocity dynamics.

    Noise is scaled by the box size passed to :meth:`predict` and
    :meth:`update`, so the same relative settings fit any magnification.
    """

    def __init__(self, x: float, y: float, size: float, *,
                 process_noise: float, measurement_noise: float):
        self._q = float(process_noise)
        self._r = float(measurement_noise)
        self.restart(x, y, size)

    def restart(self, x: float, y: float, size: float) -> None:
        """Start again at ``(x, y)``, knowing no more than a new track does."""
        size = max(float(size), 1.0)
        self.state = np.array([float(x), float(y), 0.0, 0.0])
        pos_var = (self._r * size) ** 2
        # Nothing is known about the velocity of a track on its first frame:
        # allow it to be about a box size per frame either way.
        vel_var = size ** 2
        self.cov = np.diag([pos_var, pos_var, vel_var, vel_var])

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.state[2]), float(self.state[3])

    def predict(self, dt: float, size: float) -> None:
        dt = float(dt)
        f = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Piecewise-constant white acceleration over the step.
        a = (self._q * max(float(size), 1.0)) ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        q = a * np.array([
            [dt4 / 4, 0.0, dt3 / 2, 0.0],
            [0.0, dt4 / 4, 0.0, dt3 / 2],
            [dt3 / 2, 0.0, dt2, 0.0],
            [0.0, dt3 / 2, 0.0, dt2],
        ])
        self.state = f @ self.state
        self.cov = f @ self.cov @ f.T + q

    def innovation(self, x: float, y: float, size: float):
        """Residual and its covariance for a measurement at ``(x, y)``."""
        r = (self._r * max(float(size), 1.0)) ** 2
        residual = np.array([float(x), float(y)]) - _H @ self.state
        s = _H @ self.cov @ _H.T + r * np.eye(2)
        return residual, s

    def mahalanobis_sq(self, x: float, y: float, size: float) -> float:
        residual, s = self.innovation(x, y, size)
        return float(residual @ np.linalg.solve(s, residual))

    def implausible(self, x: float, y: float, size: float) -> bool:
        """Is ``(x, y)`` somewhere this model gives less than a 1% chance?"""
        return self.mahalanobis_sq(x, y, size) > IMPLAUSIBLE

    def update(self, x: float, y: float, size: float) -> None:
        residual, s = self.innovation(x, y, size)
        gain = self.cov @ _H.T @ np.linalg.inv(s)
        self.state = self.state + gain @ residual
        # Joseph form: stays symmetric and positive definite over long runs.
        k_h = _I4 - gain @ _H
        r = (self._r * max(float(size), 1.0)) ** 2
        self.cov = k_h @ self.cov @ k_h.T + r * gain @ gain.T


class StationaryModel:
    """No motion prediction: a track is expected where it was last seen.

    Used when ``kalman.enabled`` is false.  The velocity is still measured
    (from the last two sightings) so that it can be reported and exported; it
    just never moves the prediction.
    """

    def __init__(self, x: float, y: float, size: float, **_):
        self.restart(x, y, size)

    def restart(self, x: float, y: float, size: float) -> None:
        self._pos = (float(x), float(y))
        self._vel = (0.0, 0.0)
        self._since_update = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return self._pos

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._vel

    def implausible(self, x: float, y: float, size: float) -> bool:
        # It never extrapolates, so no observation can throw a prediction off.
        return False

    def predict(self, dt: float, size: float) -> None:
        self._since_update += float(dt)

    def update(self, x: float, y: float, size: float) -> None:
        dt = self._since_update or 1.0
        self._vel = ((float(x) - self._pos[0]) / dt, (float(y) - self._pos[1]) / dt)
        self._pos = (float(x), float(y))
        self._since_update = 0.0
