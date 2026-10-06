"""Motion models: constant velocity, frame gaps, and the stationary fallback."""

from __future__ import annotations

import numpy as np
import pytest

from yoru_tracker.tracking.kalman import ConstantVelocityModel, StationaryModel


def _model(x=0.0, y=0.0):
    return ConstantVelocityModel(x, y, 30.0, process_noise=0.2, measurement_noise=0.1)


def test_velocity_converges_on_steady_motion():
    m = _model()
    for t in range(1, 30):
        m.predict(1, 30.0)
        m.update(5.0 * t, -2.0 * t, 30.0)
    vx, vy = m.velocity
    assert vx == pytest.approx(5.0, abs=0.05)
    assert vy == pytest.approx(-2.0, abs=0.05)


def test_a_frame_gap_is_a_longer_step():
    m = _model()
    for t in range(1, 30):
        m.predict(1, 30.0)
        m.update(4.0 * t, 0.0, 30.0)
    x_before = m.position[0]
    m.predict(3, 30.0)
    assert m.position[0] == pytest.approx(x_before + 12.0, abs=0.2)


def test_uncertainty_grows_while_unobserved():
    m = _model()
    m.update(0.0, 0.0, 30.0)
    before = np.trace(m.cov[:2, :2])
    m.predict(5, 30.0)
    assert np.trace(m.cov[:2, :2]) > before


def test_covariance_stays_symmetric_positive_definite():
    m = _model()
    rng = np.random.default_rng(0)
    for t in range(500):
        m.predict(1 + t % 3, 30.0)
        m.update(*rng.normal(0, 50, 2), 30.0)
    assert np.allclose(m.cov, m.cov.T)
    assert np.all(np.linalg.eigvalsh(m.cov) > 0)


def test_implausible_is_measured_against_the_grown_uncertainty():
    m = _model()
    for t in range(1, 30):
        m.predict(1, 30.0)
        m.update(4.0 * t, 0.0, 30.0)
    m.predict(1, 30.0)
    assert not m.implausible(124.0, 0.0, 30.0)      # where it should be
    assert m.implausible(124.0, 150.0, 30.0)         # 5 body lengths off
    for _ in range(20):
        m.predict(1, 30.0)                          # unseen for long: far is possible
    assert not m.implausible(204.0, 150.0, 30.0)


def test_restart_forgets_the_motion():
    m = _model()
    for t in range(1, 30):
        m.predict(1, 30.0)
        m.update(4.0 * t, 0.0, 30.0)
    m.restart(500.0, 20.0, 30.0)
    assert m.position == (500.0, 20.0) and m.velocity == (0.0, 0.0)
    assert np.allclose(m.cov, _model(500.0, 20.0).cov)


def test_stationary_model_never_extrapolates_but_reports_velocity():
    m = StationaryModel(10.0, 10.0, 30.0)
    m.predict(1, 30.0)
    m.update(14.0, 10.0, 30.0)
    m.predict(2, 30.0)
    assert m.position == (14.0, 10.0)
    assert m.velocity == (4.0, 0.0)
