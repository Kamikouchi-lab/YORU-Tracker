"""Overlays: trails with predicted positions shown or hidden."""

from __future__ import annotations

import numpy as np
import pytest

from yoru_tracker.core.types import TrackedDetection, TrackingResult, TrackState
from yoru_tracker.drawing.overlays import (
    OverlayOptions,
    TrailBook,
    draw_tracking,
    trails_from_results,
)

ONLY_TRAILS = dict(track_boxes=False, track_ids=False, trajectories=True)


def _row(track_id, x, y, predicted):
    return TrackedDetection(
        track_id=track_id, track_state=TrackState.LOST if predicted else TrackState.ACTIVE,
        box=(float(x), float(y), 10.0, 6.0, 0.0), predicted=predicted, detection=None,
        class_id=0, class_name="fly", track_confidence=0.9, velocity=(0.0, 0.0),
        hits=1, age=1, missed_frames=int(predicted))


def _results(path, track_id=0):
    """One result per ``(x, y, predicted)`` of *path*."""
    return [TrackingResult(f, None, (_row(track_id, x, y, p),)) for f, (x, y, p) in enumerate(path)]


# Seen along y=100; while lost, predicted off along y=140; seen again at y=100.
LOST_AND_FOUND = [(10, 100, False), (30, 100, False), (50, 140, True), (70, 140, True),
                  (90, 100, False)]


def _drawn_rows(results, predicted, *, live=False):
    img = np.zeros((200, 120, 3), np.uint8)
    if live:
        book = TrailBook(30)
        for result in results:
            book.add(result)
        trails = book.trails()
    else:
        trails = trails_from_results(results, len(results) - 1, 30)
    draw_tracking(img, results[-1], OverlayOptions(predicted=predicted, **ONLY_TRAILS),
                  trails=trails)
    return {y for y in range(img.shape[0]) if img[y].any()}


@pytest.mark.parametrize("live", [False, True])
def test_with_predictions_hidden_a_trail_runs_through_where_the_animal_was_seen(live):
    hidden = _drawn_rows(_results(LOST_AND_FOUND), predicted=False, live=live)
    assert hidden and all(abs(y - 100) <= 2 for y in hidden)
    shown = _drawn_rows(_results(LOST_AND_FOUND), predicted=True, live=live)
    assert 140 in shown


@pytest.mark.parametrize("live", [False, True])
def test_a_lost_track_shows_no_trail_while_its_box_is_hidden(live):
    lost_now = LOST_AND_FOUND[:4]
    assert not _drawn_rows(_results(lost_now), predicted=False, live=live)
    assert _drawn_rows(_results(lost_now), predicted=True, live=live)
