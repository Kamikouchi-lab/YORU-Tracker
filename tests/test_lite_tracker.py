"""Lite tracker behaviour: the cases the design requires before Advanced work.

Track creation, update and deletion; MAX_AGE; stable IDs during smooth motion;
crossing tracks; detector dropout; animals appearing and disappearing; the
spare boxes a detector adds where animals crowd; OBB angle wrap; invalid
detections; empty frames; reset; determinism.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from yoru_tracker.core import Detection, EventKind, TrackerConfig, TrackState, create_tracker
from yoru_tracker.core.config import AssociationConfig, LifecycleConfig

from conftest import det, walk


def lite(**lifecycle):
    return create_tracker(TrackerConfig(lifecycle=LifecycleConfig(**lifecycle)))


def ids(result, predicted=False):
    return [t.track_id for t in result.tracked if t.predicted == predicted]


def kinds(result):
    return [(e.kind, e.track_id) for e in result.events]


# -- creation / update / deletion ---------------------------------------------

def test_first_frame_detections_get_ids_at_once():
    t = lite()
    r = t.update([det(10, 10), det(200, 10)], 0)
    assert ids(r) == [0, 1]
    assert kinds(r) == [(EventKind.CREATED, 0), (EventKind.CREATED, 1)]


def test_a_later_newcomer_waits_for_min_hits():
    t = lite(min_hits=3)
    t.update([det(10, 10)], 0)
    r1 = t.update([det(12, 10), det(300, 300)], 1)
    assert ids(r1) == [0] and len(r1.unassigned) == 1
    r2 = t.update([det(14, 10), det(301, 300)], 2)
    assert ids(r2) == [0]
    r3 = t.update([det(16, 10), det(302, 300)], 3)
    assert ids(r3) == [0, 1]
    assert (EventKind.CREATED, 1) in kinds(r3)


def test_matched_track_follows_its_detection():
    t = lite()
    t.update([det(10, 10)], 0)
    r = t.update([det(15, 12)], 1)
    track = r.tracked[0]
    assert (track.cx, track.cy) == (15, 12)
    assert track.detection is not None and not track.predicted
    assert track.track_state is TrackState.ACTIVE and track.hits == 2


def test_track_is_retired_after_max_age():
    t = lite(max_age=3)
    t.update([det(10, 10)], 0)
    states = []
    for f in range(1, 6):
        r = t.update([], f)
        states.append([(x.track_id, x.track_state.value) for x in r.tracked])
        if f == 4:
            assert (EventKind.RETIRED, 0) in kinds(r)
    assert states == [[(0, "lost")], [(0, "lost")], [(0, "lost")], [], []]


def test_max_age_boundary_is_still_recoverable():
    t = lite(max_age=3)
    t.update([det(10, 10)], 0)
    for f in (1, 2, 3):
        t.update([], f)
    r = t.update([det(10, 10)], 4)
    assert ids(r) == [0]
    assert (EventKind.RECOVERED, 0) in kinds(r)


def test_one_frame_false_positive_does_not_spend_an_id():
    t = lite()
    t.update([det(10, 10)], 0)
    t.update([det(12, 10), det(400, 400)], 1)        # false positive
    for f in range(2, 5):
        t.update([det(10 + 2 * f, 10)], f)
    r = t.update([det(22, 10), det(300, 50)], 5)      # a real newcomer
    r = t.update([det(24, 10), det(300, 52)], 6)
    assert ids(r) == [0, 1]                           # numbering has no gap


# -- motion --------------------------------------------------------------------

def test_stable_ids_during_smooth_motion():
    t = lite()
    paths = [walk(120, (50, 50), (3, 1)), walk(120, (500, 60), (-2, 2)),
             walk(120, (300, 400), (1, -3))]
    for f in range(120):
        r = t.update([det(*p[f]) for p in paths], f)
        assert ids(r) == [0, 1, 2]
        for track, path in zip(r.tracked, paths):
            assert (track.cx, track.cy) == pytest.approx(path[f])


@pytest.mark.parametrize("swap_order", [False, True])
def test_crossing_tracks_keep_their_ids(swap_order):
    t = lite()
    a = walk(60, (100, 200), (6, 2))
    b = walk(60, (100, 320), (6, -2))   # the paths cross at frame 30
    for f in range(60):
        dets = [det(*a[f]), det(*b[f])]
        if swap_order and f % 2:
            dets.reverse()               # input order carries no identity
        r = t.update(dets, f)
        where = {x.track_id: (x.cx, x.cy) for x in r.tracked}
        assert where[0] == pytest.approx(a[f])
        assert where[1] == pytest.approx(b[f])


def test_detector_dropout_keeps_the_id_and_predicts_meanwhile():
    t = lite()
    path = walk(30, (100, 100), (4, 0))
    for f in range(30):
        dets = [] if 10 <= f < 14 else [det(*path[f])]
        r = t.update(dets, f)
        if 10 <= f < 14:
            assert ids(r) == [] and ids(r, predicted=True) == [0]
            assert r.tracked[0].track_state is TrackState.LOST
            # The prediction carries on along the path.
            assert r.tracked[0].cx == pytest.approx(path[f][0], abs=3)
        else:
            assert ids(r) == [0]


def test_a_track_found_where_its_motion_said_keeps_its_velocity():
    t = lite()
    path = walk(30, (100, 100), (4, 0))
    for f in range(30):
        r = t.update([] if 10 <= f < 14 else [det(*path[f])], f)
        if f == 14:
            assert r.tracked[0].velocity == pytest.approx((4, 0), abs=0.5)


def test_a_track_found_far_from_its_prediction_does_not_report_the_gap_as_speed():
    t = lite()
    for f in range(20):
        t.update([det(100, 100), det(300 + 2 * f, 300)], f)
    for f in range(20, 23):
        t.update([det(100, 100)], f)                     # the walker is missed...
    r = t.update([det(100, 100), det(480, 300)], 23)    # ...and found 134 px ahead
    assert ids(r) == [0, 1]
    assert abs(r.tracked[1].velocity[0]) < 3            # not 45 px/frame
    for f in range(24, 30):
        r = t.update([det(100, 100), det(480 + 2 * (f - 23), 300)], f)
    assert r.tracked[1].velocity == pytest.approx((2, 0), abs=0.3)


def test_frame_gaps_are_longer_motion_steps():
    t = lite()
    path = walk(40, (100, 100), (5, 0))
    for f in range(0, 20):
        t.update([det(*path[f])], f)
    # The camera dropped frames 20-24: the next frame the tracker sees is 25.
    r = t.update([det(*path[25])], 25)
    assert ids(r) == [0]
    assert r.tracked[0].association_cost < 0.6


# -- population ----------------------------------------------------------------

def test_new_animal_appearing_gets_the_next_id():
    t = lite()
    for f in range(10):
        dets = [det(100 + f, 100)] + ([det(400, 300 + f)] if f >= 5 else [])
        r = t.update(dets, f)
    assert ids(r) == [0, 1]


def test_animal_disappearing_is_retired_and_others_keep_ids():
    t = lite(max_age=4)
    for f in range(20):
        dets = [det(100 + f, 100)] + ([det(400, 300)] if f < 8 else [])
        r = t.update(dets, f)
    assert ids(r) == [0]
    assert t.tracks()[0].track_id == 0 and len(t.tracks()) == 1


def test_known_population_recovers_a_jump_without_a_new_id():
    t = lite(population=2)
    for f in range(20):
        t.update([det(100 + f, 100), det(400, 300)], f)
    # The second animal jumps 250 px in one frame.
    r = t.update([det(120, 100), det(400, 50)], 20)
    assert ids(r) == [0] and ids(r, predicted=True) == [1]
    r = t.update([det(121, 100), det(401, 50)], 21)
    assert ids(r) == [0, 1]
    assert {x.track_id: x.cy for x in r.tracked}[1] == 50


def test_a_handed_over_track_moves_at_the_animals_speed_not_the_jumps():
    t = lite(population=2)
    for f in range(20):
        t.update([det(100, 100), det(300 + 2 * f, 300)], f)
    for k, f in enumerate(range(20, 26)):
        # It jumped 270 px and walks on at 2 px/frame.
        r = t.update([det(100, 100), det(340 + 2 * k, 30)], f)
        if r.lost_ids:
            continue
        assert ids(r) == [0, 1]
        assert r.tracked[1].velocity == pytest.approx((2, 0), abs=0.5)   # not -149
    assert not r.lost_ids


def test_known_population_never_retires_or_exceeds_the_count():
    t = lite(population=2, max_age=3)
    t.update([det(100, 100), det(400, 300)], 0)
    for f in range(1, 40):
        r = t.update([det(100, 100)], f)            # the second one is gone a long time
    assert ids(r, predicted=True) == [1]
    for f in range(40, 45):
        r = t.update([det(100, 100), det(300, 300), det(50, 400)], f)
    # Two IDs in total, whatever turns up.
    assert sorted(x.track_id for x in r.tracked) == [0, 1]


def test_known_population_hand_over_keeps_within_class():
    t = create_tracker(TrackerConfig(lifecycle=LifecycleConfig(population=2, min_hits=1),
                                     association=AssociationConfig(class_aware=True)))
    t.update([det(100, 100), det(300, 100)], 0)
    t.update([], 1)                                    # both animals lost
    # Beyond every gate: one detection of their class, one of another class.
    r = t.update([det(100, 400), det(500, 400, cls=1, name="other")], 2)
    assert ids(r) == [0] and ids(r, predicted=True) == [1]
    assert {x.track_id: (x.cx, x.cy) for x in r.tracked}[0] == (100, 400)
    assert [d.class_id for d in r.unassigned] == [1]


def test_known_population_ignores_one_frame_false_positives():
    t = lite(population=2)
    for f in range(10):
        t.update([det(100 + f, 100), det(400, 300)], f)
    r = t.update([det(110, 100), det(50, 450)], 10)   # animal 1 missed, a false positive
    r = t.update([det(111, 100), det(400, 300)], 11)  # it is back where it was
    assert {x.track_id: (x.cx, x.cy) for x in r.tracked}[1] == (400, 300)


# -- spare boxes ---------------------------------------------------------------

def box(x1, y1, x2, y2, conf=0.9, cls=0, name="fly"):
    return Detection.from_xyxy(x1, y1, x2, y2, confidence=conf, class_id=cls, class_name=name)


def test_a_doubtful_detection_never_starts_a_track():
    t = lite()
    r = t.update([det(100, 100), det(300, 300, conf=0.3)], 0)   # not even on the first frame
    assert ids(r) == [0] and len(r.unassigned) == 1
    for f in range(1, 6):
        r = t.update([det(100, 100), det(300, 300, conf=0.3)], f)
    assert ids(r) == [0]


def test_a_doubtful_detection_continues_a_track_where_it_is():
    t = lite()
    for f in range(5):
        t.update([det(100 + 2 * f, 100)], f)
    r = t.update([det(110, 100, conf=0.3)], 5)   # seen less clearly, right where expected
    assert ids(r) == [0] and r.tracked[0].detection.confidence == 0.3


def test_a_doubtful_detection_does_not_bring_back_a_lost_track():
    t = lite()
    for f in range(5):
        t.update([det(100, 100)], f)
    t.update([], 5)
    r = t.update([det(100, 100, conf=0.3)], 6)
    assert ids(r) == [] and ids(r, predicted=True) == [0]


@pytest.mark.parametrize("high_confidence", [0.5, 0.0])
def test_a_box_around_two_touching_animals_is_not_a_third(high_confidence):
    # Both animals are detected, and the detector adds a weaker box around
    # the pair, overlapping each by IoU 0.44.
    t = create_tracker(TrackerConfig(
        association=AssociationConfig(high_confidence=high_confidence)))
    for f in range(20):
        r = t.update([det(80 + f, 100), det(80 + f, 120), box(60 + f, 92, 100 + f, 128, conf=0.45)], f)
    if not high_confidence:
        assert ids(r) == [0, 1, 2]   # what it was before: a third ID on no animal
        return
    assert ids(r) == [0, 1]
    # One animal missed: the box around both does not stand in for it.
    r = t.update([det(100, 100), box(80, 92, 120, 128, conf=0.45)], 20)
    assert ids(r) == [0] and ids(r, predicted=True) == [1]
    r = t.update([det(101, 100), det(101, 120)], 21)
    assert {x.track_id: x.cy for x in r.tracked} == {0: 100, 1: 120}


@pytest.mark.parametrize("class_aware", [False, True])
def test_a_box_of_another_class_on_one_animal(class_aware):
    t = create_tracker(TrackerConfig(association=AssociationConfig(class_aware=class_aware)))
    for f in range(10):
        wing = det(101 + f, 101, conf=0.8, cls=1, name="wing_extension")
        r = t.update([det(100 + f, 100, conf=0.9), wing], f)
    if class_aware:
        assert ids(r) == [0, 1]   # classes are different animals: never folded together
        return
    assert ids(r) == [0] and len(r.unassigned) == 1
    # The more confident box is the one followed, whatever its class.
    r = t.update([det(110, 100, conf=0.7), det(111, 101, conf=0.95, cls=1, name="wing_extension")], 10)
    assert ids(r) == [0] and r.tracked[0].class_name == "wing_extension"


def _hide_and_offer_a_spare(hidden_guard):
    """Three animals; the first two are seen as one box, and a confident spare box sits half on the third."""
    t = create_tracker(TrackerConfig(lifecycle=LifecycleConfig(population=3),
                                     association=AssociationConfig(hidden_guard=hidden_guard)))
    for f in range(10):
        t.update([det(100, 100), det(140, 100), det(400, 300)], f)
    for f in range(10, 16):
        r = t.update([box(80, 92, 160, 108), det(400, 300), det(400, 312, conf=0.8)], f)
    return r


def test_an_animal_hidden_in_anothers_box_is_not_given_a_spare_box():
    r = _hide_and_offer_a_spare(hidden_guard=True)
    assert len(ids(r)) == 2 and len(ids(r, predicted=True)) == 1   # hidden, waiting
    assert all(t.cy != 312 for t in r.tracked)
    unguarded = _hide_and_offer_a_spare(hidden_guard=False)
    assert sorted(t.cy for t in unguarded.tracked if not t.predicted) == [100, 300, 312]


def _hide_then_appear_nearby(hidden_guard):
    t = create_tracker(TrackerConfig(
        association=AssociationConfig(max_distance=30.0, hidden_guard=hidden_guard)))
    for f in range(5):
        t.update([det(100, 100), det(100, 118)], f)   # side by side
    for f in range(5, 8):
        t.update([box(80, 92, 120, 126)], f)          # seen as one box
    return t.update([box(80, 92, 120, 126), det(160, 109)], 8)


def test_a_track_hidden_in_anothers_box_is_looked_for_there():
    # Lost for three frames, its recovery gate would have widened to 90 px;
    # last seen inside the box that holds its neighbour, it stays at 30.
    assert len(ids(_hide_then_appear_nearby(hidden_guard=True))) == 1
    assert len(ids(_hide_then_appear_nearby(hidden_guard=False))) == 2


def test_version_1_settings_track_as_before():
    old = TrackerConfig.from_dict({"config_version": 1, "tracker": {}})
    t = create_tracker(old)
    for f in range(10):
        r = t.update([det(80 + f, 100), det(80 + f, 120), box(60 + f, 92, 100 + f, 128, conf=0.45)], f)
    assert ids(r) == [0, 1, 2]


# -- geometry ------------------------------------------------------------------

def test_obb_angle_wrap_does_not_break_a_track():
    t = lite()
    angle = math.pi / 2 - 0.01
    r = None
    for f in range(20):
        # The detector reports the same rectangle as angle ~ +pi/2 or ~ -pi/2,
        # or with w and h exchanged: all the same box.
        if f % 3 == 0:
            d = det(100 + f, 100, 40, 12, angle)
        elif f % 3 == 1:
            d = det(100 + f, 100, 40, 12, angle - math.pi)
        else:
            d = det(100 + f, 100, 12, 40, angle - math.pi / 2)
        r = t.update([d], f)
        assert ids(r) == [0]
        if f:
            assert r.tracked[0].association_cost < 0.5


# -- robustness ------------------------------------------------------------------

def test_invalid_detections_are_rejected_not_tracked():
    t = lite()
    bad = [dataclasses.replace(det(10, 10), cx=math.nan), dataclasses.replace(det(50, 50), w=0.0)]
    r = t.update(bad + [det(200, 200)], 0)
    assert ids(r) == [0]
    assert len(r.rejected) == 2


def test_empty_frames():
    t = lite()
    r = t.update([], 0)
    assert r.tracked == () and r.events == ()
    # Not the first frame any more: a detection now is a newcomer, which
    # needs min_hits sightings like any other.
    t.update([det(10, 10)], 1)
    r = t.update([det(10, 10)], 2)
    assert ids(r) == [0]
    r = t.update([], 3)
    assert ids(r, predicted=True) == [0]


def test_reset_restarts_ids_and_reports_it():
    t = lite()
    t.update([det(10, 10), det(100, 10)], 0)
    t.update([det(12, 10), det(102, 10)], 1)
    t.reset()
    r = t.update([det(300, 300)], 0)      # frame IDs may start over after a reset
    assert ids(r) == [0]
    assert kinds(r)[0] == (EventKind.RESET, None)


def test_frames_out_of_order_are_an_error_not_a_silent_reset():
    t = lite()
    t.update([det(10, 10)], 5)
    with pytest.raises(ValueError, match="increasing order"):
        t.update([det(10, 10)], 5)
    with pytest.raises(ValueError):
        t.update([det(10, 10)], 3)


def test_class_aware_matching():
    aware = create_tracker(TrackerConfig(association=AssociationConfig(class_aware=True)))
    aware.update([det(10, 10, cls=0, name="solo")], 0)
    r = aware.update([det(11, 10, cls=1, name="copulation")], 1)
    assert ids(r) == [] and ids(r, predicted=True) == [0]

    plain = create_tracker()  # default: behaviours may change under one ID
    plain.update([det(10, 10, cls=0, name="solo")], 0)
    r = plain.update([det(11, 10, cls=1, name="copulation")], 1)
    assert ids(r) == [0]
    assert (r.tracked[0].class_id, r.tracked[0].class_name) == (1, "copulation")


def test_recovery_pass_finds_a_track_beyond_the_primary_gate():
    config = TrackerConfig(association=AssociationConfig(max_distance=30.0))
    t = create_tracker(config)
    for f in range(5):
        t.update([det(100, 100)], f)
    for f in range(5, 8):
        t.update([], f)
    r = t.update([det(150, 100)], 8)  # 50 px: outside 30, inside the widened gate
    assert ids(r) == [0]
    no_recovery = create_tracker(dataclasses.replace(
        config, association=dataclasses.replace(config.association, recovery=False)))
    for f in range(5):
        no_recovery.update([det(100, 100)], f)
    for f in range(5, 8):
        no_recovery.update([], f)
    r = no_recovery.update([det(150, 100)], 8)
    assert ids(r) == []


def test_determinism():
    import random

    rng = random.Random(1)
    frames = []
    for _ in range(150):
        # Doubtful detections and second classes too: every path through an update.
        frames.append([det(rng.uniform(0, 600), rng.uniform(0, 400), conf=rng.uniform(0.2, 1.0),
                           cls=rng.randint(0, 1))
                       for _ in range(rng.randint(0, 6))])
    runs = []
    for _ in range(2):
        t = lite()
        runs.append([t.update(d, f, f / 30) for f, d in enumerate(frames)])
    assert runs[0] == runs[1]
