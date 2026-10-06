"""Tracks CSV, the metadata beside it, and stored detections in every layout."""

from __future__ import annotations

import csv

import pytest
from yoru.libs.detector_base import DETECTION_COLUMNS

from yoru_tracker.core import TrackerConfig, create_tracker
from yoru_tracker.export import (
    TRACK_COLUMNS,
    TrackCsvWriter,
    build_metadata,
    config_from_metadata,
    read_metadata,
    write_metadata,
    write_tracks_csv,
)
from yoru_tracker.runtime.detections_file import load_detections, write_detections_csv
from yoru_tracker.runtime.frames import FrameDetections, track_frames

from conftest import det


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _results():
    t = create_tracker()
    frames = [[det(10, 10), det(100, 10)], [det(12, 10), det(102, 10)], [det(14, 10)], []]
    return [t.update(d, i, i / 30) for i, d in enumerate(frames)]


def test_columns_start_with_yoru_detection_columns():
    assert TRACK_COLUMNS[:len(DETECTION_COLUMNS)] == DETECTION_COLUMNS
    for name in ("frame_id", "track_id", "track_state", "predicted", "vx", "vy",
                 "track_confidence", "association_cost"):
        assert name in TRACK_COLUMNS


def test_predicted_rows_are_left_out_unless_asked_for(tmp_path):
    results = _results()
    n = write_tracks_csv(tmp_path / "a.csv", results)
    rows = _rows(tmp_path / "a.csv")
    assert n == len(rows) == 5
    assert {r["predicted"] for r in rows} == {"0"}
    n = write_tracks_csv(tmp_path / "b.csv", results, include_predicted=True)
    rows = _rows(tmp_path / "b.csv")
    predicted = [r for r in rows if r["predicted"] == "1"]
    assert len(predicted) == 3   # track 1 in frame 2, both tracks in frame 3
    assert all(r["confidence"] == "" and r["track_state"] == "lost" for r in predicted)


def test_rows_carry_the_timestamp_in_total_time(tmp_path):
    write_tracks_csv(tmp_path / "a.csv", _results())
    row = _rows(tmp_path / "a.csv")[2]
    assert float(row["total_time"]) == pytest.approx(1 / 30)
    assert row["frame_id"] == "1" and row["track_id"] == "0"


def test_metadata_reproduces_the_configuration(tmp_path):
    config = TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    max_age: 4\n")
    tracker = create_tracker(config)
    meta = build_metadata(tracker, detector={"model_path": "m.pt"}, source={"kind": "test"})
    path = write_metadata(tmp_path / "x.json", meta)
    loaded = read_metadata(path)
    assert config_from_metadata(loaded) == config
    assert loaded["tracker"]["name"] == "YORU Lite"
    assert loaded["tracker"]["tracker_api_version"] == 1
    assert loaded["software"]["yoru"]
    assert loaded["track_columns"] == list(TRACK_COLUMNS)


def test_incremental_writer(tmp_path):
    with TrackCsvWriter(tmp_path / "live.csv") as writer:
        for r in _results():
            writer.write(r)
            writer.flush()
    assert len(_rows(tmp_path / "live.csv")) == 5


# -- stored detections --------------------------------------------------------

def test_detections_round_trip_keeps_empty_frames(tmp_path):
    frames = [FrameDetections(0, 0.0, (det(1, 2), det(30, 40, cls=1, name="b"))),
              FrameDetections(1, 0.5, ()),
              FrameDetections(3, 1.5, (det(5, 6, angle=0.3),))]
    write_detections_csv(tmp_path / "d.csv", frames)
    layout, loaded = load_detections(tmp_path / "d.csv")
    assert layout == "yoru-tracker"
    assert loaded == frames


def test_yoru_analysis_csv_fills_frames_without_detections(tmp_path):
    path = tmp_path / "movie.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "x_center", "y_center", "w", "h", "angle",
                    "confidence", "class", "class_name"])
        w.writerow([0, 0, 0, 10, 10, 5, 5, 10, 10, 0, 0.9, 0, "fly"])
        w.writerow([2, 2, 0, 12, 10, 7, 5, 10, 10, 0, 0.8, 0, "fly"])
    layout, frames = load_detections(path)
    assert layout == "yoru-analysis"
    assert [f.frame_id for f in frames] == [0, 1, 2]
    assert frames[1].detections == ()


def test_yoru_realtime_detect_csv_drops_repeated_results(tmp_path):
    """YORU writes the latest detections with every video frame until the next arrive."""
    detect = tmp_path / "run_detect.csv"
    log = tmp_path / "run_log.csv"
    a = [0, 0, 10, 10, 0.9, 0, "fly", 0.10, 5, 5, 10, 10, 0]
    b = [50, 0, 60, 10, 0.8, 0, "fly", 0.10, 55, 5, 10, 10, 0]
    c = [1, 0, 11, 10, 0.9, 0, "fly", 0.30, 6, 5, 10, 10, 0]
    with open(detect, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DETECTION_COLUMNS)
        w.writerows([a, b, a, b, a, b, c])   # frame at t=0.10 written three times
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "total_time"])
        w.writerows([[0, 0.0], [1, 0.1], [2, 0.2], [3, 0.3]])
    layout, frames = load_detections(detect)
    assert layout == "yoru-realtime"
    assert [(f.frame_id, len(f.detections)) for f in frames] == [(1, 2), (3, 1)]


def test_a_byte_order_mark_from_a_spreadsheet_is_skipped(tmp_path):
    frames = [FrameDetections(0, 0.0, (det(1, 2),)), FrameDetections(1, 0.5, ())]
    write_detections_csv(tmp_path / "d.csv", frames)
    text = (tmp_path / "d.csv").read_text(encoding="utf-8")
    (tmp_path / "saved.csv").write_text(text, encoding="utf-8-sig")
    assert load_detections(tmp_path / "saved.csv") == ("yoru-tracker", frames)


def test_unknown_csv_is_refused(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="not a detections file"):
        load_detections(path)


def test_tracking_stored_detections_equals_tracking_them_directly(tmp_path):
    frames = [FrameDetections(i, i / 30, (det(10 + 3 * i, 10), det(200 - 2 * i, 50)))
              for i in range(30)]
    direct = track_frames(create_tracker(), frames)
    write_detections_csv(tmp_path / "d.csv", frames)
    _, loaded = load_detections(tmp_path / "d.csv")
    assert track_frames(create_tracker(), loaded) == direct
