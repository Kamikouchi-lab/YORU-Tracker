"""Video, batch and live runtimes -- and that they all track identically (§39).

The same ordered detections must give the same tracking decisions whether
they come from a video, a stored detections file, or a live source.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from yoru_tracker.core import TrackerConfig, create_tracker
from yoru_tracker.runtime.batch import BatchItem, failure_report, find_videos, run_batch
from yoru_tracker.runtime.detections_file import load_detections
from yoru_tracker.runtime.frames import FrameDetections, track_frames
from yoru_tracker.runtime.realtime import RealtimeTracking
from yoru_tracker.runtime.video import detect_and_track, export_run, render_video, retrack

from conftest import BlobDetector, det, write_video


# -- video ---------------------------------------------------------------------

def test_video_keeps_two_ids_and_exports(tmp_path, video_file, blob_detector):
    run = detect_and_track(video_file, blob_detector, TrackerConfig())
    assert run.complete and run.processed == 40
    assert run.track_ids() == [0, 1]
    # Blob 0 starts on the left, at y=35.
    for result in run.results:
        by_id = {t.track_id: t for t in result.tracked if not t.predicted}
        assert by_id[0].cy == pytest.approx(35, abs=2)
        assert by_id[1].cy == pytest.approx(85, abs=2)

    written = export_run(run, tmp_path / "out")
    assert set(written) == {"tracks_csv", "detections_csv", "metadata"}
    with open(written["tracks_csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80
    assert rows[0]["class_name"] == "blob"


def test_retrack_reuses_detections_and_is_identical_with_the_same_settings(video_file):
    detector = BlobDetector()
    run = detect_and_track(video_file, detector, TrackerConfig())
    calls = detector.calls
    again = retrack(run, TrackerConfig())
    assert detector.calls == calls          # the detector did not run again
    assert again.results == run.results


def test_video_and_stored_detections_track_identically(tmp_path, video_file, blob_detector):
    run = detect_and_track(video_file, blob_detector, TrackerConfig())
    written = export_run(run, tmp_path)
    _, frames = load_detections(written["detections_csv"])
    assert track_frames(create_tracker(), frames) == run.results


def test_stopping_early_leaves_a_valid_shorter_run(video_file, blob_detector):
    run = detect_and_track(video_file, blob_detector, TrackerConfig(),
                           should_stop=lambda: blob_detector.calls >= 10)
    assert not run.complete and run.processed == 10


def test_render_writes_a_video(tmp_path, video_file, blob_detector):
    import cv2

    run = detect_and_track(video_file, blob_detector, TrackerConfig())
    out = render_video(run, tmp_path / "r.mp4")
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    cap.release()


# -- batch ---------------------------------------------------------------------

def test_batch_reports_failures_and_carries_on(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    write_video(folder / "a.avi", frames=15)
    (folder / "broken.mp4").write_bytes(b"this is not a video")
    write_video(folder / "c.avi", frames=12)
    videos = find_videos(folder)
    assert [p.split("\\")[-1].split("/")[-1] for p in videos] == ["a.avi", "broken.mp4", "c.avi"]
    seen = []
    items = run_batch([BatchItem(v) for v in videos], BlobDetector(), TrackerConfig(),
                      tmp_path / "out", on_update=lambda i: seen.append((i.name, i.status)))
    status = {i.name: i.status for i in items}
    assert status == {"a.avi": "done", "broken.mp4": "failed", "c.avi": "done"}
    assert items[0].track_ids == 2 and items[0].frames == 15
    assert "Could not open video" in items[1].error
    report = failure_report(items)
    assert "1 file(s) failed" in report and "broken.mp4" in report
    assert (tmp_path / "out" / "a_tracks.csv").is_file()
    assert ("a.avi", "running") in seen


# -- live ----------------------------------------------------------------------

class CountingSource:
    """Frames whose first pixel encodes their index; slow enough to be processed."""

    def __init__(self, n=60, interval=0.004):
        self.n = n
        self.i = 0
        self.interval = interval
        self.closed = False

    def read(self):
        if self.i >= self.n:
            time.sleep(0.01)
            raise EOFError("done")
        time.sleep(self.interval)
        frame = np.zeros((8, 8, 3), np.uint8)
        frame[0, 0, 0] = self.i % 256
        frame[0, 0, 1] = self.i // 256
        self.i += 1
        return frame

    def close(self):
        self.closed = True


class ScriptedDetector:
    """Detections looked up by the index the frame carries."""

    names = {0: "fly"}

    def __init__(self, script, delay=0.0):
        self.script = script
        self.delay = delay

    def detect(self, frame):
        if self.delay:
            time.sleep(self.delay)
        index = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        return list(self.script[index])


def _script(n):
    return [(det(10 + 2 * i, 10), det(300 - 2 * i, 200)) for i in range(n)]


def _run_live(script, *, delay, timeout=10.0):
    snapshots = []
    source = CountingSource(len(script))
    rt = RealtimeTracking(lambda: source, lambda: ScriptedDetector(script, delay),
                          TrackerConfig(), on_result=snapshots.append)
    rt.start()
    deadline = time.time() + timeout
    while rt.running and time.time() < deadline:
        time.sleep(0.02)
    rt.stop()
    return rt, source, snapshots


def test_live_results_equal_offline_tracking_of_the_same_frames():
    script = _script(60)
    # A detector slower than the camera, so frames are dropped on the way.
    rt, source, snapshots = _run_live(script, delay=0.009)
    assert isinstance(rt.error, EOFError)          # the source ran out: reported, not hidden
    assert source.closed
    assert len(snapshots) > 10
    assert rt.stats().dropped_frames > 0
    # Replay exactly the frames the live run processed, under their IDs.
    frames = [FrameDetections(s.frame_id, s.result.timestamp, s.detections) for s in snapshots]
    offline = track_frames(create_tracker(), frames)
    assert offline == [s.result for s in snapshots]
    # Frame IDs count camera frames, so drops show as gaps the motion model sees.
    ids = [s.frame_id for s in snapshots]
    assert ids == sorted(ids) and ids[-1] - ids[0] + 1 > len(ids)


def test_live_snapshots_carry_latency_stamps():
    rt, _, snapshots = _run_live(_script(30), delay=0.0)
    s = snapshots[-1]
    assert s.captured_at <= s.detect_started_at <= s.detected_at <= s.tracked_at
    assert s.tracker_ms >= 0 and s.detector_ms >= 0
    stats = rt.stats()
    assert stats.processed_frames == len(snapshots)


def test_live_reset_restarts_ids_and_bumps_the_generation():
    script = [(det(10, 10),)] * 400
    source = CountingSource(400, interval=0.002)
    snapshots = []
    rt = RealtimeTracking(lambda: source, lambda: ScriptedDetector(script), TrackerConfig(),
                          on_result=snapshots.append)
    rt.start()
    while len(snapshots) < 20 and rt.running:
        time.sleep(0.01)
    rt.reset_tracker(TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    max_age: 3\n"))
    seen = len(snapshots)
    while len(snapshots) < seen + 20 and rt.running:
        time.sleep(0.01)
    rt.stop()
    after = [s for s in snapshots if s.generation == 1]
    assert after, "no frame processed after the reset"
    assert after[0].result.events[0].kind.value == "reset" or after[0].result.tracked[0].track_id == 0
    assert rt.tracker_config.lifecycle.max_age == 3


def test_live_recording_writes_csv_and_metadata(tmp_path):
    script = _script(80)
    source = CountingSource(80, interval=0.003)
    rt = RealtimeTracking(lambda: source, lambda: ScriptedDetector(script), TrackerConfig(),
                          source_description="test source")
    rt.start()
    rt.start_recording(tmp_path / "live_tracks.csv")
    while rt.running:
        time.sleep(0.02)
    rt.stop()
    rows = list(csv.DictReader(open(tmp_path / "live_tracks.csv", newline="")))
    assert rows and {r["track_id"] for r in rows} <= {"0", "1"}
    meta = (tmp_path / "live_tracks.json").read_text(encoding="utf-8")
    assert '"kind": "live"' in meta and "test source" in meta


def test_a_failing_detector_stops_the_pipeline_with_its_error():
    class Broken:
        names = {}

        def detect(self, frame):
            raise RuntimeError("CUDA out of memory")

    rt = RealtimeTracking(lambda: CountingSource(50), lambda: Broken(), TrackerConfig())
    rt.start()
    deadline = time.time() + 5
    while rt.running and time.time() < deadline:
        time.sleep(0.01)
    rt.stop()
    assert "CUDA out of memory" in str(rt.error)
    assert rt.phase == "failed"
