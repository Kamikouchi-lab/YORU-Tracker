"""Video, batch and live runtimes -- and that they all track identically (§39).

The same ordered detections must give the same tracking decisions whether
they come from a video, a stored detections file, or a live source.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from yoru.libs.detector_base import DETECTION_COLUMNS, detection_row

from yoru_tracker.core import TrackerConfig, TrackerUnavailableError, create_tracker
from yoru_tracker.export import config_from_metadata, read_metadata
from yoru_tracker.runtime.batch import (
    BatchItem,
    failure_report,
    find_videos,
    output_names,
    run_batch,
)
from yoru_tracker.runtime.detections_file import (
    align_to_video,
    load_detections,
    write_detections_csv,
)
from yoru_tracker.runtime.frames import FrameDetections, track_frames
from yoru_tracker.runtime.realtime import RealtimeTracking
from yoru_tracker.runtime.sources import VideoFileSource, VideoInfo, frame_of
from yoru_tracker.runtime.video import (
    VideoTracking,
    detect_and_track,
    export_run,
    render_video,
    retrack,
)

from conftest import BlobDetector, det, read_all, write_video


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -- video sources ---------------------------------------------------------------

def test_frame_of_finds_a_frame_by_its_time():
    times = [0.0, 40.0, 80.0, 120.0]
    assert [frame_of(times, ms) for ms in (0.0, 80.0, 81.0, 120.0)] == [0, 2, 2, 3]
    assert frame_of(times, 60.0) is None              # between two frames
    assert frame_of(times, 400.0) == 4                # past every recorded frame
    assert frame_of([], 40.0) is None and frame_of(times, None) is None


def test_a_seek_that_lands_elsewhere_is_corrected_by_the_frame_times(tmp_path, late_seeks):
    path = write_video(tmp_path / "v.avi", frames=120)
    frames, times = read_all(path)
    with VideoFileSource(path) as source:
        source.frame_at(100)
        # Unchecked, a seek back shows a frame three later than asked for.
        assert np.array_equal(source.frame_at(10), frames[13])
        for target in (10, 90, 0, 119, 60, 61, 70):
            assert np.array_equal(source.frame_at(target, times), frames[target]), target


def test_the_tracking_pass_records_every_frames_time(video_file, blob_detector):
    run = detect_and_track(video_file, blob_detector, TrackerConfig())
    assert len(run.frame_ms) == run.processed == 40
    assert all(b > a for a, b in zip(run.frame_ms, run.frame_ms[1:]))
    assert retrack(run, TrackerConfig()).frame_ms is run.frame_ms


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


# -- stored detections over their video ----------------------------------------

def _yoru_realtime_recording(tmp_path, every=2, first=1, n=60):
    """A YORU real-time recording whose detector took every *every*-th frame.

    YORU writes the latest detections again with every video frame until the
    next ones are ready, stamped with the capture time of the frame they came
    from; the log maps each video frame to its capture time.
    """
    paths = [lambda t: (20 + 3 * t, 35), lambda t: (300 - 3 * t, 85)]
    detect, log = tmp_path / "rec_detect.csv", tmp_path / "rec_log.csv"
    with open(detect, "w", newline="") as fd, open(log, "w", newline="") as fl:
        dw, lw = csv.writer(fd), csv.writer(fl)
        dw.writerow(DETECTION_COLUMNS)
        lw.writerow(["frame", "total_time"])
        rows = []
        for i in range(n):
            t = i / 30.0
            if i >= first and (i - first) % every == 0:
                rows = []
                for path in paths:
                    d = det(*path(i))
                    rows.append(detection_row({"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                                               "conf": 0.9, "class_id": 0, "class_name": "fly"},
                                              t))
            lw.writerow([i, t])
            dw.writerows(rows)
    return detect


def test_a_yoru_realtime_recording_lined_up_with_its_video_tracks_as_the_file_does(tmp_path):
    layout, frames = load_detections(_yoru_realtime_recording(tmp_path))
    assert layout == "yoru-realtime" and [f.frame_id for f in frames[:3]] == [1, 3, 5]
    direct = track_frames(create_tracker(), frames)
    aligned = align_to_video(layout, frames, 30.0)
    assert [f.frame_id for f in aligned] == list(range(frames[-1].frame_id + 1))
    # Frames the detector never ran on are not empty frames: nothing was missed.
    assert [f.observed for f in aligned[:4]] == [False, True, False, True]
    results = track_frames(create_tracker(), aligned)
    assert [r for r, f in zip(results, aligned) if f.observed] == direct
    assert {t.track_id for r in results for t in r.tracked} == {0, 1}
    assert not any(e.kind.value == "lost" for r in results for e in r.events)


def test_in_yoru_analysis_tables_a_missing_frame_had_no_detections():
    frames = [FrameDetections(2, None, (det(10, 10),))]
    assert [f.observed for f in align_to_video("yoru-analysis", frames, 30.0)] == [True] * 3
    assert [f.observed for f in align_to_video("yoru-tracker", frames, 30.0)] == [False, False, True]


def test_unobserved_frames_stay_gaps_through_export_and_display(tmp_path):
    frames = [FrameDetections(0, 0.0, (det(10, 10),)),
              FrameDetections(1, 1 / 30, (), observed=False),
              FrameDetections(2, 2 / 30, (det(14, 10),))]
    results = track_frames(create_tracker(), frames)
    assert results[1].tracked == () and results[2].tracked[0].association_cost is not None
    write_detections_csv(tmp_path / "d.csv", frames)
    _, loaded = load_detections(tmp_path / "d.csv")
    assert [f.frame_id for f in loaded] == [0, 2]          # still a gap, not an empty frame
    run = VideoTracking(VideoInfo("v.avi", 30.0, 3, 64, 48), TrackerConfig(),
                        frames=frames, results=results)
    assert [run.shown_index(i) for i in range(3)] == [0, 0, 2]


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


def test_output_names_never_collide(tmp_path):
    root = tmp_path / "videos"
    names = output_names([str(root / "day1" / "fly.avi"), str(root / "day2" / "fly.avi"),
                          str(root / "day2" / "fly.mp4"), str(root / "day2" / "other.avi")])
    assert [n.as_posix() for n in names] == ["day1/fly", "day2/fly_avi", "day2/fly_mp4",
                                             "day2/other"]
    # A flat folder keeps the plain names.
    assert [n.as_posix() for n in output_names([str(root / "a.avi"), str(root / "b.avi")])] == [
        "a", "b"]


def test_batch_videos_with_one_name_do_not_overwrite_each_other(tmp_path):
    folder = tmp_path / "videos"
    for sub, frames in (("day1", 15), ("day2", 12)):
        (folder / sub).mkdir(parents=True)
        write_video(folder / sub / "fly.avi", frames=frames)
    items = run_batch([BatchItem(v) for v in find_videos(folder, recursive=True)],
                      BlobDetector(), TrackerConfig(), tmp_path / "out")
    assert [i.status for i in items] == ["done", "done"]
    out = {i.outputs["tracks_csv"] for i in items}
    assert len(out) == 2
    frames = sorted(len({r["frame_id"] for r in _csv_rows(p)}) for p in out)
    assert frames == [12, 15]


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


class LaggingDetector(ScriptedDetector):
    """Done with a frame only once the camera is three frames further on.

    Slower than the camera by construction, not by sleeping: how long a
    sleep lasts on Windows depends on the timer resolution some other
    program may have set.
    """

    def __init__(self, script, source):
        super().__init__(script)
        self.source = source

    def detect(self, frame):
        index = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        deadline = time.time() + 5
        while self.source.i < min(index + 4, self.source.n) and time.time() < deadline:
            time.sleep(0.001)
        return super().detect(frame)


def _script(n):
    return [(det(10 + 2 * i, 10), det(300 - 2 * i, 200)) for i in range(n)]


def _run_live(script, *, delay=0.0, lag=False, timeout=10.0):
    snapshots = []
    source = CountingSource(len(script))
    detector = LaggingDetector(script, source) if lag else ScriptedDetector(script, delay)
    rt = RealtimeTracking(lambda: source, lambda: detector,
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
    rt, source, snapshots = _run_live(script, lag=True)
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


def _wait_for(snapshots, n, rt, timeout=10.0):
    deadline = time.time() + timeout
    while len(snapshots) < n and rt.running and time.time() < deadline:
        time.sleep(0.01)


def test_a_reset_while_recording_continues_in_a_new_file(tmp_path):
    script = [(det(10, 10), det(300, 200))] * 1000
    snapshots = []
    rt = RealtimeTracking(lambda: CountingSource(1000, interval=0.002),
                          lambda: ScriptedDetector(script), TrackerConfig(),
                          on_result=snapshots.append)
    rt.start()
    _wait_for(snapshots, 5, rt)
    rt.start_recording(tmp_path / "live_tracks.csv")
    _wait_for(snapshots, 25, rt)
    rt.reset_tracker(TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    max_age: 3\n"))
    seen = len(snapshots)
    _wait_for(snapshots, seen + 20, rt)
    rt.stop()
    assert rt.error is None
    first, second = tmp_path / "live_tracks.csv", tmp_path / "live_part2_tracks.csv"
    before = [int(r["frame_id"]) for r in _csv_rows(first)]
    after = [int(r["frame_id"]) for r in _csv_rows(second)]
    assert before and after and max(before) < min(after)
    # Each file says which settings its IDs came from.
    assert config_from_metadata(read_metadata(first.with_suffix(".json"))) == TrackerConfig()
    assert config_from_metadata(
        read_metadata(second.with_suffix(".json"))).lifecycle.max_age == 3


def test_settings_that_cannot_run_live_are_refused_and_the_run_goes_on():
    script = [(det(10, 10),)] * 1000
    snapshots = []
    rt = RealtimeTracking(lambda: CountingSource(1000, interval=0.002),
                          lambda: ScriptedDetector(script), TrackerConfig(),
                          on_result=snapshots.append)
    rt.start()
    _wait_for(snapshots, 5, rt)
    with pytest.raises(TrackerUnavailableError):
        rt.reset_tracker(TrackerConfig(mode="advanced"))
    seen = len(snapshots)
    _wait_for(snapshots, seen + 5, rt)
    rt.stop()
    assert rt.error is None and len(snapshots) >= seen + 5
    assert {s.generation for s in snapshots} == {0}


def test_a_tracker_that_is_not_realtime_capable_is_refused_live(monkeypatch):
    import dataclasses

    from yoru_tracker.tracking.lite_tracker import LiteTracker

    monkeypatch.setattr(LiteTracker, "info",
                        dataclasses.replace(LiteTracker.info, realtime_capable=False))
    with pytest.raises(ValueError, match="realtime"):
        RealtimeTracking(lambda: CountingSource(), lambda: ScriptedDetector([]), TrackerConfig())


def test_a_stopped_pipeline_starts_again_from_id_zero():
    script = [(det(10, 10),)] * 2000
    source = CountingSource(2000, interval=0.002)
    snapshots = []
    rt = RealtimeTracking(lambda: source, lambda: ScriptedDetector(script), TrackerConfig(),
                          on_result=snapshots.append)
    rt.start()
    _wait_for(snapshots, 10, rt)
    rt.stop()
    first_run = len(snapshots)
    rt.start()
    _wait_for(snapshots, first_run + 10, rt)
    rt.stop()
    assert rt.error is None
    again = snapshots[first_run:]
    assert again and {s.generation for s in again} == {1}
    assert again[0].result.events[0].kind.value == "reset"
    assert [t.track_id for t in again[-1].result.tracked] == [0]


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
