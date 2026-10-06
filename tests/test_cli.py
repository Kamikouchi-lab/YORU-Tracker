"""The command line: ``python -m yoru_tracker`` and its subcommands."""

from __future__ import annotations

import subprocess
import sys

import pytest

from yoru_tracker.app import build_parser, main
from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.runtime.detections_file import write_detections_csv
from yoru_tracker.runtime.frames import FrameDetections

from conftest import det


def test_no_arguments_means_the_gui():
    args = build_parser().parse_args([])
    assert args.command == "gui" and args.func.__name__ == "_cmd_gui"


def test_help_and_version_do_not_load_the_heavy_stack():
    code = (
        "import sys, yoru_tracker.app as a; a.build_parser().parse_args(['track', 'x.csv']); "
        "heavy = [m for m in ('cv2', 'torch', 'dearpygui', 'ultralytics') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([sys.executable, "-m", "yoru_tracker", "--version"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "yoru-tracker 0.1.0" in proc.stdout


def test_config_prints_loadable_defaults(capsys):
    assert main(["config"]) == 0
    assert TrackerConfig.from_yaml(capsys.readouterr().out) == TrackerConfig()


def test_track_a_detections_csv(tmp_path, capsys):
    frames = [FrameDetections(i, i / 30, (det(10 + i, 10), det(200, 10 + i))) for i in range(20)]
    write_detections_csv(tmp_path / "clip_detections.csv", frames)
    assert main(["track", str(tmp_path / "clip_detections.csv"), "--out", str(tmp_path)]) == 0
    assert "20 frames, 2 track IDs" in capsys.readouterr().out
    assert (tmp_path / "clip_tracks.csv").is_file()
    assert (tmp_path / "clip_tracks.json").is_file()


def test_a_bad_config_fails_with_a_message(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tracker:\n  lifecycle:\n    max_agee: 3\n")
    frames = [FrameDetections(0, 0.0, (det(1, 1),))]
    write_detections_csv(tmp_path / "d.csv", frames)
    assert main(["track", str(tmp_path / "d.csv"), "--config", str(bad)]) == 1
    assert "max_agee" in capsys.readouterr().err


def test_a_video_without_a_model_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--model"):
        main(["track", str(tmp_path / "movie.mp4")])


def test_render_with_a_detections_csv_is_refused_not_ignored(tmp_path):
    write_detections_csv(tmp_path / "d.csv", [FrameDetections(0, 0.0, (det(1, 1),))])
    with pytest.raises(SystemExit, match="--render"):
        main(["track", str(tmp_path / "d.csv"), "--render"])
    assert not (tmp_path / "d_tracks.csv").exists()


def test_bench_runs(capsys, tmp_path):
    assert main(["bench", "--scenario", "single", "--json", str(tmp_path / "b.json")]) == 0
    out = capsys.readouterr().out
    assert "single" in out and "baseline" in out
    assert (tmp_path / "b.json").is_file()
