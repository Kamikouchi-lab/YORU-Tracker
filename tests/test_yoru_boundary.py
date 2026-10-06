"""The boundary with YORU, checked in the source.

* YORU-Tracker uses only the YORU names YORU documents as its external API
  (YORU's ``docs/external_api.md``), plus the one function the baseline
  tracker exists to wrap.  Anything else would break silently the next time
  YORU refactors its internals.
* YORU never imports YORU-Tracker.
* Detection goes through YORU's detector registry -- YORU-Tracker carries no
  detector code of its own.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "yoru_tracker"

ALLOWED = {
    "yoru": {"__version__"},
    "yoru.libs.plugins": {"get_detector", "list_detector_backends",
                          "DEFAULT_CONF_THRESH", "DEFAULT_IOU_THRESH"},
    "yoru.libs.detector_base": {"DETECTION_COLUMNS", "obb_of", "detection_row", "DetectorBase"},
    "yoru.libs.obb": {"normalize_angle", "obb_corners", "corners_to_obb", "obb_to_aabb",
                      "aabb_to_obb", "point_in_obb", "rotate_points", "is_rotated", "box_axes"},
    "yoru.libs.camera": {"open_camera"},
    "yoru.gui_base": {"apply_default_theme", "process_frame", "frame_to_data_rgb",
                      "frame_to_data_rgba"},
    "yoru.gui_layout": {"GuiSession"},
    "yoru.libs.gui_error": {"GuiErrorMixin"},
    "yoru.libs.user_paths": {"get_yoru_home", "get_log_dir", "get_log_file", "setup_logging",
                             "log_exception", "log_message"},
    # The evaluation baseline is YORU's own matcher, deliberately not a copy.
    "yoru.libs.analysis": {"match_to_previous"},
}


def _yoru_imports():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "yoru" or node.module.startswith("yoru.")):
                for alias in node.names:
                    yield path, node.lineno, node.module, alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "yoru" or alias.name.startswith("yoru."):
                        yield path, node.lineno, alias.name, None


def test_only_documented_yoru_names_are_used():
    bad = []
    for path, line, module, name in _yoru_imports():
        if name is None:
            if module != "yoru":
                bad.append(f"{path.name}:{line}: import {module}")
            continue
        if name not in ALLOWED.get(module, set()):
            bad.append(f"{path.name}:{line}: from {module} import {name}")
    assert not bad, "Undocumented YORU internals used:\n" + "\n".join(bad)


def test_yoru_does_not_import_the_tracker():
    import yoru

    root = Path(yoru.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "yoru_tracker" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, offenders


def test_no_detector_code_lives_here():
    forbidden = {"ultralytics", "torch", "torchvision", "onnxruntime"}
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in forbidden:
                    found.append(f"{path.name}:{node.lineno}: {name}")
    assert not found, "Detector frameworks imported directly:\n" + "\n".join(found)


def _yoru_model():
    explicit = os.environ.get("YORU_TRACKER_TEST_MODEL")
    if explicit:
        return Path(explicit)
    import yoru

    candidate = Path(yoru.__file__).resolve().parents[1] / "yolo11n.pt"
    return candidate if candidate.is_file() else None


@pytest.mark.slow
def test_a_real_yoru_model_loads_and_detects_through_the_adapter():
    model = _yoru_model()
    if model is None:
        pytest.skip("no model: set YORU_TRACKER_TEST_MODEL or keep yolo11n.pt in the YORU checkout")
    import numpy as np

    from yoru_tracker.runtime.detection import DetectorConfig, YoruDetector

    detector = YoruDetector.load(DetectorConfig(model_path=str(model)))
    assert detector.names
    frame = np.full((320, 320, 3), 128, np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    for d in detections:
        assert d.is_valid()
