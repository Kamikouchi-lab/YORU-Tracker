# YORU Tracker — Claude Code project guide

YORU Tracker is the tracking sister application of YORU (`../YORU`): persistent
IDs, trajectories and tracking export on top of YORU's detectors. Windows,
Python 3.10, DearPyGui. The architecture and its boundaries are in
`../YORU_YORU-Tracker_Architecture_Design.md`; read it before moving a feature
between the two repositories.

## Logs — check these first when something fails

`~/.yoru/logs/yoru_tracker.log` (beside YORU's `yoru.log`; relocated by
`YORU_HOME`). GUI errors, failed batch files, tracker resets and config loads
are there with full tracebacks.

## The boundary with YORU

- YORU-Tracker depends on YORU; YORU never imports `yoru_tracker`.
- Use only the YORU names in YORU's `docs/external_api.md`.
  `tests/test_yoru_boundary.py` enforces the list; if a new YORU primitive is
  genuinely needed, add it to YORU's external API (and YORU's
  `tests/test_public_api.py`) first.
- No detector code here: detection goes through `yoru.libs.plugins.get_detector`
  (`runtime/detection.py`). Never import ultralytics/torch directly.
- Do not subclass or copy YORU's GUI screens; build from `yoru.gui_base`,
  `yoru.gui_layout.GuiSession` and `yoru.libs.gui_error`.

## Independent implementation

Tracking concepts (Kalman, Hungarian, IoU, gating, multi-pass association)
are general. AMADEUS or any other project's source must not be copied,
translated, or used as a template — no borrowed names, file layout, stage
numbering, thresholds or comments.

## Rules the tracker keeps

- Same detections + frame IDs + timestamps + config → equal `TrackingResult`s.
  No randomness, no wall-clock time inside a result.
- Never fail silently: no fallback from one tracker mode to another, no
  silent ID reset (frames out of order raise), no config key silently ignored
  (`ConfigError`).
- Realtime, video, batch and stored detections all go through
  `TrackerBase.update` with the same tracker — no second tracking path.
- Public result types stay plain frozen dataclasses; internals (Kalman state,
  `TrackRecord`) never leak into them.
- `TRACKER_API_VERSION` changes only with an incompatible contract change.

## Changing tracking behaviour

Run `uv run yoru-tracker bench` before and after, and report ID switches,
fragmentation, IDF1 and ms/frame per scenario. `tests/test_evaluation.py`
fails if any scenario gets worse than `tests/data/lite_reference.json`; after
an intended improvement regenerate it with
`uv run yoru-tracker bench --json tests/data/lite_reference.json`.
Do not tune on one video; add a scenario to `evaluation/scenarios.py` for a
new failure case instead.

## Environment

`uv sync` creates `.venv` with YORU installed editable from `../YORU`.
Run with `.venv/Scripts/python.exe` or `uv run`; bare `python` on this machine
is a broken Windows Store stub. Tests: `uv run pytest` (GUI window tests:
`YORU_TRACKER_GUI_TESTS=1`).

CI (`.github/workflows/ci.yml`, Windows) checks YORU out beside the
repository at `vars.YORU_REF` (default `v2.0.0-beta.4`) and runs
`uv sync --locked`: when YORU's dependencies change, run `uv lock` here, or CI
fails. Keep tests free of sleep-based timing -- a Windows sleep lasts as long
as the system timer resolution allows (1 to 15.6 ms).
