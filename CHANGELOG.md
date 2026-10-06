# Changelog

## 0.1.0 — unreleased

First version: the application, the tracker API and the Lite tracker.

- `python -m yoru_tracker` / `yoru-tracker`: start screen with Realtime,
  Video and Batch tracking and the tracker settings; YORU-family logo and theme.
- Tracker API v1: `Detection`, `TrackedDetection`, `Track`, `TrackingResult`,
  `TrackEvent`, `TrackerBase`, `TrackerConfig` (versioned, strictly validated),
  `TrackerInfo`, mode registry with entry-point plugins and API-version checks.
- Lite tracker: constant-velocity Kalman prediction, distance / oriented-IoU /
  long-axis cost, gated Hungarian assignment, primary and recovery passes,
  track lifecycle (tentative, active, lost, retired) with contiguous IDs, and a
  known-population mode.
- Baseline tracker wrapping YORU's `match_to_previous`, for comparison.
- Detection through YORU's detector registry; OBB support throughout.
- Video tracking with timeline, frame stepping, overlays, track inspection,
  re-tracking from stored detections, CSV export and overlay rendering.
- Realtime tracking from a camera or a video played in real time, with latency
  and dropped-frame statistics and live CSV recording.
- Batch tracking with per-file status and a failure report.
- Tracks CSV starting with YORU's detection columns; JSON metadata sufficient
  to reproduce a run; readers for YORU's analysis and real-time detection CSVs.
- Metrics (ID switches, fragmentation, recovery, false new IDs, IDF1, MOTA),
  16 synthetic behavioural scenarios, benchmark and regression gate.
