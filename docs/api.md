# YORU Tracker API (tracker API v1)

Everything here is importable from `yoru_tracker.core`. Importing it loads
neither OpenCV, a detector, nor the GUI.

```python
from yoru_tracker.core import Detection, TrackerConfig, create_tracker

tracker = create_tracker(TrackerConfig())          # mode "lite"
for frame_id, detections in enumerate(frames):     # detections: list[Detection]
    result = tracker.update(detections, frame_id, timestamp=frame_id / fps)
    for t in result.tracked:
        print(frame_id, t.track_id, t.track_state.value, t.cx, t.cy, t.predicted)
```

## The contract

```python
class TrackerBase:
    info: ClassVar[TrackerInfo]
    api_version: ClassVar[int] = TRACKER_API_VERSION   # 1

    def __init__(self, config: TrackerConfig | None = None): ...
    def reset(self) -> None: ...
    def update(self, detections, frame_id: int, timestamp: float | None = None,
               frame=None) -> TrackingResult: ...
    def tracks(self) -> tuple[Track, ...]: ...         # optional, for inspection
```

* **`update`** is called once per frame, in frame order. `frame_id` must
  increase from call to call — an equal or smaller one raises `ValueError`
  rather than resetting anything silently. A gap in `frame_id` means frames
  were skipped (a live camera running faster than the detector) and is a
  longer motion step. `timestamp` is carried into the result untouched.
  `frame` is the image, for trackers whose `info.requires_frame` is true; Lite
  ignores it.
* **`reset`** drops every track; IDs restart at 0. The first result after a
  reset carries a `reset` event.
* **Determinism.** Given the same detections, frame IDs, timestamps and
  configuration, a tracker whose `info.deterministic` is true returns equal
  results — `==` on `TrackingResult`. Realtime, video and stored-detection
  runs therefore make the same decisions for the same input.

The configuration is validated in `TrackerBase.__init__`; no tracker runs on a
configuration that would be refused on load.

## Data types

All are frozen dataclasses of plain values, with `to_dict()` for JSON. No
filter state, history or detector object is reachable from them.

### `Detection`

One detector output, independent of the detector family.

| Field | |
|---|---|
| `x1, y1, x2, y2` | Upright box (pixels). |
| `cx, cy, w, h, angle` | The box itself; rotated for OBB models, `angle` in radians; identical to the upright box with `angle` 0 otherwise. |
| `confidence`, `class_id`, `class_name` | |

Constructors: `from_yoru(dict)` (a YORU `DetectorBase.detect()` item),
`from_row(row)` (YORU's `DETECTION_COLUMNS` order), `from_obb(...)`,
`from_xyxy(...)`, `from_dict(...)`. `is_valid()` is false for non-finite
numbers or an empty box; such detections end up in `TrackingResult.rejected`.

A rectangle's `angle` is defined modulo π. It is an axis, never a heading.

### `TrackingResult`

| Field | |
|---|---|
| `frame_id`, `timestamp` | As passed to `update`. |
| `tracked` | One `TrackedDetection` per confirmed track, sorted by `track_id`. Lost tracks are included with `predicted=True`. |
| `unassigned` | Valid detections not part of a confirmed track this frame (a newcomer still tentative, a detection left over when the population is full). |
| `rejected` | Invalid detections. |
| `events` | `TrackEvent`s of this frame. |

Properties: `active_ids`, `lost_ids`. A result holds no timing: wall-clock
numbers would make equal inputs give unequal results. Runtimes measure
latency around the call instead.

### `TrackedDetection`

| Field | |
|---|---|
| `track_id` | Contiguous from 0; given when a track is confirmed. |
| `track_state` | `TrackState`: `tentative`, `active`, `lost`, `occluded` (reserved). |
| `box` | `(cx, cy, w, h, angle)` this frame — the detection's, or the prediction when `predicted`. |
| `predicted` | True when no detection was matched this frame. |
| `detection` | The matched `Detection`, or `None`. |
| `class_id`, `class_name` | Of the latest matched detection. |
| `track_confidence` | Smoothed detector confidence, discounted while unseen. |
| `velocity` | `(vx, vy)`, pixels per frame. |
| `hits`, `age`, `missed_frames` | Counted in tracker updates. |
| `association_cost` | Cost of this frame's match, for debugging; `None` for new or predicted rows. |

### `Track`

A summary of every track the tracker holds (`tracker.tracks()`), tentative
ones included with `track_id=None`: state, class, box, velocity, counts,
confidence, `first_frame_id`, `last_seen_frame_id`.

### `TrackEvent`

`kind` (`EventKind`: `created`, `lost`, `recovered`, `retired`, `reset`),
`frame_id`, `track_id`, `detail`. With `log_events: true` in the configuration
they are also written to the log.

## Configuration — `TrackerConfig`

Frozen and validated; `to_dict()` / `from_dict()`, `to_yaml()` / `from_yaml()`,
`save(path)` / `load(path)` (YAML, or JSON by extension). On disk:

```yaml
config_version: 1
tracker:
  mode: lite
  lifecycle:   {max_age: 10, min_hits: 2, population: 0}
  association: {distance_weight: 1.0, iou_weight: 1.0, axis_weight: 0.25,
                size_weight: 0.0, max_distance: 100.0, min_iou: 0.0,
                class_aware: false, recovery: true, recovery_gate_scale: 3.0}
  kalman:      {enabled: true, process_noise: 0.2, measurement_noise: 0.1}
  advanced:    {reid: false, occlusion_recovery: false}
  log_events: false
```

Unknown keys, wrong types, out-of-range values, an unsupported
`config_version`, and `advanced.*` options on a mode other than `advanced` are
all `ConfigError`s, whose `problems` list every reason at once. A file may give
only some keys; the rest take their defaults.

`config_version` changes when a key is renamed or changes meaning;
`from_dict` then learns to read the old version.

## Capabilities — `TrackerInfo`

`name`, `version`, `api_version`, `description`, `realtime_capable`,
`requires_frame`, `requires_gpu`, `supports_obb`,
`supports_variable_population`, `deterministic`. The GUI reads these rather
than tracker names — a tracker that is not `realtime_capable` is refused for
live tracking.

## Modes and plugins — `registry`

`create_tracker(config)` builds the tracker `config.mode` names.
`available_modes()` lists the built-ins (`lite`, `baseline`) and installed
plugins. `advanced` is reserved for the planned Advanced tracker: asking for it
is a `TrackerUnavailableError` saying so — never a silent fallback to Lite.

A plugin package registers a `TrackerBase` subclass:

```toml
[project.entry-points."yoru_tracker.backends"]
my_tracker = "my_tracker.plugin:MyTracker"
```

It must set `info` and `api_version`. One written for another tracker API
version is refused with `IncompatibleTrackerError` ("requires tracker API v2;
this application provides v1"); one that fails to import is a
`TrackerUnavailableError` carrying the import error.

## Versioning

`TRACKER_API_VERSION` (now 1) changes when the contract above changes
incompatibly: a method's meaning, a required field, the result types. Adding
an optional field to a result type is not a change of version.
