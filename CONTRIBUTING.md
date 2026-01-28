# Contributing to CameraKit

Thanks for helping improve CameraKit. This document explains the architecture, data flow, and the current roadmap so changes stay aligned with the tool's direction.

## Architecture overview

CameraKit is a small CLI-first tool with a Python API. It is structured into a few key modules:

- `camerakit/cli.py`: unified CLI entry point (`camerakit ...`).
- `camerakit/calibration.py`: calibration implementation.
- `camerakit/capture.py`: synchronized capture implementation.
- `camerakit/devices.py`: device discovery implementation.
- `camerakit/report.py`: calibration summary implementation.
- `camerakit/utils/common.py`: device discovery utilities and logging setup.
- `camerakit/utils/calibration.py`: calibration pipeline (intrinsics + extrinsics) and UI helpers.
- `camerakit/utils/calib.py`: low-level helpers for frame extraction and TOML output.
- `camerakit/utils/sync.py`: threaded capture for multi-camera video reads.
- `camerakit/core/calibration_file.py`: reader for calibration TOML.
- `camerakit/core/calibration_data.py`: per-camera calibration data model.

## Data flow

1. A project directory contains `Config.toml` and a `calibration/` folder.
2. `camerakit calibrate` reads `Config.toml`, finds calibration media, and runs the solver.
3. Outputs are written to `calibration/Calib_<method>.toml`.
4. `camerakit report` reads the TOML and prints a per-camera summary.
5. `camerakit capture` records synchronized video sessions under `data/Session_*/Trial_*`.

Project layout:

```
project/
  Config.toml
  calibration/
    intrinsics/
      cam_00/ intrinsics.mp4
    extrinsics/
      cam_00/ extrinsics.png
```

## Logging

All logging should go through the CameraKit logger (`logging.getLogger("camerakit")`).
Use `setup_logging()` from `camerakit/utils/common.py` when you need file output.

## Roadmap

The roadmap is mirrored on the website. If you add or remove items here, update the page too.

- Calibration capture pipeline: record intrinsics/extrinsics clips from the tool.
- Calibration UX hardening: improve manual annotation flow, window handling, and edge-case messaging.
- Sync reports + timestamps: per-frame timestamps and drift summaries.
- Pairwise extrinsics chaining: calibrate with partial target visibility and global pose-graph solve.
- Hardware-grade sync: LED/GPIO triggering and tighter alignment.
- Exports & visualizers: TRACX export formats, multi-view previews, pose visualization.
- Expanded device support: improved USB discovery and future IP camera backends.

## Half-implemented or fragile areas

- Manual annotation UX in calibration has TODOs (window close handling, repeat "C" behavior).
- Extrinsics use a single image/frame per camera; no multi-view chaining yet.
- Synchronization is best-effort (threaded reads) and lacks timestamps/drift analysis.
- macOS device discovery is based on `system_profiler` and may be incomplete.

## Guidelines

- Keep configuration and output formats TOML-based.
- Avoid adding heavy dependencies unless required by the roadmap.
- Update README and the website when CLI behavior changes.
