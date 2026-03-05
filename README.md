<p align="center">
  <img src="docs/_static/camerakit-header.svg" alt="CameraKit" width="100%" />
</p>

# CameraKit

CameraKit is a CLI-first Python package for camera calibration and synchronized capture.

Current package version: **v2.0.0**

## Stable Features

- Unified CLI with subcommands: `devices`, `init`, `calibrate`, `capture`, `report`
- Camera discovery with supported resolution/FPS/codec probing
- Checkerboard-based calibration (intrinsics + optional extrinsics)
- Optional fisheye intrinsics/extrinsics path in calibration workflow
- Calibration TOML outputs with separate per-camera errors:
  - `intrinsics_error_px`
  - `extrinsics_error_px`
- Synchronized software capture with per-trial recordings
- Calibration summary reporting via CLI

## Installation

```bash
pip install camerakit
```

For capture workflows, use GUI-enabled OpenCV (`opencv-python`).
`opencv-python-headless` intentionally disables the interactive capture command.

## Quickstart

```bash
# 1) Create a project
camerakit init --path /path/to/project --cameras 2

# 2) List devices
camerakit devices --max-cameras 6

# 3) Run calibration
camerakit calibrate --config /path/to/project

# 4) Summarize calibration output
camerakit report --input /path/to/project/calibration/Calib_board_outer.toml

# 5) Record synchronized videos
camerakit capture --data-dir data --max-cameras 6
```

Project layout example:

```text
project/
  Config.toml
  calibration/
    intrinsics/
      cam_00/ intrinsics.mp4
      cam_01/ intrinsics.mp4
    extrinsics/
      cam_00/ extrinsics.png
      cam_01/ extrinsics.png
```

## Python API

```python
from camerakit.calibration import run_calibration
from camerakit.utils import CameraEnumerator

run_calibration("/path/to/project")
cameras = CameraEnumerator(max_cameras=6).list()
```

## Documentation

Detailed Sphinx docs are in [`docs/`](docs/). Build locally with:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

Open `docs/_build/html/index.html`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture notes, logging conventions,
documentation workflow, and roadmap priorities.

## Release Notes

See [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Third-Party Notices

Some portions of this project include code derived from third-party sources under
their original licenses. See:

- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [licenses/BSD-3-Clause-pose2sim.txt](licenses/BSD-3-Clause-pose2sim.txt)

## License

CameraKit is licensed under the MIT License. See [LICENSE](LICENSE), plus
third-party notices and licenses listed above.
