# CameraKit: CLI Tools and Python API for Camera Calibration and Synchronized Capture

CameraKit is a Python package that provides command-line tools and an API for camera calibration and synchronized capture. It has the following features:

- Camera calibration using checkerboard images or video
- Synchronized capture from multiple cameras (software sync)

Current release: **v2.0.0** (see RELEASE_NOTES.md)

## Installation

You can install CameraKit using pip:

```bash
pip install camerakit
```

## Usage

### Command-Line Interface

You can use the command-line interface to perform camera calibration and capture images. Here are some examples:

```bash
# List available cameras and supported resolutions
camerakit devices --max-cameras 6
```

```bash
# Initialize a calibration project
camerakit init --path /path/to/project
```

```bash
# Run calibration using Config.toml and a calibration folder
camerakit calibrate --config /path/to/project
```

Project layout example:

```text
project/
  Config.toml
  calibration/
    intrinsics/
      cam_00/ intrinsics.mp4
    extrinsics/
      cam_00/ extrinsics.png
```

```bash
# Record synchronized video from multiple cameras
camerakit capture --data-dir data --required-fps 60
```

```bash
# Summarize calibration results
camerakit report --input calibration/Calib_board_outer.toml
```

Calibration outputs include per-camera reprojection errors in the TOML:

```
[cam_00]
intrinsics_error_px = 0.42
extrinsics_error_px = 0.88
```

### Python API
You can also use CameraKit as a Python library. Here is an example of how to use it:

```python
from camerakit.calibration import run_calibration
run_calibration("/path/to/project")
```

## Contributing
Contributions are welcome! See CONTRIBUTING.md for architecture notes and the roadmap. If you find a bug or have a feature request, please open an issue on GitHub. You can also submit a pull request with your changes.

## Release notes
See RELEASE_NOTES.md for version history and breaking changes.

## License
CameraKit is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
