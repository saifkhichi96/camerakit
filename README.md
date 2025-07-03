# CameraKit: CLI Tools and Python API for Camera Calibration and Synchronized Capture

CameraKit is a Python package that provides command-line tools and an API for camera calibration and synchronized capture. It has the following features:

- Camera calibration using checkerboard images or video
- Synchronized capture from multiple cameras

## Installation

You can install CameraKit using pip:

```bash
pip install camerakit
```

## Usage

### Command-Line Interface

You can use the command-line interface to perform camera calibration and capture images. Here are some examples:

```bash
# Calibrate a camera using checkerboard images
ck-calibrate --images path/to/checkerboard/images --output calibration.toml
```

```bash
# Capture images from multiple cameras
ck-capture --cameras camera1,camera2 --output path/to/output
```

### Python API
You can also use CameraKit as a Python library. Here is an example of how to use it:

```python
from camerakit.calibration import calibrate_camera
calibration_data = calibrate_camera(
    images='path/to/checkerboard/images',
    output='calibration.toml'
)
print(calibration_data)
```

## Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue on GitHub. You can also submit a pull request with your changes.

## License
CameraKit is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
