import contextlib
import logging
import logging.handlers
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
from easydict import EasyDict as edict

LOGGER_NAME = "camerakit"


def get_logger():
    """Return the shared CameraKit logger instance.

    Returns:
        logging.Logger: Package-level logger.
    """
    return logging.getLogger(LOGGER_NAME)


def configure_opencv_logging(silent: bool = True):
    """Reduce or silence OpenCV internal logging.

    Args:
        silent: If `True`, prefer the most restrictive OpenCV log level.
    """
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    # OpenCV Python API varies across versions; try the available hooks.
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            if silent and hasattr(cv2.utils.logging, "LOG_LEVEL_SILENT"):
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
            elif hasattr(cv2.utils.logging, "LOG_LEVEL_ERROR"):
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        if hasattr(cv2, "LOG_LEVEL_SILENT") and silent:
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
        elif hasattr(cv2, "LOG_LEVEL_ERROR"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass


def setup_logging(session_dir: Optional[str] = None, level=logging.INFO):
    """Configure global logging handlers for a session.

    Args:
        session_dir: Optional directory where `logs.txt` should be written.
        level: Logging level for CameraKit logger and handlers.

    Returns:
        logging.Logger: The package logger configured for the session.
    """
    handlers = [logging.StreamHandler()]
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)
        handlers.append(
            logging.handlers.TimedRotatingFileHandler(
                os.path.join(session_dir, "logs.txt"), when="D", interval=7
            )
        )

    # Configure logging with timestamps and log level.
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False

    # Clear and close existing handlers to avoid duplicates and leaked file handles.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        with contextlib.suppress(Exception):
            existing.close()

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)

    return logger


@dataclass
class CalibrationParams:
    """Container for calibration outputs shared by calculators and converters."""

    ret_intrinsics: list[float]
    ret_extrinsics: list[float]
    C: list[str]
    S: list[Any]
    D: list[Any]
    K: list[Any]
    R: list[Any]
    T: list[Any]


class suppress_stderr:
    """Context manager that temporarily redirects stderr to `/dev/null`."""

    def __enter__(self):
        """Enter context and suppress stderr output.

        Returns:
            suppress_stderr: This context manager instance.
        """
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, *args):
        """Restore stderr stream on context exit.

        Args:
            *args: Standard context manager exception arguments.
        """
        sys.stderr.close()
        sys.stderr = self._stderr


def find_supported_resolutions_and_fps(
    cap,
    codecs=("MJPG", "YUYV", "H264"),
    aspect_ratios=None,
    common_widths=None,
):
    """Probe supported camera settings for common resolutions.

    Args:
        cap: OpenCV `VideoCapture` instance.
        codecs: Candidate codec FourCC strings to probe.
        aspect_ratios: Aspect ratios to probe as `(w, h)` tuples.
        common_widths: Candidate frame widths to probe.

    Returns:
        list[tuple[int, int, float, str]]: Supported `(width, height, fps, codec)` tuples.
    """
    if not cap.isOpened():
        return []

    # Typical resolutions.
    if aspect_ratios is None:
        aspect_ratios = [
            (3, 2),  # 3:2 aspect ratio
            (4, 3),  # 4:3 aspect ratio
            (16, 9),  # 16:9 aspect ratio
        ]
    if common_widths is None:
        common_widths = [
            640,  # VGA
            800,  # SVGA
            960,  # XGA
            1024,  # XGA (1024x768, 1024x576)
            1280,  # HD 720p
            1600,  # UXGA
            1920,  # Full HD 1080p
            # 2560,  # 2K QHD
            # 3840,  # 4K UHD
            # 4096,  # 4K DCI
        ]
    resolutions = set()
    for width in common_widths:
        for ratio in aspect_ratios:
            height = round(width * ratio[1] / ratio[0])
            resolutions.add((width, height))
    resolutions = sorted(resolutions)

    available = []
    for width, height in resolutions:
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if actual_width == width and actual_height == height and fps > 0:
                available.append((width, height, fps, codec))
                break  # One working codec is enough
    return available


def get_camera_hardware_linux(cam_id):
    """Read Linux camera hardware details via `udevadm`.

    Args:
        cam_id: Camera device index.

    Returns:
        dict[str, str]: Manufacturer, model number, and serial number.
    """
    # Prepare the external command to extract serial_number number.
    p = subprocess.Popen(
        f'udevadm info --query=all /dev/video{cam_id} | grep "ID_"',
        stdout=subprocess.PIPE,
        shell=True,
    )

    # Run the command
    (output, err) = p.communicate()

    # Wait for it to finish
    p.status = p.wait()

    # Decode the output
    response = output.decode("utf-8")

    # Parse response to get hardware info
    response = response.split("\n")
    info = {}
    for line in response:
        line = line.strip()
        if line:
            key = line.split("=")[0].split(":")[1].strip()
            value = line.split("=")[1]
            info[key] = value
    return {
        "manufacturer": info["ID_VENDOR"].replace("_", " "),
        "model_number": info["ID_MODEL"].replace("_", " ") + f" {info['ID_MODEL_ID']}",
        "serial_number": info["ID_SERIAL"].replace("_", " "),
    }


def get_camera_hardware_windows(cam_id):
    """Read Windows camera hardware details via WMIC.

    Args:
        cam_id: Camera device index.

    Returns:
        dict[str, str]: Manufacturer, model number, and serial number when available.
    """
    # Use WMIC to get camera info
    # WMIC class Win32_PnPEntity can be used to get device details
    cmd = "wmic path Win32_PnPEntity where \"Service='usbvideo'\" get DeviceID,Manufacturer,Name,PNPDeviceID /format:list"
    output = subprocess.check_output(cmd, shell=True, text=True)

    devices = []
    device = {}
    for line in output.splitlines():
        if line.strip() == "":
            if device:
                devices.append(device)
                device = {}
        else:
            if "=" in line:
                key, value = line.split("=", 1)
                device[key.strip()] = value.strip()
    if device:
        devices.append(device)

    info = {}
    if cam_id < len(devices):
        cam = devices[cam_id]
        info["manufacturer"] = cam.get("Manufacturer", "")
        info["model_number"] = cam.get("Name", "")
        # Attempt to extract serial_number from PNPDeviceID
        pnp_id = cam.get("PNPDeviceID", "")
        serial_match = re.search(
            r"VID_\w+&PID_\w+&(?:REV_\w+&)?(?:SERIALNUMBER_|SERNUM_)([^&]+)",
            pnp_id,
            re.IGNORECASE,
        )
        if serial_match:
            info["serial_number"] = serial_match.group(1)
        else:
            info["serial_number"] = ""
    else:
        get_logger().error(f"No camera found with id {cam_id} on Windows.")

    return info


def get_camera_hardware_macos():
    """Retrieve macOS camera metadata and map it to OpenCV camera indices.

    Returns:
        list[easydict.EasyDict]: Camera metadata dictionaries enriched with OpenCV IDs.
    """
    cmd = "system_profiler SPCameraDataType"
    output = subprocess.check_output(cmd, shell=True, text=True)

    cameras = []
    current_camera = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Model ID:"):
            if current_camera:
                cameras.append(current_camera)
                current_camera = {}
            current_camera["model_number"] = line.split(":", 1)[1].strip()
        elif line.startswith("Vendor ID:"):
            current_camera["manufacturer"] = line.split(":", 1)[1].strip()
        elif line.startswith("Unique ID:"):
            current_camera["serial_number"] = line.split(":", 1)[1].strip()

    if current_camera:
        cameras.append(current_camera)

    # Verify which cameras OpenCV can actually access
    mapped_cameras = []
    index = 0
    for cam in cameras:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cam["id"] = index  # Assign OpenCV-compatible index
            cam["manufacturer"] = cam.get("manufacturer", "Unknown")
            cam["model_number"] = cam.get("model_number", f"CAM{index}")
            cam["serial_number"] = cam.get("serial_number", "")
            cam["name"] = f"{cam['manufacturer']} {cam['model_number']}"
            mapped_cameras.append(edict(cam))
            index += 1
        cap.release()

    return mapped_cameras


def get_camera_hardware(cam_id: int) -> Optional[Dict[str, str]]:
    """Retrieve hardware metadata for a camera index on the active OS.

    Args:
        cam_id: Camera device ID (for example `0` for `/dev/video0` on Linux).

    Returns:
        dict[str, str] | None: Hardware metadata dictionary or `None` for unsupported
        cases on macOS.
    """
    system = platform.system()
    default = {
        "manufacturer": "Unknown",
        "model_number": f"CAM{cam_id}",
        "serial_number": "",
    }

    try:
        if system == "Linux":
            return get_camera_hardware_linux(cam_id)

        if system == "Windows":
            return get_camera_hardware_windows(cam_id)

        if system == "Darwin":
            return None  # macOS cameras are handled separately

        raise Exception(f"Unsupported platform: {system}")
    except Exception as ex:
        get_logger().error(f"Error retrieving camera hardware information: {ex}")
        return default


def get_camera_properties(
    camera_id,
    aspect_ratios=None,
    common_widths=None,
    codecs=("MJPG", "YUYV", "H264"),
) -> Optional[edict]:
    """Open a camera and collect metadata plus supported capture settings.

    Args:
        camera_id: Camera index, as integer or digit-like string.
        aspect_ratios: Aspect ratios to probe as `(w, h)` tuples.
        common_widths: Candidate frame widths to probe.
        codecs: Candidate codec FourCC strings to probe.

    Returns:
        easydict.EasyDict | None: Camera properties if the device is usable,
        otherwise `None`.
    """
    if isinstance(camera_id, int) or camera_id.isdigit():
        camera = cv2.VideoCapture(int(camera_id))
        if not camera.isOpened():
            get_logger().debug(f"Failed to open camera {camera_id}.")
            return None

        info = {
            "id": camera_id,
            **get_camera_hardware(camera_id),
        }

        name = f"{info['manufacturer']} {info['model_number']}"
        info["name"] = name

        # Get supported settings.
        settings = find_supported_resolutions_and_fps(
            camera,
            codecs=codecs,
            aspect_ratios=aspect_ratios,
            common_widths=common_widths,
        )
        if not settings:
            get_logger().debug(
                f"No supported resolutions found for camera {camera_id}."
            )
            camera.release()
            return None

        info["settings"] = settings

        camera.release()
        return edict(info)

    return None
