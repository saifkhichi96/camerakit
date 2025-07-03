import logging
import logging.handlers
import os
import platform
import re
import subprocess
from typing import Dict, List, Optional

import cv2
from easydict import EasyDict as edict


def setup_logging(session_dir: Optional[str] = None, level=logging.INFO):
    """
    Create logging file and stream handlers
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
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers to avoid duplicates
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)

    return logger


def find_supported_resolutions_and_fps(camera_index, codec):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error(f"Failed to open camera {camera_index}.")
        return []

    # Define common resolutions.
    resolutions = [
        (320, 240),  # QVGA
        (640, 480),  # VGA
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1920, 1080),  # Full HD
        (2560, 1440),  # 2K
        (3840, 2160),  # 4K
        (4096, 2160),  # DCI 4K
        (7680, 4320),  # 8K
    ]

    available_options = []
    for width, height in resolutions:
        # Set resolution and mp4v codec.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution and FPS.
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if actual_width == width and actual_height == height:
            available_options.append((width, height, fps))
    cap.release()
    return available_options


def get_camera_hardware_linux(cam_id):
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
        logging.error(f"No camera found with id {cam_id} on Windows.")

    return info


def get_camera_hardware_macos():
    """
    Retrieve camera hardware information for macOS.
    Maps camera indices from OpenCV to their respective Unique IDs.
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
    """
    Retrieves camera hardware information including manufacturer, model_number, and serial_number number.

    Parameters:
        cam_id (int): The camera device ID (e.g., 0 for /dev/video0 on Linux).

    Returns:
        dict: A dictionary containing 'manufacturer', 'model_number', and 'serial_number' keys.
              Returns None if the information cannot be retrieved.
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
        logging.error(f"Error retrieving camera hardware information: {ex}")
        return default


def get_camera_properties(camera_id, codec="mp4v") -> Optional[edict]:
    if isinstance(camera_id, int) or camera_id.isdigit():
        camera = cv2.VideoCapture(int(camera_id))
        if not camera.isOpened():
            return None

        info = {
            "id": camera_id,
            **get_camera_hardware(camera_id),
        }

        name = f"{info['manufacturer']} {info['model_number']}"
        info["name"] = name

        # Get supported settings.
        resolutions = find_supported_resolutions_and_fps(camera_id, codec)
        if not resolutions:
            return None

        info["available_resolutions"] = resolutions
        info["codec"] = codec

        camera.release()
        return edict(info)

    return None


def find_cameras(max_cameras=5, codec="mp4v") -> List[Dict]:
    """Find available cameras.

    Args:
        max_cameras (int): Maximum number of cameras to search for.
    """
    system = platform.system()

    info = []
    if system == "Darwin":  # macOS
        info_darwin = get_camera_hardware_macos()
        for cam in info_darwin:
            resolutions = find_supported_resolutions_and_fps(cam["id"], codec)
            if not resolutions:
                continue

            cam["available_resolutions"] = resolutions
            cam["codec"] = codec
            info.append(cam)

    else:
        for i in range(2 * max_cameras):  # 2x because OpenCV skips indices
            camera = get_camera_properties(i)
            if camera:
                info.append(camera)

    return info
