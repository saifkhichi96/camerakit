import logging
import platform
import re
import subprocess
from typing import Dict, List, Optional

import cv2
from easydict import EasyDict as edict


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
            cam["resolution"] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            cam["frame_rate"] = int(cap.get(cv2.CAP_PROP_FPS))
            cam["manufacturer"] = cam.get("manufacturer", "Unknown")
            cam["model_number"] = cam.get("model_number", f"CAM{index}")
            cam["serial_number"] = cam.get("serial_number", "")
            cam["name"] = (
                f"{cam['manufacturer']} {cam['model_number']} ({cam['resolution'][0]}x{cam['resolution'][1]} @ {cam['frame_rate']} FPS)"
            )
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


def get_camera_properties(camera_id):
    if isinstance(camera_id, int) or camera_id.isdigit():
        camera = cv2.VideoCapture(int(camera_id))
        if not camera.isOpened():
            return None

        info = {
            "id": camera_id,
            "resolution": (
                int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
            "frame_rate": int(camera.get(cv2.CAP_PROP_FPS)),
            **get_camera_hardware(camera_id),
        }

        name = f"{info['manufacturer']} {info['model_number']} ({info['resolution'][0]}x{info['resolution'][1]} @ {info['frame_rate']} FPS)"
        info["name"] = name

        camera.release()
        return edict(info)

    return None


def find_cameras(max_cameras=5) -> List[Dict]:
    """Find available cameras.

    Args:
        max_cameras (int): Maximum number of cameras to search for.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        return get_camera_hardware_macos()

    info = []
    for i in range(max_cameras):
        camera = get_camera_properties(i)
        if camera:
            info.append(camera)
    return info
