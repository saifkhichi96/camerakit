import logging

import cv2
import numpy as np
from lxml import etree

from ..common import CalibrationParams
from .common import natural_sort_key, rotate_cam, world_to_camera_persp

logger = logging.getLogger(__name__)


def calib_qca_fun(file_to_convert_path, binning_factor=1) -> CalibrationParams:
    """Convert a Qualisys `.qca.txt` calibration file.

    Args:
        file_to_convert_path: Path to the Qualisys calibration file.
        binning_factor: Sensor binning factor used during recording.

    Returns:
        CalibrationParams: Converted calibration arrays and residuals.
    """

    logger.info(f"Converting {file_to_convert_path} to .toml calibration file...")
    residuals_mm, C, S, D, K, R, T = read_qca(file_to_convert_path, binning_factor)

    RT = [world_to_camera_persp(r, t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)

    ret_intrinsics = [0.0] * len(C)
    ret_extrinsics = [float(np.around(r, decimals=6)) for r in residuals_mm]
    return CalibrationParams(
        ret_intrinsics=ret_intrinsics,
        ret_extrinsics=ret_extrinsics,
        C=C,
        S=S,
        D=D,
        K=K,
        R=R,
        T=T,
    )


def read_qca(qca_path, binning_factor):
    """Parse raw Qualisys calibration XML into calibration arrays.

    Args:
        qca_path: Path to `.qca.txt` calibration file.
        binning_factor: Sensor binning factor used during recording.

    Returns:
        tuple[list, list, list, list, list, list, list]: Residuals, camera names,
        image sizes, distortions, intrinsics, rotation matrices, and translations.
    """

    root = etree.parse(qca_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    res = []
    vid_id = []

    # Camera name
    for i, tag in enumerate(root.findall("cameras/camera")):
        ret += [float(tag.attrib.get("avg-residual"))]
        C += [tag.attrib.get("serial")]
        res += [
            int(tag.attrib.get("video_resolution")[:-1])
            if tag.attrib.get("video_resolution") is not None
            else 1080
        ]
        if tag.attrib.get("model") in ("Miqus Video", "Miqus Video UnderWater", "none"):
            vid_id += [i]

    # Image size
    for i, tag in enumerate(root.findall("cameras/camera/fov_video")):
        w = (
            (float(tag.attrib.get("right")) - float(tag.attrib.get("left")) + 1)
            / binning_factor
            / (1080 / res[i])
        )
        h = (
            (float(tag.attrib.get("bottom")) - float(tag.attrib.get("top")) + 1)
            / binning_factor
            / (1080 / res[i])
        )
        S += [[w, h]]

    # Intrinsic parameters: distortion and intrinsic matrix
    for i, tag in enumerate(root.findall("cameras/camera/intrinsic")):
        k1 = float(tag.get("radialDistortion1")) / 64 / binning_factor
        k2 = float(tag.get("radialDistortion2")) / 64 / binning_factor
        p1 = float(tag.get("tangentalDistortion1")) / 64 / binning_factor
        p2 = float(tag.get("tangentalDistortion2")) / 64 / binning_factor
        D += [np.array([k1, k2, p1, p2])]

        fu = float(tag.get("focalLengthU")) / 64 / binning_factor / (1080 / res[i])
        fv = float(tag.get("focalLengthV")) / 64 / binning_factor / (1080 / res[i])
        cu = (
            float(tag.get("centerPointU")) / 64 / binning_factor
            - float(root.findall("cameras/camera/fov_video")[i].attrib.get("left"))
        ) / (1080 / res[i])
        cv = (
            float(tag.get("centerPointV")) / 64 / binning_factor
            - float(root.findall("cameras/camera/fov_video")[i].attrib.get("top"))
        ) / (1080 / res[i])
        K += [np.array([fu, 0.0, cu, 0.0, fv, cv, 0.0, 0.0, 1.0]).reshape(3, 3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for tag in root.findall("cameras/camera/transform"):
        tx = float(tag.get("x")) / 1000
        ty = float(tag.get("y")) / 1000
        tz = float(tag.get("z")) / 1000
        r11 = float(tag.get("r11"))
        r12 = float(tag.get("r12"))
        r13 = float(tag.get("r13"))
        r21 = float(tag.get("r21"))
        r22 = float(tag.get("r22"))
        r23 = float(tag.get("r23"))
        r31 = float(tag.get("r31"))
        r32 = float(tag.get("r32"))
        r33 = float(tag.get("r33"))

        # Rotation (by-column to by-line)
        R += [np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3, 3).T]
        T += [np.array([tx, ty, tz])]

    # Cameras names by natural order
    C_vid = [C[v] for v in vid_id]
    C_vid_id = [C_vid.index(c) for c in sorted(C_vid, key=natural_sort_key)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    ret = [ret[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]

    return ret, C, S, D, K, R, T
