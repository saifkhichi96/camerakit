import logging

import cv2
import numpy as np
from lxml import etree

from ..common import CalibrationParams
from .common import natural_sort_key, quat2mat, world_to_camera_persp

logger = logging.getLogger(__name__)


def calib_vicon_fun(file_to_convert_path, binning_factor=1) -> CalibrationParams:
    """Convert a Vicon `.xcp` calibration file.

    Args:
        file_to_convert_path: Path to `.xcp` calibration file.
        binning_factor: Unused for Vicon conversions.

    Returns:
        CalibrationParams: Converted calibration arrays and residuals.
    """

    logger.info(f"Converting {file_to_convert_path} to .toml calibration file...")
    residuals_mm, C, S, D, K, R, T = read_vicon(file_to_convert_path)

    RT = [world_to_camera_persp(r, t) for r, t in zip(R, T)]
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


def read_vicon(vicon_path):
    """Parse raw Vicon calibration XML into calibration arrays.

    Args:
        vicon_path: Path to `.xcp` calibration file.

    Returns:
        tuple[list, list, list, list, list, list, list]: Residuals, camera names,
        image sizes, distortions, intrinsics, rotation matrices, and translations.
    """

    root = etree.parse(vicon_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    vid_id = []

    # Camera name and image size
    for i, tag in enumerate(root.findall("Camera")):
        C += [tag.attrib.get("DEVICEID")]
        S += [[float(t) for t in tag.attrib.get("SENSOR_SIZE").split()]]
        ret += [float(tag.findall("KeyFrames/KeyFrame")[0].attrib.get("WORLD_ERROR"))]
        vid_id += [i]

    # Intrinsic parameters: distortion and intrinsic matrix
    for cam_elem in root.findall("Camera"):
        try:
            dist = (
                cam_elem.findall("KeyFrames/KeyFrame")[0]
                .attrib.get("VICON_RADIAL2")
                .split()[3:5]
            )
        except:
            dist = (
                cam_elem.findall("KeyFrames/KeyFrame")[0]
                .attrib.get("VICON_RADIAL")
                .split()
            )
        D += [[float(d) for d in dist] + [0.0, 0.0]]

        fu = float(cam_elem.findall("KeyFrames/KeyFrame")[0].attrib.get("FOCAL_LENGTH"))
        fv = fu / float(cam_elem.attrib.get("PIXEL_ASPECT_RATIO"))
        cam_center = (
            cam_elem.findall("KeyFrames/KeyFrame")[0]
            .attrib.get("PRINCIPAL_POINT")
            .split()
        )
        cu, cv = [float(c) for c in cam_center]
        K += [np.array([fu, 0.0, cu, 0.0, fv, cv, 0.0, 0.0, 1.0]).reshape(3, 3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for cam_elem in root.findall("Camera"):
        rot = (
            cam_elem.findall("KeyFrames/KeyFrame")[0].attrib.get("ORIENTATION").split()
        )
        R_quat = [float(r) for r in rot]
        R_mat = quat2mat(R_quat, scalar_idx=3)
        R += [R_mat]

        trans = cam_elem.findall("KeyFrames/KeyFrame")[0].attrib.get("POSITION").split()
        T += [[float(t) / 1000 for t in trans]]

    # Camera names by natural order
    C_vid_id = [
        v for v in vid_id if ("VIDEO") in root.findall("Camera")[v].attrib.get("TYPE")
    ]
    C_vid = [root.findall("Camera")[v].attrib.get("DEVICEID") for v in C_vid_id]
    C = sorted(C_vid, key=natural_sort_key)
    C_id_sorted = [
        i
        for v_sorted in C
        for i, v in enumerate(root.findall("Camera"))
        if v.attrib.get("DEVICEID") == v_sorted
    ]
    S = [S[c] for c in C_id_sorted]
    D = [D[c] for c in C_id_sorted]
    K = [K[c] for c in C_id_sorted]
    R = [R[c] for c in C_id_sorted]
    T = [T[c] for c in C_id_sorted]

    return ret, C, S, D, K, R, T
