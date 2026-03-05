import logging
import os
import pickle

import cv2
import numpy as np

from ..common import CalibrationParams
from .common import rotate_cam, world_to_camera_persp

logger = logging.getLogger(__name__)


def calib_opencap_fun(files_to_convert_paths, binning_factor=1) -> CalibrationParams:
    """Convert OpenCap pickle calibrations into CameraKit calibration arrays.

    OpenCap extrinsics are expressed with a vertical board reference, so rotations
    are adjusted to match CameraKit's world-frame conventions.

    Args:
        files_to_convert_paths: Paths to OpenCap `.pickle` calibration files.
        binning_factor: Unused for OpenCap conversions.

    Returns:
        CalibrationParams: Converted calibration arrays and residuals.
    """

    logger.info(
        f"Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file..."
    )

    C, S, D, K, R, T = [], [], [], [], [], []
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path, "rb") as f_pickle:
            calib_data = pickle.load(f_pickle)
            C += [f"cam_{str(i).zfill(2)}"]
            S += [list(calib_data["imageSize"].squeeze()[::-1])]
            D += [list(calib_data["distortion"][0][:-1])]
            K += [calib_data["intrinsicMat"]]
            R_cam = calib_data["rotation"]
            T_cam = calib_data["translation"].squeeze()

            # Rotate cameras by Pi/2 around x in world frame -> could have just switched some columns in matrix
            # camera frame to world frame
            R_w, T_w = world_to_camera_persp(R_cam, T_cam)
            # x_rotate -Pi/2 and z_rotate Pi
            R_w_90, T_w_90 = rotate_cam(
                R_w, T_w, ang_x=-np.pi / 2, ang_y=0, ang_z=np.pi
            )
            # world frame to camera frame
            R_c_90, T_c_90 = world_to_camera_persp(R_w_90, T_w_90)
            # Store a consistent extrinsic pair after frame conversion.
            R += [cv2.Rodrigues(R_c_90)[0].squeeze()]
            T += [T_c_90 / 1000]

    zero_errors = [0.0] * len(C)
    return CalibrationParams(
        ret_intrinsics=zero_errors.copy(),
        ret_extrinsics=zero_errors,
        C=C,
        S=S,
        D=D,
        K=K,
        R=R,
        T=T,
    )
