import logging
import os

import cv2
import numpy as np

from ..common import CalibrationParams

logger = logging.getLogger(__name__)


def calib_biocv_fun(files_to_convert_paths, binning_factor=1) -> CalibrationParams:
    """Convert bioCV calibration files into CameraKit calibration arrays.

    Args:
        files_to_convert_paths: Paths to `.calib` files.
        binning_factor: Unused for bioCV conversions.

    Returns:
        CalibrationParams: Converted calibration arrays and residuals.
    """

    logger.info(
        f"Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file..."
    )

    C, S, D, K, R, T = [], [], [], [], [], []
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path) as f:
            calib_data = f.read().split("\n")
            C += [f"cam_{str(i).zfill(2)}"]
            S += [[int(calib_data[0]), int(calib_data[1])]]
            D += [[float(d) for d in calib_data[-2].split(" ")[:4]]]
            K += [np.array([k.strip().split(" ") for k in calib_data[2:5]], np.float32)]
            RT = np.array([k.strip().split(" ") for k in calib_data[6:9]], np.float32)
            R += [cv2.Rodrigues(RT[:, :3])[0].squeeze()]
            T += [RT[:, 3] / 1000]

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
