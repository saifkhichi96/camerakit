import cv2

from ..common import CalibrationParams


def read_intrinsic_yml(intrinsic_path):
    """Read an EasyMocap intrinsics YAML file.

    Args:
        intrinsic_path: Path to `intri.yml`.

    Returns:
        tuple[list, list, list, list]: Camera names, image sizes, intrinsic matrices,
        and distortion coefficients.
    """
    intrinsic_yml = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)
    cam_number = intrinsic_yml.getNode("names").size()
    N, S, D, K = [], [], [], []
    for i in range(cam_number):
        name = intrinsic_yml.getNode("names").at(i).string()
        N.append(name)
        K.append(intrinsic_yml.getNode(f"K_{name}").mat())
        D.append(intrinsic_yml.getNode(f"dist_{name}").mat().flatten()[:-1])
        S.append([K[i][0, 2] * 2, K[i][1, 2] * 2])
    return N, S, K, D


def read_extrinsic_yml(extrinsic_path):
    """Read an EasyMocap extrinsics YAML file.

    Args:
        extrinsic_path: Path to `extri.yml`.

    Returns:
        tuple[list, list, list]: Camera names, Rodrigues rotations, and translations.
    """
    extrinsic_yml = cv2.FileStorage(extrinsic_path, cv2.FILE_STORAGE_READ)
    cam_number = extrinsic_yml.getNode("names").size()
    N, R, T = [], [], []
    for i in range(cam_number):
        name = extrinsic_yml.getNode("names").at(i).string()
        N.append(name)
        R.append(
            extrinsic_yml.getNode(f"R_{name}").mat().flatten()
        )  # R_1 pour Rodrigues, Rot_1 pour matrice
        T.append(extrinsic_yml.getNode(f"T_{name}").mat().flatten())
    return N, R, T


def calib_easymocap_fun(files_to_convert_paths, binning_factor=1) -> CalibrationParams:
    """Convert EasyMocap calibration files into CameraKit calibration arrays.

    Args:
        files_to_convert_paths: Paths for `extri.yml` and `intri.yml`.
        binning_factor: Unused for EasyMocap conversions.

    Returns:
        CalibrationParams: Converted calibration arrays and residuals.
    """

    extrinsic_path, intrinsic_path = files_to_convert_paths
    C, S, K, D = read_intrinsic_yml(intrinsic_path)
    _, R, T = read_extrinsic_yml(extrinsic_path)
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
