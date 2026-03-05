"""
###########################################################################
## CAMERAS CALIBRATION                                                   ##
###########################################################################

Use this module to calibrate your cameras and save results to a .toml file.

It calibrates cameras from checkerboard images or videos.

Checkerboard calibration is based on
https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.

INPUTS:
- folders 'calibration/intrinsics' (populated with video or about 30 images) and
  'calibration/extrinsics' (populated with video or one image)
- a Config.toml file in the project folder

OUTPUTS:
- a calibration file in the 'calibration' folder (.toml extension)
"""

# TODO: DETECT WHEN WINDOW IS CLOSED
# TODO: WHEN 'Y', CATCH IF NUMBER OF IMAGE POINTS CLICKED NOT EQUAL TO NB OBJ POINTS

import contextlib
import glob
import os
import re
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import toml
from mpl_interactions import panhandler, zoom_factory

from .calib import (
    euclidean_distance,
    extract_frames,
    toml_write,
)
from .common import get_logger
from .pose import (
    find_outer_corners,
)

## SETUP LOGGING
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
logger = get_logger()


def get_frame(img_or_video):
    """Load the first frame from an image or video path.

    Args:
        img_or_video: Path to an image or video file.

    Returns:
        np.ndarray | None: Loaded frame, or `None` if reading fails.
    """
    img = cv2.imread(img_or_video)
    if img is None:
        cap = cv2.VideoCapture(img_or_video)
        ret, img = cap.read()
        cap.release()
        if not ret:
            logger.error(f"Could not read image or video from {img_or_video}")
            return None
    return img


## FUNCTIONS
def calib_calc_fun(calib_dir, intrinsics_config_dict, extrinsics_config_dict):
    """Compute calibration from checkerboard data or existing calibration file.

    Args:
        calib_dir: Calibration directory containing `intrinsics/` and `extrinsics/`.
        intrinsics_config_dict: Intrinsics calibration configuration dictionary.
        extrinsics_config_dict: Extrinsics calibration configuration dictionary.

    Returns:
        tuple[list, list, list, list, list, list, list]: Residuals, camera names,
        image sizes, distortions, intrinsics, Rodrigues rotations, and translations.
    """

    overwrite_intrinsics = intrinsics_config_dict.get("overwrite_intrinsics")
    calculate_extrinsics = extrinsics_config_dict.get("calculate_extrinsics")

    # retrieve intrinsics if calib_file found and if overwrite_intrinsics=False
    with contextlib.suppress(Exception):
        calib_file = glob.glob(os.path.join(calib_dir, "Calib*.toml"))[0]
    if not overwrite_intrinsics and "calib_file" in locals():
        logger.info(f"\nPreexisting calibration file found: '{calib_file}'.")
        logger.info(
            '\nRetrieving intrinsic parameters from file. Set "overwrite_intrinsics" to true in Config.toml to recalculate them.'
        )
        calib_file = glob.glob(os.path.join(calib_dir, "Calib*.toml"))[0]
        calib_data = toml.load(calib_file)

        ret_intrinsics, C, S, D, K, R, T = [], [], [], [], [], [], []
        meta_intr = calib_data.get("metadata", {}).get("intrinsics_error_px")
        if meta_intr is None:
            meta_intr = calib_data.get("metadata", {}).get("error")
        cam_index = 0
        for cam in calib_data:
            if cam != "metadata":
                cam_block = calib_data[cam]
                intr_err = cam_block.get("intrinsics_error_px")
                if (
                    intr_err is None
                    and isinstance(meta_intr, list)
                    and cam_index < len(meta_intr)
                ):
                    intr_err = meta_intr[cam_index]
                if intr_err is None:
                    intr_err = 0.0
                ret_intrinsics += [intr_err]
                C += [cam_block["name"]]
                S += [cam_block["size"]]
                K += [np.array(cam_block["matrix"])]
                D += [cam_block["distortions"]]
                R += [[0.0, 0.0, 0.0]]
                T += [[0.0, 0.0, 0.0]]
                cam_index += 1
        nb_cams_intrinsics = len(C)

    # calculate intrinsics otherwise
    else:
        logger.info("\nCalculating intrinsic parameters...")
        ret_intrinsics, C, S, D, K, R, T = calibrate_intrinsics(
            calib_dir, intrinsics_config_dict
        )
        nb_cams_intrinsics = len(C)

    # calculate extrinsics
    ret_extrinsics = None
    if calculate_extrinsics:
        logger.info("\nCalculating extrinsic parameters...")

        # check that the number of cameras is consistent
        nb_cams_extrinsics = len(
            next(os.walk(os.path.join(calib_dir, "extrinsics")))[1]
        )
        if nb_cams_intrinsics != nb_cams_extrinsics:
            raise Exception(
                f"Error: The number of cameras is not consistent:\
                    Found {nb_cams_intrinsics} cameras based on the number of intrinsic folders or on calibration file data,\
                    and {nb_cams_extrinsics} cameras based on the number of extrinsic folders."
            )
        ret_extrinsics, C, S, D, K, R, T = calibrate_extrinsics(
            calib_dir, extrinsics_config_dict, C, S, K, D
        )
    else:
        logger.info(
            '\nExtrinsic parameters won\'t be calculated. Set "calculate_extrinsics" to true in Config.toml to calculate them.'
        )

    return ret_intrinsics, ret_extrinsics, C, S, D, K, R, T


def calibrate_intrinsics(calib_dir, intrinsics_config_dict):
    """Estimate intrinsic parameters for all cameras in the project.

    Args:
        calib_dir: Calibration directory containing `intrinsics/`.
        intrinsics_config_dict: Intrinsics calibration configuration dictionary.

    Returns:
        tuple[list, list, list, list, list, list, list]: Residuals, camera names,
        image sizes, distortions, intrinsics, placeholder rotations, and translations.
    """

    try:
        intrinsics_cam_listdirs_names = next(
            os.walk(os.path.join(calib_dir, "intrinsics"))
        )[1]
    except StopIteration:
        logger.exception(
            f"Error: No {os.path.join(calib_dir, 'intrinsics')} folder found."
        )
        raise Exception(
            f"Error: No {os.path.join(calib_dir, 'intrinsics')} folder found."
        )
    intrinsics_extension = intrinsics_config_dict.get("intrinsics_extension")
    extract_every_N_sec = intrinsics_config_dict.get("extract_every_N_sec")
    overwrite_extraction = False
    show_detection_intrinsics = intrinsics_config_dict.get("show_detection_intrinsics")
    intrinsics_corners_nb = intrinsics_config_dict.get("intrinsics_corners_nb")
    intrinsics_square_size = (
        intrinsics_config_dict.get("intrinsics_square_size") / 1000
    )  # convert to meters
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []

    for i, cam in enumerate(intrinsics_cam_listdirs_names):
        # Prepare object points
        objp = np.zeros(
            (intrinsics_corners_nb[0] * intrinsics_corners_nb[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : intrinsics_corners_nb[0], 0 : intrinsics_corners_nb[1]
        ].T.reshape(-1, 2)
        objp[:, :2] = objp[:, 0:2] * intrinsics_square_size
        objpoints = []  # 3d points in world space
        imgpoints = []  # 2d points in image plane

        logger.info(f"\nCamera {cam}:")
        img_vid_files = glob.glob(
            os.path.join(calib_dir, "intrinsics", cam, f"*.{intrinsics_extension}")
        )
        if len(img_vid_files) == 0:
            logger.exception(
                f"The folder {os.path.join(calib_dir, 'intrinsics', cam)} does not exist or does not contain any files with extension .{intrinsics_extension}."
            )
            raise ValueError(
                f"The folder {os.path.join(calib_dir, 'intrinsics', cam)} does not exist or does not contain any files with extension .{intrinsics_extension}."
            )
        img_vid_files = sorted(
            img_vid_files, key=lambda c: [int(n) for n in re.findall(r"\d+", c)]
        )  # sorting paths with numbers

        # extract frames from video if video
        try:
            cap = cv2.VideoCapture(img_vid_files[0])
            cap.read()
            if not cap.read()[0]:
                raise
            extract_frames(img_vid_files[0], extract_every_N_sec, overwrite_extraction)
            img_vid_files = glob.glob(
                os.path.join(calib_dir, "intrinsics", cam, "*.png")
            )
            img_vid_files = sorted(
                img_vid_files, key=lambda c: [int(n) for n in re.findall(r"\d+", c)]
            )
        except Exception:
            pass

        # find corners
        for img_path in img_vid_files:
            if show_detection_intrinsics:
                imgp_confirmed, objp_confirmed = findCorners(
                    img_path,
                    intrinsics_corners_nb,
                    objp=objp,
                    show=show_detection_intrinsics,
                )
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp_confirmed)
            else:
                imgp_confirmed = findCorners(
                    img_path,
                    intrinsics_corners_nb,
                    objp=objp,
                    show=show_detection_intrinsics,
                )
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp)
                elif (
                    isinstance(imgp_confirmed, (list, tuple))
                    and len(imgp_confirmed) == 2
                    and isinstance(imgp_confirmed[0], np.ndarray)
                ):
                    imgpoints.append(imgp_confirmed[0])
                    objpoints.append(imgp_confirmed[1])
        if len(imgpoints) < 10:
            logger.info(
                f"Corners were detected only on {len(imgpoints)} images for camera {cam}. Calibration of intrinsic parameters may not be accurate with fewer than 10 good images of the board."
            )

        # calculate intrinsics
        img = cv2.imread(str(img_path))
        objpoints = np.array(objpoints)

        ret_cam, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            img.shape[1::-1],
            None,
            None,
            flags=(cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT),
        )
        h, w = [np.float32(i) for i in img.shape[:-1]]
        ret.append(ret_cam)
        C.append(cam)
        S.append([w, h])
        D.append(dist[0])
        K.append(mtx)
        R.append([0.0, 0.0, 0.0])
        T.append([0.0, 0.0, 0.0])

        logger.info(
            f"Intrinsics error: {np.around(ret_cam, decimals=3)} px for camera {cam}."
        )

    return ret, C, S, D, K, R, T


def calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D):
    """Estimate extrinsic parameters for all cameras.

    Args:
        calib_dir: Calibration directory containing `extrinsics/`.
        extrinsics_config_dict: Extrinsics calibration configuration dictionary.
        C: Camera names from intrinsics stage.
        S: Image sizes from intrinsics stage.
        K: Intrinsic matrices from intrinsics stage.
        D: Distortion coefficients from intrinsics stage.

    Returns:
        tuple[list, list, list, list, list, list, list]: Residuals, camera names,
        image sizes, distortions, intrinsics, Rodrigues rotations, and translations.
    """

    try:
        extrinsics_cam_listdirs_names = next(
            os.walk(os.path.join(calib_dir, "extrinsics"))
        )[1]
    except StopIteration:
        logger.exception(
            f"Error: No {os.path.join(calib_dir, 'extrinsics')} folder found."
        )
        raise Exception(
            f"Error: No {os.path.join(calib_dir, 'extrinsics')} folder found."
        )

    extrinsics_method = extrinsics_config_dict.get("extrinsics_method")
    ret, R, T = [], [], []

    if extrinsics_method in ("board", "board_outer", "scene"):
        # Define 3D object points
        if extrinsics_method in ["board", "board_outer"]:
            extrinsics_corners_nb = extrinsics_config_dict.get("board").get(
                "extrinsics_corners_nb"
            )
            extrinsics_square_size = (
                extrinsics_config_dict.get("board").get("extrinsics_square_size") / 1000
            )  # convert to meters
            if extrinsics_method == "board":
                object_coords_3d = np.zeros(
                    (extrinsics_corners_nb[0] * extrinsics_corners_nb[1], 3), np.float32
                )
                object_coords_3d[:, :2] = np.mgrid[
                    0 : extrinsics_corners_nb[0], 0 : extrinsics_corners_nb[1]
                ].T.reshape(-1, 2)
                object_coords_3d[:, :2] = (
                    object_coords_3d[:, 0:2] * extrinsics_square_size
                )
            else:  # board_outer
                object_coords_3d = np.array(
                    [
                        [0, 0, 0],
                        [(extrinsics_corners_nb[0]) * extrinsics_square_size, 0, 0],
                        [0, (extrinsics_corners_nb[1]) * extrinsics_square_size, 0],
                        [
                            (extrinsics_corners_nb[0]) * extrinsics_square_size,
                            (extrinsics_corners_nb[1]) * extrinsics_square_size,
                            0,
                        ],
                    ],
                    dtype=np.float32,
                )
        elif extrinsics_method == "scene":
            object_coords_3d = np.array(
                extrinsics_config_dict.get("scene").get("object_coords_3d"), np.float32
            )

        for i, cam in enumerate(extrinsics_cam_listdirs_names):
            logger.info(f"\nCamera {cam}:")

            # Read images or video
            extrinsics_extension = [
                extrinsics_config_dict.get("board").get("extrinsics_extension")
                if extrinsics_method in ["board", "board_outer"]
                else extrinsics_config_dict.get("scene").get("extrinsics_extension")
            ][0]
            show_reprojection_error = [
                extrinsics_config_dict.get("board").get("show_reprojection_error")
                if extrinsics_method in ["board", "board_outer"]
                else extrinsics_config_dict.get("scene").get("show_reprojection_error")
            ][0]
            img_vid_files = glob.glob(
                os.path.join(calib_dir, "extrinsics", cam, f"*.{extrinsics_extension}")
            )
            if len(img_vid_files) == 0:
                logger.exception(
                    f"The folder {os.path.join(calib_dir, 'extrinsics', cam)} does not exist or does not contain any files with extension .{extrinsics_extension}."
                )
                raise ValueError(
                    f"The folder {os.path.join(calib_dir, 'extrinsics', cam)} does not exist or does not contain any files with extension .{extrinsics_extension}."
                )
            img_vid_files = sorted(
                img_vid_files, key=lambda c: [int(n) for n in re.findall(r"\d+", c)]
            )  # sorting paths with numbers

            # extract frames from image, or from video if imread is None
            img = get_frame(img_vid_files[0])
            if img is None:
                raise ValueError(
                    f"Could not read image or video from {img_vid_files[0]}"
                )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Find corners or label by hand
            if extrinsics_method in ["board", "board_outer"]:
                imgp, objp = findCorners(
                    img_vid_files[0],
                    extrinsics_corners_nb,
                    objp=object_coords_3d,
                    show=show_reprojection_error,
                    outermost_only=extrinsics_method == "board_outer",
                )
                if len(imgp) == 0:
                    logger.exception(
                        'No corners found. Set "show_detection_extrinsics" to true to click corners by hand, or change extrinsics_method to "scene".'
                    )
                    raise ValueError(
                        'No corners found. Set "show_detection_extrinsics" to true to click corners by hand, or change extrinsics_method to "scene".'
                    )

            elif extrinsics_method == "scene":
                imgp, objp = imgp_objp_visualizer_clicker(
                    img, imgp=[], objp=object_coords_3d, img_path=img_vid_files[0]
                )
                if len(imgp) == 0:
                    logger.exception(
                        "No points clicked (or fewer than 6). Press 'C' when the image is displayed, and then click on the image points corresponding to the 'object_coords_3d' you measured and wrote down in the Config.toml file."
                    )
                    raise ValueError(
                        "No points clicked (or fewer than 6). Press 'C' when the image is displayed, and then click on the image points corresponding to the 'object_coords_3d' you measured and wrote down in the Config.toml file."
                    )
                if len(objp) < 10:
                    logger.info(
                        f"Only {len(objp)} reference points for camera {cam}. Calibration of extrinsic parameters may not be accurate with fewer than 10 reference points, as spread out in the captured volume as possible."
                    )

                # refine manually clicked points
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    50,
                    1e-10,
                )
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgp = cv2.cornerSubPix(gray, imgp, (1, 1), (-1, -1), criteria)

            # Calculate extrinsics
            mtx, dist = np.array(K[i]), np.array(D[i])
            _, r, t = cv2.solvePnP(np.array(objp), imgp, mtx, dist)
            r, t = r.flatten(), t.flatten()

            # Projection of object points to image plane
            # # Former way, distortions used to be ignored
            # Kh_cam = np.block([mtx, np.zeros(3).reshape(3,1)])
            # r_mat, _ = cv2.Rodrigues(r)
            # H_cam = np.block([[r_mat,t.reshape(3,1)], [np.zeros(3), 1 ]])
            # P_cam = Kh_cam @ H_cam
            # proj_obj = [ ( P_cam[0] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)),  P_cam[1] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)) ) for o in objp]
            proj_obj = np.squeeze(cv2.projectPoints(objp, r, t, mtx, dist)[0])

            # Check calibration results
            if show_reprojection_error:
                # Reopen image, otherwise 2 sets of text are overlaid
                img = get_frame(img_vid_files[0])
                if img is None:
                    raise ValueError(
                        f"Could not read image or video from {img_vid_files[0]}"
                    )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for o in proj_obj:
                    cv2.circle(img, (int(o[0]), int(o[1])), 8, (0, 0, 255), -1)
                for i in imgp:
                    cv2.drawMarker(
                        img,
                        (int(i[0][0]), int(i[0][1])),
                        (0, 255, 0),
                        cv2.MARKER_CROSS,
                        15,
                        2,
                    )
                cv2.putText(
                    img,
                    "Verify calibration results, then close window.",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    7,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    "Verify calibration results, then close window.",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.drawMarker(img, (20, 40), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(
                    img,
                    "    Clicked points",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    7,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    "    Clicked points",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(img, (20, 60), 8, (0, 0, 255), -1)
                cv2.putText(
                    img,
                    "    Reprojected object points",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    7,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    "    Reprojected object points",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                plt.figure()
                plt.imshow(img)
                plt.title(os.path.basename(img_vid_files[0]))
                plt.axis("off")
                plt.show()

            # Calculate reprojection error
            imgp_to_objreproj_dist = [
                euclidean_distance(proj_obj[n], imgp[n]) for n in range(len(proj_obj))
            ]
            rms_px = np.sqrt(np.sum([d**2 for d in imgp_to_objreproj_dist]))
            ret.append(rms_px)
            R.append(r)
            T.append(t)

    else:
        raise ValueError("Wrong value for extrinsics_method")

    return ret, C, S, D, K, R, T


def findCorners(img_path, corner_nb, objp=[], show=True, outermost_only=False):
    """
    Find corners in a checkerboard image with options for manual confirmation.

    Press 'Y' to accept detection, 'N' to dismiss this image, 'C' to click points by hand.
    Left click to add a point, right click to remove the last point.
    Use mouse wheel to zoom in and out and to pan.

    Make sure that:
    - the checkerboard is surrounded by a white border
    - rows != lines, and row is even if lines is odd (or conversely)
    - it is flat and without reflections
    - corner_nb correspond to _internal_ corners

    Args:
        img_path: Path to image or video input.
        corner_nb: Chessboard internal corner layout as `(cols, rows)`.
        objp: Optional 3D object points matching expected corners.
        show: Whether to show the interactive verification window.
        outermost_only: Whether to keep only four outer corners.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray] | list | None: Confirmed image points,
        optionally paired with object points, depending on inputs and detection outcome.
    """

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )  # stop refining after 30 iterations or if error less than 0.001px

    img = get_frame(img_path)
    if img is None:
        logger.error(f"Could not read image or video from {img_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, corner_nb, None)

    # If corners are found, refine corners
    if ret == True:
        imgp = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        logger.info(f"{os.path.basename(img_path)}: Corners found.")

        if outermost_only:
            # Use only outer corners
            logger.info(f"{os.path.basename(img_path)}: Extracting outer corners.")
            imgp = find_outer_corners(imgp, corner_nb).reshape(-1, 1, 2)

        if show:
            if outermost_only:
                # Draw outer corners
                for i, corner in enumerate(imgp):
                    x, y = corner.ravel()
                    cv2.drawMarker(
                        img, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 15, 2
                    )
                    cv2.putText(
                        img,
                        str(i + 1),
                        (int(x) - 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        7,
                    )
                    cv2.putText(
                        img,
                        str(i + 1),
                        (int(x) - 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )

            else:
                # Draw all corners
                cv2.drawChessboardCorners(img, corner_nb, imgp, ret)
                for i, corner in enumerate(imgp):
                    if i in [
                        0,
                        corner_nb[0] - 1,
                        corner_nb[0] * (corner_nb[1] - 1),
                        corner_nb[0] * corner_nb[1] - 1,
                    ]:
                        x, y = corner.ravel()
                        cv2.putText(
                            img,
                            str(i + 1),
                            (int(x) - 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            7,
                        )
                        cv2.putText(
                            img,
                            str(i + 1),
                            (int(x) - 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2,
                        )

            # Visualizer and key press event handler
            for var_to_delete in ["imgp_confirmed", "objp_confirmed"]:
                if var_to_delete in globals():
                    del globals()[var_to_delete]
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(
                img, imgp=imgp, objp=objp, img_path=img_path
            )
        else:
            imgp_objp_confirmed = imgp, objp

    # If corners are not found, dismiss or click points by hand
    else:
        logger.info(
            f'{os.path.basename(img_path)}: Corners not found. To label them by hand, set "show_detection_intrinsics" to true in the Config.toml file.'
        )
        if show:
            # Visualizer and key press event handler
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(
                img, imgp=[], objp=objp, img_path=img_path
            )
        else:
            imgp_objp_confirmed = [], []

    return imgp_objp_confirmed


def imgp_objp_visualizer_clicker(img, imgp=[], objp=[], img_path=""):
    """Open an interactive visualizer for validating or clicking calibration points.

    Args:
        img: Image as NumPy array.
        imgp: Optional detected image points to review.
        objp: Optional 3D object points for guidance.
        img_path: Optional path used as the window title.

    Returns:
        tuple[np.ndarray, np.ndarray] | np.ndarray | None: Confirmed image/object points
        when accepted, or `None` when dismissed.
    """
    global old_image_path
    old_image_path = img_path

    def on_key(event):
        """Handle key presses for accepting, rejecting, or editing detections.

        Args:
            event: Matplotlib key event.
        """

        global \
            imgp_confirmed, \
            objp_confirmed, \
            objp_confirmed_notok, \
            scat, \
            ax_3d, \
            fig_3d, \
            events, \
            count

        if event.key == "y":
            # If 'y', close all
            # If points have been clicked, imgp_confirmed is returned, else imgp
            # If objp is given, objp_confirmed is returned in addition
            if "scat" not in globals() or "imgp_confirmed" not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = np.array(
                    [imgp.astype("float32") for imgp in imgp_confirmed]
                )
                objp_confirmed = objp_confirmed
            # OpenCV needs at leas 4 correspondance points to calibrate
            if len(imgp_confirmed) < 6:
                objp_confirmed = []
                imgp_confirmed = []
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close("all")
            if len(objp) == 0 and "objp_confirmed" in globals():
                del objp_confirmed

        if event.key == "n" or event.key == "q":
            # If 'n', close all and return nothing
            plt.close("all")
            imgp_confirmed = []
            objp_confirmed = []

        if event.key == "c":
            # TODO: RIGHT NOW, IF 'C' IS PRESSED ANOTHER TIME, OBJP_CONFIRMED AND IMGP_CONFIRMED ARE RESET TO []
            # We should reopen a figure without point on it
            img_for_pointing = get_frame(old_image_path)
            if img_for_pointing is None:
                logger.error(f"Could not read image or video from {old_image_path}")
                return
            img_for_pointing = cv2.cvtColor(img_for_pointing, cv2.COLOR_BGR2RGB)
            ax.imshow(img_for_pointing)
            # To update the image
            plt.draw()

            if "objp_confirmed" in globals():
                del objp_confirmed
            # If 'c', allows retrieving imgp_confirmed by clicking them on the image
            scat = ax.scatter([], [], s=100, marker="+", color="g")
            plt.connect("button_press_event", on_click)
            # If objp is given, display 3D object points in black
            if len(objp) != 0 and not plt.fignum_exists(2):
                fig_3d = plt.figure()
                fig_3d.tight_layout()
                fig_3d.canvas.manager.set_window_title("Object points to be clicked")
                ax_3d = fig_3d.add_subplot(projection="3d")
                plt.rc("xtick", labelsize=5)
                plt.rc("ytick", labelsize=5)
                for i, (xs, ys, zs) in enumerate(np.float32(objp)):
                    ax_3d.scatter(xs, ys, zs, marker=".", color="k")
                    ax_3d.text(
                        xs, ys, zs, f"{str(i + 1)}", size=10, zorder=1, color="k"
                    )
                set_axes_equal(ax_3d)
                ax_3d.set_xlabel("X")
                ax_3d.set_ylabel("Y")
                ax_3d.set_zlabel("Z")
                if np.all(objp[:, 2] == 0):
                    ax_3d.view_init(elev=-90, azim=0)
                fig_3d.show()

        if event.key == "h":
            # If 'h', indicates that one of the objp is not visible on image
            # Displays it in red on 3D plot
            if len(objp) != 0 and "ax_3d" in globals():
                count = [0 if "count" not in globals() else count + 1][0]
                if "events" not in globals():
                    # retrieve first objp_confirmed_notok and plot 3D
                    events = [event]
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker="o", color="r")
                    fig_3d.canvas.draw()
                elif count == len(objp) - 1:
                    # if all objp have been clicked or indicated as not visible, close all
                    objp_confirmed = np.array(
                        [
                            [objp[count]]
                            if "objp_confirmed" not in globals()
                            else objp_confirmed + [objp[count]]
                        ][0]
                    )[:-1]
                    imgp_confirmed = np.array(
                        np.expand_dims(scat.get_offsets(), axis=1), np.float32
                    )
                    plt.close("all")
                    for var_to_delete in [
                        "events",
                        "count",
                        "scat",
                        "fig_3d",
                        "ax_3d",
                        "objp_confirmed_notok",
                    ]:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve other objp_confirmed_notok and plot 3D
                    events.append(event)
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker="o", color="r")
                    fig_3d.canvas.draw()
            else:
                pass

    def on_click(event):
        """Handle mouse clicks while manually selecting image points.

        Args:
            event: Matplotlib mouse event.
        """

        global \
            imgp_confirmed, \
            objp_confirmed, \
            objp_confirmed_notok, \
            scat, \
            ax_3d, \
            fig_3d, \
            events, \
            count, \
            xydata

        # Left click: Add clicked point to imgp_confirmed
        # Display it on image and on 3D plot
        if event.button == 1:
            # To remember the event to cancel after right click
            if "events" in globals():
                events.append(event)
            else:
                events = [event]

            # Add clicked point to image
            xydata = scat.get_offsets()
            new_xydata = np.concatenate((xydata, [[event.xdata, event.ydata]]))
            scat.set_offsets(new_xydata)
            imgp_confirmed = np.expand_dims(scat.get_offsets(), axis=1)
            plt.draw()

            # Add clicked point to 3D object points if given
            if len(objp) != 0:
                count = [0 if "count" not in globals() else count + 1][0]
                if count == 0:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [objp[count]]
                    ax_3d.scatter(*objp[count], marker="o", color="g")
                    fig_3d.canvas.draw()
                elif count == len(objp) - 1:
                    # close all
                    plt.close("all")
                    # retrieve objp_confirmed
                    objp_confirmed = np.array(
                        [
                            [objp[count]]
                            if "objp_confirmed" not in globals()
                            else objp_confirmed + [objp[count]]
                        ][0]
                    )
                    imgp_confirmed = np.array(imgp_confirmed, np.float32)
                    # delete all
                    for var_to_delete in [
                        "events",
                        "count",
                        "scat",
                        "scat_3d",
                        "fig_3d",
                        "ax_3d",
                        "objp_confirmed_notok",
                    ]:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [
                        [objp[count]]
                        if "objp_confirmed" not in globals()
                        else objp_confirmed + [objp[count]]
                    ][0]
                    ax_3d.scatter(*objp[count], marker="o", color="g")
                    fig_3d.canvas.draw()

        # Right click:
        # If last event was left click, remove last point and if objp given, from objp_confirmed
        # If last event was 'H' and objp given, remove last point from objp_confirmed_notok
        elif event.button == 3:  # right click
            if "events" in globals():
                # If last event was left click:
                if "button" in dir(events[-1]):
                    if events[-1].button == 1:
                        # Remove lastpoint from image
                        new_xydata = scat.get_offsets()[:-1]
                        scat.set_offsets(new_xydata)
                        plt.draw()
                        # Remove last point from imgp_confirmed
                        imgp_confirmed = imgp_confirmed[:-1]
                        if len(objp) != 0:
                            if count >= 0:
                                count -= 1
                            # Remove last point from objp_confirmed
                            objp_confirmed = objp_confirmed[:-1]
                            # remove from plot
                            if len(ax_3d.collections) > len(objp):
                                ax_3d.collections[-1].remove()
                                fig_3d.canvas.draw()

                # If last event was 'h' key
                elif events[-1].key == "h" and len(objp) != 0:
                    if count >= 1:
                        count -= 1
                    # Remove last point from objp_confirmed_notok
                    objp_confirmed_notok = objp_confirmed_notok[:-1]
                    # remove from plot
                    if len(ax_3d.collections) > len(objp):
                        ax_3d.collections[-1].remove()
                        fig_3d.canvas.draw()

    def set_axes_equal(ax):
        """Set equal scaling on all axes of a 3D plot.

        Args:
            ax: Matplotlib 3D axis.
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Write instructions
    cv2.putText(
        img,
        'Type "Y" to accept point detection.',
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        'Type "Y" to accept point detection.',
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "If points are wrongfully (or not) detected:",
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "If points are wrongfully (or not) detected:",
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '- type "N" to dismiss this image,',
        (20, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '- type "N" to dismiss this image,',
        (20, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '- type "C" to click points by hand (beware of their order).',
        (20, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '- type "C" to click points by hand (beware of their order).',
        (20, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ',
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ',
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '   Confirm with "Y", cancel with "N".',
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        '   Confirm with "Y", cancel with "N".',
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Use mouse wheel to zoom in and out and to pan",
        (20, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        7,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Use mouse wheel to zoom in and out and to pan",
        (20, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    # Put image in a matplotlib figure for more controls
    plt.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(os.path.basename(img_path))
    ax.axis("off")
    for corner in imgp:
        x, y = corner.ravel()
        cv2.drawMarker(img, (int(x), int(y)), (128, 128, 128), cv2.MARKER_CROSS, 10, 2)
    ax.imshow(img)
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()
    plt.tight_layout()

    # Allow for zoom and pan in image
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    # Handles key presses to Accept, dismiss, or click points by hand
    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.draw()
    plt.show(block=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams["toolbar"] = "toolmanager"

    for var_to_delete in [
        "events",
        "count",
        "scat",
        "fig_3d",
        "ax_3d",
        "objp_confirmed_notok",
    ]:
        if var_to_delete in globals():
            del globals()[var_to_delete]

    if "imgp_confirmed" in globals() and "objp_confirmed" in globals():
        return imgp_confirmed, objp_confirmed
    if "imgp_confirmed" in globals() and "objp_confirmed" not in globals():
        return imgp_confirmed
    return None


def recap_calibrate(ret_intrinsics, ret_extrinsics, calib_path):
    """
    Print a log message giving calibration results. Also stored in User/logs.txt.
    """

    if ret_intrinsics is not None:
        ret_intrinsics = [float(np.around(r, decimals=3)) for r in ret_intrinsics]
        logger.info(
            f"\n--> Intrinsics reprojection errors (RMS) per camera: {ret_intrinsics} px."
        )

    if ret_extrinsics is not None:
        ret_extrinsics = [float(np.around(r, decimals=3)) for r in ret_extrinsics]
        logger.info(
            f"\n--> Extrinsics reprojection errors (RMS) per camera: {ret_extrinsics} px."
        )

    logger.info(f"Calibration file is stored at {calib_path}.")
    return ret_intrinsics, ret_extrinsics


def calibrate_cams_all(config_dict):
    """Run the full calibration flow from configuration.

    Depending on `calibration_type`, this either converts a third-party calibration
    file or computes intrinsics/extrinsics from raw calibration media.

    Args:
        config_dict: Parsed calibration configuration dictionary.
    """

    # Read config_dict
    project_dir = config_dict.get("project").get("project_dir")
    calib_dir = [
        os.path.join(project_dir, c)
        for c in os.listdir(project_dir)
        if ("Calib" in c or "calib" in c)
    ][0]
    calib_type = config_dict.get("calibration").get("calibration_type")

    if calib_type == "calculate":
        intrinsics_config_dict = (
            config_dict.get("calibration").get("calculate").get("intrinsics")
        )
        extrinsics_config_dict = (
            config_dict.get("calibration").get("calculate").get("extrinsics")
        )
        extrinsics_method = (
            config_dict.get("calibration")
            .get("calculate")
            .get("extrinsics")
            .get("extrinsics_method")
        )

        calib_output_path = os.path.join(calib_dir, f"Calib_{extrinsics_method}.toml")
        calib_full_type = calib_type
        args_calib_fun = [calib_dir, intrinsics_config_dict, extrinsics_config_dict]

    else:
        raise ValueError(
            'Unsupported calibration_type. Use calibration_type = "calculate".'
        )

    # Map calib function
    calib_mapping = {
        "calculate": calib_calc_fun,
    }
    calib_fun = calib_mapping[calib_full_type]

    # Calibrate
    ret_intrinsics, ret_extrinsics, C, S, D, K, R, T = calib_fun(*args_calib_fun)

    # Recap message
    ret_intrinsics, ret_extrinsics = recap_calibrate(
        ret_intrinsics, ret_extrinsics, calib_output_path
    )

    # Write calibration file with error metadata
    toml_write(
        calib_output_path,
        C,
        S,
        D,
        K,
        R,
        T,
        intrinsics_error_px=ret_intrinsics,
        extrinsics_error_px=ret_extrinsics,
    )
