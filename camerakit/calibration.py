import logging
import logging.handlers
import os
import time
from copy import deepcopy
from datetime import datetime

import toml
from easydict import EasyDict as edict

from .utils import calibrate_cams_all

__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


def setup_logging(session_dir):
    """
    Create logging file and stream handlers
    """

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.handlers.TimedRotatingFileHandler(
                os.path.join(session_dir, "logs.txt"), when="D", interval=7
            ),
            logging.StreamHandler(),
        ],
    )


def recursive_update(dict_to_update, dict_with_new_values):
    """
    Update nested dictionaries without overwriting existing keys in any level of nesting

    Example:
    dict_to_update = {'key': {'key_1': 'val_1', 'key_2': 'val_2'}}
    dict_with_new_values = {'key': {'key_1': 'val_1_new'}}
    returns {'key': {'key_1': 'val_1_new', 'key_2': 'val_2'}}
    while dict_to_update.update(dict_with_new_values) would return {'key': {'key_1': 'val_1_new'}}
    """

    for key, value in dict_with_new_values.items():
        if (
            key in dict_to_update
            and isinstance(value, dict)
            and isinstance(dict_to_update[key], dict)
        ):
            # Recursively update nested dictionaries
            dict_to_update[key] = recursive_update(dict_to_update[key], value)
        else:
            # Update or add new key-value pairs
            dict_to_update[key] = value

    return dict_to_update


def determine_level(config_dir):
    """
    Determine the level at which the function is called.
    Level = 1: Trial folder
    Level = 2: Root folder
    """

    len_paths = [
        len(root.split(os.sep))
        for root, dirs, files in os.walk(config_dir)
        if "Config.toml" in files
    ]
    if len_paths == []:
        raise FileNotFoundError(
            "You need a Config.toml file in each trial or root folder."
        )
    return max(len_paths) - min(len_paths) + 1


def read_config_files(config):
    """
    Read Root and Trial configuration files,
    and output a dictionary with all the parameters.
    """

    if isinstance(config, dict):
        level = 2  # log_dir = os.getcwd()
        config_dicts = [config]
        if config_dicts[0].get("project").get("project_dir") is None:
            raise ValueError(
                'Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_PROJECT_DIRECTORY>"})'
            )
    else:
        # if launched without an argument, config is None, else it is the path to the config directory
        config_dir = ["." if config is None else config][0]
        level = determine_level(config_dir)

        # Trial level
        if level == 1:  # Trial
            try:
                # if batch
                session_config_dict = toml.load(
                    os.path.join(config_dir, "..", "Config.toml")
                )
                trial_config_dict = toml.load(os.path.join(config_dir, "Config.toml"))
                session_config_dict = recursive_update(
                    session_config_dict, trial_config_dict
                )
            except Exception:
                # if single trial
                session_config_dict = toml.load(os.path.join(config_dir, "Config.toml"))
            session_config_dict.get("project").update({"project_dir": config_dir})
            config_dicts = [session_config_dict]

        # Root level
        if level == 2:
            session_config_dict = toml.load(os.path.join(config_dir, "Config.toml"))
            config_dicts = []
            # Create config dictionaries for all trials of the participant
            for root, dirs, files in os.walk(config_dir):
                if "Config.toml" in files and root != config_dir:
                    trial_config_dict = toml.load(os.path.join(root, files[0]))
                    # deep copy, otherwise session_config_dict is modified at each iteration within the config_dicts list
                    temp_dict = deepcopy(session_config_dict)
                    temp_dict = recursive_update(temp_dict, trial_config_dict)
                    temp_dict.get("project").update(
                        {"project_dir": os.path.join(config_dir, os.path.relpath(root))}
                    )
                    if os.path.basename(root) not in temp_dict.get("project").get(
                        "exclude_from_batch"
                    ):
                        config_dicts.append(temp_dict)

    return level, config_dicts


def run_calibration(config=None):
    """
    Cameras calibration from checkerboards or from qualisys files.

    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    """

    level, config_dicts = read_config_files(config)
    config_dict = config_dicts[0]
    try:
        session_dir = os.path.realpath(
            [os.getcwd() if level == 2 else os.path.join(os.getcwd(), "..")][0]
        )
        [
            os.path.join(session_dir, c)
            for c in os.listdir(session_dir)
            if "calib" in c.lower() and not c.lower().endswith(".py")
        ][0]
    except Exception:
        session_dir = os.path.realpath(os.getcwd())
    config_dict.get("project").update({"project_dir": session_dir})

    # Set up logging
    setup_logging(session_dir)
    currentDateAndTime = datetime.now()

    # Run calibration
    calib_dir = [
        os.path.join(session_dir, c)
        for c in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, c)) and "calib" in c.lower()
    ][0]
    logging.info(
        "\n---------------------------------------------------------------------"
    )
    logging.info("Camera calibration")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info(f"Calibration directory: {calib_dir}")
    logging.info(
        "---------------------------------------------------------------------\n"
    )
    start = time.time()

    calibrate_cams_all(config_dict)

    end = time.time()
    logging.info(f"\nCalibration took {end - start:.2f} s.\n")


def calibrate_camera_intrinsics(
    session_dir,
    overwrite_intrinsics=True,
    show_detection_intrinsics=False,
    intrinsics_extension="jpg",
    extract_every_N_sec=1,
    intrinsics_corners_nb=(8, 13),
    intrinsics_square_size=20,  # mm
    calculate_extrinsics=True,
    extrinsics_method="board",
    extrinsics_extension="jpg",
    extrinsics_corners_nb=(8, 13),
    extrinsics_square_size=20,  # mm
    show_reprojection_error=True,
):
    # Load the config
    config_file = os.path.join(session_dir, "Config.toml")
    with open(config_file, "r") as f:
        cfg = edict(toml.load(f))

    # Set to calculation mode
    cfg.calibration.calibration_type = "calculate"

    # Set intrinsics calculation parameters
    # [calibration.calculate.intrinsics]
    cfg.calibration.calculate.intrinsics.overwrite_intrinsics = overwrite_intrinsics
    cfg.calibration.calculate.intrinsics.show_detection_intrinsics = (
        show_detection_intrinsics
    )
    cfg.calibration.calculate.intrinsics.intrinsics_extension = intrinsics_extension
    cfg.calibration.calculate.intrinsics.extract_every_N_sec = extract_every_N_sec
    cfg.calibration.calculate.intrinsics.intrinsics_corners_nb = intrinsics_corners_nb
    cfg.calibration.calculate.intrinsics.intrinsics_square_size = intrinsics_square_size

    if calculate_extrinsics:
        # [calibration.calculate.extrinsics]
        cfg.calibration.calculate.extrinsics.calculate_extrinsics = calculate_extrinsics
        cfg.calibration.calculate.extrinsics.extrinsics_method = extrinsics_method
        cfg.calibration.calculate.extrinsics.board.extrinsics_extension = (
            extrinsics_extension
        )
        cfg.calibration.calculate.extrinsics.board.extrinsics_corners_nb = (
            extrinsics_corners_nb
        )
        cfg.calibration.calculate.extrinsics.board.extrinsics_square_size = (
            extrinsics_square_size
        )
        cfg.calibration.calculate.extrinsics.board.show_reprojection_error = (
            show_reprojection_error
        )
    else:
        cfg.calibration.calculate.extrinsics.calculate_extrinsics = False

    # Save the config
    with open(config_file, "w") as f:
        toml.dump(cfg, f)

    # Execute the calibration
    cwd = os.getcwd()
    os.chdir(session_dir)
    calibrated = False
    try:
        run_calibration()
        calibrated = True
    except Exception as e:
        logging.error(f"Failed to calibrate cameras: {e}")

        import traceback

        logging.debug(traceback.format_exc())
    finally:
        os.chdir(cwd)

    return calibrated


def calibrate_camera_pose(
    experiment,
    person_height=1.74,  # meters
    extrinsics_extension="jpg",
):
    extrinsics_folder = os.path.join(experiment.calibration_dir, "extrinsics")

    def estimate_pose(image_path):
        pass

    poses = {}
    for cam in os.listdir(extrinsics_folder):
        cam_folder = os.path.join(extrinsics_folder, cam)
        if not os.path.isdir(cam_folder):
            continue

        # Get the first image in the camera folder
        img_files = [
            f for f in os.listdir(cam_folder) if f.endswith(extrinsics_extension)
        ]
        if len(img_files) == 0:
            raise ValueError(f"No images found for camera '{cam}'")

        # NOTE: This must be a T-pose image showing the full body!!!
        #       which should be automatically verified and an error
        #       should be raised if the image is not valid.
        # TODO: Implement the image validation
        img_file = os.path.join(cam_folder, img_files[0])

        # Estimate the pose from the image
        poses[cam] = estimate_pose(img_file)

    # Define a set of 3D points for the person
    points = [
        (0, 0, 0),  # Origin at middle of heels
        (0, 0, person_height),  # Z-axis pointing upwards (using head top)
    ]

    # Set the middle of heels as the origin
    L_HEEL_IDX = 24
    R_HEEL_IDX = 25
    heel_mid = (poses["cam1"][L_HEEL_IDX] + poses["cam2"][R_HEEL_IDX]) / 2


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Camera calibration script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration directory or file. If not provided, uses the current directory.",
    )
    args = parser.parse_args()
    run_calibration(args.config)


if __name__ == "__main__":
    main()
