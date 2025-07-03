import os
import time
from copy import deepcopy
from datetime import datetime

import toml

from .utils import calibrate_cams_all, setup_logging


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
    logger = setup_logging(session_dir)
    currentDateAndTime = datetime.now()

    # Run calibration
    calib_dir = [
        os.path.join(session_dir, c)
        for c in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, c)) and "calib" in c.lower()
    ][0]
    logger.info(
        "\n---------------------------------------------------------------------"
    )
    logger.info("Camera calibration")
    logger.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logger.info(f"Calibration directory: {calib_dir}")
    logger.info(
        "---------------------------------------------------------------------\n"
    )
    start = time.time()

    calibrate_cams_all(config_dict)

    end = time.time()
    logger.info(f"\nCalibration took {end - start:.2f} s.\n")


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
