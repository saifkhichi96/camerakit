import argparse
from pathlib import Path

from .utils.common import get_logger, setup_logging

logger = get_logger()


DEFAULT_CONFIG = """[project]
project_dir = "."

[calibration]
calibration_type = "calculate"

[calibration.calculate.intrinsics]
overwrite_intrinsics = false
show_detection_intrinsics = true
intrinsics_extension = "mp4"
extract_every_N_sec = 1
intrinsics_corners_nb = [9, 6]
intrinsics_square_size = 25
fisheye = false

[calibration.calculate.extrinsics]
calculate_extrinsics = true
extrinsics_method = "board_outer"

[calibration.calculate.extrinsics.board]
extrinsics_corners_nb = [9, 6]
extrinsics_square_size = 25
extrinsics_extension = "png"
show_reprojection_error = true
"""


def main():
    """Create a CameraKit project layout and default config."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Initialize a CameraKit calibration project."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Destination path for the project.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=2,
        help="Number of camera subfolders to create (default: 2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing Config.toml if present.",
    )
    args = parser.parse_args()

    project_path = Path(args.path).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    config_path = project_path / "Config.toml"
    if config_path.exists() and not args.force:
        logger.error(
            f"Config.toml already exists at {config_path}. Use --force to overwrite."
        )
        return

    calibration_dir = project_path / "calibration"
    intrinsics_dir = calibration_dir / "intrinsics"
    extrinsics_dir = calibration_dir / "extrinsics"

    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(max(args.cameras, 1)):
        cam_name = f"cam_{idx:02d}"
        (intrinsics_dir / cam_name).mkdir(parents=True, exist_ok=True)
        (extrinsics_dir / cam_name).mkdir(parents=True, exist_ok=True)

    config_path.write_text(DEFAULT_CONFIG)
    logger.info(f"Initialized CameraKit project at {project_path}")
    logger.info("Created calibration folders and Config.toml")


if __name__ == "__main__":
    main()
