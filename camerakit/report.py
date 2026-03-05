import argparse
import os

from .core import CalibrationFile


def main():
    """Print a summary of a calibration TOML file."""
    parser = argparse.ArgumentParser(description="Summarize a calibration TOML file.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Calib_*.toml",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Calibration file not found: {args.input}")

    calib = CalibrationFile(args.input)
    intr_errors = calib.metadata.get("intrinsics_error_px")
    extr_errors = calib.metadata.get("extrinsics_error_px")
    legacy_errors = calib.metadata.get("error")

    print(f"Calibration file: {args.input}")
    print(f"Cameras: {len(calib)}")
    if calib.metadata:
        print(f"Metadata: {calib.metadata}")

    for idx, cam in enumerate(calib.cameras):
        intr_err = getattr(cam, "intrinsics_error_px", None)
        extr_err = getattr(cam, "extrinsics_error_px", None)
        if (
            intr_err is None
            and isinstance(intr_errors, list)
            and idx < len(intr_errors)
        ):
            intr_err = intr_errors[idx]
        if (
            intr_err is None
            and isinstance(legacy_errors, list)
            and idx < len(legacy_errors)
        ):
            intr_err = legacy_errors[idx]
        if (
            extr_err is None
            and isinstance(extr_errors, list)
            and idx < len(extr_errors)
        ):
            extr_err = extr_errors[idx]
        size = tuple(cam.S.tolist())
        dist_len = len(cam.dist)
        intr_str = f"{intr_err:.3f} px" if isinstance(intr_err, (float, int)) else "n/a"
        extr_str = f"{extr_err:.3f} px" if isinstance(extr_err, (float, int)) else "n/a"
        print(
            f"- {cam.id} | size={size[0]}x{size[1]} | dist={dist_len} | intr={intr_str} | extr={extr_str}"
        )


if __name__ == "__main__":
    main()
