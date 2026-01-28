import argparse

from .utils import find_cameras
from .utils.common import configure_opencv_logging


def main():
    configure_opencv_logging(silent=True)
    parser = argparse.ArgumentParser(description="List available cameras.")
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=5,
        help="Maximum number of cameras to probe (default: 5).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec used for probing (default: mp4v).",
    )
    args = parser.parse_args()

    cameras = find_cameras(max_cameras=args.max_cameras, codec=args.codec)
    if not cameras:
        print("No cameras found.")
        return

    print(f"Found {len(cameras)} camera(s):")
    for cam in cameras:
        cam_id = cam.get("id")
        name = cam.get("name", "Unknown")
        manufacturer = cam.get("manufacturer", "Unknown")
        model_number = cam.get("model_number", "Unknown")
        serial_number = cam.get("serial_number", "")
        print(f"- [{cam_id}] {name}")
        print(f"  Manufacturer: {manufacturer}")
        print(f"  Model: {model_number}")
        if serial_number:
            print(f"  Serial: {serial_number}")
        for idx, (width, height, fps) in enumerate(
            cam.get("available_resolutions", [])
        ):
            print(f"  {idx + 1}. {width}x{height} @ {fps:.1f} FPS")


if __name__ == "__main__":
    main()
