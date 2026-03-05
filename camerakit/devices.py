import argparse

from .utils import CameraEnumerator
from .utils.common import configure_opencv_logging


def main():
    """List discoverable cameras and their metadata."""
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
        default=None,
        help="Optional comma-separated FourCC codecs used for probing (e.g., MJPG,YUYV).",
    )
    args = parser.parse_args()

    codecs = None
    if args.codec:
        codecs = tuple(
            codec.strip().upper() for codec in args.codec.split(",") if codec.strip()
        )

    enumerator = CameraEnumerator(
        max_cameras=args.max_cameras,
        codecs=codecs or ("MJPG", "YUYV", "H264"),
    )
    cameras = enumerator.list()
    if not cameras:
        print("No cameras found.")
        return

    print(f"Found {len(cameras)} camera(s):")
    for cam in cameras:
        cam_id = cam.id
        name = cam.name or "Unknown"
        manufacturer = cam.manufacturer or "Unknown"
        model_number = cam.model_number or "Unknown"
        serial_number = cam.serial_number or ""
        print(f"- [{cam_id}] {name}")
        print(f"  Manufacturer: {manufacturer}")
        print(f"  Model: {model_number}")
        if serial_number:
            print(f"  Serial: {serial_number}")
        if not cam.settings:
            print("  No supported settings detected.")
            continue
        for idx, setting in enumerate(cam.settings):
            print(
                f"  {idx + 1}. "
                f"{setting.width}x{setting.height} @ {setting.fps:.1f} FPS ({setting.codec})"
            )


if __name__ == "__main__":
    main()
