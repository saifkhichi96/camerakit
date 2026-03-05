import platform
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from .common import (
    find_supported_resolutions_and_fps,
    get_camera_hardware_macos,
    get_camera_properties,
    get_logger,
    suppress_stderr,
)


@dataclass
class CaptureSettings:
    """Capture mode supported by a camera.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate.
        codec: FourCC codec string.
    """

    width: int
    height: int
    fps: float
    codec: str

    def __str__(self):
        """Return a human-readable description of this capture setting.

        Returns:
            str: Formatted setting string.
        """
        return f"{self.width}x{self.height} @ {self.fps} FPS ({self.codec})"


@dataclass
class CameraMetadata:
    """Metadata and supported settings for one discovered camera.

    Attributes:
        id: Numeric camera index.
        name: Human-readable camera name.
        settings: Supported capture settings.
        manufacturer: Optional manufacturer name.
        model_number: Optional model identifier.
        serial_number: Optional serial identifier.
    """

    id: int
    name: str
    settings: List[CaptureSettings]
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    serial_number: Optional[str] = None

    @property
    def width(self) -> int:
        """Return width from the first configured setting.

        Returns:
            int: Width in pixels, or `0` when no settings exist.
        """
        if len(self.settings) > 0:
            return self.settings[0].width
        return 0

    @property
    def height(self) -> int:
        """Return height from the first configured setting.

        Returns:
            int: Height in pixels, or `0` when no settings exist.
        """
        if len(self.settings) > 0:
            return self.settings[0].height
        return 0

    @property
    def fps(self) -> float:
        """Return FPS from the first configured setting.

        Returns:
            float: Frames per second, or `0.0` when no settings exist.
        """
        if len(self.settings) > 0:
            return self.settings[0].fps
        return 0.0

    @property
    def codec(self) -> str:
        """Return codec from the first configured setting.

        Returns:
            str: Codec string, or `"MJPG"` when no settings exist.
        """
        if len(self.settings) > 0:
            return self.settings[0].codec
        return "MJPG"

    def supports(self, width: int, height: int, fps: float, codec: str) -> bool:
        """Check whether a specific capture mode is supported.

        Args:
            width: Requested width in pixels.
            height: Requested height in pixels.
            fps: Requested frame rate.
            codec: Requested codec.

        Returns:
            bool: `True` when the setting is available.
        """
        return any(
            setting.width == width
            and setting.height == height
            and setting.fps == fps
            and setting.codec == codec
            for setting in self.settings
        )


class CameraEnumerator:
    """Class to enumerate available cameras."""

    def __init__(
        self,
        max_cameras=5,
        aspect_ratios=[(3, 2), (4, 3), (16, 9)],
        common_widths=[640, 800, 960, 1024, 1280, 1600, 1920, 2560, 3840, 4096],
        codecs=("MJPG", "YUYV", "H264"),
    ):
        """Initialize camera discovery settings.

        Args:
            max_cameras: Maximum number of camera indices to probe.
            aspect_ratios: Aspect ratios used for resolution probing.
            common_widths: Candidate widths used for resolution probing.
            codecs: Candidate codec names used during probing.
        """
        self.max_cameras = max_cameras
        self.aspect_ratios = aspect_ratios
        self.common_widths = common_widths
        self.codecs = codecs

    def list(self) -> List[CameraMetadata]:
        """List all available cameras with metadata and supported settings.

        Returns:
            list[CameraMetadata]: Discovered camera metadata entries.
        """
        with suppress_stderr():
            system = platform.system()
            discovered = []

            if system == "Darwin":  # macOS
                for cam in get_camera_hardware_macos():
                    cap = cv2.VideoCapture(cam["id"])
                    if not cap.isOpened():
                        get_logger().error(
                            f"Failed to open camera {cam['id']} on macOS."
                        )
                        continue

                    settings = find_supported_resolutions_and_fps(
                        cap,
                        codecs=self.codecs,
                        aspect_ratios=self.aspect_ratios,
                        common_widths=self.common_widths,
                    )
                    cap.release()
                    if not settings:
                        continue
                    cam["settings"] = settings
                    discovered.append(cam)
            else:
                for camera_id in range(2 * self.max_cameras):  # OpenCV often skips IDs.
                    camera = get_camera_properties(
                        camera_id,
                        aspect_ratios=self.aspect_ratios,
                        common_widths=self.common_widths,
                        codecs=self.codecs,
                    )
                    if camera:
                        discovered.append(camera)

        metadata = []
        for cam in discovered:
            metadata.append(
                CameraMetadata(
                    id=cam["id"],
                    name=cam["name"],
                    manufacturer=cam.get("manufacturer"),
                    model_number=cam.get("model_number"),
                    serial_number=cam.get("serial_number"),
                    settings=[
                        CaptureSettings(width=w, height=h, fps=fps, codec=codec)
                        for w, h, fps, codec in cam["settings"]
                    ],
                )
            )

        return metadata

    @staticmethod
    def _setting_key(setting: CaptureSettings) -> Tuple[int, int, float, str]:
        """
        Build a stable key for setting comparisons.

        Args:
            setting: Capture setting to normalize.

        Returns:
            tuple[int, int, float, str]: Comparable setting key with normalized FPS/codec.
        """
        return (
            setting.width,
            setting.height,
            round(setting.fps, 3),
            setting.codec.upper(),
        )

    def list_synchronizable(self) -> List[CameraMetadata]:
        """List cameras that share compatible capture settings.

        Returns:
            list[CameraMetadata]: Cameras with settings filtered to synchronizable modes.
        """
        cameras = self.list()
        if not cameras:
            return []

        # Map each setting to the set of camera IDs that support it.
        # Use per-camera uniqueness so duplicate entries from one camera
        # are not treated as multi-camera compatibility.
        setting_to_camera_ids = {}
        for cam in cameras:
            unique_keys = {self._setting_key(s) for s in cam.settings}
            for key in unique_keys:
                setting_to_camera_ids.setdefault(key, set()).add(cam.id)

        min_shared_cameras = 1 if len(cameras) == 1 else 2
        common_settings = {
            key
            for key, camera_ids in setting_to_camera_ids.items()
            if len(camera_ids) >= min_shared_cameras
        }
        if not common_settings:
            return []

        # Filter each camera's settings to only common ones, preserving order.
        synchronizable_cameras = []
        for cam in cameras:
            filtered_settings = []
            seen = set()
            for setting in cam.settings:
                key = self._setting_key(setting)
                if key in common_settings and key not in seen:
                    filtered_settings.append(setting)
                    seen.add(key)

            if filtered_settings:
                synchronizable_cameras.append(
                    CameraMetadata(
                        id=cam.id,
                        name=cam.name,
                        manufacturer=cam.manufacturer,
                        model_number=cam.model_number,
                        serial_number=cam.serial_number,
                        settings=filtered_settings,
                    )
                )

        return synchronizable_cameras


def find_cameras(
    max_cameras=5,
    aspect_ratios=[(3, 2), (4, 3), (16, 9)],
    common_widths=[640, 800, 960, 1024, 1280, 1600, 1920, 2560, 3840, 4096],
    codecs=("MJPG", "YUYV", "H264"),
) -> List[CameraMetadata]:
    """Find available cameras as typed metadata objects.

    Args:
        max_cameras: Maximum number of camera indices to probe.
        aspect_ratios: Aspect ratios used for resolution probing.
        common_widths: Candidate widths used for resolution probing.
        codecs: Candidate codec names used during probing.

    Returns:
        list[CameraMetadata]: Discovered camera metadata entries.
    """
    return CameraEnumerator(
        max_cameras=max_cameras,
        aspect_ratios=aspect_ratios,
        common_widths=common_widths,
        codecs=codecs,
    ).list()
