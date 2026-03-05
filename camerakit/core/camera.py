import hashlib


class CameraInfo:
    """
    Class representing camera information.

    Attributes:
        id (str): Unique ID of the camera. Unique auto-generated identifier.
        manufacturer (str): Manufacturer of the camera.
        serial_number (str): Serial number of the camera.
        model_number (str): Model number of the camera.
        resolution (tuple): Resolution of the camera in pixels (width, height).
        name (str): Name of the camera. Unique user-defined identifier. Optional.
        frame_rate (int): Frame rate of the camera. Optional, default is -1.
        K (list): Camera intrinsic matrix. 3x3 matrix. Optional, default is a zero matrix.
        D (list): Distortion coefficients. 1x5 list. Optional, default is a zero list.
        reprojection_error (float): Reprojection error. Optional, default is 0.0.
    """

    def __init__(
        self,
        manufacturer: str,
        serial_number: str,
        model_number: str,
        resolution: tuple,
        name: str = None,
        frame_rate: int = -1,
        K: list = None,
        D: list = None,
        reprojection_error: float = 0.0,
    ):
        """Initialize camera metadata and calibration values.

        Args:
            manufacturer: Camera manufacturer name.
            serial_number: Camera serial number.
            model_number: Camera model number.
            resolution: Resolution as `(width, height)`.
            name: Optional display name. If omitted, generated from metadata.
            frame_rate: Frame rate in FPS. Defaults to `-1` when unknown.
            K: Intrinsic matrix.
            D: Distortion coefficients.
            reprojection_error: Reprojection error in pixels.
        """
        self.name = name
        self.manufacturer = manufacturer
        self.serial_number = serial_number
        self.model_number = model_number
        self.resolution = resolution
        self.frame_rate = frame_rate

        # Initialize camera parameters
        self.K = K or [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.D = D or [0, 0, 0, 0, 0]
        self.reprojection_error = reprojection_error

        if self.name is None:
            # Use a default name if not provided
            self.name = f"{self.manufacturer} {self.model_number} ({self.resolution})"
            if self.frame_rate > 0:
                self.name += f" @ {self.frame_rate} FPS"

    @staticmethod
    def from_dict(data: dict) -> "CameraInfo":
        """Build a `CameraInfo` object from a dictionary.

        Args:
            data: Camera data dictionary.

        Returns:
            CameraInfo: Parsed camera info instance.
        """
        return CameraInfo(
            name=data.get("name"),
            manufacturer=data["manufacturer"],
            serial_number=data["serial_number"],
            model_number=data["model_number"],
            resolution=data["resolution"],
            frame_rate=data.get("frame_rate", -1),
            K=data.get("K", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            D=data.get("D", [0, 0, 0, 0, 0]),
            reprojection_error=data.get("reprojection_error", 0.0),
        )

    @property
    def id(self) -> str:
        """Generate a deterministic camera ID from hardware properties.

        Returns:
            str: MD5 hash computed from manufacturer, serial, model, and resolution.
        """
        unique_str = f"{self.manufacturer}_{self.serial_number}_{self.model_number}_{self.resolution}"
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        """Serialize camera info to a dictionary.

        Returns:
            dict: Dictionary representation of the camera.
        """
        return {
            "id": self.id,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "serial_number": self.serial_number,
            "model_number": self.model_number,
            "resolution": self.resolution,
            "frame_rate": self.frame_rate,
            "K": self.K,
            "D": self.D,
            "reprojection_error": self.reprojection_error,
        }

    def __repr__(self):
        """Return a developer-friendly string representation.

        Returns:
            str: Printable representation of this camera metadata object.
        """
        return f"CameraInfo(manufacturer={self.manufacturer}, serial_number={self.serial_number}, model_number={self.model_number}, resolution={self.resolution}, name={self.name})"

    def __eq__(self, other):
        """Compare cameras by immutable hardware-identifying fields.

        Args:
            other: Object to compare against.

        Returns:
            bool: `True` when both objects represent the same hardware camera.
        """
        return (
            isinstance(other, CameraInfo)
            and self.manufacturer == other.manufacturer
            and self.serial_number == other.serial_number
            and self.model_number == other.model_number
            and self.resolution == other.resolution
        )

    def __hash__(self):
        """Hash camera identity fields for set/dict usage.

        Returns:
            int: Hash value for this camera.
        """
        return hash(
            (self.manufacturer, self.serial_number, self.model_number, self.resolution)
        )
