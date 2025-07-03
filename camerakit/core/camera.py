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
        """
        Create a CameraInfo object from a dictionary.
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
        """
        Generate a unique ID for the camera based on its properties.
        The ID is generated using the MD5 algorithm.
        """
        unique_str = f"{self.manufacturer}_{self.serial_number}_{self.model_number}_{self.resolution}"
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        """
        Convert the camera information to a dictionary.
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
        return f"CameraInfo(manufacturer={self.manufacturer}, serial_number={self.serial_number}, model_number={self.model_number}, resolution={self.resolution}, name={self.name})"

    def __eq__(self, other):
        return (
            isinstance(other, CameraInfo)
            and self.manufacturer == other.manufacturer
            and self.serial_number == other.serial_number
            and self.model_number == other.model_number
            and self.resolution == other.resolution
        )

    def __hash__(self):
        """
        Generate a hash for the camera information.
        This is used to ensure that the camera information can be used as a key in dictionaries or sets.
        """
        return hash(
            (self.manufacturer, self.serial_number, self.model_number, self.resolution)
        )
