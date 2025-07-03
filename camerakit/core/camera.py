import hashlib

import cv2
import numpy as np


class Camera:
    """
    Represents a single camera with its calibration parameters.

    Attributes:
        id (str): Unique identifier for the camera.
        image_size (np.ndarray): Image size as [height, width].
        K (np.ndarray): Intrinsic camera matrix (3x3).
        dist (np.ndarray): Distortion coefficients (array of length 4 or 5).
        rotation_vector (np.ndarray): Rotation vector (3x1).
        translation_vector (np.ndarray): Translation vector (3x1).
        fisheye (bool): Flag indicating if fisheye distortion is used.
        optim_K (np.ndarray): Optimal new camera matrix for undistorting points (3x3).
        inv_K (np.ndarray): Inverse of the intrinsic camera matrix (3x3).
        R_mat (np.ndarray): Rotation matrix obtained from the rotation vector (3x3).
    """

    def __init__(self, config):
        """
        Initializes the Camera instance with calibration parameters.

        Args:
            config (dict): Calibration configuration dictionary for the camera.
        """
        self.id = config["name"]
        self.image_size = np.array(config.get("size", [1920, 1080]), dtype=int)
        self.K = np.array(
            config.get("matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=float
        )
        self.dist = np.array(config.get("distortions", [0, 0, 0, 0]), dtype=float)
        self.rotation_vector = np.array(config.get("rotation", [0, 0, 0]), dtype=float)
        self.translation_vector = np.array(
            config.get("translation", [0, 0, 0]), dtype=float
        )
        self.fisheye = config.get("fisheye", False)

        # Compute the optimal new camera matrix for undistorting points
        self.optim_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, tuple(self.image_size), 1, tuple(self.image_size)
        )

        # Compute the inverse of the intrinsic camera matrix
        self.inv_K = np.linalg.inv(self.K)

        # Convert rotation vector to rotation matrix
        self.R_mat, _ = cv2.Rodrigues(self.rotation_vector)

        inverted = config.get("inverted", False)
        if inverted:
            self.R_mat = self.R_mat.T
            self.translation_vector = -self.R_mat @ self.translation_vector

        if "projection_matrix" in config:
            self.P = np.array(config["projection_matrix"], dtype=float)
        else:
            self.P = self.compute_projection_matrix()

    def compute_projection_matrix(self, undistort=False):
        """
        Computes the projection matrix for the camera.

        Args:
            undistort (bool): If True, uses the optimal camera matrix for undistorting points.

        Returns:
            np.ndarray: Projection matrix (3x4).
        """
        if undistort:
            Kh = np.hstack([self.optim_K, np.zeros((3, 1))])
        else:
            Kh = np.hstack([self.K, np.zeros((3, 1))])

        # Construct the [R | T] matrix
        RT = np.hstack([self.R_mat, self.translation_vector.reshape(3, 1)])

        # Append a row [0, 0, 0, 1] to make it a 4x4 matrix
        H = np.vstack([RT, [0, 0, 0, 1]])

        # Compute the projection matrix
        return Kh @ H  # Resulting in a 3x4 matrix

    @property
    def intrinsic_parameters(self):
        """
        Returns the intrinsic parameters of the camera.

        Returns:
            dict: Dictionary containing intrinsic parameters.
        """
        return {
            "K": self.K,
            "optim_K": self.optim_K,
            "inv_K": self.inv_K,
            "dist": self.dist,
        }

    @property
    def extrinsic_parameters(self):
        """
        Returns the extrinsic parameters of the camera.

        Returns:
            dict: Dictionary containing extrinsic parameters.
        """
        return {
            "rotation_matrix": self.R_mat,
            "rotation_vector": self.rotation_vector,
            "translation_vector": self.translation_vector,
        }


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
