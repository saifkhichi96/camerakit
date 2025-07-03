import logging
import os
from typing import Dict, List, Optional

import numpy as np
import toml

from .calibration_data import CalibrationData


class CalibrationFile:
    """
    Represents a system of multiple cameras with their calibration parameters.

    Attributes:
        cameras (list of Camera): List of Camera instances in the system.
    """

    def __init__(self, path: str):
        """
        Initializes the CalibrationFile by loading calibration data from a TOML file.

        Args:
            calib_file (str): Path to the calibration TOML file.
        """
        self.calib_file = path
        self.cameras: List[CalibrationData] = []
        self.metadata: Dict[str, any] = {}
        self.refresh()

    def _parse(self, calib_file: str) -> List[CalibrationData]:
        """
        Loads calibration parameters from a TOML file and initializes Camera instances.

        Args:
            calib_file (str): Path to the calibration TOML file.

        Returns:
            list of Camera: List of initialized Camera instances.
        """
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file {calib_file} does not exist.")

        calib = toml.load(calib_file)

        metadata = calib.get("metadata", {})
        if not isinstance(metadata, dict):
            logging.warning(
                "Metadata in calibration file is not a dictionary. Using empty metadata."
            )
            metadata = {}

        # Filter out irrelevant keys and ensure we're only processing camera configurations
        cal_keys = [
            c
            for c in calib
            if c not in ["metadata", "capture_volume", "charuco", "checkerboard"]
            and isinstance(calib[c], dict)
        ]

        # Initialize Camera instances for each camera in the calibration file
        cameras = [CalibrationData(calib[cam]) for cam in cal_keys]

        if not cameras:
            logging.warning(
                "No valid camera configurations found in the calibration file."
            )

        return cameras, metadata

    @property
    def id(self):
        """
        Retrieves the identifiers for all cameras.

        Returns:
            list of str: List of camera IDs.
        """
        return [camera.id for camera in self.cameras]

    @property
    def S(self) -> np.ndarray:
        """
        Retrieves the image sizes for all cameras.

        Returns:
            np.ndarray: Array of camera resolutions with shape (N, 2),
                        where N is the number of cameras and each entry is [height, width].
        """
        return np.array([camera.S for camera in self.cameras], dtype=int)

    @property
    def K(self) -> np.ndarray:
        """
        Retrieves the intrinsic matrices (K) for all cameras.

        Returns:
            np.ndarray: Array of intrinsic matrices with shape (N, 3, 3),
                        where N is the number of cameras and each entry is a 3x3 matrix.
        """
        return np.array([camera.K for camera in self.cameras], dtype=float)

    @property
    def K_optim(self) -> np.ndarray:
        """
        Retrieves the optimized intrinsic matrices for all cameras.

        Returns:
            np.ndarray: Array of optimized intrinsic matrices with shape (N, 3, 3).
        """
        return np.array([camera.K_optim for camera in self.cameras], dtype=float)

    @property
    def K_inv(self) -> np.ndarray:
        """
        Retrieves the inverse intrinsic matrices for all cameras.

        Returns:
            np.ndarray: Array of inverse intrinsic matrices with shape (N, 3, 3).
        """
        return np.array([camera.K_inv for camera in self.cameras], dtype=float)

    @property
    def dist(self) -> np.ndarray:
        """
        Retrieves the distortion coefficients for all cameras.

        Returns:
            np.ndarray: Array of distortion coefficients with shape (N, 4) or (N, 5).
        """
        return np.array([camera.dist for camera in self.cameras], dtype=float)

    @property
    def rvec(self) -> np.ndarray:
        """
        Retrieves the rotation vectors for all cameras.

        Returns:
            np.ndarray: Array of rotation vectors with shape (N, 3, 1).
        """
        return np.array([camera.rvec for camera in self.cameras], dtype=float)

    @property
    def R(self) -> np.ndarray:
        """
        Retrieves the rotation matrices for all cameras.

        Returns:
            np.ndarray: Array of rotation matrices with shape (N, 3, 3).
        """
        return np.array([camera.R for camera in self.cameras], dtype=float)

    @property
    def t(self) -> np.ndarray:
        """
        Retrieves the translation vectors for all cameras.

        Returns:
            np.ndarray: Array of translation vectors with shape (N, 3, 1).
        """
        return np.array([camera.t for camera in self.cameras], dtype=float)

    @property
    def P(self) -> np.ndarray:
        """
        Retrieves the projection matrices for all cameras in the system.

        Returns:
            np.ndarray: Array of projection matrices with shape (N, 3, 4).
        """
        return np.array([camera.P for camera in self.cameras], dtype=float)

    @property
    def intrinsics(self) -> List[Dict[str, np.ndarray]]:
        """
        Retrieves intrinsic parameters for all cameras.

        Returns:
            list of dict: List of dictionaries containing intrinsic parameters for each camera.
        """
        return [camera.intrinsics for camera in self.cameras]

    @property
    def extrinsics(self) -> List[Dict[str, np.ndarray]]:
        """
        Retrieves extrinsic parameters for all cameras.

        Returns:
            list of dict: List of dictionaries containing extrinsic parameters for each camera.
        """
        return [camera.extrinsics for camera in self.cameras]

    def get(self, camera_id: str, default=None) -> Optional[CalibrationData]:
        """
        Retrieves a specific Camera instance from the system.

        Args:
            camera_id (str): Identifier of the camera to retrieve.

        Returns:
            Camera: The Camera instance with the specified ID.
        """
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera

        return default

    def refresh(self):
        """
        Refreshes the CalibrationFile by reloading calibration data from the TOML file.
        """
        cams, metadata = self._parse(self.calib_file)
        self.cameras = cams
        self.metadata = metadata

    def to_dict(self):
        """
        Converts the CalibrationFile to a dictionary.

        Returns:
            dict: Dictionary representation of the CalibrationFile.
        """
        return {
            "S": self.S,
            "K": self.K,
            "dist": self.dist,
            "K_inv": self.K_inv,
            "K_optim": self.K_optim,
            "rvec": self.rvec,
            "R": self.R,
            "t": self.t,
            "P": self.P,
            "metadata": self.metadata,
        }

    def __repr__(self):
        """
        Returns a string representation of the CalibrationFile.

        Returns:
            str: String representation of the CalibrationFile.
        """
        return f"CalibrationFile(calib_file={self.calib_file}, cameras={self.cameras})"

    def __len__(self):
        """
        Returns the number of cameras in the CalibrationFile.

        Returns:
            int: Number of cameras.
        """
        return len(self.cameras)

    def __getitem__(self, index):
        """
        Retrieves a Camera instance by index.

        Args:
            index (int): Index of the camera to retrieve.

        Returns:
            Camera: The Camera instance at the specified index.
        """
        return self.cameras[index]

    def __iter__(self):
        """
        Returns an iterator over the Camera instances in the CalibrationFile.

        Returns:
            iterator: Iterator over Camera instances.
        """
        return iter(self.cameras)

    def __contains__(self, camera_id: str) -> bool:
        """
        Checks if a camera with the specified ID exists in the CalibrationFile.

        Args:
            camera_id (str): Identifier of the camera to check.

        Returns:
            bool: True if the camera exists, False otherwise.
        """
        return any(camera.id == camera_id for camera in self.cameras)
