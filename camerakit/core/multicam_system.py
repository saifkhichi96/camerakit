import logging
import os

import toml

from .camera import Camera


class MultiCameraManager:
    """
    Represents a system of multiple cameras with their calibration parameters.

    Attributes:
        cameras (list of Camera): List of Camera instances in the system.
    """

    def __init__(self, calib_file):
        """
        Initializes the CameraSystem by loading calibration data from a TOML file.

        Args:
            calib_file (str): Path to the calibration TOML file.
        """
        self.calib_file = calib_file
        self.cameras = self._load_calibration(calib_file)

    def refresh(self):
        """
        Refreshes the CameraSystem by reloading calibration data from the TOML file.
        """
        self.cameras = self._load_calibration(self.calib_file)

    def _load_calibration(self, calib_file):
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

        # Filter out irrelevant keys and ensure we're only processing camera configurations
        cal_keys = [
            c
            for c in calib
            if c not in ["metadata", "capture_volume", "charuco", "checkerboard"]
            and isinstance(calib[c], dict)
        ]

        # Initialize Camera instances for each camera in the calibration file
        cameras = [Camera(calib[cam]) for cam in cal_keys]

        if not cameras:
            logging.warning(
                "No valid camera configurations found in the calibration file."
            )

        return cameras

    def get_camera(self, camera_id):
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

        raise ValueError(f"Camera with ID {camera_id} not found in the system.")

    def get_projection_matrices(self, undistort=False):
        """
        Retrieves the projection matrices for all cameras in the system.

        Args:
            undistort (bool): If True, uses the optimal camera matrix for undistorting points.

        Returns:
            list of np.ndarray: List of projection matrices (each 3x4).
        """
        return [camera.compute_projection_matrix(undistort) for camera in self.cameras]

    @property
    def intrinsic_matrices(self):
        """
        Retrieves the intrinsic matrices (K) for all cameras.

        Returns:
            list of np.ndarray: List of intrinsic matrices (3x3).
        """
        return [camera.K for camera in self.cameras]

    @property
    def distortion_coefficients(self):
        """
        Retrieves the distortion coefficients for all cameras.

        Returns:
            list of np.ndarray: List of distortion coefficients arrays.
        """
        return [camera.dist for camera in self.cameras]

    @property
    def rotation_matrices(self):
        """
        Retrieves the rotation matrices for all cameras.

        Returns:
            list of np.ndarray: List of rotation matrices (3x3).
        """
        return [camera.R_mat for camera in self.cameras]

    @property
    def translation_vectors(self):
        """
        Retrieves the translation vectors for all cameras.

        Returns:
            list of np.ndarray: List of translation vectors (3x1).
        """
        return [camera.translation_vector for camera in self.cameras]

    @property
    def image_sizes(self):
        """
        Retrieves the image sizes for all cameras.

        Returns:
            list of np.ndarray: List of image sizes as [height, width].
        """
        return [camera.image_size for camera in self.cameras]

    @property
    def camera_ids(self):
        """
        Retrieves the identifiers for all cameras.

        Returns:
            list of str: List of camera IDs.
        """
        return [camera.id for camera in self.cameras]

    def get_intrinsic_parameters(self):
        """
        Retrieves intrinsic parameters for all cameras.

        Returns:
            list of dict: List of dictionaries containing intrinsic parameters for each camera.
        """
        return [camera.intrinsic_parameters for camera in self.cameras]

    def get_extrinsic_parameters(self):
        """
        Retrieves extrinsic parameters for all cameras.

        Returns:
            list of dict: List of dictionaries containing extrinsic parameters for each camera.
        """
        return [camera.extrinsic_parameters for camera in self.cameras]

    def to_dict(self, undistort=False):
        """
        Converts the CameraSystem to a dictionary.

        Args:
            undistort (bool): If True, compute projection matrices using optimal intrinsic matrices.

        Returns:
            dict: Dictionary representation of the CameraSystem.
        """
        # Compute undistorted intrinsic matrices if requested
        optim_K = [camera.optim_K if undistort else camera.K for camera in self.cameras]

        return {
            "S": self.image_sizes,
            "K": self.intrinsic_matrices,
            "dist": self.distortion_coefficients,
            "inv_K": [camera.inv_K for camera in self.cameras],
            "optim_K": optim_K,
            "R": [camera.rotation_vector for camera in self.cameras],
            "R_mat": self.rotation_matrices,
            "T": self.translation_vectors,
            "P": self.get_projection_matrices(undistort=undistort),
        }
