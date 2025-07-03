from typing import Dict

import cv2
import numpy as np
from easydict import EasyDict as edict


class CalibrationData:
    DEFAULT_RESOLUTION = (640, 480)
    DEFAULT_K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    DEFAULT_DISTORTIONS = [0, 0, 0, 0]
    DEFAULT_ROTATION = [0, 0, 0]
    DEFAULT_TRANSLATION = [0, 0, 0]

    """
    Represents a single camera with its calibration parameters.

    Attributes:
        id (str): Unique identifier for the camera.
        S (np.ndarray): Image size as [height, width].
        K (np.ndarray): Intrinsic camera matrix (3x3).
        dist (np.ndarray): Distortion coefficients (array of length 4 or 5).
        rvec (np.ndarray): Rotation vector (3x1).
        t (np.ndarray): Translation vector (3x1).
        fisheye (bool): Flag indicating if fisheye distortion is used.
        K_optim (np.ndarray): Optimal new camera matrix for undistorting points (3x3).
        K_inv (np.ndarray): Inverse of the intrinsic camera matrix (3x3).
        R (np.ndarray): Rotation matrix obtained from the rotation vector (3x3).
    """

    def __init__(
        self,
        info: Dict[str, any],
        alpha: float = 1.0,
        undistort: bool = False,
    ):
        """
        Initializes the Camera instance with calibration parameters.

        Args:
            info (Dict[str, any]): Dictionary containing camera calibration parameters, including:
                - "name": Unique identifier for the camera. Required.
                - "size": Image size as [width, height]. Optional, defaults to (640, 480).
                - "matrix": Intrinsic camera matrix (3x3). Optional, defaults to identity matrix.
                - "distortions": Distortion coefficients (array of length 4 or 5). Optional, defaults to [0, 0, 0, 0].
                - "rotation": Rotation vector (3x1). Optional, defaults to [0, 0, 0].
                - "translation": Translation vector (3x1). Optional, defaults to [0, 0, 0].
                - "fisheye": Flag indicating if fisheye distortion is used. Optional, defaults to False.
            alpha (float): Free scaling parameter for the optimal new camera matrix. Optional, defaults to 1.0.
            undistort (bool): If True, computes the projection matrix using the optimal camera matrix for undistorting points. Optional, defaults to False.
        """
        self.id = info["name"]
        self.S = np.array(info.get("size", self.DEFAULT_RESOLUTION), dtype=int)
        self.K = np.array(info.get("matrix", self.DEFAULT_K), dtype=float)
        self.dist = np.array(
            info.get("distortions", self.DEFAULT_DISTORTIONS), dtype=float
        )
        self.rvec = np.array(info.get("rotation", self.DEFAULT_ROTATION), dtype=float)
        self.t = np.array(
            info.get("translation", self.DEFAULT_TRANSLATION), dtype=float
        )
        self.fisheye = info.get("fisheye", False)

        # Compute the optimal new camera matrix for undistorting points
        # The free scaling parameter (alpha) is set to 1, which means the entire image is retained.
        # If alpha is set to 0, the optimal camera matrix will be computed such that
        # the undistorted image is cropped to the largest rectangle that fits inside the original image.
        self.K_optim, _ = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.K,
            distCoeffs=self.dist,
            imageSize=tuple(self.S),
            alpha=alpha,
            newImgSize=tuple(self.S),
            centerPrincipalPoint=False,
        )

        # Compute the inverse of the intrinsic camera matrix
        self.K_inv = np.linalg.inv(self.K_optim) if undistort else np.linalg.inv(self.K)

        # Convert rotation vector to rotation matrix
        self.R, _ = cv2.Rodrigues(self.rvec)

        inverted = info.get("inverted", False)
        if inverted:
            self.R = self.R.T
            self.t = -self.R @ self.t

        if "projection_matrix" in info:
            self.P = np.array(info["projection_matrix"], dtype=float)
        else:
            self.P = self._compute_P(undistort=undistort)

    def _compute_P(self, undistort=False):
        """
        Computes the projection matrix for the camera.

        Args:
            undistort (bool): If True, uses the optimal camera matrix for undistorting points.

        Returns:
            np.ndarray: Projection matrix (3x4).
        """
        if undistort:
            Kh = np.hstack([self.K_optim, np.zeros((3, 1))])
        else:
            Kh = np.hstack([self.K, np.zeros((3, 1))])

        # Construct the [R | T] matrix
        RT = np.hstack([self.R, self.t.reshape(3, 1)])

        # Append a row [0, 0, 0, 1] to make it a 4x4 matrix
        H = np.vstack([RT, [0, 0, 0, 1]])

        # Compute the projection matrix
        return Kh @ H  # Resulting in a 3x4 matrix

    @property
    def intrinsics(self):
        """
        Returns the intrinsic parameters of the camera.

        Returns:
            dict: Dictionary containing intrinsic parameters.
        """
        return edict(
            {
                "K": self.K,
                "K_optim": self.K_optim,
                "K_inv": self.K_inv,
                "dist": self.dist,
            }
        )

    @property
    def extrinsics(self):
        """
        Returns the extrinsic parameters of the camera.

        Returns:
            dict: Dictionary containing extrinsic parameters.
        """
        return edict(
            {
                "rvec": self.rvec,
                "R": self.R,
                "t": self.t,
            }
        )
