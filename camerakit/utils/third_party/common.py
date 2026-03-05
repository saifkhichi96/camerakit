import re

import cv2
import numpy as np


def natural_sort_key(s):
    """Build a natural-sort key for strings containing numbers.

    Args:
        s: Input value convertible to string.

    Returns:
        list[int | str]: Tokenized key suitable for natural ordering.
    """
    s = str(s)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def world_to_camera_persp(r, t):
    """Convert extrinsics between world-centered and camera-centered forms.

    Args:
        r: Rotation matrix.
        t: Translation vector.

    Returns:
        tuple[np.ndarray, np.ndarray]: Converted rotation and translation.
    """

    r = r.T
    t = -r @ t

    return r, t


def rotate_cam(r, t, ang_x=0, ang_y=0, ang_z=0):
    """Rotate camera extrinsics around camera axes.

    Args:
        r: Rotation as matrix or Rodrigues vector.
        t: Translation vector.
        ang_x: Rotation around X axis in radians.
        ang_y: Rotation around Y axis in radians.
        ang_z: Rotation around Z axis in radians.

    Returns:
        tuple[np.ndarray, np.ndarray]: Rotated rotation matrix and translation vector.
    """

    r, t = np.array(r), np.array(t)
    if r.shape == (3, 3):
        rt_h = np.block([[r, t.reshape(3, 1)], [np.zeros(3), 1]])
    elif r.shape == (3,):
        rt_h = np.block([[cv2.Rodrigues(r)[0], t.reshape(3, 1)], [np.zeros(3), 1]])

    r_ax_x = np.array(
        [1, 0, 0, 0, np.cos(ang_x), -np.sin(ang_x), 0, np.sin(ang_x), np.cos(ang_x)]
    ).reshape(3, 3)
    r_ax_y = np.array(
        [np.cos(ang_y), 0, np.sin(ang_y), 0, 1, 0, -np.sin(ang_y), 0, np.cos(ang_y)]
    ).reshape(3, 3)
    r_ax_z = np.array(
        [np.cos(ang_z), -np.sin(ang_z), 0, np.sin(ang_z), np.cos(ang_z), 0, 0, 0, 1]
    ).reshape(3, 3)
    r_ax = r_ax_z @ r_ax_y @ r_ax_x

    r_ax_h = np.block([[r_ax, np.zeros(3).reshape(3, 1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h @ rt_h

    r = r_ax_h__rt_h[:3, :3]
    t = r_ax_h__rt_h[:3, 3]

    return r, t


def quat2mat(quat, scalar_idx=0):
    """Convert a quaternion to a rotation matrix.

    Args:
        quat: Quaternion values.
        scalar_idx: Index of scalar part (`0` or `3`).

    Returns:
        np.ndarray: Rotation matrix of shape `(3, 3)`.
    """

    if scalar_idx == 0:
        w, qx, qy, qz = np.array(quat)
    elif scalar_idx == 3:
        qx, qy, qz, w = np.array(quat)
    else:
        print("Error: scalar_idx should be 0 or 3")

    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx * qy - qz * w)
    r13 = 2 * (qx * qz + qy * w)
    r21 = 2 * (qx * qy + qz * w)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy * qz - qx * w)
    r31 = 2 * (qx * qz - qy * w)
    r32 = 2 * (qy * qz + qx * w)
    r33 = 1 - 2 * (qx**2 + qy**2)
    return np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3, 3).T
