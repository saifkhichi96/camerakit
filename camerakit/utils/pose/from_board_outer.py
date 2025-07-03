import cv2
import numpy as np


def get_line(points):
    """
    Fit a line to a set of 2D points using cv2.fitLine.

    Args:
        points (np.ndarray): Nx2 array of points.

    Returns:
        p (np.ndarray): A point on the line.
        v (np.ndarray): The direction vector of the line.
    """
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def intersect_lines(p1, v1, p2, v2):
    """
    Compute the intersection of two lines given in parametric form:
        L1: p1 + t*v1, L2: p2 + s*v2.

    Args:
        p1, p2 (np.ndarray): Points on each line.
        v1, v2 (np.ndarray): Direction vectors.

    Returns:
        np.ndarray: Intersection point (2D). If the lines are nearly parallel, returns the average of p1 and p2.
    """
    A = np.array([v1, -v2]).T  # 2x2 system
    if np.abs(np.linalg.det(A)) < 1e-6:
        # Nearly parallel: fallback to the average point.
        return (p1 + p2) / 2.0
    t_s = np.linalg.solve(A, p2 - p1)
    return p1 + t_s[0] * v1


def find_outer_corners(corners, corner_nb):
    """
    Given a full set of detected chessboard corners and the grid dimensions (cols, rows),
    compute corrected outer corners via robust line fitting on each edge.

    Args:
        corners (np.ndarray): Detected corners with shape (N, 1, 2).
        corner_nb (list or tuple): [cols, rows] of internal corners.

    Returns:
        np.ndarray: Array of 4 corrected outer corners in order:
                    [top-left, top-right, bottom-left, bottom-right] (shape (4, 2)).
    """
    pts = corners.reshape(-1, 2)
    num_cols, num_rows = int(corner_nb[0]), int(corner_nb[1])

    # Extract edge points (assuming the returned order is row-wise left-to-right)
    top_row = pts[0:num_cols]
    bottom_row = pts[(num_rows - 1) * num_cols : num_rows * num_cols]
    left_col = pts[0::num_cols]
    right_col = pts[num_cols - 1 :: num_cols]

    # Fit lines on each edge
    p_top, v_top = get_line(top_row)
    p_bottom, v_bottom = get_line(bottom_row)
    p_left, v_left = get_line(left_col)
    p_right, v_right = get_line(right_col)

    # Compute intersections: these will be our corrected corners.
    top_left = intersect_lines(p_top, v_top, p_left, v_left)
    top_right = intersect_lines(p_top, v_top, p_right, v_right)
    bottom_left = intersect_lines(p_bottom, v_bottom, p_left, v_left)
    bottom_right = intersect_lines(p_bottom, v_bottom, p_right, v_right)

    return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
