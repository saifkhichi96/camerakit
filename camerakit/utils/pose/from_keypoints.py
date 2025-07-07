import glob
import logging
import os
import re

import cv2
import numpy as np


def read_extrinsics_cam_img(calib_dir, cam, extrinsics_extension):
    img_vid_files = glob.glob(
        os.path.join(calib_dir, "extrinsics", cam, f"*.{extrinsics_extension}")
    )
    if len(img_vid_files) == 0:
        logging.exception(
            f"The folder {os.path.join(calib_dir, 'extrinsics', cam)} does not exist or does not contain any files with extension .{extrinsics_extension}."
        )
        raise ValueError(
            f"The folder {os.path.join(calib_dir, 'extrinsics', cam)} does not exist or does not contain any files with extension .{extrinsics_extension}."
        )
    img_vid_files = sorted(
        img_vid_files, key=lambda c: [int(n) for n in re.findall(r"\d+", c)]
    )  # sorting paths with numbers

    # extract frames from image, or from video if imread is None
    img = cv2.imread(img_vid_files[0])
    if img is None:
        cap = cv2.VideoCapture(img_vid_files[0])
        r, img = cap.read()
        if not r:
            raise
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_keypoints(
    config_dict, 
    extrinsics_cam_listdirs_names, 
    calib_dir, 
    extrinsics_extension,
):
    """
    Estimates rough 3D object points from keypoints using measured pixel distances and a known participant height.

    For each camera (listed in extrinsics_cam_listdirs_names), an image is loaded and a 2D pose estimator
    (BodyWithFeet in 'performance' mode) extracts 26 HALPE keypoints. Instead of using only the first camera,
    this function averages keypoints over all cameras to account for differences in scale.

    We assume the person stands straight in a T-pose, so that all keypoints lie in the same plane (y=0).
    The following keypoints (in pixel coordinates) are used:
      - Heel midpoint: average of left heel (index 24) and right heel (index 25)
      - Head top: index 17
      - Left shoulder: index 5
      - Right shoulder: index 6
      - Left elbow: index 7
      - Right elbow: index 8
      - Left wrist: index 9
      - Right wrist: index 10
      - Left hip: index 11
      - Right hip: index 12
      - Left knee: index 13
      - Right knee: index 14

    The participant height in meters (from config_dict.process.participant_height) is compared against
    the average pixel height (distance from heel midpoint to head top) to compute a scale factor (m/px).
    Then, each keypoint is converted into a 3D object point (in meters) as follows:
      X = (pt_x - heel_mid_x) * s,
      Y = 0,
      Z = (heel_mid_y - pt_y) * s.

    In addition, the function returns, for each camera, the corresponding 2D image points (in pixels)
    for these keypoints, preserving the same order.

    Returns:
        obj_points: numpy array of shape (12, 3) containing the computed 3D object points.
        img_points_list: list of length n_cams, each element a numpy array of shape (12, 2) containing the 2D image points.
    """

    logging.info("Loading 2D pose estimator for extrinsics...")
    from posetrack import BodyWithFeet
    pose_estimator = BodyWithFeet(mode="performance")

    keypoints_list = []
    scores_list = []
    for cam in extrinsics_cam_listdirs_names:
        img = read_extrinsics_cam_img(calib_dir, cam, extrinsics_extension)
        keypoints, scores = pose_estimator(img)
        if len(keypoints) == 0:
            raise ValueError(
                f"No keypoints detected in camera {cam}. Please check the image or video."
            )
        # Use only the first detected person (26 keypoints)
        keypoints_list.append(keypoints[0])
        scores_list.append(scores[0])
        logging.info(f"Keypoints detected in camera {cam}.")

    # Define keypoint indices for HALPE 26
    left_heel_idx = 24
    right_heel_idx = 25
    head_top_idx = 17
    # Additional indices: from 5 to 19, excluding 17.
    disallowed_indices = [0, 1, 2, 3, 4, 7, 8, 13, 14, 15, 16, 17, 18]
    additional_indices = [i for i in range(20) if i not in disallowed_indices]

    # Compute per-camera keypoints:
    # For each camera, compute:
    #   - Heel midpoint = average of indices 24 and 25.
    #   - Then store additional points for indices in additional_indices.
    n_cams = len(keypoints_list)
    cam_points = []  # Each element: list of points in order: [heel_mid, head_top] + extra points.
    measured_heights = []  # head_top to heel_mid distances in pixels.
    for kp in keypoints_list:
        heel_mid = (np.array(kp[left_heel_idx]) + np.array(kp[right_heel_idx])) / 2.0
        head_top = np.array(kp[head_top_idx])
        extra_points = [np.array(kp[i]) for i in additional_indices]
        cam_points.append([heel_mid, head_top] + extra_points)
        measured_heights.append(np.linalg.norm(head_top - heel_mid))
    cam_points = np.array(cam_points)  # shape: (n_cams, (2+len(additional_indices)), 2)
    measured_height_px = np.mean(measured_heights)

    # Compute average scores for the extra indices over all cameras.
    avg_scores = {}
    for idx in additional_indices:
        scores = [scores_list[cam][idx] for cam in range(n_cams)]
        avg_scores[idx] = np.mean(scores)

    # Choose the best two extra indices based on average scores.
    best_two = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    # Sort the chosen indices in increasing order to ensure consistent ordering.
    best_two_indices = sorted([idx for idx, score in best_two])
    logging.info(f"Selected extra indices for calibration: {best_two_indices}")

    # Final keypoint order: [heel_mid, head_top] + best_two_indices.
    final_order = [None] * (2 + len(best_two_indices))
    final_order[0] = "heel_mid"
    final_order[1] = "head_top"
    for j, idx in enumerate(best_two_indices):
        final_order[2 + j] = idx

    # Compute average 2D points (across cameras) for each selected keypoint.
    # For heel_mid and head_top, use indices 0 and 1 in cam_points.
    avg_points = []
    # Heel mid:
    avg_points.append(np.mean(cam_points[:, 0, :], axis=0))
    # Head top:
    avg_points.append(np.mean(cam_points[:, 1, :], axis=0))
    # For each extra index in best_two_indices:
    for idx in best_two_indices:
        # Determine position in cam_points: extra indices are stored starting at index 2.
        pos = additional_indices.index(idx) + 2
        avg_points.append(np.mean(cam_points[:, pos, :], axis=0))
    avg_points = np.array(avg_points)  # shape: (6, 2)

    # Known participant height in meters (default to 1.75 m if not provided)
    person_height = config_dict.get("project", {}).get("participant_height", 1.75)
    if person_height is None:
        raise ValueError(
            "Participant height (project.participant_height) not provided in config."
        )
    s = person_height / measured_height_px  # Scale factor in m/px
    logging.info(
        f"Estimated scale factor: {s:.6f} m/px (measured pixel height: {measured_height_px:.2f} px)"
    )

    # Define a helper function to compute a 3D point from a 2D keypoint relative to heel_mid.
    # In image coordinates, y increases downward. We define z = (heel_mid_y - pt_y) * s.
    def compute_obj_point(pt, heel_mid):
        pt = np.array(pt)
        return np.array(
            [(pt[0] - heel_mid[0]) * s, 0, (heel_mid[1] - pt[1]) * s], dtype=np.float32
        )

    heel_mid_avg = avg_points[0]
    obj_points = []
    for pt in avg_points:
        obj_points.append(compute_obj_point(pt, heel_mid_avg))
    obj_points = np.array(obj_points, dtype=np.float32)
    obj_points = np.around(obj_points, 6)
    logging.info(f"3D Object Points: {obj_points.tolist()}")

    # For each camera, extract corresponding 2D image points in the same order.
    img_points_list = []
    for cam_pts in cam_points:
        # cam_pts is a list: [heel_mid, head_top] + extra_points (in same order as additional_indices)
        selected_pts = [cam_pts[0], cam_pts[1]]  # heel_mid, head_top
        for idx in best_two_indices:
            pos = additional_indices.index(idx) + 2
            selected_pts.append(cam_pts[pos])
        img_points_list.append(np.array(selected_pts, dtype=np.float32))

    return obj_points, img_points_list
