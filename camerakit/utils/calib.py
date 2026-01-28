import os

import cv2
import numpy as np

from .common import get_logger

logger = get_logger()


def euclidean_distance(q1, q2):
    """
    Euclidean distance between 2 points (N-dim).

    INPUTS:
    - q1: list of N_dimensional coordinates of point
         or list of N points of N_dimensional coordinates
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    """

    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    if np.isnan(dist).all():
        dist = np.empty_like(dist)
        dist[...] = np.inf

    if len(dist.shape) == 1:
        euc_dist = np.sqrt(np.nansum([d**2 for d in dist]))
    else:
        euc_dist = np.sqrt(np.nansum([d**2 for d in dist], axis=1))

    return euc_dist


def extract_frames(video_path, extract_every_N_sec=1, overwrite_extraction=False):
    """
    Extract frames from video
    if has not been done yet or if overwrite==True

    INPUT:
    - video_path: path to video whose frames need to be extracted
    - extract_every_N_sec: extract one frame every N seconds (can be <1)
    - overwrite_extraction: if True, overwrite even if frames have already been extracted

    OUTPUT:
    - extracted frames in folder
    """

    if (
        not os.path.exists(os.path.splitext(video_path)[0] + "_00000.png")
        or overwrite_extraction
    ):
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            frame_nb = 0
            logger.info("Extracting frames...")
            while cap.isOpened():
                r, frame = cap.read()
                if r:
                    if frame_nb % (fps * extract_every_N_sec) == 0:
                        img_path = (
                            os.path.splitext(video_path)[0]
                            + "_"
                            + str(frame_nb).zfill(5)
                            + ".png"
                        )
                        cv2.imwrite(str(img_path), frame)
                    frame_nb += 1
                else:
                    break


def toml_write(
    calib_path,
    C,
    S,
    D,
    K,
    R,
    T,
    intrinsics_error_px=None,
    extrinsics_error_px=None,
):
    """
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distortion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    Optional:
    - intrinsics_error_px: list of per-camera intrinsic reprojection errors (px)
    - extrinsics_error_px: list of per-camera extrinsic reprojection errors (px)

    OUTPUTS:
    - a .toml file cameras calibrations
    """

    intrinsics_error_px = (
        None if intrinsics_error_px is None else np.array(intrinsics_error_px).tolist()
    )
    extrinsics_error_px = (
        None if extrinsics_error_px is None else np.array(extrinsics_error_px).tolist()
    )

    with open(os.path.join(calib_path), "w+") as cal_f:
        for c in range(len(C)):
            cam = f"[{C[c]}]\n"
            name = f'name = "{C[c]}"\n'
            size = f"size = [ {S[c][0]}, {S[c][1]}]\n"
            mat = f"matrix = [ [ {K[c][0, 0]}, 0.0, {K[c][0, 2]}], [ 0.0, {K[c][1, 1]}, {K[c][1, 2]}], [ 0.0, 0.0, 1.0]]\n"
            dist = f"distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]}]\n"
            rot = f"rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]}]\n"
            tran = f"translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]}]\n"
            intr_error = None
            if isinstance(intrinsics_error_px, list) and c < len(intrinsics_error_px):
                intr_error = intrinsics_error_px[c]
            extr_error = None
            if isinstance(extrinsics_error_px, list) and c < len(extrinsics_error_px):
                extr_error = extrinsics_error_px[c]

            error_lines = ""
            if intr_error is not None:
                error_lines += f"intrinsics_error_px = {intr_error}\n"
            if extr_error is not None:
                error_lines += f"extrinsics_error_px = {extr_error}\n"

            cal_f.write(
                cam + name + size + mat + dist + rot + tran + error_lines + "\n"
            )

        meta = "[metadata]\nadjusted = false\n"
        if intrinsics_error_px is not None:
            meta += f"intrinsics_error_px = {intrinsics_error_px}\n"
        if extrinsics_error_px is not None:
            meta += f"extrinsics_error_px = {extrinsics_error_px}\n"
        cal_f.write(meta)
