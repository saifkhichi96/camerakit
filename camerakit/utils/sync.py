import logging
import threading
import time
from typing import List, Tuple

import cv2


class SynchronizedVideoCapture:
    """
    Captures frames from multiple cameras and provides synchronized frame reading.
    """

    def __init__(self, cameras: List[dict]):
        """
        Initialize the camera streams and synchronization setup.

        Args:
            camera_ids (List[int]): List of camera IDs.
        """
        camera_ids = [camera["id"] for camera in cameras]
        self.camera_ids = camera_ids
        self.cameras = [cv2.VideoCapture(cam_id) for cam_id in camera_ids]

        # Ensure all cameras are opened
        for idx, cam in enumerate(self.cameras):
            if not cam.isOpened():
                raise ValueError(f"Failed to open camera {camera_ids[idx]}")

            # Set camera resolutions
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, cameras[idx]["width"])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cameras[idx]["height"])
            cam.set(cv2.CAP_PROP_FPS, cameras[idx]["fps"])

        # Synchronization setup
        self.lock = threading.Lock()
        self.running = True
        self.frames = [None] * len(camera_ids)
        self.frame_ready = threading.Event()

        # Background thread for capturing frames
        self.capture_threads = [
            threading.Thread(target=self._capture_frames, args=(idx,))
            for idx in range(len(camera_ids))
        ]
        for thread in self.capture_threads:
            thread.start()

    def _capture_frames(self, index: int):
        """
        Captures frames continuously for a specific camera.

        Args:
            index (int): Camera index in the list.
        """
        while self.running:
            ret, frame = self.cameras[index].read()
            if not ret:
                logging.warning(
                    f"Camera {self.camera_ids[index]} failed to capture frame."
                )
                time.sleep(0.01)  # Avoid busy-waiting
                continue

            with self.lock:
                self.frames[index] = frame

            self.frame_ready.set()  # Signal that a frame is ready

    def read(self) -> Tuple[bool, List]:
        """
        Reads synchronized frames from all cameras.

        Returns:
            Tuple[bool, List]: (True, List of frames) if successful, else (False, [])
        """
        self.frame_ready.wait()  # Wait for at least one frame to be captured

        with self.lock:
            if any(frame is None for frame in self.frames):
                return True, []  # Not all frames are ready

            frames_copy = self.frames.copy()  # Avoid race conditions

        self.frame_ready.clear()  # Reset event for the next capture cycle
        return True, frames_copy

    def release(self):
        """
        Release all camera resources and stop threads.
        """
        self.running = False
        for thread in self.capture_threads:
            thread.join()

        for cam in self.cameras:
            cam.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
