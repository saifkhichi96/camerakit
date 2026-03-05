import argparse
import os
import time
from datetime import datetime

import cv2

from .utils import (
    CameraEnumerator,
    CameraMetadata,
    SynchronizedVideoCapture,
    setup_logging,
)


def parse_args():
    """Parse CLI arguments for synchronized capture.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Record synchronized video from multiple cameras."
    )
    parser.add_argument(
        "--required-fps",
        type=float,
        default=None,
        help="Optional: required FPS for recorded video (if different from actual, video will be resampled).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Optional: codec to use for recording video (default: mp4v).",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=5,
        help="Optional: maximum number of cameras to detect (default: 5). Only for Linux and Windows.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Optional: directory to save recorded videos (default: 'data').",
    )
    return parser.parse_args()


def select_resolution_and_fps(options):
    """Prompt the user to pick a capture setting.

    Args:
        options: List of selectable capture settings.

    Returns:
        CaptureSettings: Selected setting object.

    Raises:
        ValueError: If the selected index is out of range.
    """
    print("\nSelected cameras support the following common resolutions/FPS:")
    for idx, settings in enumerate(options):
        print(f"{idx + 1}: {settings}")

    selected_idx = int(
        input("Select a setting by number (1-{}): ".format(len(options)))
    )
    if selected_idx < 1 or selected_idx > len(options):
        raise ValueError("Invalid selection, please try again.")
    return options[selected_idx - 1]


def _setting_key(setting):
    """Build a normalized comparison key for capture settings."""
    return (
        setting.width,
        setting.height,
        round(setting.fps, 3),
        setting.codec.upper(),
    )


def _opencv_has_gui_support():
    """Check whether the installed OpenCV build provides HighGUI support."""
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        # If build info cannot be queried, don't block capture preemptively.
        return True

    for line in build_info.splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            backend = stripped.split(":", 1)[1].strip().upper()
            return backend != "NONE"
    return True


def _common_settings(cameras):
    """Compute settings common to all selected cameras."""
    if not cameras:
        return []

    setting_sets = [
        {_setting_key(setting) for setting in cam.settings} for cam in cameras
    ]
    common = set.intersection(*setting_sets) if setting_sets else set()
    if not common:
        return []

    # Preserve ordering from the first selected camera.
    result = []
    seen = set()
    for setting in cameras[0].settings:
        key = _setting_key(setting)
        if key in common and key not in seen:
            result.append(setting)
            seen.add(key)
    return result


def reencode_video(input_filename, output_filename, target_fps, width, height, codec):
    """Re-encode a saved video file to a target FPS and codec.

    Args:
        input_filename: Source video path.
        output_filename: Destination video path.
        target_fps: Output frame rate.
        width: Output frame width.
        height: Output frame height.
        codec: FourCC codec string.

    Returns:
        bool: `True` if re-encoding succeeds, else `False`.
    """
    # Wait a short moment to ensure the file is flushed to disk.
    time.sleep(0.5)

    print(
        f"Re-encoding video:\n  Input: {input_filename}\n  Output: {output_filename}\n  Target FPS: {target_fps:.2f}"
    )
    cap = cv2.VideoCapture(input_filename)
    if not cap.isOpened():
        print(f"Failed to open input file for re-encoding: {input_filename}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_filename, fourcc, target_fps, (width, height))
    if not out.isOpened():
        print(f"Failed to open output file for re-encoding: {output_filename}")
        cap.release()
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return True


def main():
    """Run the interactive synchronized capture workflow."""
    args = parse_args()
    if not _opencv_has_gui_support():
        print(
            "Capture is disabled because the installed OpenCV build has no GUI support "
            "(likely `opencv-python-headless`). Install `opencv-python` to use "
            "`camerakit capture`."
        )
        return

    codec = args.codec
    data_dir = args.data_dir

    # Create a directory for this session's recordings.
    currentDateAndTime = datetime.now()
    session_dir = f"{data_dir}/Session_{currentDateAndTime.strftime('%Y%m%dT%H%M%S')}"
    os.makedirs(session_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(session_dir)

    # Discover available cameras.
    print("---------------------------------------------------------------------")
    print("Discovering connected cameras...")
    cameras = CameraEnumerator(max_cameras=args.max_cameras).list_synchronizable()
    if not cameras:
        print("No cameras found. Exiting.")
        return

    # Display available cameras.
    print("The following cameras were found:")
    for cam in cameras:
        print(f"  - {cam.name} (ID: {cam.id})")
    print("---------------------------------------------------------------------\n")

    # Ask the user to select camera IDs to use.
    selected_ids = input(
        "Select the camera(s) to use for recording. For synchronized multi-camera\n"
        "capture, two or more cameras with matching resolutions and FPS are\n"
        "required. Enter comma-separated camera IDs (e.g., 0 or 0,1,2): "
    ).strip()
    try:
        selected_ids = [int(cam_id.strip()) for cam_id in selected_ids.split(",")]
    except ValueError:
        logger.error(
            "Invalid input. Please enter numeric camera IDs separated by commas."
        )
        return
    cameras = [cam for cam in cameras if cam.id in selected_ids]
    if not cameras:
        logger.error("No valid camera IDs were selected.")
        return

    common_settings = _common_settings(cameras)

    print("Using selected cameras:")
    for cam in cameras:
        print(f"  - {cam.id}: {cam.name}")
    cameras = [(cam.id, cam.name, cam.settings) for cam in cameras]

    if not common_settings:
        logger.error(
            "No common resolutions and FPS found across selected cameras. "
            "Please ensure all cameras support the same settings."
        )
        return

    # Ask the user to select resolution and FPS for each camera.
    selected_cams = []
    selected_setting = select_resolution_and_fps(common_settings)
    for cam_id, cam_name, _ in cameras:
        selected_cams.append(
            CameraMetadata(id=cam_id, name=cam_name, settings=[selected_setting])
        )
    cam_ids = [cam.id for cam in selected_cams]

    # Create the synchronized capture.
    sync = SynchronizedVideoCapture(selected_cams)

    logger.info("---------------------------------------------------------------------")
    logger.info("Starting video capture session.")
    logger.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logger.info(f"Session directory: {session_dir}")
    logger.info(
        "Selected cameras: "
        + ", ".join(f"{cam.id} ({cam.name})" for cam in selected_cams)
    )
    logger.info(
        f"Resolution: {selected_setting.width}x{selected_setting.height}, FPS: {selected_setting.fps:.1f}, Codec: {selected_setting.codec}"
    )
    logger.info(
        "---------------------------------------------------------------------\n"
    )

    # Recording state variables.
    recording = False
    current_writers = {}  # Active VideoWriter objects, keyed by camera ID.
    recording_frame_count = 0
    recording_start_time = None
    current_session = 0

    # Variables for computing display FPS.
    last_display_time = time.time()
    display_fps = 0.0

    print("Press 'r' to start recording, 's' to stop & save, 'q' to quit.")

    try:
        while True:
            ret, frames = sync.read()
            if not ret or len(frames) == 0:
                continue

            now = time.time()
            dt = now - last_display_time
            if dt > 0:
                display_fps = 1.0 / dt
            last_display_time = now

            # Prepare copies of frames for display and overlay status text.
            disp_frames = [frame.copy() for frame in frames]
            if recording:
                elapsed = now - recording_start_time
                avg_rec_fps = recording_frame_count / elapsed if elapsed > 0 else 0.0
                status_text = (
                    f"Recording | Time: {elapsed:.1f}s | Avg FPS: {avg_rec_fps:.1f}"
                )
            else:
                status_text = "Not Recording | Press 'r' to record"
            status_text += f" | Display FPS: {display_fps:.1f}"

            # Stack frames horizontally and show them.
            stacked = cv2.hconcat(disp_frames)
            cv2.putText(
                stacked,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if recording else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Synchronized Cameras", stacked)

            # If recording, write frames to file.
            if recording:
                for idx, frame in enumerate(frames):
                    cam_id = cam_ids[idx]
                    current_writers[cam_id].write(frame)
                recording_frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                # Start a new recording session.
                recording = True
                current_session += 1
                recording_start_time = time.time()
                recording_frame_count = 0
                current_writers = {}
                for cam in selected_cams:
                    cam_id = cam.id
                    cam_settings = cam.settings[
                        0
                    ]  # Assuming single setting per camera.
                    width = cam_settings.width
                    height = cam_settings.height
                    fps = cam_settings.fps
                    codec = cam_settings.codec
                    output_path = os.path.join(
                        session_dir,
                        f"Trial_{current_session}/{cam_id}_raw.mp4",
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise IOError(
                            f"Failed to open video writer for camera {cam_id}."
                        )
                    current_writers[cam_id] = writer
                print(f"Started recording session {current_session}.")
            elif key == ord("s") and recording:
                # Stop the current recording session.
                recording = False
                rec_end = time.time()
                elapsed = rec_end - recording_start_time
                avg_rec_fps = recording_frame_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"Stopped recording session {current_session}. Average recorded FPS: {avg_rec_fps:.2f}"
                )

                # Release all video writers and keep track of the raw filenames.
                raw_files = {}
                for cam in selected_cams:
                    cam_id = cam.id
                    current_writers[cam_id].release()
                    raw_files[cam_id] = os.path.join(
                        session_dir,
                        f"Trial_{current_session}/{cam_id}_raw.mp4",
                    )

                # Give the OS a moment to flush the files.
                time.sleep(0.5)

                # Determine the target FPS.
                target_fps = avg_rec_fps
                if args.required_fps is not None:
                    if abs(args.required_fps - avg_rec_fps) > 0.1:
                        target_fps = args.required_fps
                        print(
                            f"User-specified FPS {args.required_fps} differs from computed FPS {avg_rec_fps:.2f}. "
                            f"Re-encoding using {target_fps:.2f} FPS."
                        )
                    else:
                        target_fps = args.required_fps

                # For each camera, re-encode if the originally set FPS differs from target FPS.
                for cam in selected_cams:
                    cam_id = cam.id
                    width, height = cam.settings[0].width, cam.settings[0].height
                    raw_filename = raw_files[cam_id]
                    final_filename = os.path.join(
                        session_dir, f"Trial_{current_session}/{cam_id}.mp4"
                    )
                    if abs(cam.settings[0].fps - target_fps) > 0.1:
                        success = reencode_video(
                            raw_filename,
                            final_filename,
                            target_fps,
                            width,
                            height,
                            codec,
                        )
                        if success:
                            os.remove(raw_filename)
                            print(
                                f"Re-encoded video for camera {cam_id} to {target_fps:.2f} FPS."
                            )
                        else:
                            print(
                                f"Re-encoding failed for camera {cam_id}. Keeping raw video."
                            )
                    else:
                        os.rename(raw_filename, final_filename)
                        print(
                            f"Video for camera {cam_id} kept at original FPS ({cam.settings[0].fps:.2f})."
                        )
            elif key == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        sync.release()
        print("Session ended.")


if __name__ == "__main__":
    main()
