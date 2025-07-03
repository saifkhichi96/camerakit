import argparse
import logging
import os
import time

import cv2

from .utils import SynchronizedVideoCapture, find_cameras

# Configure logging with timestamps and log level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
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
    return parser.parse_args()


def list_resolutions_and_fps(camera_index, codec):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error(f"Failed to open camera {camera_index}.")
        return []

    # Define common resolutions.
    resolutions = [
        (320, 240),  # QVGA
        (640, 480),  # VGA
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1920, 1080),  # Full HD
    ]

    available_options = []
    logging.info(f"Checking resolutions with mp4v codec for camera {camera_index}...")
    for width, height in resolutions:
        # Set resolution and mp4v codec.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution and FPS.
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if actual_width == width and actual_height == height:
            available_options.append((width, height, fps))
            logging.info(f"Resolution: {width}x{height} | FPS: {fps:.1f}")
    cap.release()
    return available_options


def select_resolution_and_fps(camera_index, codec):
    print(f"Select resolution and FPS for camera {camera_index}:")
    options = list_resolutions_and_fps(camera_index, codec)
    if not options:
        logging.error(f"No resolutions available for camera {camera_index}.")
        return None, None, None

    print("\nSelect an option by entering the index:")
    for idx, (width, height, fps) in enumerate(options):
        print(f"{idx + 1}: {width}x{height} @ {fps:.1f} FPS")

    selected_idx = int(input("Enter the index of the desired resolution: "))
    if selected_idx < 1 or selected_idx > len(options):
        raise ValueError("Invalid selection, please try again.")
    selected_width, selected_height, selected_fps = options[selected_idx - 1]
    return selected_width, selected_height, selected_fps


def reencode_video(input_filename, output_filename, target_fps, width, height, codec):
    """
    Re-encode the video at input_filename using target_fps and write to output_filename.
    """
    # Wait a short moment to ensure the file is flushed to disk.
    time.sleep(0.5)

    logging.info(
        f"Re-encoding video:\n  Input: {input_filename}\n  Output: {output_filename}\n  Target FPS: {target_fps:.2f}"
    )
    cap = cv2.VideoCapture(input_filename)
    if not cap.isOpened():
        logging.error(f"Failed to open input file for re-encoding: {input_filename}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_filename, fourcc, target_fps, (width, height))
    if not out.isOpened():
        logging.error(f"Failed to open output file for re-encoding: {output_filename}")
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
    args = parse_args()
    codec = args.codec

    # Discover available cameras.
    logging.info("Looking for available cameras...")
    cameras = find_cameras()
    if not cameras:
        logging.error("No cameras found. Exiting.")
        return

    # Display available cameras.
    logging.info("Available cameras:")
    for cam in cameras:
        logging.info(f"  - {cam['id']}: {cam['name']}")

    # Ask the user to select camera IDs to use.
    print("Enter the IDs of the cameras you want to use, separated by commas:")
    selected_ids = input("Camera IDs: ").strip()
    try:
        selected_ids = [int(cam_id.strip()) for cam_id in selected_ids.split(",")]
    except ValueError:
        logging.error(
            "Invalid input. Please enter numeric camera IDs separated by commas."
        )
        return
    cameras = [cam for cam in cameras if cam["id"] in selected_ids]
    if not cameras:
        logging.error("No valid cameras selected. Exiting.")
        return

    logging.info("Using selected cameras:")
    for cam in cameras:
        logging.info(f"  - {cam['id']}: {cam['name']}")
    cameras = [(cam["id"], cam["name"]) for cam in cameras]

    # Ask the user to select resolution and FPS for each camera.
    print(
        "Listing supported resolutions and FPS for each camera. "
        "Please select the same resolution and FPS for all cameras."
    )
    selected_cams = []
    for cam_id, cam_name in cameras:
        width, height, fps = select_resolution_and_fps(cam_id, codec)
        selected_cams.append(
            {
                "id": cam_id,
                "name": cam_name,
                "width": width,
                "height": height,
                "fps": fps,
            }
        )
    cam_ids = [cam["id"] for cam in selected_cams]

    # Create the synchronized capture.
    sync = SynchronizedVideoCapture(selected_cams)

    # Create a directory for this session's recordings.
    session_save_dir = f"data/Session_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(session_save_dir, exist_ok=True)
    logging.info(f"Saving recordings to directory: {session_save_dir}")

    # Create the calibration directory.
    calibration_dir = os.path.join(session_save_dir, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)

    # Copy the config file.
    default_config = "data/Session_Sample_1/Config.toml"
    config_file = os.path.join(session_save_dir, "Config.toml")
    os.system(f"cp {default_config} {config_file}")
    logging.info(f"Copied default config file to: {config_file}")

    # Recording state variables.
    recording = False
    current_writers = {}  # Active VideoWriter objects, keyed by camera ID.
    recording_frame_count = 0
    recording_start_time = None
    current_session = 0

    # Variables for computing display FPS.
    last_display_time = time.time()
    display_fps = 0.0

    logging.info("Press 'r' to start recording, 's' to stop & save, 'q' to quit.")

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
                    cam_id = cam["id"]
                    width, height, fps = cam["width"], cam["height"], cam["fps"]
                    output_path = os.path.join(
                        session_save_dir,
                        f"Trial_{current_session}/{cam_id}_raw.mp4",
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        logging.error(
                            f"Failed to open video writer for camera {cam_id}."
                        )
                        raise IOError(
                            f"Failed to open video writer for camera {cam_id}."
                        )
                    current_writers[cam_id] = writer
                logging.info(f"Started recording session {current_session}.")
            elif key == ord("s") and recording:
                # Stop the current recording session.
                recording = False
                rec_end = time.time()
                elapsed = rec_end - recording_start_time
                avg_rec_fps = recording_frame_count / elapsed if elapsed > 0 else 0.0
                logging.info(
                    f"Stopped recording session {current_session}. Average recorded FPS: {avg_rec_fps:.2f}"
                )

                # Release all video writers and keep track of the raw filenames.
                raw_files = {}
                for cam in selected_cams:
                    cam_id = cam["id"]
                    current_writers[cam_id].release()
                    raw_files[cam_id] = os.path.join(
                        session_save_dir,
                        f"Trial_{current_session}/{cam_id}_raw.mp4",
                    )

                # Give the OS a moment to flush the files.
                time.sleep(0.5)

                # Determine the target FPS.
                target_fps = avg_rec_fps
                if args.required_fps is not None:
                    if abs(args.required_fps - avg_rec_fps) > 0.1:
                        target_fps = args.required_fps
                        logging.info(
                            f"User-specified FPS {args.required_fps} differs from computed FPS {avg_rec_fps:.2f}. "
                            f"Re-encoding using {target_fps:.2f} FPS."
                        )
                    else:
                        target_fps = args.required_fps

                # For each camera, re-encode if the originally set FPS differs from target FPS.
                for cam in selected_cams:
                    cam_id = cam["id"]
                    width, height = cam["width"], cam["height"]
                    raw_filename = raw_files[cam_id]
                    final_filename = os.path.join(
                        session_save_dir, f"Trial_{current_session}/{cam_id}.mp4"
                    )
                    if abs(cam["fps"] - target_fps) > 0.1:
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
                            logging.info(
                                f"Re-encoded video for camera {cam_id} to {target_fps:.2f} FPS."
                            )
                        else:
                            logging.error(
                                f"Re-encoding failed for camera {cam_id}. Keeping raw video."
                            )
                    else:
                        os.rename(raw_filename, final_filename)
                        logging.info(
                            f"Video for camera {cam_id} kept at original FPS ({cam['fps']:.2f})."
                        )
            elif key == ord("q"):
                break

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        sync.release()
        logging.info("Session ended.")


if __name__ == "__main__":
    main()
