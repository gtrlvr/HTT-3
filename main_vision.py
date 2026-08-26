import time

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import ROOT, YAML


CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30
WINDOW_NAME = "Vision Control Test"

SPEED_SCALE = 1.0
SPEED_STEP = 0.25
SPEED_SCALE_MIN = 0.1
SPEED_SCALE_MAX = 5.0

SPEED_LOW = 0.5
SPEED_MEDIUM = 1.0
SPEED_HIGH = 2.0
SPEED_SUPER = 3.5

# Keep these tracking settings in sync with main.py.
IDLE_MOVEMENT_THRESHOLD_PX = 25.0
IDLE_CONFIRMATION_TIME_S = 10.0
MAX_TRACKING_SPEED_PX_S = 600.0
TRACK_HISTORY_TIMEOUT_S = 1.0
TARGET_SELECTION_INTERVAL_S = 5.0

KP_X = 0.1
KP_Y = 0.1
DEADZONE_X = 0
DEADZONE_Y = 0

PAN_MIN = 0
PAN_MAX = 4095
TILT_MIN = 1623
TILT_MAX = 3878

# Virtual motor state begins at the same positions commanded by main.py.
INITIAL_PAN_POSITION = 903
INITIAL_TILT_POSITION = 3100


def set_speed_scale(value):
    global SPEED_SCALE
    SPEED_SCALE = float(np.clip(value, SPEED_SCALE_MIN, SPEED_SCALE_MAX))
    print(f"Speed scale: {SPEED_SCALE:.2f}x")


def handle_key(key):
    if key == ord("q"):
        return True
    if key == ord("a"):
        set_speed_scale(SPEED_SCALE + SPEED_STEP)
    elif key == ord("d"):
        set_speed_scale(SPEED_SCALE - SPEED_STEP)
    elif key == ord("1"):
        set_speed_scale(SPEED_LOW)
    elif key == ord("2"):
        set_speed_scale(SPEED_MEDIUM)
    elif key == ord("3"):
        set_speed_scale(SPEED_HIGH)
    elif key == ord("4"):
        set_speed_scale(SPEED_SUPER)
    return False


def build_tracker_config():
    botsort_cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
    botsort_cfg["with_reid"] = True
    botsort_cfg["model"] = "auto"

    tracker_config = "botsort_reid.yaml"
    YAML.save(tracker_config, botsort_cfg)
    return tracker_config


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open camera {CAMERA_INDEX}")

    cx_C = FRAME_WIDTH // 2
    cy_C = FRAME_HEIGHT // 2
    current_motor_position_pan = INITIAL_PAN_POSITION
    current_motor_position_tilt = INITIAL_TILT_POSITION

    model = YOLO("yolov8n.pt")
    tracker_config = build_tracker_config()
    previous_positions = {}
    idle_start_times = {}
    active_target_id = None
    last_target_selection_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            frame_time = time.perf_counter()
            results = model.track(
                frame,
                tracker=tracker_config,
                persist=True,
                classes=[0],
                verbose=False,
            )
            result = results[0]
            tracked_objects = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    speed = None
                    previous_position = previous_positions.get(track_id)

                    if previous_position is not None:
                        previous_cx, previous_cy, previous_time = previous_position
                        dt = frame_time - previous_time

                        if dt > 0:
                            movement_px = float(
                                np.hypot(cx - previous_cx, cy - previous_cy)
                            )
                            if movement_px >= IDLE_MOVEMENT_THRESHOLD_PX:
                                idle_start_times.pop(track_id, None)
                                speed = movement_px / dt
                            else:
                                idle_start_time = idle_start_times.setdefault(
                                    track_id, frame_time
                                )
                                speed = (
                                    0.0
                                    if frame_time - idle_start_time
                                    >= IDLE_CONFIRMATION_TIME_S
                                    else movement_px / dt
                                )

                    previous_positions[track_id] = (cx, cy, frame_time)
                    tracked_objects.append(
                        {
                            "track_id": track_id,
                            "box": (x1, y1, x2, y2),
                            "cx": cx,
                            "cy": cy,
                            "speed": speed,
                        }
                    )

            stale_track_ids = [
                track_id
                for track_id, (_, _, last_seen) in previous_positions.items()
                if frame_time - last_seen > TRACK_HISTORY_TIMEOUT_S
            ]
            for track_id in stale_track_ids:
                del previous_positions[track_id]
                idle_start_times.pop(track_id, None)

            eligible_objects = [
                tracked_object
                for tracked_object in tracked_objects
                if tracked_object["speed"] is not None
                and tracked_object["speed"] <= MAX_TRACKING_SPEED_PX_S
            ]
            current_target = next(
                (
                    tracked_object
                    for tracked_object in tracked_objects
                    if tracked_object["track_id"] == active_target_id
                    and tracked_object["speed"] is not None
                ),
                None,
            )

            if (
                current_target is not None
                and frame_time - last_target_selection_time
                < TARGET_SELECTION_INTERVAL_S
            ):
                target = current_target
            else:
                target = max(
                    eligible_objects,
                    key=lambda tracked_object: tracked_object["speed"],
                    default=None,
                )
                last_target_selection_time = frame_time
                active_target_id = None if target is None else target["track_id"]

            for tracked_object in tracked_objects:
                track_id = tracked_object["track_id"]
                x1, y1, x2, y2 = tracked_object["box"]
                cx = tracked_object["cx"]
                cy = tracked_object["cy"]
                speed = tracked_object["speed"]
                is_target = target is not None and track_id == target["track_id"]

                if is_target:
                    box_color = (0, 255, 255)
                    speed_status = f"{speed:.1f} px/s | TARGET"
                elif speed is None:
                    box_color = (0, 255, 0)
                    speed_status = "calculating"
                elif speed > MAX_TRACKING_SPEED_PX_S:
                    box_color = (0, 0, 255)
                    speed_status = f"{speed:.1f} px/s | TOO FAST"
                elif speed == 0.0:
                    box_color = (128, 128, 128)
                    speed_status = "0.0 px/s | IDLE"
                else:
                    box_color = (0, 255, 0)
                    speed_status = f"{speed:.1f} px/s"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    3 if is_target else 2,
                )
                cv2.putText(
                    frame,
                    f"ID: {track_id} | {speed_status}",
                    (int(x1), max(int(y1) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            if target is not None:
                target_id = target["track_id"]
                target_speed = target["speed"]
                error_x = cx_C - target["cx"]
                error_y = target["cy"] - cy_C

                if abs(error_x) < DEADZONE_X:
                    error_x = 0
                if abs(error_y) < DEADZONE_Y:
                    error_y = 0

                correction_pan = KP_X * error_x * SPEED_SCALE
                correction_tilt = KP_Y * error_y * SPEED_SCALE
                current_motor_position_pan = int(
                    np.clip(
                        current_motor_position_pan + correction_pan, PAN_MIN, PAN_MAX
                    )
                )
                current_motor_position_tilt = int(
                    np.clip(
                        current_motor_position_tilt + correction_tilt,
                        TILT_MIN,
                        TILT_MAX,
                    )
                )

                cv2.line(
                    frame,
                    (cx_C, cy_C),
                    (int(target["cx"]), int(target["cy"])),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"TARGET ID: {target_id} | Speed: {target_speed:.1f} px/s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Error X: {error_x:.1f} | Error Y: {error_y:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                print(
                    f"Tracking ID: {target_id} | Speed: {target_speed:.1f} px/s | "
                    f"Error: ({error_x:.1f}, {error_y:.1f}) | "
                    f"Virtual correction: ({correction_pan:.1f}, {correction_tilt:.1f})"
                )
            else:
                cv2.putText(
                    frame,
                    "No target at or below maximum speed",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )

            cv2.circle(frame, (cx_C, cy_C), 5, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"Motor speed: {SPEED_SCALE:.2f}x | Keys 1-4 change speed",
                (10, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)

            if handle_key(cv2.waitKey(1) & 0xFF):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
