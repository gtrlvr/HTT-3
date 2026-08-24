import time

import cv2
import numpy as np

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from ultralytics import YOLO
from ultralytics.utils import ROOT, YAML


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
WINDOW_NAME = "Human Tracking Turret"

# Ignore small frame-to-frame center changes caused by bounding-box jitter.
IDLE_MOVEMENT_THRESHOLD_PX = 20.0

# Only objects moving within this range can become the active target.
MIN_TRACKING_SPEED_PX_S = 35.0
MAX_TRACKING_SPEED_PX_S = 1000.0

# Forget old positions after an ID has been missing for this long.
TRACK_HISTORY_TIMEOUT_S = 1.0

# Proportional control gains.
KP_X = 0.1
KP_Y = 0.1

# Stop correcting once the target is close enough to center.
DEADZONE_X = 0
DEADZONE_Y = 0

PROTOCOL_VERSION = 1.0
DEVICENAME = "COM12"
BAUDRATE = 57600

PAN_MOTOR_ID = 5
TILT_MOTOR_ID = 1

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
MOVING_SPEED = 1023

SPEED_SCALE = 1.0
SPEED_STEP = 0.25
SPEED_SCALE_MIN = 0.1
SPEED_SCALE_MAX = 5.0

SPEED_LOW = 0.5
SPEED_MEDIUM = 1.0
SPEED_HIGH = 2.0
SPEED_SUPER = 3.5

PAN_MIN = 0
PAN_MAX = 4095
TILT_MIN = 1623
TILT_MAX = 3878

# Startup positions from the Dynamixel Wizard screenshots.
PAN_HOME_DEGREES = 183.08
TILT_HOME_DEGREES = 278.00
MX_28_MAX_DEGREES = 359.91
MX_28_MAX_POSITION = 4095
HOME_SETTLE_TIME_S = 1.5


def degrees_to_position(degrees):
    clamped_degrees = float(np.clip(degrees, 0.0, MX_28_MAX_DEGREES))
    return int(round((clamped_degrees / MX_28_MAX_DEGREES) * MX_28_MAX_POSITION))


def check_dynamixel_result(packet_handler, dxl_comm_result, dxl_error, action):
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(
            f"{action} failed: {packet_handler.getTxRxResult(dxl_comm_result)}"
        )
    if dxl_error != 0:
        raise RuntimeError(
            f"{action} failed: {packet_handler.getRxPacketError(dxl_error)}"
        )


def set_dynamixel_torque(port_handler, packet_handler, motor_id, enabled):
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
        port_handler,
        motor_id,
        ADDR_MX_TORQUE_ENABLE,
        TORQUE_ENABLE if enabled else TORQUE_DISABLE,
    )
    action = f"{'Enable' if enabled else 'Disable'} torque for ID {motor_id}"
    check_dynamixel_result(packet_handler, dxl_comm_result, dxl_error, action)


def set_dynamixel_position(port_handler, packet_handler, motor_id, position, retries=2):
    for _ in range(retries + 1):
        port_handler.clearPort()
        dxl_comm_result, dxl_error = packet_handler.write2ByteTxRx(
            port_handler, motor_id, ADDR_MX_GOAL_POSITION, int(position)
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            return True
        time.sleep(0.002)

    print(f"[warn] write goal position for ID {motor_id} failed")
    return False


def set_dynamixel_speed(port_handler, packet_handler, motor_id, speed, retries=2):
    for _ in range(retries + 1):
        port_handler.clearPort()
        dxl_comm_result, dxl_error = packet_handler.write2ByteTxRx(
            port_handler, motor_id, ADDR_MX_MOVING_SPEED, int(speed)
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            return True
        time.sleep(0.002)

    print(f"[warn] write moving speed for ID {motor_id} failed")
    return False


def set_speed_scale(value):
    global SPEED_SCALE
    SPEED_SCALE = float(np.clip(value, SPEED_SCALE_MIN, SPEED_SCALE_MAX))
    print(f"Speed scale: {SPEED_SCALE:.2f}x")


def apply_speed_change(delta):
    set_speed_scale(SPEED_SCALE + delta)


def handle_key(key):
    if key == ord("q"):
        return True
    if key == ord("a"):
        apply_speed_change(SPEED_STEP)
    elif key == ord("d"):
        apply_speed_change(-SPEED_STEP)
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


def initialize_dynamixel():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Failed to open Dynamixel port {DEVICENAME}")

    if not port_handler.setBaudRate(BAUDRATE):
        port_handler.closePort()
        raise RuntimeError(f"Failed to set Dynamixel baudrate to {BAUDRATE}")

    set_dynamixel_torque(port_handler, packet_handler, PAN_MOTOR_ID, True)
    set_dynamixel_torque(port_handler, packet_handler, TILT_MOTOR_ID, True)

    if not set_dynamixel_speed(port_handler, packet_handler, PAN_MOTOR_ID, MOVING_SPEED):
        raise RuntimeError(f"Failed to set moving speed for ID {PAN_MOTOR_ID}")
    if not set_dynamixel_speed(port_handler, packet_handler, TILT_MOTOR_ID, MOVING_SPEED):
        raise RuntimeError(f"Failed to set moving speed for ID {TILT_MOTOR_ID}")

    pan_home_position = int(np.clip(degrees_to_position(PAN_HOME_DEGREES), PAN_MIN, PAN_MAX))
    tilt_home_position = int(
        np.clip(degrees_to_position(TILT_HOME_DEGREES), TILT_MIN, TILT_MAX)
    )

    print("Moving motors to startup positions before tracking begins...")
    print(
        f"Pan  ID {PAN_MOTOR_ID}: {PAN_HOME_DEGREES:.2f} deg -> {pan_home_position}"
    )
    print(
        f"Tilt ID {TILT_MOTOR_ID}: {TILT_HOME_DEGREES:.2f} deg -> {tilt_home_position}"
    )

    if not set_dynamixel_position(
        port_handler, packet_handler, PAN_MOTOR_ID, pan_home_position
    ):
        raise RuntimeError(f"Failed to move ID {PAN_MOTOR_ID} to its startup position")
    if not set_dynamixel_position(
        port_handler, packet_handler, TILT_MOTOR_ID, tilt_home_position
    ):
        raise RuntimeError(f"Failed to move ID {TILT_MOTOR_ID} to its startup position")

    time.sleep(HOME_SETTLE_TIME_S)
    return port_handler, packet_handler, pan_home_position, tilt_home_position


def main():
    port_handler = None
    packet_handler = None
    cap = None

    try:
        port_handler, packet_handler, current_pan_position, current_tilt_position = (
            initialize_dynamixel()
        )

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {CAMERA_INDEX}")

        cx_C = FRAME_WIDTH // 2
        cy_C = FRAME_HEIGHT // 2

        model = YOLO("yolov8n.pt")
        tracker_config = build_tracker_config()
        previous_positions = {}

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
                            dx = cx - previous_cx
                            dy = cy - previous_cy
                            movement_px = float(np.hypot(dx, dy))

                            if movement_px < IDLE_MOVEMENT_THRESHOLD_PX:
                                speed = 0.0
                            else:
                                speed = movement_px / dt

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

            eligible_objects = [
                tracked_object
                for tracked_object in tracked_objects
                if tracked_object["speed"] is not None
                and MIN_TRACKING_SPEED_PX_S
                <= tracked_object["speed"]
                <= MAX_TRACKING_SPEED_PX_S
            ]

            target = max(
                eligible_objects,
                key=lambda tracked_object: tracked_object["speed"],
                default=None,
            )

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
                elif speed < MIN_TRACKING_SPEED_PX_S:
                    box_color = (128, 128, 128)
                    speed_status = f"{speed:.1f} px/s | BELOW MIN"
                else:
                    box_color = (0, 255, 0)
                    speed_status = f"{speed:.1f} px/s"

                box_thickness = 3 if is_target else 2

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    box_thickness,
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

                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    5,
                    (0, 0, 255),
                    -1,
                )

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

                new_pan_position = int(
                    np.clip(current_pan_position + correction_pan, PAN_MIN, PAN_MAX)
                )
                new_tilt_position = int(
                    np.clip(current_tilt_position + correction_tilt, TILT_MIN, TILT_MAX)
                )

                pan_ok = set_dynamixel_position(
                    port_handler, packet_handler, PAN_MOTOR_ID, new_pan_position
                )
                tilt_ok = set_dynamixel_position(
                    port_handler, packet_handler, TILT_MOTOR_ID, new_tilt_position
                )

                if pan_ok:
                    current_pan_position = new_pan_position
                if tilt_ok:
                    current_tilt_position = new_tilt_position

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

                cv2.putText(
                    frame,
                    (
                        f"Pan: {current_pan_position} | Tilt: {current_tilt_position} | "
                        f"Scale: {SPEED_SCALE:.2f}x"
                    ),
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                print(
                    f"Tracking ID: {target_id} | "
                    f"Speed: {target_speed:.1f} px/s | "
                    f"Error: ({error_x:.1f}, {error_y:.1f}) | "
                    f"Pan: {current_pan_position} | Tilt: {current_tilt_position}"
                )
            else:
                cv2.putText(
                    frame,
                    "No target within velocity limits",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Scale: {SPEED_SCALE:.2f}x",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )

            cv2.circle(
                frame,
                (cx_C, cy_C),
                5,
                (255, 0, 0),
                -1,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if handle_key(cv2.waitKey(1) & 0xFF):
                break

    finally:
        if packet_handler is not None and port_handler is not None:
            try:
                set_dynamixel_torque(port_handler, packet_handler, PAN_MOTOR_ID, False)
            except Exception:
                pass

            try:
                set_dynamixel_torque(port_handler, packet_handler, TILT_MOTOR_ID, False)
            except Exception:
                pass

            port_handler.closePort()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
