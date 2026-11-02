import time
import threading

import cv2
import numpy as np
import serial

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from ultralytics import YOLO
from ultralytics.utils import ROOT, YAML


CAMERA_INDEX = 0

ARDUINO_PORT = "COM4"
ARDUINO_BAUDRATE = 9600

PROTOCOL_VERSION = 1.0
DEVICENAME = "COM6"
BAUDRATE = 57600

PAN_MOTOR_ID = 5
TILT_MOTOR_ID = 1

# Known startup positions from Dynamixel Wizard.
INITIAL_PAN_POSITION = 903
INITIAL_TILT_POSITION = 3100

ADDR_MX_TORQUE_ENABLE = 24  
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Keep the Dynamixel's internal slew-speed limit high. Tracking speed is
# controlled separately with SPEED_SCALE.
MOVING_SPEED = 1023
SPEED_SCALE = 1.0
SPEED_STEP = 0.25
SPEED_SCALE_MIN = 0.1
SPEED_SCALE_MAX = 5.0

SPEED_LOW = 0.5
SPEED_MEDIUM = 1.0
SPEED_HIGH = 2.0
SPEED_SUPER = 3.5

# Ignore small frame-to-frame center changes caused by bounding-box jitter.
IDLE_MOVEMENT_THRESHOLD_PX = 25.0

# Movement must stay below the threshold this long before the track is idle.
IDLE_CONFIRMATION_TIME_S = 10.0

# Ignore objects moving faster than this when selecting a target.
MAX_TRACKING_SPEED_PX_S = 600.0

# Forget old positions after an ID has been missing for this long.
TRACK_HISTORY_TIMEOUT_S = 1.0

# Re-evaluate which tracked object is fastest at this interval.
TARGET_SELECTION_INTERVAL_S = 5.0


def check_dynamixel_result(dxl_comm_result, dxl_error, action):
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(
            f"{action} failed: {packet_handler.getTxRxResult(dxl_comm_result)}"
        )
    if dxl_error != 0:
        raise RuntimeError(
            f"{action} failed: {packet_handler.getRxPacketError(dxl_error)}"
        )


def get_dynamixel_position(motor_id, retries=3):
    for _ in range(retries + 1):
        port_handler.clearPort()
        position, dxl_comm_result, dxl_error = packet_handler.read2ByteTxRx(
            port_handler, motor_id, ADDR_MX_PRESENT_POSITION
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            return position
        time.sleep(0.002)

    print(f"[warn] read position for ID {motor_id} failed")
    return None


def set_dynamixel_position(motor_id, position, retries=2):
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


def set_dynamixel_speed(motor_id, speed, retries=2):
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


def handle_key(key):
    # Lowercase q exits; uppercase Q is reserved for MOSFET OFF.
    if key == ord("x"):
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
    elif key in (ord("e"), ord("E")):
        send_arduino_command("E")
    elif key == ord("Q"):
        send_arduino_command("Q")
    elif key in (ord("f"), ord("F")):
        send_arduino_command("F")
    return False


def send_arduino_command(command):
    """Send one command selected from the OpenCV camera window."""
    try:
        arduino_serial.write(command.encode("ascii"))
        arduino_serial.flush()
        print(f"Sent {command} to Arduino")
    except serial.SerialException as exc:
        print(f"[warn] Arduino write failed: {exc}")


def arduino_response_listener():
    """Print messages returned by the Arduino."""
    while not arduino_stop_event.is_set():
        try:
            response = arduino_serial.readline().decode("utf-8", errors="replace").strip()
        except serial.SerialException as exc:
            if not arduino_stop_event.is_set():
                print(f"[warn] Arduino read failed: {exc}")
            return

        if response:
            print(f"Arduino: {response}")


# DYNAMIXEL SETUP

port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)

if not port_handler.openPort():
    raise RuntimeError(f"Failed to open Dynamixel port {DEVICENAME}")

if not port_handler.setBaudRate(BAUDRATE):
    port_handler.closePort()
    raise RuntimeError(f"Failed to set Dynamixel baudrate to {BAUDRATE}")

dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
    port_handler, PAN_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE
)
check_dynamixel_result(
    dxl_comm_result,
    dxl_error,
    f"Enable torque for ID {PAN_MOTOR_ID}",
)

dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
    port_handler, TILT_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE
)
check_dynamixel_result(
    dxl_comm_result,
    dxl_error,
    f"Enable torque for ID {TILT_MOTOR_ID}",
)

set_dynamixel_speed(PAN_MOTOR_ID, MOVING_SPEED)
set_dynamixel_speed(TILT_MOTOR_ID, MOVING_SPEED)

# Move both motors to their known starting orientation before tracking begins.
if not set_dynamixel_position(PAN_MOTOR_ID, INITIAL_PAN_POSITION):
    raise RuntimeError("Failed to initialize the pan motor position")
if not set_dynamixel_position(TILT_MOTOR_ID, INITIAL_TILT_POSITION):
    raise RuntimeError("Failed to initialize the tilt motor position")


# CAMERA

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

frame_width = 1280
frame_height = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, 30)



# CAMERA CENTER
cx_C = frame_width // 2
cy_C = frame_height // 2


# MOTOR CONTROL

Kp_x = 0.1
Kp_y = 0.1

# Stop correcting once the target is this close to the camera center.
deadzone_x = 0
deadzone_y = 0

pan_min = 0
pan_max = 4095
tilt_min = 1623
tilt_max = 3878

# Start software position tracking from the commanded initialization positions.
current_motor_position_pan = INITIAL_PAN_POSITION
current_motor_position_tilt = INITIAL_TILT_POSITION


# ARDUINO SERIAL

try:
    arduino_serial = serial.Serial(
        ARDUINO_PORT,
        ARDUINO_BAUDRATE,
        timeout=0.1,
        write_timeout=1.0,
    )
except serial.SerialException as exc:
    packet_handler.write1ByteTxRx(
        port_handler, PAN_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
    )
    packet_handler.write1ByteTxRx(
        port_handler, TILT_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
    )
    port_handler.closePort()
    cap.release()
    raise RuntimeError(f"Failed to open Arduino on {ARDUINO_PORT}: {exc}") from exc

# Opening an Arduino serial port resets the board. Give setup() time to finish.
time.sleep(2.0)

arduino_stop_event = threading.Event()
arduino_reader_thread = threading.Thread(
    target=arduino_response_listener,
    daemon=True,
)
arduino_reader_thread.start()



# YOLO DETECTOR

model = YOLO("yolov8n.pt")

# Load the normal Ultralytics BoT-SORT config
botsort_cfg = YAML.load(ROOT / "cfg/trackers/botsort.yaml")

# Turn ReID on
botsort_cfg["with_reid"] = True

# Let Ultralytics use the detector's native features
botsort_cfg["model"] = "auto"

# Save our custom config
tracker_config = "botsort_reid.yaml"
YAML.save(tracker_config, botsort_cfg)


# Last known center and timestamp for each BoT-SORT track ID.
previous_positions = {}

# Time when each track first dropped below the idle movement threshold.
idle_start_times = {}

# Active target selection state.
active_target_id = None
last_target_selection_time = 0.0


try:
    while True:

        
        # CAPTURE FRAME
        
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame_time = time.perf_counter()


        
        # BOTSORT TRACKING
        

        results = model.track(
            frame,
            tracker=tracker_config,
            persist=True,
            classes=[0],

            verbose=False
        )

        result = results[0]


        
        # READ TRACKS

        tracked_objects = []

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):

                x1, y1, x2, y2 = box


                
                # BOUNDING BOX CENTER
                
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)



                # OBJECT VELOCITY

                speed = None
                previous_position = previous_positions.get(track_id)

                if previous_position is not None:
                    previous_cx, previous_cy, previous_time = previous_position
                    dt = frame_time - previous_time

                    if dt > 0:
                        dx = cx - previous_cx
                        dy = cy - previous_cy
                        movement_px = float(np.hypot(dx, dy))

                        if movement_px >= IDLE_MOVEMENT_THRESHOLD_PX:
                            idle_start_times.pop(track_id, None)
                            speed = movement_px / dt
                        else:
                            idle_start_time = idle_start_times.setdefault(
                                track_id,
                                frame_time,
                            )

                            if (
                                frame_time - idle_start_time
                                >= IDLE_CONFIRMATION_TIME_S
                            ):
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



        # REMOVE STALE TRACK HISTORY

        stale_track_ids = [
            track_id
            for track_id, (_, _, last_seen) in previous_positions.items()
            if frame_time - last_seen > TRACK_HISTORY_TIMEOUT_S
        ]

        for track_id in stale_track_ids:
            del previous_positions[track_id]
            idle_start_times.pop(track_id, None)



        # SELECT THE FASTEST OBJECT THAT IS NOT MOVING TOO FAST

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
            # max() naturally falls back to the second-fastest object when the
            # fastest object was excluded for exceeding the maximum speed.
            target = max(
                eligible_objects,
                key=lambda tracked_object: tracked_object["speed"],
                default=None,
            )

            last_target_selection_time = frame_time

            if target is None:
                active_target_id = None
            else:
                active_target_id = target["track_id"]



        # DRAW ALL TRACKS AND HIGHLIGHT THE ACTIVE TARGET

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

            box_thickness = 3 if is_target else 2


                
            # DRAW BOX
                
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                box_thickness
            )


                
            # DRAW TRACK ID AND SPEED

            cv2.putText(
                frame,
                f"ID: {track_id} | {speed_status}",
                (int(x1), max(int(y1) - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


                
            # DRAW OBJECT CENTER
                
            cv2.circle(
                frame,
                (int(cx), int(cy)),
                5,
                (0, 0, 255),
                -1
            )



        # CALCULATE ERROR FOR ONLY THE FASTEST ELIGIBLE TARGET

        if target is not None:
            target_id = target["track_id"]
            target_speed = target["speed"]
            error_x = cx_C - target["cx"]
            error_y = target["cy"] - cy_C

            # Apply the same proportional Dynamixel control used by the
            # original motor-control program.
            if abs(error_x) < deadzone_x:
                error_x = 0
            if abs(error_y) < deadzone_y:
                error_y = 0

            correction_pan = Kp_x * error_x * SPEED_SCALE
            correction_tilt = Kp_y * error_y * SPEED_SCALE

            new_motor_position_pan = int(
                np.clip(
                    current_motor_position_pan + correction_pan,
                    pan_min,
                    pan_max,
                )
            )
            new_motor_position_tilt = int(
                np.clip(
                    current_motor_position_tilt + correction_tilt,
                    tilt_min,
                    tilt_max,
                )
            )

            set_dynamixel_position(PAN_MOTOR_ID, new_motor_position_pan)
            set_dynamixel_position(TILT_MOTOR_ID, new_motor_position_tilt)

            current_motor_position_pan = new_motor_position_pan
            current_motor_position_tilt = new_motor_position_tilt

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
                f"Tracking ID: {target_id} | "
                f"Speed: {target_speed:.1f} px/s | "
                f"Error: ({error_x:.1f}, {error_y:.1f})"
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


        
        # DRAW CAMERA CENTER
        
        cv2.circle(
            frame,
            (cx_C, cy_C),
            5,
            (255, 0, 0),
            -1
        )


        
        # DISPLAY

        cv2.putText(
            frame,
            f"Motor speed: {SPEED_SCALE:.2f}x | Keys 1-4 change speed",
            (10, frame_height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "E: MOSFET ON | Shift+Q: OFF | F: FIRE | x: quit",
            (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        
        cv2.imshow("Camera Feed", frame)


        # Handle controls while the camera window has keyboard focus.
        if handle_key(cv2.waitKey(1) & 0xFF):
            break


finally:

    arduino_stop_event.set()
    if arduino_serial.is_open:
        try:
            # Always leave the MOSFET off when the program exits normally or
            # because of a handled exception.
            arduino_serial.write(b"Q")
            arduino_serial.flush()
            time.sleep(0.1)
            print("Shutdown: MOSFET OFF command sent to Arduino")
        except serial.SerialException as exc:
            print(f"[warn] Could not send MOSFET OFF during shutdown: {exc}")
        finally:
            arduino_serial.close()

    packet_handler.write1ByteTxRx(
        port_handler,
        PAN_MOTOR_ID,
        ADDR_MX_TORQUE_ENABLE,
        TORQUE_DISABLE,
    )
    packet_handler.write1ByteTxRx(
        port_handler,
        TILT_MOTOR_ID,
        ADDR_MX_TORQUE_ENABLE,
        TORQUE_DISABLE,
    )
    port_handler.closePort()
    cap.release()
    cv2.destroyAllWindows()
