import numpy as np
import cv2
import time
import threading

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from ultralytics import YOLO


PROTOCOL_VERSION = 1.0
DEVICENAME = "COM12"
BAUDRATE = 57600
CAMERA_INDEX = 1

PAN_MOTOR_ID = 5
TILT_MOTOR_ID = 1

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Motor slew-speed register (joint mode): pinned HIGH so the motor can always execute each
# commanded step within a frame. This is NOT the tracking-speed knob (see SPEED_SCALE).
MOVING_SPEED = 1023   # 0-1023 magnitude (1023 = MX-28 top speed)

# Speed knob: multiplies the PID correction = how far the turret jumps per frame.
# Bigger = faster tracking. This is what 'a'/'d' and the 1-4 presets actually change.
SPEED_SCALE = 1.0
SPEED_STEP = 0.25       # a/d nudge amount
SPEED_SCALE_MIN = 0.1
SPEED_SCALE_MAX = 5.0

# Speed presets (keys 1-4 in the video window) -> correction multiplier
SPEED_LOW = 0.5
SPEED_MEDIUM = 1.0
SPEED_HIGH = 2.0
SPEED_SUPER = 3.5


TARGET_TRACK_ID = int(input("Track ID: ").strip() or "1")
print(f"Tracking ID: {TARGET_TRACK_ID}")


def track_id_input_listener():
    """Background thread: type a new track ID in the terminal to switch targets live."""
    global TARGET_TRACK_ID
    while True:
        try:
            entry = input().strip()
        except EOFError:
            break
        if not entry:
            continue
        try:
            TARGET_TRACK_ID = int(entry)
            print(f"Now tracking ID: {TARGET_TRACK_ID}")
        except ValueError:
            print(f"Invalid track ID: {entry!r}")


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
    # Used only ONCE at startup now (loop tracks position in software). Returns None on
    # a transient bus glitch instead of crashing.
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
    # Non-fatal: a bad write just gets retried next frame instead of crashing the program.
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
    # Non-fatal, same as above.
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
    """Set the correction multiplier (the speed knob) and clamp it."""
    global SPEED_SCALE
    SPEED_SCALE = float(np.clip(value, SPEED_SCALE_MIN, SPEED_SCALE_MAX))
    print(f"Speed scale: {SPEED_SCALE:.2f}x")


def apply_speed_change(delta):
    """Nudge the speed knob (used by the a/d keys)."""
    set_speed_scale(SPEED_SCALE + delta)


def handle_key(key):
    """Process a keypress from the OpenCV window. Returns True if quit was requested."""
    if key == ord("q"):
        return True
    elif key == ord("a"):
        apply_speed_change(SPEED_STEP)   # nudge up
    elif key == ord("d"):
        apply_speed_change(-SPEED_STEP)  # nudge down
    elif key == ord("1"):
        set_speed_scale(SPEED_LOW)
    elif key == ord("2"):
        set_speed_scale(SPEED_MEDIUM)
    elif key == ord("3"):
        set_speed_scale(SPEED_HIGH)
    elif key == ord("4"):
        set_speed_scale(SPEED_SUPER)
    return False


port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)

if not port_handler.openPort():
    raise RuntimeError(f"Failed to open Dynamixel port {DEVICENAME}")

if not port_handler.setBaudRate(BAUDRATE):
    raise RuntimeError(f"Failed to set Dynamixel baudrate to {BAUDRATE}")

dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
    port_handler, PAN_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE
)
check_dynamixel_result(dxl_comm_result, dxl_error, f"Enable torque for ID {PAN_MOTOR_ID}")

dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
    port_handler, TILT_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE
)
check_dynamixel_result(dxl_comm_result, dxl_error, f"Enable torque for ID {TILT_MOTOR_ID}")

# Pin motor slew speed high so it can always execute each commanded step
set_dynamixel_speed(PAN_MOTOR_ID, MOVING_SPEED)
set_dynamixel_speed(TILT_MOTOR_ID, MOVING_SPEED)


# Initialize the video capture
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, 30)

# Calculating center of camera using resolution
cx_C = frame_width / 2
cy_C = frame_height / 2

#Proportional gains (P controller)
Kp_x = 0.1
Kp_y = 0.1

#Deadzone (px): once the target is within this many pixels of center, stop correcting so
#the turret settles instead of jittering. This is the "stop" threshold. 0 = disabled.
deadzone_x = 0
deadzone_y = 0

#Set motor limits
pan_min = 0
pan_max = 4095
tilt_min = 1623   # ~142 deg, camera-mount lower limit (measured)
tilt_max = 3878   # ~341 deg, camera-mount upper limit (measured)


#Set tracking state:
# selected_track_id = none
lost_target_count = 0

# Read starting positions ONCE, then track in software (no per-loop reads = far fewer bus errors)
current_motor_position_pan = get_dynamixel_position(PAN_MOTOR_ID)
current_motor_position_tilt = get_dynamixel_position(TILT_MOTOR_ID)
if current_motor_position_pan is None:
    current_motor_position_pan = (pan_min + pan_max) // 2
if current_motor_position_tilt is None:
    current_motor_position_tilt = (tilt_min + tilt_max) // 2

model = YOLO("yolov8n.pt")   # standard person detection; BoT-SORT handles IDs

# Start terminal listener so the track ID can be changed live
input_thread = threading.Thread(target=track_id_input_listener, daemon=True)
input_thread.start()
print("Type a new track ID + Enter at any time to switch targets.")

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            classes=[0],          # person class in COCO
            tracker="botsort_reid.yaml",
            persist=True,
            verbose=False
        )

        target_found = False
        for box in results[0].boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            if track_id != TARGET_TRACK_ID:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2   # body box center
            cy = (y1 + y2) / 2

            target_found = True
            break

        if not target_found:
            annoted_frame = results[0].plot()
            cv2.imshow("human tracking turret version 3", annoted_frame)

            if handle_key(cv2.waitKey(1) & 0xFF):
                break
            continue

        #calculate error between SORT target center and camera center
        error_x = cx_C - cx
        error_y = cy - cy_C
        print(f"Tracking ID: {TARGET_TRACK_ID} | Error X: {error_x:.2f} | Error Y: {error_y:.2f}")

        #apply deadzone to error
        if abs(error_x) < deadzone_x:
            error_x = 0
        if abs(error_y) < deadzone_y:
            error_y = 0

        #proportional control: correction proportional to error, scaled by the speed knob
        correction_pan = Kp_x * error_x * SPEED_SCALE
        correction_tilt = Kp_y * error_y * SPEED_SCALE

        #Calculate new motor positions (position is tracked in software, no reads)
        new_motor_position_pan = current_motor_position_pan + correction_pan
        new_motor_position_tilt = current_motor_position_tilt + correction_tilt

        #Clamp
        new_motor_position_pan = int(np.clip(new_motor_position_pan, pan_min, pan_max))
        new_motor_position_tilt = int(np.clip(new_motor_position_tilt, tilt_min, tilt_max))

        #Send commands to motors
        set_dynamixel_position(PAN_MOTOR_ID, new_motor_position_pan)
        set_dynamixel_position(TILT_MOTOR_ID, new_motor_position_tilt)

        #Update stored motor positions
        current_motor_position_pan = new_motor_position_pan
        current_motor_position_tilt = new_motor_position_tilt

        annoted_frame = results[0].plot()
        cv2.imshow("human tracking turret version 3", annoted_frame)

        if handle_key(cv2.waitKey(1) & 0xFF):
            break
finally:
    packet_handler.write1ByteTxRx(
        port_handler, PAN_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
    )
    packet_handler.write1ByteTxRx(
        port_handler, TILT_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
    )
    port_handler.closePort()
    cap.release()
    cv2.destroyAllWindows()
