import numpy as np
import cv2
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from ultralytics import YOLO


PROTOCOL_VERSION = 1.0
DEVICENAME = "COM3"
BAUDRATE = 57600
CAMERA_INDEX = 1

PAN_MOTOR_ID = 1
# TILT_MOTOR_ID = 2

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

TUNING_WINDOW = "Live PID Tuning"
KP_SCALE = 100
KI_SCALE = 100
KD_SCALE = 100


def noop(_value):
    pass


def check_dynamixel_result(dxl_comm_result, dxl_error, action):
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeError(
            f"{action} failed: {packet_handler.getTxRxResult(dxl_comm_result)}"
        )
    if dxl_error != 0:
        raise RuntimeError(
            f"{action} failed: {packet_handler.getRxPacketError(dxl_error)}"
        )


def get_dynamixel_position(motor_id, retries=1):
    last_result = COMM_SUCCESS
    last_error = 0

    for _ in range(retries + 1):
        port_handler.clearPort()
        position, dxl_comm_result, dxl_error = packet_handler.read2ByteTxRx(
            port_handler, motor_id, ADDR_MX_PRESENT_POSITION
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            return position

        last_result = dxl_comm_result
        last_error = dxl_error
        time.sleep(0.01)

    check_dynamixel_result(last_result, last_error, f"Read position for ID {motor_id}")
    return 0


def set_dynamixel_position(motor_id, position, retries=1):
    last_result = COMM_SUCCESS
    last_error = 0

    for _ in range(retries + 1):
        port_handler.clearPort()
        dxl_comm_result, dxl_error = packet_handler.write2ByteTxRx(
            port_handler, motor_id, ADDR_MX_GOAL_POSITION, int(position)
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            return

        last_result = dxl_comm_result
        last_error = dxl_error
        time.sleep(0.01)

    check_dynamixel_result(last_result, last_error, f"Write goal position for ID {motor_id}")


def get_live_pid_values():
    kp_x = cv2.getTrackbarPos("Kp x100", TUNING_WINDOW) / KP_SCALE
    ki_x = cv2.getTrackbarPos("Ki x100", TUNING_WINDOW) / KI_SCALE
    kd_x = cv2.getTrackbarPos("Kd x100", TUNING_WINDOW) / KD_SCALE
    deadzone_x = cv2.getTrackbarPos("Deadzone", TUNING_WINDOW)
    return kp_x, ki_x, kd_x, deadzone_x


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

# Set PID gains pan
Kp_x = 0
Ki_x = 0
Kd_x = 0

# Set PID memory
previous_error_x = 0
integral_x = 0

# Set deadzone
deadzone_x = 0

# Set motor limits
pan_min = 0
pan_max = 4095

# Set starting motor position
current_motor_position_pan = get_dynamixel_position(PAN_MOTOR_ID)

# PID control loop timing
previous_time = time.time()

cv2.namedWindow(TUNING_WINDOW)
cv2.createTrackbar("Kp x100", TUNING_WINDOW, int(Kp_x * KP_SCALE), 500, noop)
cv2.createTrackbar("Ki x100", TUNING_WINDOW, int(Ki_x * KI_SCALE), 500, noop)
cv2.createTrackbar("Kd x100", TUNING_WINDOW, int(Kd_x * KD_SCALE), 500, noop)
cv2.createTrackbar("Deadzone", TUNING_WINDOW, deadzone_x, 100, noop)

model = YOLO("yolov8n.pt")

try:
    while True:
        current_time = time.time()
        dt = max(current_time - previous_time, 1e-6)
        previous_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        Kp_x, Ki_x, Kd_x, deadzone_x = get_live_pid_values()

        results = model.track(
            frame,
            classes=[0],
            tracker="botsort.yaml",
            persist=True
        )

        target_found = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            track_id = box.id
            print(cx, cy, track_id)
            target_found = True
            break

        annoted_frame = results[0].plot()

        if not target_found:
            cv2.putText(
                annoted_frame,
                f"Kp:{Kp_x:.2f} Ki:{Ki_x:.2f} Kd:{Kd_x:.2f} DZ:{deadzone_x}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("human tracking turret live PID", annoted_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        error_x = cx_C - cx

        if abs(error_x) < deadzone_x:
            error_x = 0

        proportional_x = Kp_x * error_x
        integral_x += Ki_x * error_x * dt
        derivative_x = Kd_x * (error_x - previous_error_x) / dt
        pid_output_x = proportional_x + integral_x + derivative_x

        correction_pan = pid_output_x

        new_motor_position_pan = current_motor_position_pan + correction_pan
        new_motor_position_pan = int(np.clip(new_motor_position_pan, pan_min, pan_max))

        set_dynamixel_position(PAN_MOTOR_ID, new_motor_position_pan)
        current_motor_position_pan = new_motor_position_pan
        previous_error_x = error_x

        cv2.putText(
            annoted_frame,
            f"Kp:{Kp_x:.2f} Ki:{Ki_x:.2f} Kd:{Kd_x:.2f} DZ:{deadzone_x}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            annoted_frame,
            f"ErrX:{error_x:.1f} Pan:{current_motor_position_pan}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("human tracking turret live PID", annoted_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    packet_handler.write1ByteTxRx(
        port_handler, PAN_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
    )
    port_handler.closePort()
    cap.release()
    cv2.destroyAllWindows()
