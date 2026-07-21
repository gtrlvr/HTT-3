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
TILT_MOTOR_ID = 2

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


TARGET_TRACK_ID = int(input("Track ID: ").strip() or "1")
print(f"Tracking ID: {TARGET_TRACK_ID}")


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

#Set PID gains pan
Kp_x = 0.1
Ki_x = 0
Kd_x = 0

#Set PID gains tilt
Kp_y = 0
Ki_y = 0
Kd_y = 0

#Set PID memory
previous_error_x = 0
previous_error_y = 0
integral_x = 0
integral_y = 0

#Set deadzone
deadzone_x = 0
deadzone_y = 0

#Set motor limits
pan_min = 0
pan_max = 4095
tilt_min = 0
tilt_max = 0


#Set starting motor position
current_motor_position_pan = get_dynamixel_position(PAN_MOTOR_ID)
current_motor_position_tilt = get_dynamixel_position(TILT_MOTOR_ID)

#Set tracking state:
# selected_track_id = none
lost_target_count = 0

# PID control loop timing
previous_time = time.time()

model = YOLO("yolov8n.pt")

try:
    while True:

        current_time = time.time()
        dt = current_time - previous_time
        previous_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            classes=[0],
            tracker="botsort.yaml",
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
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            target_found = True
            break

        if not target_found:
            annoted_frame = results[0].plot()
            cv2.imshow("human tracking turret version 3", annoted_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
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

        #calculate PID for pan axis:
        proportional_x = Kp_x * error_x
        integral_x += Ki_x * error_x * dt
        derivative_x = Kd_x * (error_x - previous_error_x) / dt
        PID_output_x = proportional_x + integral_x + derivative_x

        #calculate PID for tilt axis:
        proportional_y = Kp_y * error_y
        integral_y += Ki_y * error_y * dt
        derivative_y = Kd_y * (error_y - previous_error_y) / dt
        PID_output_y = proportional_y + integral_y + derivative_y

        #set motor corrections
        correction_pan = PID_output_x
        # correction_tilt = PID_output_y

        #Get current position in dynamixel motors.
        current_motor_position_pan = get_dynamixel_position(PAN_MOTOR_ID)
        # current_motor_position_tilt = get_dynamixel_position(TILT_MOTOR_ID)

        #Calculate new motor positions
        new_motor_position_pan = current_motor_position_pan + correction_pan
        # new_motor_position_tilt = current_motor_position_tilt + correction_tilt

        #Clamp
        new_motor_position_pan = int(np.clip(new_motor_position_pan, pan_min, pan_max))
        # new_motor_position_tilt = np.clip(new_motor_position_tilt, tilt_min, tilt_max)

        #Send commands to motors
        set_dynamixel_position(PAN_MOTOR_ID, new_motor_position_pan)
        # set_dynamixel_position(TILT_MOTOR_ID, new_motor_position_tilt)

        #Update stored motor positions
        current_motor_position_pan = new_motor_position_pan
        # current_motor_position_tilt = new_motor_position_tilt

        #Update PID memory
        previous_error_x = error_x
        previous_error_y = error_y
        previous_time = current_time

        annoted_frame = results[0].plot()
        cv2.imshow("human tracking turret version 3", annoted_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    packet_handler.write1ByteTxRx(
        port_handler, PAN_MOTOR_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
    )
    port_handler.closePort()
    cap.release()
    cv2.destroyAllWindows()
