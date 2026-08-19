import numpy as np
import cv2

from ultralytics import YOLO


CAMERA_INDEX = 0   # was 1 -> "Camera index out of range" means no camera exists at 1 on this PC

# Speed knob: kept so the keybinds/printouts behave identically, but nothing is sent anywhere.
SPEED_SCALE = 1.0
SPEED_STEP = 0.25       # a/d nudge amount
SPEED_SCALE_MIN = 0.1
SPEED_SCALE_MAX = 5.0

SPEED_LOW = 0.5
SPEED_MEDIUM = 1.0
SPEED_HIGH = 2.0
SPEED_SUPER = 3.5


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


# Initialize the video capture
"""cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)"""  # only use if using a usb camera that is not a normal logi or whatever.
cap = cv2.VideoCapture(CAMERA_INDEX)  # use this for normal cameras (logitech, etc)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera at index {CAMERA_INDEX}. Try a different index (0, 1, 2...)."
    )

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

#Deadzone (px)
deadzone_x = 0
deadzone_y = 0

model = YOLO("yolov8n.pt")   # standard person detection; BoT-SORT handles IDs

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

        # Track the FIRST person detected in the frame (no ID selection)
        target_found = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2   # body box center
            cy = (y1 + y2) / 2

            target_found = True
            break

        if not target_found:
            annoted_frame = results[0].plot()
            cv2.imshow("human tracking turret version 3 (vision only)", annoted_frame)

            if handle_key(cv2.waitKey(1) & 0xFF):
                break
            continue

        #calculate error between target center and camera center
        error_x = cx_C - cx
        error_y = cy - cy_C

        #apply deadzone to error
        if abs(error_x) < deadzone_x:
            error_x = 0
        if abs(error_y) < deadzone_y:
            error_y = 0

        #what WOULD be sent to the motors (printed only, no commands go out)
        correction_pan = Kp_x * error_x * SPEED_SCALE
        correction_tilt = Kp_y * error_y * SPEED_SCALE
        print(
            f"Error X: {error_x:.2f} | Error Y: {error_y:.2f} "
            f"| would-be pan corr: {correction_pan:.2f} | tilt corr: {correction_tilt:.2f}"
        )

        annoted_frame = results[0].plot()
        cv2.imshow("human tracking turret version 3 (vision only)", annoted_frame)

        if handle_key(cv2.waitKey(1) & 0xFF):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()