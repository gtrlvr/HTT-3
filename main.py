
import numpy as np 
import cv2 
import time

from ultralytics import YOLO




# Initialize the video capture
cap = cv2.VideoCapture(0)
frame_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
frame_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_fps = cap.set(cv2.CAP_PROP_FPS, 30)

# Calculating center of camera using resolution
cx_C = frame_width / 2
cy_C = frame_height / 2

#Set PID gains pan 
Kp_x = 0 
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
pan_max = 0
tilt_min = 0
tilt_max = 0


#Set starting motor position 
# current_motor_position_pan = 0
# current_motor_position_tilt =  0

#Set tracking state:
# selected_track_id = none
lost_target_count = 0 

# PID control loop timing
previous_time = time.time()

model = YOLO('yolov8n.pt')

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
        persist=True
    )

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        track_id = box.id
        print(cx, cy, track_id)

    #calculate error between SORT target center and camera center
    error_x = cx - cx_C
    error_y = cy - cy_C
    
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
    correction_tilt = PID_output_y

    #Get current position in dynamixel motors.
    # current_motor_position_pan = get_dynamixel_position(pan_motor_id)
    # current_motor_position_tilt = get_dynamixel_position(tilt_motor_id)

    #Calculate new motor positions
    new_motor_position_pan = current_motor_position_pan + correction_pan
    new_motor_position_tilt = current_motor_position_tilt + correction_tilt

    #Clamp
    new_motor_position_pan = np.clip(new_motor_position_pan, pan_min, pan_max)
    new_motor_position_tilt = np.clip(new_motor_position_tilt, tilt_min, tilt_max)

    #Send commands to motors 
    # set dynamixel position to new_motor_position_pan
    # set dynamixel position to new_motor_position_tilt

    #Update stored motor positions
    current_motor_position_pan = new_motor_position_pan
    current_motor_position_tilt = new_motor_position_tilt

    #Update PID memory 
    previous_error_x = error_x
    previous_error_y = error_y
    previous_time = current_time



    annoted_frame = results[0].plot()
    cv2.imshow('human tracking turret version 3', annoted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

