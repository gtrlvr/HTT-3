
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

    
    annoted_frame = results[0].plot()
    cv2.imshow('human tracking turret version 3', annoted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

