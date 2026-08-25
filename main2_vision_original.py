import time

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import ROOT, YAML


CAMERA_INDEX = 0

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


# CAMERA

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

frame_width = 640
frame_height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, 30)



# CAMERA CENTER
cx_C = frame_width // 2
cy_C = frame_height // 2



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
            classes=[67],

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
            error_y = cy_C - target["cy"]

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
        
        cv2.imshow("Camera Feed", frame)


        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    cap.release()
    cv2.destroyAllWindows()
