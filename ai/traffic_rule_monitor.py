from ultralytics import YOLO
import cv2
import requests

from rule_engine import crossed_line, get_center
from traffic_light_color import detect_light_color


print("Starting AI Traffic Rule Monitor...")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera")
    exit()

# Virtual stop line
line_y = 280

# ---------------------------------
# TRAFFIC LIGHT STATE
# ---------------------------------
# For testing only.
# Later this will come from the
# traffic-light detection system.

traffic_light_state = "RED"

print(f"Traffic light: {traffic_light_state}")

# Store previous Y position
previous_positions = {}

# Prevent duplicate alerts
already_reported = set()

vehicle_classes = {"car", "motorcycle", "bus", "truck"}


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    # Traffic-light region: top-left
    light_roi = frame[0:200, 0:250]
    traffic_light_state = detect_light_color(light_roi)

    # YOLO tracking
    results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)

    annotated_frame = results[0].plot()

    # ---------------------------------
    # DRAW STOP LINE
    # ---------------------------------

    cv2.line(
        annotated_frame,
        (0, line_y),
        (annotated_frame.shape[1], line_y),
        (255, 255, 255),
        3
    )

    cv2.putText(
        annotated_frame,
        "STOP LINE",
        (20, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ---------------------------------
    # DISPLAY TRAFFIC LIGHT STATE
    # ---------------------------------

    cv2.putText(
        annotated_frame,
        f"TRAFFIC LIGHT: {traffic_light_state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ---------------------------------
    # TRACKING
    # ---------------------------------
    print(f"Detections: {len(results[0].boxes)}, "
           f"Classes: {[results[0].names[int(c)] for c in results[0].boxes.cls]}, "
              f"Tracking IDs: {results[0].boxes.id}"
               )
     
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        track_ids = (
            results[0]
            .boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):

            class_name = results[0].names[class_id]

            if class_name not in vehicle_classes:
                continue

            center_x, center_y = get_center(box)

            print(f"Vehicle ID {track_id}: Y={center_y}")

            # Show ID
            cv2.putText(
                annotated_frame,
                f"ID: {track_id}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # ---------------------------------
            # CHECK PREVIOUS POSITION
            # ---------------------------------

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # Check stop-line crossing
                if crossed_line(
                    previous_y,
                    center_y,
                    line_y
                ):

                    # ---------------------------------
                    # RED LIGHT RULE
                    # ---------------------------------

                    if traffic_light_state == "RED":

                        if track_id not in already_reported:

                            print(
                                "🚨 RED LIGHT VIOLATION!"
                            )

                            print(
                                f"Vehicle ID {track_id} "
                                "crossed the stop line "
                                "while the light was RED."
                            )

                            already_reported.add(track_id)

                            try:
                                response = requests.post(
                                    "http://127.0.0.1:8000/violation",
                                    timeout=5
                                )
                                print("Backend:", response.json())
                            except requests.RequestException as error:
                                print("Backend error:", error)

            # Save current position
            previous_positions[track_id] = center_y

    # ---------------------------------
    # SHOW CAMERA
    # ---------------------------------

    cv2.imshow(
        "AI Traffic Rule Monitor",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("AI Traffic Rule Monitor stopped.")
