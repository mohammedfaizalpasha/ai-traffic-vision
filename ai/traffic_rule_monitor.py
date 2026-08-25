from ultralytics import YOLO
import cv2

from rule_engine import crossed_line, get_center


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


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
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

        for box, track_id in zip(boxes, track_ids):

            center_x, center_y = get_center(box)

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