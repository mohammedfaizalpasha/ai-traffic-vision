from ultralytics import YOLO
import cv2

print("Starting AI Traffic Monitor...")

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame")
        break

    # Vehicle/object tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Display
    cv2.putText(
        annotated_frame,
        "AI Traffic Monitor",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Traffic Monitor",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("AI Traffic Monitor stopped.")