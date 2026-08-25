from ultralytics import YOLO
import cv2

print("Starting AI traffic tracking...")

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("ERROR: Could not open camera")
    exit()

while True:
    success, frame = video.read()

    if not success:
        print("Could not read camera frame")
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("AI Traffic Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Traffic tracking stopped.")