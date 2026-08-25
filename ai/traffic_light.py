from ultralytics import YOLO
import cv2

print("Starting traffic object detection...")

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

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            if confidence > 0.30:
                print(f"Detected: {name} ({confidence:.2f})")

    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Detection stopped.")