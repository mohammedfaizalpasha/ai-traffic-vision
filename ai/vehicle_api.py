from ultralytics import YOLO
import requests

MODEL = "yolo11n.pt"
BACKEND = "http://127.0.0.1:8000"

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
}

print("Starting AI vehicle API...")

model = YOLO(MODEL)

image = "https://ultralytics.com/images/bus.jpg"

results = model(image)

vehicle_count = 0

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = result.names[class_id]

        if name in VEHICLE_CLASSES:
            vehicle_count += 1
            print(f"Detected: {name} ({confidence:.2f})")

print(f"Vehicles detected: {vehicle_count}")

response = requests.post(
    f"{BACKEND}/vehicles/{vehicle_count}",
    timeout=5,
)

print("Backend response:", response.json())
print("AI vehicle API completed!")
