from ultralytics import YOLO

print("Starting AI traffic detection...")

model = YOLO("yolo11n.pt")

image = "https://ultralytics.com/images/bus.jpg"

results = model(image)

for result in results:
    result.show()

print("Detection completed!")