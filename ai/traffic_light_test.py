import cv2
from traffic_light_color import detect_light_color


image = cv2.imread("traffic_light.jpg")

if image is None:
    print("ERROR: traffic_light.jpg not found")
    exit()

color = detect_light_color(image)

print(f"Traffic light state: {color}")