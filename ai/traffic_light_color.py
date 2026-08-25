import cv2
import numpy as np


def detect_light_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_lower_1 = np.array([0, 100, 100])
    red_upper_1 = np.array([10, 255, 255])

    red_lower_2 = np.array([170, 100, 100])
    red_upper_2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 70, 70])
    green_upper = np.array([90, 255, 255])

    red_mask_1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
    red_mask_2 = cv2.inRange(hsv, red_lower_2, red_upper_2)

    red_pixels = (
        cv2.countNonZero(red_mask_1)
        + cv2.countNonZero(red_mask_2)
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    colors = {
        "RED": red_pixels,
        "YELLOW": yellow_pixels,
        "GREEN": green_pixels,
    }

    detected_color = max(colors, key=colors.get)

    if colors[detected_color] < 20:
        return "UNKNOWN"

    return detected_color


def create_test_image(color):
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    if color == "RED":
        bgr = (0, 0, 255)
    elif color == "YELLOW":
        bgr = (0, 255, 255)
    else:
        bgr = (0, 255, 0)

    cv2.circle(image, (100, 100), 60, bgr, -1)

    return image


if __name__ == "__main__":
    for color in ["RED", "YELLOW", "GREEN"]:
        test_image = create_test_image(color)
        result = detect_light_color(test_image)

        print(f"Expected: {color} | Detected: {result}")