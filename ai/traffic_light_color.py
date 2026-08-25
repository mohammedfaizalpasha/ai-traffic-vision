import cv2
import numpy as np


def detect_light_color(image):
    """
    Detect the active traffic-light color.

    Returns:
        RED, YELLOW, GREEN, or UNKNOWN
    """

    if image is None or image.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV ranges
    red_lower_1 = np.array([0, 100, 100])
    red_upper_1 = np.array([10, 255, 255])

    red_lower_2 = np.array([170, 100, 100])
    red_upper_2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 70, 70])
    green_upper = np.array([90, 255, 255])

    red_mask = (
        cv2.inRange(hsv, red_lower_1, red_upper_1)
        | cv2.inRange(hsv, red_lower_2, red_upper_2)
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

    # Calculate brightness-weighted scores
    value_channel = hsv[:, :, 2]

    red_score = int(
        np.sum(value_channel[red_mask > 0])
    )

    yellow_score = int(
        np.sum(value_channel[yellow_mask > 0])
    )

    green_score = int(
        np.sum(value_channel[green_mask > 0])
    )

    scores = {
        "RED": red_score,
        "YELLOW": yellow_score,
        "GREEN": green_score,
    }

    best_color = max(scores, key=scores.get)

    if scores[best_color] < 5000:
        return "UNKNOWN"

    return best_color


if __name__ == "__main__":
    print("Traffic-light color detector loaded.")