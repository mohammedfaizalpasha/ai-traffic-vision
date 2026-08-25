import cv2


def crossed_line(previous_y, current_y, line_y):
    """
    Returns True when an object moves from above the line
    to below the line.
    """

    if previous_y < line_y <= current_y:
        return True

    return False


def get_center(box):
    """
    Get the center point of a bounding box.
    """

    x1, y1, x2, y2 = box

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return center_x, center_y