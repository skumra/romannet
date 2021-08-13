import math

import cv2
import numpy as np


def draw_angled_rec(x0, y0, angle, img, width=50, height=25):
    b = math.cos(angle) * 0.5
    a = math.sin(angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)

    return img


def draw_arrow(x, y, angle, img, length=50):
    _x = int(x + length * math.cos(angle))
    _y = int(y + length * math.sin(angle))
    img = cv2.arrowedLine(img, (x, y), (_x, _y), (255, 255, 255), 3)

    return img


def draw_circle(x, y, img):
    img = cv2.circle(img, (x, y), 10, (255, 255, 255), 3)
    return img


def get_prediction_vis(predictions, color_heightmap, best_pix_ind, action):
    num_rotations = predictions.shape[0]
    angle = math.radians(best_pix_ind[0] * (360.0 / num_rotations))
    prediction_vis = predictions[best_pix_ind[0], :, :].copy()
    cv2.normalize(prediction_vis, prediction_vis, 0, 255, norm_type=cv2.NORM_MINMAX)
    prediction_vis = cv2.applyColorMap(prediction_vis.astype(np.uint8), cv2.COLORMAP_HOT)
    prediction_vis = (0.5 * cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis).astype(np.uint8)
    if action == 'grasp':
        prediction_vis = draw_angled_rec(int(best_pix_ind[2]), int(best_pix_ind[1]), angle, prediction_vis)
    elif action == 'place':
        prediction_vis = draw_circle(int(best_pix_ind[2]), int(best_pix_ind[1]), prediction_vis)
    elif action == 'push':
        prediction_vis = draw_arrow(int(best_pix_ind[2]), int(best_pix_ind[1]), angle, prediction_vis)
    return prediction_vis
