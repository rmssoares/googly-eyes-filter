import cv2
import numpy as np
import math
import random


def generate_dummy_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def calculate_googly_center_and_size(eye_corners, settings):
    def calculateDistance(p1, p2):
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        return dist

    def findCentroid(p1, p2):
        return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

    size = int(calculateDistance(*eye_corners) * settings['size_multiplier'])
    size += int(size * random.uniform(0, settings['random_max_percent_inc']))
    # To simplify calculations, eyes will have even sides
    if size % 2 == 1:
        size += 1
    center = findCentroid(*eye_corners)
    return center, size
