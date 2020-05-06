import base64

import cv2
import numpy as np


def decode_base64_str_to_img(string):
    buffer = np.frombuffer(base64.b64decode(string), np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return img


def encode_img_to_base64_str(img, filetype='jpg'):
    _, buffer = cv2.imencode(f".{filetype}", img)
    img_base64_bytes = base64.b64encode(buffer)
    return img_base64_bytes.decode()

