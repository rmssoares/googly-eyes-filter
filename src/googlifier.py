import random

import cv2
import dlib
import numpy as np
from imutils import face_utils

from src.utils.image_utils import generate_dummy_alpha_channel, calculate_googly_center_and_size
from src.load_models import get_models


class Googlifier:
    def __init__(self, params):
        self.face_detector, self.eye_detector = get_models(params['model'])
        self.googly_params = params['service']
        self.googly_img = cv2.imread(params['googly_path'], cv2.IMREAD_UNCHANGED)

    def googlify_picture(self, img):
        faces = self.detect_faces(img)
        eyes = self.detect_eyes(img, faces)
        return self.generate_googly_eyes(img, eyes)

    def detect_faces(self, img):
        """
        In the provided image, we detect as many faces as possible through the DNN model. Then, according to a measure
        of confidence, we filter out the detections that we are not positive enough about.
        """
        # Obtain face detections
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        faces = []
        # Get coordinates for each detected face, filter out cases with little confidence
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.googly_params['confidence_threshold']:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append(dlib.rectangle(startX, startY, endX, endY))
        return faces

    def detect_eyes(self, img, faces):
        """
        For each face, we identify the facial landmarks through the HOG model. We find the eye corners and we use
        these to define the measures of our goggly eyes.
        """
        eyes = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for face in faces:
            # Obtain eye landmarks
            shape = self.eye_detector(gray, face)
            shape = face_utils.shape_to_np(shape)
            # Identify corners and calculate ((center), size) per eye
            left_eye_corners, right_eye_corners = (shape[10], shape[13]), (shape[16], shape[19])
            eyes.append(calculate_googly_center_and_size(left_eye_corners, self.googly_params['googly_settings']))
            eyes.append(calculate_googly_center_and_size(right_eye_corners, self.googly_params['googly_settings']))
        return eyes

    def generate_googly_eyes(self, img, eyes):
        """
        For each eye, we resize accordingly, rotate, and we add to the original picture. Eyes are defined by the
         coordinate that represents its center, and its size. -> ((c_x, c_y), size)
        """
        img = generate_dummy_alpha_channel(img)
        for (x, y), size in eyes:
            half = size // 2
            # Resize eye image according to metrics
            eye_original = cv2.resize(self.googly_img, (size, size), interpolation=cv2.INTER_AREA)
            # Handle transparency around googly eyes
            roi = img[y - half:y + half, x - half:x + half]
            mask_eye = (eye_original[:, :, 3] != 0).astype(np.uint8) * 255
            mask_not_eye = cv2.bitwise_not(mask_eye)
            eye = cv2.bitwise_and(eye_original, eye_original, mask=mask_eye)
            eye_background = cv2.bitwise_and(roi, roi, mask=mask_not_eye)
            # Eye rotation
            rotation = cv2.getRotationMatrix2D((half, half), random.randrange(360), 1)
            eye = cv2.warpAffine(eye, rotation, (size, size))
            # merge eye into main picture
            dst = cv2.add(eye_background, eye)
            img[y - half:y + half, x - half:x + half] = dst
        return img
