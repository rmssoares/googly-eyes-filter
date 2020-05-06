import cv2
import dlib


def get_models(model_params: dict):
    face_detector = load_face_detector(model_params['face_model'])
    eye_detector = load_eye_detector(model_params['eye_model'])
    return face_detector, eye_detector


def load_face_detector(face_config):
    if face_config['type'] == "DNN":
        dnn_config = face_config['DNN']
        return cv2.dnn.readNetFromCaffe(dnn_config['prototxt'], dnn_config['caffemodel'])
    else:
        raise NotImplementedError("Face detector defined in config is not supported.")


def load_eye_detector(eye_config):
    if eye_config['type'] == "HOG":
        hog_config = eye_config['HOG']
        return dlib.shape_predictor(hog_config['eyes_landmarks'])
    else:
        raise NotImplementedError("Eye detector defined in config is not supported.")
