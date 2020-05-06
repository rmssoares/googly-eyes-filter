import copy
import cv2
import dlib
import pytest

from src import load_models

MODEL_CONFIG = {
    'face_model': {
        "type": "DNN",
        "DNN": {
            "prototxt": "models/deploy.prototxt",
            "caffemodel": "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        }
    }, 'eye_model': {
        "type": "HOG",
        "HOG": {
            "eyes_landmarks": "models/eye_eyebrows_22.dat"
        }
    }
}


def test_get_models():
    # Given a valid model configuration
    model_config = MODEL_CONFIG
    # When we attempt to load the models
    face_model, eye_model = load_models.get_models(model_config)
    # Then we get the correct models
    assert isinstance(face_model, cv2.dnn_Net)
    assert isinstance(eye_model, dlib.shape_predictor)


def test_load_face_detector_success():
    # Given a model configuration with a valid face_model type
    model_config = MODEL_CONFIG
    # When we attempt to load the model
    face_model = load_models.load_face_detector(model_config['face_model'])
    # Then we get the correct model
    assert isinstance(face_model, cv2.dnn_Net)


def test_load_face_detector_failure():
    # Given a model configuration with an invalid face_model type
    model_config = copy.deepcopy(MODEL_CONFIG)
    model_config['face_model']['type'] = "haarcascade"
    # When we attempt to load the models
    with pytest.raises(NotImplementedError)as exc:
        load_models.load_face_detector(model_config['face_model'])
    # Then we get a ValidationError exception
    assert "Face detector defined in config is not supported." in str(exc)


def test_load_eye_detector_success():
    # Given a model configuration with a valid eye_model type
    model_config = MODEL_CONFIG
    # When we attempt to load the model
    eye_model = load_models.load_eye_detector(model_config['eye_model'])
    # Then we get the correct model
    assert isinstance(eye_model, dlib.shape_predictor)


def test_load_eye_detector_failure():
    # Given a model configuration with an invalid eye_model type
    model_config = copy.deepcopy(MODEL_CONFIG)
    model_config['eye_model']['type'] = "haarcascade"
    # When we attempt to load the models
    with pytest.raises(NotImplementedError) as exc:
        load_models.load_eye_detector(model_config['eye_model'])
    # Then we get a ValidationError exception
    assert "Eye detector defined in config is not supported." in str(exc)
