import numpy as np
from src.googlifier import Googlifier
from dlib import rectangle, full_object_detection, point


class FakeDNNModel:
    # [_, _, confidence, xStart, yStart, xEnd, yEnd]
    face_detections = [[0, 0, 0.6, 1, 2, 5, 6], [0, 0, 0.4, 3, 3, 5, 5]]

    @staticmethod
    def setInput(X):
        return

    def forward(self):
        return np.array([[self.face_detections]])


def FakeHOGModel(_, face):
    return full_object_detection(face, [point(i, i) for i in range(20)])


def test_googlify_picture(mocker):
    # Given a googlifier and a blank_image
    googlifier = init_googlifier(mocker)
    blank_image = np.zeros((10, 10, 3), np.uint8)
    # When we mock method behaviour, to be able to just observe googlify's behaviour
    mocker.patch("src.googlifier.Googlifier.detect_faces")
    mocker.patch("src.googlifier.Googlifier.detect_eyes")
    mocker.patch("src.googlifier.Googlifier.generate_googly_eyes")
    # Then nothing breaks
    googlifier.googlify_picture(blank_image)


def test_detect_faces(mocker):
    # Given a googlifier and a blank image
    googlifier = init_googlifier(mocker)
    blank_image = np.zeros((10, 10, 3), np.uint8)
    # When we attempt to run detect_faces (check FakeDNNModel to peruse output)
    faces = googlifier.detect_faces(blank_image)
    # Then we get one face out of two, due to the threshold, and with the expected coordinates
    assert len(faces) == 1
    assert faces[0] == rectangle(10, 20, 50, 60)


def test_detect_eyes(mocker):
    # Given a face and an image
    faces = [rectangle(10, 20, 50, 60)]
    googlifier = init_googlifier(mocker)
    blank_image = np.zeros((10, 10, 3), np.uint8)
    # When we get the eyes from a list containing a single face (check FakeHOGModel to peruse output)
    left_eye, right_eye = googlifier.detect_eyes(blank_image, faces)
    # Then, we obtain the expected results
    left_center, left_size = left_eye
    right_center, right_size = right_eye
    assert left_center == (11, 11)
    assert right_center == (17, 17)
    assert left_size == right_size == 8


def test_generate_googly_eyes_success(mocker):
    # Given an image and an array with one eye
    blank_image = np.zeros((10, 10, 3), np.uint8)
    eyes = [((3, 3), 2)]
    googlifier = init_googlifier(mocker)
    # When we generate the googly eye
    img = googlifier.generate_googly_eyes(blank_image, eyes)
    # Then the image is modified
    assert not (img[:, :, :3] == 0).all()


def test_generate_googly_eyes_empty(mocker):
    # Given an image and an array with no eyes
    blank_image = np.zeros((10, 10, 3), np.uint8)
    eyes = []
    googlifier = init_googlifier(mocker)
    # When we attempt to generate googly eyes
    img = googlifier.generate_googly_eyes(blank_image, eyes)
    # Then the image is NOT modified
    assert (img[:, :, :3] == 0).all()


def init_googlifier(mocker):
    config = {"model": "", "googly_path": "tests/data/googly.png",
              "service": {
                  "confidence_threshold": 0.5,
                  "googly_settings": {
                      "random_max_percent_inc": 0,
                      "size_multiplier": 2
                  }
              }}
    mocker.patch('src.googlifier.get_models', return_value=(FakeDNNModel(), FakeHOGModel))
    return Googlifier(config)
