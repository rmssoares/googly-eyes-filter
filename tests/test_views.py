from io import BytesIO

import numpy as np
import pytest

from src.views import app


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_googlify_success(client, mocker):
    # Given a blue pixel and the relevant methods mocked
    data = np.array([[[255, 0, 0]]])
    mocker.patch('src.views.googlifier')
    mocker.patch('os.makedirs')
    mock_cv2 = mocker.patch('cv2.imwrite')
    # When we invoke the POST endpoint and pass the pixel
    rv = client.post('/googlify', data=dict(file=(BytesIO(data.tobytes()), "blue_pixel.png")),
                     follow_redirects=True,
                     content_type='multipart/form-data'
                     )
    # Then we get the expected behaviour
    assert rv.data.decode() == 'Image googlified and saved.'
    mock_cv2.assert_called_once()
    assert mock_cv2.call_args[0][0] == 'googly_images/blue_pixel_googlified.png'


def test_googlify_failure(client):
    # Given a POST invocation with the wrong data type
    rv = client.post('/googlify', json={"img": "example"})
    # Then we get the following request
    assert rv.data.decode() == 'This class only supports multipart/form-data.'


def test_googlify_base64_success(client):
    # Given a POST invocation with an image encoded by base64 to the endpoint
    rv = client.post('/googlify_base64', json={
        "img": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}
                     )
    # When we attempt to get the response
    json_data = rv.get_json()
    # Then we get a googlified image
    assert "googlified_img" in json_data
    assert isinstance(json_data['googlified_img'], str)


def test_googlify_base64_failure(client):
    # Given a POST invocation with the wrong data type
    rv = client.post('/googlify_base64', data="test")
    # Then we get the following request
    assert rv.data.decode() == 'This class only supports json data.'

