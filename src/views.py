import io

import cv2
import numpy as np
from flask import Flask, request, Response, jsonify

from src.googlifier import Googlifier
from src.utils.io_utils import load_config, get_googly_filepath
from src.utils.codec import encode_img_to_base64_str, decode_base64_str_to_img

# The flask app for serving predictions
app = Flask(__name__)

config = load_config()
googlifier = Googlifier(config)


@app.route('/googlify', methods=['POST'])
def googlify():
    if 'multipart/form-data' not in request.content_type:
        return Response(
            response='This class only supports multipart/form-data.',
            status=415,
            mimetype='application/json'
        )
    file = request.files['file']
    destination = get_googly_filepath(file.filename)
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)
    googlified = googlifier.googlify_picture(img)
    cv2.imwrite(destination, googlified)
    return Response(response='Image googlified and saved.', status=200, mimetype='application/json')


@app.route('/googlify_base64', methods=['POST'])
def googlify_base64():
    if request.content_type != 'application/json':
        return Response(response='This class only supports json data.',
                        status=415,
                        mimetype='application/json')
    data = request.get_json()
    img = decode_base64_str_to_img(data['img'])
    googlified = googlifier.googlify_picture(img)
    img_base64 = encode_img_to_base64_str(googlified[:, :, :3])
    return jsonify({"googlified_img": img_base64})
