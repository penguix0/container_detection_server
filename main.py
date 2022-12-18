import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image 
import cv2
import numpy as np
from detect import Detector
import json

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['jpg'])
detector = None

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(12).hex()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/upload_image", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        print (request)
        # check if the post request has the file part
        if 'image' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            add_padding(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            downscale_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), 640)

    return "Succues"

@app.route("/api/get_detection", methods=["GET"])
def get_detection():
    if request.method == "GET":
        response = detector.detect()
        response = json.dumps(response, indent=4, sort_keys=True)
        return response


def add_padding(path):
    # read image
    img = cv2.imread(path)
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = old_image_width
    new_image_height = old_image_width
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img

    # save result
    cv2.imwrite(path, result)

def downscale_image(path, resolution):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    scaling_factor = resolution / max(height, width)
    cv2.imwrite(path, cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor))

if __name__ == '__main__':
    detector = Detector()
    detector.load()
    app.run(port=8080, debug=True)