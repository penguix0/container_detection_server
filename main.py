import os
import shutil
from flask import Flask, flash, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
import numpy as np
from detect import Detector
import json
import zipfile

UPLOAD_FOLDER = './upload'
CONVERT_FOLDER = './converted'
ALLOWED_EXTENSIONS = set(['zip'])
detector = None

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERT_FOLDER'] = CONVERT_FOLDER
app.config['SECRET_KEY'] = os.urandom(12).hex()
app.config["RESULT_PATH"] = "./result.json"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/upload_zip", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        print (request)
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if not file or not allowed_file(file.filename):
            return

        shutil.rmtree(app.config["UPLOAD_FOLDER"])
        os.mkdir(app.config["UPLOAD_FOLDER"])

        shutil.rmtree(app.config["CONVERT_FOLDER"])
        os.mkdir(app.config["CONVERT_FOLDER"])

        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])

        os.remove(zip_path)

    return "200"

@app.route("/api/start_detection", methods=["GET"])
def start_detection():
    if request.method == "GET":
        response = detector.detect()
        response = json.dumps(response, indent=4, sort_keys=True)
        with open(app.config["RESULT_PATH"], "w") as outfile:
            outfile.write(response)
        outfile.close()

        return "200"

@app.route("/api/get_json", methods=["GET"])
def get_result():
    if request.method == "GET":
        return send_file(app.config["RESULT_PATH"])

@app.route("/api/images/<image>", methods=["GET"])
def get_image(image):
    if request.method == "GET":
        file = os.path.join(os.path.dirname(__file__), "converted", image.split("?")[0])
        if os.path.exists(file):
            return send_file("./converted/"+image.split("?")[0])
        
        return send_file("./empty.jpg")

if __name__ == '__main__':
    detector = Detector()
    detector.load()
    app.run(host='0.0.0.0',port=8080, debug=True)