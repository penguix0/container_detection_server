import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image 

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['jpg'])

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
            print ("Filename: " +filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            add_padding(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return "succes"

def add_padding(path):
    image = Image.open(path)
    width, height = image.size
    new_height = width
    new_image = Image.new(image.mode, (width, new_height), (0,0,0))
    new_image.paste(image, (100, 100))
    new_image.save('image.jpg')
    
if __name__ == '__main__':
    app.run(port=8080, debug=True)