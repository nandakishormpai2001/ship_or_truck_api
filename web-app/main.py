from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from torch.functional import split
from cnn_model.predict import predict_vehicle, Network
import base64
from PIL import Image
import os
from werkzeug.utils import secure_filename
from decouple import config
import os


app = Flask("ship_or_truck_api")
CORS(app)


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('main.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)

        image = Image.open(file_path)
        print(image)
        model = Network()
        vehicle = predict_vehicle(model, image)
    return render_template('predict.html', vehicle=vehicle)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
