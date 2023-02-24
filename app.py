from flask import Flask, redirect, render_template, request
from keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
# from tensorflow.keras.preprocessing import image
# from PIL import Image
import keras.utils as ku

# TO run this script use python -m flask ru

model = load_model('BrainTumorModel.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', bt='Predict', result=False)


@app.route('/upload', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        photo = request.files['brainPic']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename("brain.jpg"))

        print("Before saving")
        photo.save(file_path)
        print("After Saving")
        brainImg = ku.load_img(file_path)
        print(".............................................................")
        print(brainImg)
        photo = cv2.resize(cv2.imread(file_path),
                           dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file_path, photo)
        pred = np.round(model.predict(np.array([photo / 255.]))[0][0])
        if pred > 0.8:
            output = 'Brain Tumor Detected'
        else:
            output = 'There is no Brain Tumor'
    else:
        return redirect('/')
    return render_template('index.html', output=output, bt='Predict Again', result=True)


if __name__ == '__main__':
    app.run(debug=True)


# for Running  python -m flask run
