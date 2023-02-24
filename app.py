from flask import Flask, redirect, render_template, request
from keras.models import load_model
import numpy as np
import cv2

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
        photo.save('static/brain.jpg')
        photo = cv2.resize(cv2.imread('static/brain.jpg'),
                           dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('static/brain.jpg', photo)
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
