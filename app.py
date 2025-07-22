from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
model = load_model("mnist_digit_classifier.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).astype('float32') / 255.0
    img = 1 - img  # invert colors (white background, black digit)
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({'digit': digit, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)
