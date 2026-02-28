from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
import json
import base64
import logging

app = Flask(__name__)
model = load_model('dog_breed_model.h5')

# Load class index to breed name mapping
with open('class_indices.json') as f:
    class_indices = json.load(f)

def prepare_image(file):
    img = load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    app.logger.info("Rendering index page.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error("No file part in the request.")
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file.")
        return jsonify({'error': 'No selected file'})
    if file:
        app.logger.info("File received. Preparing image.")
        img = prepare_image(file)
        preds = model.predict(img)
        predicted_class_index = np.argmax(preds[0])
        breed_name = class_indices[str(predicted_class_index)]
        
        # Reset the file pointer and read the file content
        file.seek(0)
        image_content = file.read()
        
        # Encode the image content in Base64
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        
        response_data = {
            'breed': breed_name,
            'image': image_base64
        }
        
        app.logger.info(f"Prediction: {breed_name}")
        
        response = app.response_class(
            response=json.dumps(response_data, indent=4),
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
