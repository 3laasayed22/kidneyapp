from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

app = Flask(__name__)


model = tf.keras.models.load_model('kidney_disease_model (4).h5')
class_names = ['Normal', 'Cyst', 'Tumor', 'Stone']

def is_valid_image(image_path):
    try:
        img = Image.open(image_path)
       
        if img.mode == 'RGB':
            img_array = np.array(img)
            threshold = 5  
            diff_rg = np.abs(img_array[..., 0] - img_array[..., 1])
            diff_gb = np.abs(img_array[..., 1] - img_array[..., 2])
          
            if np.max(diff_rg) > threshold or np.max(diff_gb) > threshold:
                return False
           
            img = img.convert('L')
        elif img.mode != 'L':
            img = img.convert('L')
        
        
        img_array = np.array(img)
        mean_intensity = np.mean(img_array)
        if mean_intensity > 250 or mean_intensity < 5:
            return False
        
        return True
    except Exception as e:
        return False

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)
    return image

def predict_from_file(image_path):
   
      
    if not os.path.exists(image_path):
        return {"error": "Image file not found."}
    
    if not is_valid_image(image_path):
        return {"error": "This is not a kidney radiology image."}
    
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    result = {
        "prediction": class_names[predicted_class],
        "confidence": f"{confidence * 100:.2f}%"
    }
    return result

@app.route('/')
def index():
    return "Kidney Disease Detection API is running."

@app.route('/predict_backend', methods=['GET'])
def predict_backend():
    
    image_path = 'your_image.png'
    result = predict_from_file(image_path)
    return jsonify(result)

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)


