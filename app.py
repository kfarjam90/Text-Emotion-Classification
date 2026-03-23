from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Load the model and preprocessing objects
print("Loading model and preprocessing objects...")

# Load model
model = tf.keras.models.load_model('models/emotion_model.keras')
print("✓ Model loaded successfully")

# Load tokenizer
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("✓ Tokenizer loaded successfully")

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("✓ Label encoder loaded successfully")

# Load model info
with open('models/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)
print(f"✓ Model info loaded: Max length = {model_info['max_length']}")
print(f"✓ Available emotions: {model_info['classes']}")

def predict_emotion(text):
    """
    Predict emotion from text input
    """
    # Preprocess the text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=model_info['max_length'])
    
    # Make prediction
    predictions = model.predict(padded, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    predicted_emotion = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Get confidence scores for all emotions
    confidence_scores = {}
    for i, emotion in enumerate(model_info['classes']):
        confidence_scores[emotion] = float(predictions[0][i])
    
    return {
        'emotion': predicted_emotion,
        'confidence': float(predictions[0][predicted_class_index]),
        'all_scores': confidence_scores
    }

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', emotions=model_info['classes'])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Make prediction
        result = predict_emotion(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', 
                         model_info=model_info,
                         emotions=model_info['classes'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)