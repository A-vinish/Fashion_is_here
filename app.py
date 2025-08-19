# Complete Fashion AI Backend (Visual Search + Sketch-to-Image)
import numpy as np
import os
import faiss
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory  # Added send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load environment variables
load_dotenv()

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = VGG16(weights='imagenet', include_top=False, pooling='avg')
dimension = 512
index = faiss.IndexFlatL2(dimension)
filenames = []  # NEW: To track image filenames

# Start batch processing
def preprocess_images_in_batch(image_paths, target_size=(224, 224)):
    batch_images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        batch_images.append(x)
    return np.array(batch_images)
# End batch processing

def load_features(dataset_path=r"A:\DATASET\archive\fashion-dataset\images", 
                 feature_file="features.npy", 
                 filename_file="filenames.npy",  # NEW: Filename tracking
                 batch_size=1000,
                 max_images=None):
    global index, filenames
    
    if not os.path.exists(feature_file) or not os.path.exists(filename_file):
        image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        features = []
        filenames = []  # Reset filenames
        
        # Batch processing
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch = preprocess_images_in_batch(batch_paths)
            batch_features = model.predict(batch)
            features.append(batch_features)
            filenames.extend([os.path.basename(p) for p in batch_paths])  # Store filenames
        
        features = np.vstack(features).astype('float32')
        np.save(feature_file, features)
        np.save(filename_file, filenames)  # Save filenames
        print(f"Saved {len(features)} features and {len(filenames)} filenames")
    else:
        features = np.load(feature_file).astype('float32')
        filenames = np.load(filename_file).tolist()  # Load filenames
    
    index.reset()
    index.add(features)
    print(f"FAISS index loaded with {index.ntotal} vectors")

# Initialize with dataset (using raw string for Windows paths)
load_features(max_images=5000)

# Feature extraction with error handling
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

# NEW: Image serving endpoint
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(
        r"A:\DATASET\archive\fashion-dataset\images",  # Your dataset path
        filename
    )

# API Endpoints
@app.route('/api/search', methods=['POST'])
def visual_search():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format"}), 400
        
        features = extract_features(filepath)
        if features is None:
            return jsonify({"error": "Image processing failed"}), 500
            
        distances, indices = index.search(np.array([features.astype('float32')]), 5)
        
        # NEW: Map indices to filenames
        result_filenames = [filenames[idx] for idx in indices[0]]
        
        return jsonify({
            "results": indices.tolist(),
            "filenames": result_filenames,  # NEW: Return filenames
            "distances": distances.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def sketch_to_image():
    if 'sketch' not in request.files:
        return jsonify({"error": "No sketch uploaded"}), 400
    
    file = request.files['sketch']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        api_key = os.getenv("STABILITY_API_KEY")
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*"
            },
            files={"image": open(filepath, 'rb')},
            data={
                "prompt": "high-quality fashionable clothing item",
                "output_format": "webp"
            }
        )

        if response.status_code == 200:
            return response.content, 200, {'Content-Type': 'image/webp'}
        return jsonify({"error": f"API failed: {response.text}"}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
