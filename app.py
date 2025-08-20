# Complete Fashion AI Backend (Visual Search + Sketch-to-Image)
import os
import numpy as np
import faiss
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import normalize

# =========================
# Config / Environment
# =========================
# Load from key.env
# =========================
from dotenv import load_dotenv
load_dotenv("key.env")

# Debug check (safe: hides most of your key)
api_key = os.getenv("STABILITY_API_KEY")
if api_key:
    print("Stability API key loaded:", api_key[:5] + "*****")
else:
    print("STABILITY_API_KEY not found in key.env")

DATASET_DIR = r"A:\DATASET\archive\fashion-dataset\images"
FEATURES_NPY = "features.npy"
FILENAMES_NPY = "filenames.npy"
BATCH_SIZE_DEFAULT = 256
TOP_K = 12

# Optional knobs via the same key.env (leave unset to use defaults)
MAX_IMAGES = os.getenv("MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None
BATCH_SIZE = int(os.getenv("BATCH_SIZE", BATCH_SIZE_DEFAULT))

# =========================
# Flask
# =========================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# Model / FAISS
# =========================
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
dimension = 2048
# Cosine similarity = normalize vectors + inner product
index = faiss.IndexFlatIP(dimension)
filenames = []

# =========================
# Helpers
# =========================
def preprocess_images_in_batch(image_paths, target_size=(224, 224)):
    batch_images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        batch_images.append(x)
    return np.array(batch_images)

def load_features(
    dataset_path=DATASET_DIR,
    feature_file=FEATURES_NPY,
    filename_file=FILENAMES_NPY,
    batch_size=BATCH_SIZE,
    max_images=MAX_IMAGES
):
    """
    Extract features for all images (or up to max_images), normalize them,
    save to disk, and build the FAISS index.
    """
    global index, filenames

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Collect image paths
    image_paths = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_paths.sort()  # determinism
    total_found = len(image_paths)
    print(f"📁 Dataset path: {dataset_path}")
    print(f"🔎 Image files found: {total_found}")

    # Respect optional limit
    if max_images is not None:
        image_paths = image_paths[:max_images]
        print(f"⚙️  MAX_IMAGES set → processing first: {len(image_paths)} images")
    else:
        print("⚙️  MAX_IMAGES not set → processing ALL images")

    # If cached features exist, load them; else extract anew
    if not (os.path.exists(feature_file) and os.path.exists(filename_file)):
        print(f"🚀 Extracting features with batch_size={batch_size} ...")
        features_chunks = []
        filenames_local = []
        n = len(image_paths)

        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = preprocess_images_in_batch(batch_paths)
            batch_features = model.predict(batch, verbose=0)  # quiet
            features_chunks.append(batch_features)
            filenames_local.extend([os.path.basename(p) for p in batch_paths])

            # Progress
            done = min(i + batch_size, n)
            print(f"   • Processed {done}/{n}")

        features = np.vstack(features_chunks).astype('float32')
        # Normalize rows for cosine
        features = normalize(features, axis=1)

        # Save
        np.save(feature_file, features)
        np.save(filename_file, filenames_local)

        print(f"✅ Total images processed: {len(filenames_local)}")
        print(f"✅ Feature matrix shape: {features.shape}")
        print(f"💾 Saved features → {feature_file}")
        print(f"💾 Saved filenames → {filename_file}")

        filenames = filenames_local
    else:
        print("📦 Loading cached features from disk ...")
        features = np.load(feature_file).astype('float32')
        filenames = np.load(filename_file).tolist()

        # Ensure normalized (in case old cache wasn’t)
        features = normalize(features, axis=1)

        print(f"✅ Loaded {features.shape[0]} features, dim={features.shape[1]}")

    # Build FAISS
    index.reset()
    index.add(features)
    print(f"🧠 FAISS index loaded with {index.ntotal} vectors (cosine ready)")

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x, verbose=0)
        feat = normalize(feat, axis=1)  # normalize for cosine
        return feat.flatten().astype('float32')
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

# =========================
# Startup: build (or load) index
# =========================
load_features()

# =========================
# Routes
# =========================
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(DATASET_DIR, filename)

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

        q = extract_features(filepath)
        if q is None:
            return jsonify({"error": "Image processing failed"}), 500

        # Top K matches
        distances, indices = index.search(np.array([q]), TOP_K)
        result_filenames = [filenames[idx] for idx in indices[0]]

        return jsonify({
            "results": indices.tolist(),
            "filenames": result_filenames,
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
        if not api_key:
            return jsonify({"error": "STABILITY_API_KEY not set in key.env"}), 500

        with open(filepath, 'rb') as f:
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "image/*"
                },
                files={"image": f},
                data={
                    "prompt": "high-quality fashionable clothing item",
                    "output_format": "webp"
                },
                timeout=60
            )

        if response.status_code == 200:
            return response.content, 200, {'Content-Type': 'image/webp'}
        return jsonify({"error": f"API failed: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# (Removed the old /features/... route that caused warnings)

# =========================
# Main
# =========================
if __name__ == '__main__':
    # Use port 5000 as in your last run
    app.run(host='0.0.0.0', port=5000, debug=False)
