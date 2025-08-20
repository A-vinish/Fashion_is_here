import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import faiss

# Paths
DATASET_DIR = "uploads"   # change this to where your images are stored
FEATURES_PATH = "features_cache.pkl"
INDEX_PATH = "faiss_index.index"

# Load pre-trained ResNet50 (without top layer)
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Collect features
features = []
image_paths = []

print("Extracting features...")
for root, _, files in os.walk(DATASET_DIR):
    for file in tqdm(files):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                image_paths.append(path)

features = np.array(features, dtype="float32")

# Save features + paths
with open(FEATURES_PATH, "wb") as f:
    pickle.dump({"features": features, "paths": image_paths}, f)

print(f"Saved features to {FEATURES_PATH}")

# Build FAISS index
d = features.shape[1]  # feature dimension
index = faiss.IndexFlatL2(d)
index.add(features)

faiss.write_index(index, INDEX_PATH)
print(f"Saved FAISS index to {INDEX_PATH}")
