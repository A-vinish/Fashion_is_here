"""
embeddings.py
--------------
Wraps OpenCLIP (open-source, free CLIP model) to turn images and text
into vectors that live in the SAME embedding space. That's the trick
that makes "search images using text" possible: an image of a blue
saree and the text "blue saree" end up close together in vector space.

Model used: ViT-B-32 pretrained on LAION-2B (~600MB, runs fine on CPU).
Completely free, downloaded once from Hugging Face / OpenCLIP's hub.
"""

import torch
import open_clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loaded once, reused for every request (loading takes a few seconds)
_model = None
_preprocess = None
_tokenizer = None


def load_model():
    """Lazy-load the CLIP model so the Flask app starts up fast."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        print(f"[embeddings] Loading OpenCLIP model on {DEVICE} ...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.to(DEVICE)
        _model.eval()
        print("[embeddings] Model loaded.")
    return _model, _preprocess, _tokenizer


def embed_image(image_path: str) -> list[float]:
    """Turn an image file into a normalized embedding vector."""
    model, preprocess, _ = load_model()
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)  # normalize

    return features[0].cpu().tolist()


def embed_text(text: str) -> list[float]:
    """Turn a text query (e.g. 'blue wedding saree') into the same
    embedding space as images, so we can compare them directly."""
    model, _, tokenizer = load_model()
    tokens = tokenizer([text]).to(DEVICE)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


if __name__ == "__main__":
    # Quick manual test: run `python embeddings.py` after adding a
    # sample image to backend/data/
    import sys
    if len(sys.argv) > 1:
        vec = embed_image(sys.argv[1])
        print(f"Embedding length: {len(vec)}")
        print(f"First 5 values: {vec[:5]}")
    else:
        vec = embed_text("blue wedding saree")
        print(f"Text embedding length: {len(vec)}")
