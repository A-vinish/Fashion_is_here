"""
auto_tag.py
-----------
Since your 40k images have no metadata (just numbered filenames), we use
CLIP's "zero-shot classification" trick to auto-generate tags: we compare
each image's embedding against a fixed list of candidate text labels
(colors, categories, occasions) and pick whichever text is closest in
embedding space. No training needed - this is the same CLIP model
already used for search, just used a different way.

This is a real, legitimate ML technique (zero-shot classification) and
is worth mentioning explicitly on your resume/in interviews.
"""

import numpy as np
from embeddings import embed_text

CANDIDATE_COLORS = [
    "red", "blue", "green", "black", "white", "yellow", "pink", "purple",
    "orange", "brown", "grey", "navy blue", "maroon", "beige", "gold",
    "silver", "multicolor",
]

CANDIDATE_CATEGORIES = [
    "saree", "kurta", "dress", "shirt", "t-shirt", "jeans", "trousers",
    "skirt", "jacket", "sweater", "shoes", "sandals", "handbag", "watch",
    "jewelry", "top", "blouse", "suit", "shorts",
]

CANDIDATE_OCCASIONS = [
    "casual wear", "formal wear", "party wear", "wedding wear",
    "sportswear", "office wear", "festive wear", "ethnic wear",
]

# Rough placeholder price ranges per category (in Rs) - since the
# dataset has no real prices, these keep results looking realistic.
PRICE_RANGES = {
    "saree": (1500, 6000), "kurta": (600, 2500), "dress": (900, 4000),
    "shirt": (500, 2000), "t-shirt": (300, 1200), "jeans": (800, 2500),
    "trousers": (700, 2200), "skirt": (600, 2000), "jacket": (1200, 4500),
    "sweater": (800, 2500), "shoes": (900, 3500), "sandals": (400, 1800),
    "handbag": (700, 3000), "watch": (1000, 5000), "jewelry": (400, 3000),
    "top": (400, 1800), "blouse": (500, 2000), "suit": (2000, 8000),
    "shorts": (400, 1500),
}
DEFAULT_PRICE_RANGE = (500, 3000)

_label_embeddings_cache = {}


def _get_label_embeddings(labels: list[str]) -> dict:
    """Embed each candidate label once and cache it (only happens once
    per script run, not per image - keeps this fast)."""
    key = tuple(labels)
    if key not in _label_embeddings_cache:
        _label_embeddings_cache[key] = {
            label: np.array(embed_text(label)) for label in labels
        }
    return _label_embeddings_cache[key]


def _best_match(image_embedding: list[float], labels: list[str]) -> str:
    img_vec = np.array(image_embedding)
    label_vecs = _get_label_embeddings(labels)
    scores = {label: float(np.dot(img_vec, vec)) for label, vec in label_vecs.items()}
    return max(scores, key=scores.get)


def auto_tag(image_embedding: list[float]) -> dict:
    """Returns {"color": ..., "category": ..., "occasion": ..., "price": ...}"""
    color = _best_match(image_embedding, CANDIDATE_COLORS)
    category_raw = _best_match(image_embedding, CANDIDATE_CATEGORIES)
    occasion = _best_match(image_embedding, CANDIDATE_OCCASIONS).replace(" wear", "")

    low, high = PRICE_RANGES.get(category_raw, DEFAULT_PRICE_RANGE)
    import random
    price = random.randint(low, high)

    return {
        "color": color,
        "category": category_raw,
        "occasion": occasion,
        "price": price,
    }
