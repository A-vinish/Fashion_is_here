"""
app.py
------
Flask backend for StyleGPT. Three main endpoints:

  POST /upload        -> add a new fashion item (image + metadata) to the catalog
  POST /search/image   -> image-to-image visual search (your original v1 feature)
  POST /chat            -> conversational search: text refines/filters results

Run:
    cd backend
    pip install -r requirements.txt
    python app.py
Then open frontend/index.html in your browser (or serve it separately).
"""

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from flask import send_file
from embeddings import embed_image, embed_text
from vectorstore import add_item, search, count_items, get_collection
from filter_parser import parse_filters, filters_to_chroma_where

app = Flask(__name__)
CORS(app)  # allows the frontend (different port) to call this API

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory session store: keeps track of each user's last
# uploaded image + accumulated filters across a conversation.
# (For a resume project, in-memory is fine. A production version
#  would use Redis or a DB table keyed by session id.)
SESSIONS = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "items_in_catalog": count_items()})


@app.route("/upload", methods=["POST"])
def upload_item():
    """
    Add a fashion item to the searchable catalog.
    Expects multipart/form-data: image file + color, occasion, category, price
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    item_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_FOLDER, f"{item_id}_{filename}")
    file.save(save_path)

    metadata = {
        "image_path": save_path,
        "color": request.form.get("color", "unknown"),
        "occasion": request.form.get("occasion", "unknown"),
        "category": request.form.get("category", "unknown"),
        "price": float(request.form.get("price", 0)),
    }

    embedding = embed_image(save_path)
    add_item(item_id, embedding, metadata)

    return jsonify({"item_id": item_id, "metadata": metadata})


@app.route("/search/image", methods=["POST"])
def search_by_image():
    """Original v1 feature: upload an image, find visually similar items."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    session_id = request.form.get("session_id", "default")

    temp_path = os.path.join(UPLOAD_FOLDER, f"query_{uuid.uuid4()}.jpg")
    file.save(temp_path)

    query_embedding = embed_image(temp_path)

    # Remember this embedding so /chat can refine from it later
    SESSIONS[session_id] = {"base_embedding": query_embedding, "filters": {}}

    results = search(query_embedding, top_k=8)
    return jsonify({"results": results})


@app.route("/chat", methods=["POST"])
def chat_refine():
    """
    v2 feature: conversational refinement.
    Body: {"session_id": "...", "message": "show me this but in blue for a wedding"}

    Flow:
      1. Parse the message into structured filters via LangChain + LLM
      2. Merge with any filters already collected earlier in the conversation
      3. Re-run the vector search using the session's base image embedding
         (or a text embedding if no image was uploaded) + the merged filters
    """
    data = request.get_json()
    session_id = data.get("session_id", "default")
    message = data.get("message", "")

    session = SESSIONS.setdefault(session_id, {"base_embedding": None, "filters": {}})

    try:
        # 1. Extract new filters from this message
        new_filters = parse_filters(message)
    except Exception as e:
        return jsonify({"error": f"Failed to parse your message with the LLM: {e}"}), 500

    # 2. Merge with filters collected so far in this conversation
    session["filters"].update(new_filters)

    # 3. Build the query embedding: use the uploaded image if we have one,
    #    otherwise fall back to embedding the text itself (pure text search)
    if session["base_embedding"] is not None:
        query_embedding = session["base_embedding"]
    else:
        query_embedding = embed_text(message)

    where_clause = filters_to_chroma_where(session["filters"])
    results = search(query_embedding, top_k=8, filters=where_clause)

    return jsonify({
        "reply": f"Here's what I found matching: {session['filters']}",
        "applied_filters": session["filters"],
        "results": results,
    })


@app.route("/image/<item_id>", methods=["GET"])
def get_image(item_id):
    """
    Serves the actual image file for a catalog item, no matter where it
    lives on disk (static/uploads for manually-uploaded items, or your
    original 40k-image dataset folder for bulk-seeded items).
    """
    collection = get_collection()
    result = collection.get(ids=[item_id])
    if not result["ids"]:
        return jsonify({"error": "item not found"}), 404

    image_path = result["metadatas"][0]["image_path"]
    if not os.path.exists(image_path):
        return jsonify({"error": "image file missing on disk"}), 404

    return send_file(image_path)


@app.route("/session/reset", methods=["POST"])
def reset_session():
    """Clear a conversation's memory (new search from scratch)."""
    session_id = request.get_json().get("session_id", "default")
    SESSIONS.pop(session_id, None)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
