"""
vectorstore.py
--------------
ChromaDB is a free, open-source, local vector database — no server,
no API key, no cost. It stores each fashion item's CLIP embedding
plus metadata (color, occasion, price, image path), and lets us
search by "nearest neighbor" (most visually/semantically similar items).

Data is persisted to backend/data/chroma_db/ so it survives restarts.
"""

import chromadb
from chromadb.config import Settings

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "fashion_items"

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity fits CLIP embeddings
        )
    return _collection


def add_item(item_id: str, embedding: list[float], metadata: dict):
    """
    Store one fashion item.
    metadata example:
        {
          "image_path": "static/uploads/saree1.jpg",
          "color": "red",
          "occasion": "wedding",
          "category": "saree",
          "price": 2500
        }
    """
    collection = get_collection()
    collection.add(
        ids=[item_id],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def search(query_embedding: list[float], top_k: int = 8, filters: dict | None = None):
    """
    Find the top_k most similar items to query_embedding.
    filters (optional) narrows results using metadata, e.g.
        {"color": "blue"}  -> only blue items
        {"$and": [{"color": "blue"}, {"occasion": "wedding"}]}
    """
    collection = get_collection()
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],  # lower = more similar
        })
    return items


def count_items() -> int:
    return get_collection().count()


if __name__ == "__main__":
    # Quick sanity test with a fake embedding (no CLIP model needed)
    import random
    fake_vec = [random.random() for _ in range(512)]
    add_item(
        item_id="test-1",
        embedding=fake_vec,
        metadata={"color": "blue", "occasion": "wedding", "category": "saree", "price": 2500},
    )
    print("Items in DB:", count_items())
    results = search(fake_vec, top_k=3)
    print("Search results:", results)
