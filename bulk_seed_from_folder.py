#cd A:\GENAI\stylegpt\backend
# venv\Scripts\activate
# python bulk_seed_from_folder.py

"""
bulk_seed_from_folder.py
-------------------------
For large, unlabeled datasets (e.g. 40k numbered images, no metadata).
Auto-tags each image using CLIP zero-shot classification (see auto_tag.py)
and adds it directly to ChromaDB - no need to copy files, no CSV to write
by hand, no HTTP calls (much faster than going through /upload for 40k items).

IMPORTANT: This will take a while on CPU (roughly 0.1-0.3 sec/image just
for the search embedding, similar again for tagging = expect ~1-3 hours
for 40,000 images). It's SAFE TO STOP (Ctrl+C) and re-run any time -
already-processed images are automatically skipped.

HOW TO USE:
1. Set DATASET_FOLDER below to your actual dataset path
2. (Recommended) First set LIMIT = 200 and do a test run to sanity-check
   the auto-tagging quality before committing to all 40k
3. Once happy, set LIMIT = None and run it (can take hours - let it run
   in the background, e.g. overnight)
4. Run: python bulk_seed_from_folder.py
"""

import os
import time
import glob

from embeddings import embed_image
from vectorstore import get_collection, count_items
from auto_tag import auto_tag

# ── EDIT THESE TWO LINES ──────────────────────────────────────────────
DATASET_FOLDER = r"A:\DATASET\archive\fashion-dataset\images"   # <-- CHANGE to your actual folder path
LIMIT = None                                    # <-- set to None to process ALL images
# ───────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
LOG_FILE = "bulk_seed_errors.log"


def already_processed(collection, item_id: str) -> bool:
    result = collection.get(ids=[item_id])
    return len(result["ids"]) > 0


def run():
    if not os.path.isdir(DATASET_FOLDER):
        print(f"ERROR: DATASET_FOLDER does not exist: {DATASET_FOLDER}")
        print("Open bulk_seed_from_folder.py and fix the DATASET_FOLDER path at the top.")
        return

    image_paths = sorted(
        p for p in glob.glob(os.path.join(DATASET_FOLDER, "*"))
        if p.lower().endswith(VALID_EXTENSIONS)
    )
    if LIMIT:
        image_paths = image_paths[:LIMIT]

    if not image_paths:
        print(f"No images found in {DATASET_FOLDER}")
        return

    print(f"Found {len(image_paths)} images to process.")
    print(f"Already in catalog: {count_items()}")

    collection = get_collection()
    start_time = time.time()
    processed, skipped, failed = 0, 0, 0

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        for i, path in enumerate(image_paths, start=1):
            item_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "1539"

            if already_processed(collection, item_id):
                skipped += 1
                continue

            try:
                embedding = embed_image(path)
                tags = auto_tag(embedding)
                metadata = {
                    "image_path": os.path.abspath(path),
                    "color": tags["color"],
                    "occasion": tags["occasion"],
                    "category": tags["category"],
                    "price": tags["price"],
                }
                collection.add(ids=[item_id], embeddings=[embedding], metadatas=[metadata])
                processed += 1
            except Exception as e:
                failed += 1
                log.write(f"{path}: {e}\n")

            if i % 50 == 0 or i == len(image_paths):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(image_paths) - i) / rate if rate > 0 else 0
                print(
                    f"[{i}/{len(image_paths)}] processed={processed} "
                    f"skipped={skipped} failed={failed} "
                    f"| {rate:.1f} img/s | ETA {remaining/60:.1f} min"
                )

    print("\nDone.")
    print(f"  Newly added: {processed}")
    print(f"  Skipped (already in catalog): {skipped}")
    print(f"  Failed (see {LOG_FILE}): {failed}")
    print(f"  Total items now in catalog: {count_items()}")


if __name__ == "__main__":
    run()
