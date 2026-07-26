"""
seed_data.py
------------
Bulk-add fashion items to your catalog without writing curl commands.

HOW TO USE:
1. Put your fashion images in backend/sample_images/  (any .jpg/.png files)
2. Edit sample_items.csv to describe each image (filename, color, occasion, category, price)
3. Make sure app.py is already running (python app.py) in another terminal
4. Run: python seed_data.py

This just calls your own /upload API repeatedly - it's a convenience
script, not a separate system.
"""

import csv
import os
import requests

API_BASE = "http://localhost:5000"
IMAGES_FOLDER = "sample_images"
CSV_FILE = "sample_items.csv"


def seed():
    if not os.path.exists(CSV_FILE):
        print(f"'{CSV_FILE}' not found. Create it first (see sample_items.csv template).")
        return

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in CSV. Add some items first.")
        return

    print(f"Found {len(rows)} items to upload...")
    success, failed = 0, 0

    for row in rows:
        image_path = os.path.join(IMAGES_FOLDER, row["filename"])
        if not os.path.exists(image_path):
            print(f"  [skip] Image not found: {image_path}")
            failed += 1
            continue

        with open(image_path, "rb") as img_file:
            files = {"image": img_file}
            data = {
                "color": row.get("color", "unknown"),
                "occasion": row.get("occasion", "unknown"),
                "category": row.get("category", "unknown"),
                "price": row.get("price", "0"),
            }
            try:
                res = requests.post(f"{API_BASE}/upload", files=files, data=data, timeout=30)
                if res.status_code == 200:
                    print(f"  [ok] {row['filename']} -> {res.json()['item_id']}")
                    success += 1
                else:
                    print(f"  [fail] {row['filename']} -> {res.status_code} {res.text}")
                    failed += 1
            except requests.exceptions.ConnectionError:
                print("ERROR: Can't reach the backend. Is 'python app.py' running in another terminal?")
                return

    print(f"\nDone. {success} uploaded, {failed} failed/skipped.")


if __name__ == "__main__":
    seed()
