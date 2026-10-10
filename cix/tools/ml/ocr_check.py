"""Run OCR through an immich_ml server and print it next to the text stored in the database.

Usage: ocr_check.py http://127.0.0.1:3003 [--model PP-OCRv5_mobile] [--limit 6]
"""

import argparse
import json
import time

import requests
from _db import psql

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("--model", default="PP-OCRv5_mobile")
parser.add_argument("--limit", type=int, default=6)
args = parser.parse_args()

entries = {
    "ocr": {
        "detection": {"modelName": args.model, "options": {"minScore": 0.5, "maxResolution": 736}},
        "recognition": {"modelName": args.model, "options": {"minScore": 0.8}},
    }
}
# OCR text can contain "|" and line breaks, so use a separator text never has and flatten the whitespace
rows = psql(
    f"""with t as (select "assetId" from asset_ocr group by 1 order by random() limit {args.limit})
        select f.path, (select regexp_replace(string_agg(text, ' / '), '\\s+', ' ', 'g')
                        from asset_ocr o where o."assetId"=t."assetId")
        from t join asset_file f on f."assetId"=t."assetId" and f.type='preview' and f."isEdited"=false""",
    sep="\x1f",
)
for path, stored in rows:
    t = time.time()
    r = requests.post(
        f"{args.url}/predict", data={"entries": json.dumps(entries)}, files={"image": open(path, "rb")}, timeout=600
    )
    text = r.json().get("ocr", {}).get("text", [])
    print(f"DB : {stored[:120]}\nNEW: {' / '.join(text)[:120]}   ({(time.time() - t) * 1000:.0f} ms)\n")
