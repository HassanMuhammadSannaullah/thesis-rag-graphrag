"""
Script 02: Parse HybridQA + WikiTables into structured JSONL files.

Supports:
  - dev only
  - train only
  - both splits

Outputs:
  data/hybridqa/original/{split}.jsonl
  data/hybridqa/samples/{split}_sample.jsonl
"""
import argparse
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jsonlines

from src.config import settings as cfg
from src.data_pipeline.hybridqa_parser import ZIP_PREFIX, build_hybridqa_record, load_json_from_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "train", "all"],
        default="dev",
        help="Which HybridQA split to parse.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=cfg.DEV_SAMPLE_SIZE,
        help="How many parsed records to save into the sample file for each split.",
    )
    return parser.parse_args()

def parse_split(split: str, zf: zipfile.ZipFile, sample_size: int) -> None:
    input_path = cfg.RAW_DIR / f"{split}.json"
    print(f"Loading {input_path.name} ...")
    with open(input_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  {len(questions)} questions loaded")

    print(f"Parsing {split} questions ...")
    parsed = []
    skipped = 0
    for i, q in enumerate(questions):
        table_id = q["table_id"]
        table_path = f"{ZIP_PREFIX}/tables_tok/{table_id}.json"
        request_path = f"{ZIP_PREFIX}/request_tok/{table_id}.json"

        table_json = load_json_from_zip(zf, table_path)
        if table_json is None:
            skipped += 1
            continue

        passages_json = load_json_from_zip(zf, request_path) or {}
        parsed.append(
            build_hybridqa_record(
                question_payload=q,
                table_json=table_json,
                passages_json=passages_json,
                split=split,
            )
        )

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(questions)} ...")

    print(f"  Parsed: {len(parsed)}, Skipped (no table): {skipped}")

    cfg.ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    full_path = cfg.ORIGINAL_DIR / f"{split}.jsonl"
    with jsonlines.open(str(full_path), mode="w") as writer:
        writer.write_all(parsed)
    print(f"Saved full {split} set: {full_path} ({len(parsed)} records)")

    cfg.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample = parsed[:sample_size]
    sample_path = cfg.SAMPLES_DIR / f"{split}_sample.jsonl"
    with jsonlines.open(str(sample_path), mode="w") as writer:
        writer.write_all(sample)
    print(f"Saved sample: {sample_path} ({len(sample)} records)")


def main() -> None:
    args = parse_args()
    splits = ["dev", "train"] if args.split == "all" else [args.split]

    print("Opening WikiTables ZIP ...")
    zf = zipfile.ZipFile(str(cfg.RAW_DIR / "WikiTables-WithLinks.zip"))
    try:
        for split in splits:
            print(f"\n=== Parsing split: {split} ===")
            parse_split(split, zf, args.sample_size)
    finally:
        zf.close()
    print("\n=== Parsing complete ===")


if __name__ == "__main__":
    main()
