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


ZIP_PREFIX = "WikiTables-WithLinks-master"


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


def load_json_from_zip(zf: zipfile.ZipFile, path_in_zip: str):
    try:
        with zf.open(path_in_zip) as f:
            return json.load(f)
    except KeyError:
        return None


def parse_table(table_json: dict) -> dict:
    headers = [cell[0] for cell in table_json.get("header", [])]
    rows = []
    all_links = set()
    for row_data in table_json.get("data", []):
        row = {}
        row_links = []
        for i, cell in enumerate(row_data):
            col_name = headers[i] if i < len(headers) else f"col_{i}"
            text = cell[0] if isinstance(cell, list) else str(cell)
            links = cell[1] if isinstance(cell, list) and len(cell) > 1 else []
            row[col_name] = text
            row_links.extend(links)
            all_links.update(links)
        row["_links"] = row_links
        rows.append(row)

    return {
        "title": table_json.get("title", ""),
        "section_title": table_json.get("section_title", ""),
        "section_text": table_json.get("section_text", ""),
        "intro": table_json.get("intro", ""),
        "headers": headers,
        "rows": rows,
        "num_rows": len(rows),
        "all_links": sorted(all_links),
    }


def get_linked_passages(all_links: list, passages_json: dict, max_passages: int) -> list[dict]:
    if not passages_json:
        return []
    entity_links = []
    generic_links = []
    for link in all_links:
        name = link.split("/")[-1]
        if any(name.startswith(str(y)) for y in range(1800, 2100)):
            generic_links.append(link)
        else:
            entity_links.append(link)
    ordered = entity_links + generic_links

    result = []
    for link in ordered[:max_passages]:
        text = passages_json.get(link, "")
        if text:
            result.append({"link": link, "text": text})
    return result


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
        table_parsed = parse_table(table_json)
        linked_passages = get_linked_passages(
            table_parsed["all_links"], passages_json, cfg.MAX_LINKED_PASSAGES * 10
        )
        parsed.append(
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "answer": q["answer-text"],
                "table_id": table_id,
                "table": table_parsed,
                "linked_passages": linked_passages,
                "num_linked_passages": len(linked_passages),
                "split": split,
            }
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
