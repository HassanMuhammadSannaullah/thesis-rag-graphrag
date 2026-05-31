"""
Script 01: Download HybridQA dataset files.

Downloads the official HybridQA question JSON files from the GitHub repository.
Also downloads the WikiTables-WithLinks ZIP (tables + passages) from GitHub.

Output:
  data/raw/hybridqa/train.json
  data/raw/hybridqa/dev.json
  data/raw/hybridqa/WikiTables-WithLinks.zip
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from pathlib import Path
from src.config import settings as cfg

RAW = cfg.RAW_DIR
RAW.mkdir(parents=True, exist_ok=True)

# ── HybridQA question files ─────────────────────────────────────────
HYBRIDQA_BASE = "https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data"
FILES = {
    "train.json": f"{HYBRIDQA_BASE}/train.json",
    "dev.json":   f"{HYBRIDQA_BASE}/dev.json",
}

def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name} from {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Saved {dest.name} ({len(resp.content) / 1024:.0f} KB)")

print("=== Downloading HybridQA question files ===")
for fname, url in FILES.items():
    download_file(url, RAW / fname)

# ── WikiTables-WithLinks ZIP ────────────────────────────────────────
# Git clone fails on Windows due to invalid filenames in the repo.
# Use the GitHub ZIP archive instead.
WIKITABLES_ZIP_URL = "https://github.com/wenhuchen/WikiTables-WithLinks/archive/refs/heads/master.zip"
WIKITABLES_ZIP = RAW / "WikiTables-WithLinks.zip"

print("\n=== Downloading WikiTables-WithLinks ZIP ===")
if WIKITABLES_ZIP.exists():
    print(f"  Already exists: {WIKITABLES_ZIP.name}")
else:
    print(f"  Downloading ZIP (~200 MB, may take a few minutes) ...")
    resp = requests.get(WIKITABLES_ZIP_URL, timeout=600, stream=True)
    resp.raise_for_status()
    total = 0
    with open(WIKITABLES_ZIP, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            total += len(chunk)
            print(f"\r  Downloaded {total / (1024*1024):.0f} MB", end="", flush=True)
    print(f"\n  Saved WikiTables-WithLinks.zip ({total / (1024*1024):.0f} MB)")

print("\n=== Download complete ===")
print(f"Files in {RAW}:")
for p in sorted(RAW.iterdir()):
    size = p.stat().st_size / (1024 * 1024)
    print(f"  {p.name}  ({size:.1f} MB)")
