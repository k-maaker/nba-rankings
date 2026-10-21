#!/usr/bin/env python3
"""Download the BigDataBall NBA Dropbox folder and keep only the newest NBA
season file in data/bigdataball/.

The shared-folder link comes from the DROPBOX_FOLDER_URL secret. Files are
named like "10-20-2026-nba-season-team-feed.xlsx"; the newest is picked by
the date in the name (not alphabetically, since the season runs into the
next calendar year). A failed download never fails the run: the model just
keeps the games it already has, and the next run catches up.
"""
import datetime as dt
import os
import re
import shutil
import sys
import urllib.request
import zipfile

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bigdataball")
TMP_ZIP = "/tmp/bdb_nba.zip"
TMP_DIR = "/tmp/bdb_nba"
DATE_RE = re.compile(r"(\d{1,2})-(\d{1,2})-(\d{4})")


def file_date(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return dt.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


def is_nba_feed(name: str) -> bool:
    n = name.lower()
    return n.endswith((".xlsx", ".xlsm")) and "nba" in n and "wnba" not in n


def main() -> None:
    url = os.environ.get("DROPBOX_FOLDER_URL", "").strip()
    if not url:
        print("No DROPBOX_FOLDER_URL secret set; skipping Dropbox pull.")
        return
    url = re.sub(r"([?&])dl=0", r"\1dl=1", url)
    if "dl=1" not in url:
        url += ("&" if "?" in url else "?") + "dl=1"

    print("Downloading Dropbox folder...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(TMP_ZIP, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:  # noqa: BLE001
        print(f"!! Download failed: {e}. Keeping the games we already have.")
        return
    print(f"Downloaded {os.path.getsize(TMP_ZIP):,} bytes.")

    if not zipfile.is_zipfile(TMP_ZIP):
        print("!! Download is not a zip. The Dropbox link may be wrong or expired. "
              "First 200 bytes:")
        with open(TMP_ZIP, "rb") as f:
            print(f.read(200))
        return

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR)
    with zipfile.ZipFile(TMP_ZIP) as z:
        z.extractall(TMP_DIR)

    found = []
    for root, _, files in os.walk(TMP_DIR):
        for fn in files:
            print(f"  in folder: {fn}")
            if is_nba_feed(fn) and file_date(fn):
                found.append((file_date(fn), os.path.join(root, fn)))
    if not found:
        print("!! No dated NBA BigDataBall .xlsx found in the folder.")
        return

    newest_date, newest = max(found)
    os.makedirs(DEST_DIR, exist_ok=True)
    for old in os.listdir(DEST_DIR):
        if old.lower().endswith((".xlsx", ".xlsm")):
            os.remove(os.path.join(DEST_DIR, old))
    shutil.copy(newest, DEST_DIR)
    print(f"Using: {os.path.basename(newest)} (dated {newest_date})")


if __name__ == "__main__":
    main()
    sys.exit(0)
