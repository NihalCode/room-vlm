#!/usr/bin/env python3
"""Download NYU Depth V2 category archives (official URLs only)."""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

from room_vlm.constants import NYU_DATASET_PAGE, NYU_OFFICIAL_BASE_URL, NYU_TARGET_ARCHIVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_nyu")


def official_manifest() -> list[dict]:
    """Fallback manifest of official horatio.cs.nyu.edu archives."""
    return [
        {
            "name": name,
            "url": f"{NYU_OFFICIAL_BASE_URL}/{name}",
            "md5_url": f"{NYU_OFFICIAL_BASE_URL}/{name}.md5",
            "category": name.split("_part")[0].replace(".zip", ""),
        }
        for name in NYU_TARGET_ARCHIVES
    ]


def discover_from_page(timeout: float = 30.0) -> list[dict] | None:
    """Best-effort discovery of official archive links from the NYU dataset page."""
    try:
        resp = requests.get(NYU_DATASET_PAGE, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Could not fetch NYU page (%s); using fallback manifest", exc)
        return None

    # Look for horatio links to our target categories.
    urls = set(re.findall(r"https?://horatio\.cs\.nyu\.edu/[^\s\"'<>]+?\.zip", resp.text))
    # Also relative links
    for match in re.findall(r"href=[\"']([^\"']+\.zip)[\"']", resp.text, flags=re.I):
        urls.add(urljoin(NYU_DATASET_PAGE, match))

    targets = set(NYU_TARGET_ARCHIVES)
    found = []
    for url in sorted(urls):
        name = url.rstrip("/").split("/")[-1]
        if name in targets:
            found.append(
                {
                    "name": name,
                    "url": url,
                    "md5_url": url + ".md5",
                    "category": name.split("_part")[0].replace(".zip", ""),
                }
            )
    if len(found) < len(targets) // 2:
        logger.warning("Page discovery incomplete (%d archives); using fallback manifest", len(found))
        return None
    return found


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_expected_md5(md5_url: str) -> str | None:
    try:
        resp = requests.get(md5_url, timeout=30)
        if resp.status_code != 200:
            return None
        text = resp.text.strip().split()[0]
        if re.fullmatch(r"[a-fA-F0-9]{32}", text):
            return text.lower()
    except Exception:
        return None
    return None


def download_with_resume(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing = dest.stat().st_size if dest.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        logger.info("Resuming %s from byte %d", dest.name, existing)

    with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
        if resp.status_code == 416:
            logger.info("Already complete: %s", dest.name)
            return
        resp.raise_for_status()
        with dest.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/archives"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=["bathrooms", "bedrooms", "kitchens", "living_rooms"],
        help="Category prefixes to download",
    )
    args = parser.parse_args(argv)

    items = discover_from_page() or official_manifest()
    selected = [
        it
        for it in items
        if any(it["name"].startswith(cat.rstrip("s")) or it["category"].startswith(cat.rstrip("s")) for cat in args.categories)
        or any(cat in it["name"] for cat in args.categories)
    ]
    # Simpler filter: name contains category keyword
    selected = []
    for it in items:
        for cat in args.categories:
            key = cat.replace(" ", "_").lower()
            if key in it["name"] or key.rstrip("s") in it["name"]:
                selected.append(it)
                break

    print("NYU Depth V2 download plan")
    print(f"  categories: {', '.join(args.categories)}")
    print(f"  archive parts: {len(selected)}")
    print(f"  destination: {args.dest.resolve()}")
    print("  estimated size: large (tens of GB for these categories; not measured here)")
    print("  source: official NYU / horatio.cs.nyu.edu only")
    for it in selected:
        print(f"    - {it['name']}: {it['url']}")

    if args.dry_run:
        print("Dry run only — nothing downloaded.")
        return 0

    if not args.yes:
        answer = input("Proceed with download? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    for it in selected:
        dest = args.dest / it["name"]
        logger.info("Downloading %s", it["url"])
        download_with_resume(it["url"], dest)
        expected = fetch_expected_md5(it["md5_url"])
        if expected:
            actual = md5_file(dest)
            if actual != expected:
                logger.error("MD5 mismatch for %s: expected %s got %s", dest.name, expected, actual)
                return 2
            logger.info("MD5 OK for %s", dest.name)
        else:
            logger.warning("No MD5 available for %s — skipped checksum", dest.name)

    print("Download complete. Extract archives into a raw scene directory, then run prepare_nyu.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
