"""Build the cross-collection corpus shipped with Hadith Finder.

Source: per-collection JSON dumps (Bukhari, Muslim, Abu Dawud, Nasa'i, Ibn
Majah) with the shape
    {"all_books": [{"english_title": ..., "hadith_list": [
        {"title": "Sahih al-Bukhari 1", "narrator": ...,
         "english_text": ..., "arabic_text": ..., "local_num": 1}, ...]}]}

Output (compressed, committed):
    hadith_sidecar/data/crossref.json.gz
        {"version": 1,
         "collections": [{"slug","name","short"}, ...],
         "hadiths": [[collection_index, number, book_title, english], ...]}

Only the English matn is shipped: it is what the cross-reference matcher
compares against (token containment), and it keeps the corpus ~5 MB
compressed. Arabic is intentionally left out — see crossref.py.

Usage:
    py -3.13 tools/build_crossref.py --source "G:/models tester/.cache/hadith_sources"
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys

COLLECTIONS = [
    ("bukhari", "Sahih al-Bukhari", "Bukhari", "sahih_bukhari.json"),
    ("muslim", "Sahih Muslim", "Muslim", "sahih_muslim.json"),
    ("abu_dawud", "Sunan Abi Dawud", "Abu Dawud", "sunan_abi_dawud.json"),
    ("nasai", "Sunan an-Nasa'i", "Nasa'i", "sunan_an_nasai.json"),
    ("ibn_majah", "Sunan Ibn Majah", "Ibn Majah", "sunan_ibn_majah.json"),
]

MAX_TEXT = 600
_WS = re.compile(r"\s+")


def clean(text: str, limit: int = MAX_TEXT) -> str:
    t = _WS.sub(" ", (text or "").replace("\u00a0", " ")).strip()
    if len(t) > limit:
        cut = t[:limit]
        # prefer ending on a sentence boundary when one is close enough
        for mark in (". ", "! ", "? ", "; "):
            pos = cut.rfind(mark)
            if pos > limit * 0.6:
                return cut[: pos + 1].strip()
        return cut.rstrip() + "…"
    return t


def build(source_dir: str) -> dict:
    collections = []
    hadiths: list[list] = []
    for idx, (slug, name, short, filename) in enumerate(COLLECTIONS):
        path = os.path.join(source_dir, filename)
        if not os.path.isfile(path):
            print(f"warning: missing {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        collections.append({"slug": slug, "name": name, "short": short})
        coll_index = len(collections) - 1
        count = 0
        for book in data.get("all_books", []):
            book_title = clean(book.get("english_title") or "", 90)
            for hadith in book.get("hadith_list", []):
                text = clean(hadith.get("english_text") or "")
                if len(text) < 40:
                    continue
                # `local_num` restarts inside every book; the collection-wide
                # number is the one printed in the title ("Sahih al-Bukhari 52").
                number = 0
                m = re.search(r"(\d+)\s*$", str(hadith.get("title") or ""))
                if m:
                    number = int(m.group(1))
                if number <= 0:
                    try:
                        number = int(hadith.get("local_num") or 0)
                    except (TypeError, ValueError):
                        number = 0
                if number <= 0:
                    continue
                hadiths.append([coll_index, number, book_title, text])
                count += 1
        print(f"{name}: {count} narrations")
    hadiths.sort(key=lambda h: (h[0], h[1]))
    return {"version": 1, "collections": collections, "hadiths": hadiths}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "..", "hadith_sidecar", "data", "crossref.json.gz")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="folder with the collection JSON dumps")
    ap.add_argument("--out", default=os.path.normpath(default_out))
    args = ap.parse_args()

    payload = build(args.source)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(args.out, "wb", compresslevel=9) as fh:
        fh.write(raw)
    print(f"wrote {args.out}: {len(payload['hadiths'])} narrations, "
          f"{len(raw) / 1e6:.1f} MB raw, {os.path.getsize(args.out) / 1e6:.1f} MB gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
